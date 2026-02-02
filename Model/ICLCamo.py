import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from Model.decoder import Decoder


# ======================================= Basic ========================================
class BasicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


def get_masked_features(x, masks):
    sample_num, sample_size = x.size()[0], x.size()[2:]
    fore_list, back_list = [], []
    for i in range(sample_num):
        mask = F.interpolate(masks[i].unsqueeze(0), size=sample_size, mode='bilinear', align_corners=False)
        fore_list.append((x[i].unsqueeze(0) * mask).squeeze(0))
        back_list.append((x[i].unsqueeze(0) * (1 - mask)).squeeze(0))
    return torch.stack(fore_list, dim=0), torch.stack(back_list, dim=0)


class TinyNet(nn.Module):
    def __init__(self, in_channel=3, mid_channel=32, out_channel=64):
        super(TinyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.GELU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.GELU(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU())
    
    def forward(self, x):
        return self.net(x)
# ======================================= Basic ========================================

# ======================================== CrossAttention ========================================
class FeedForwardLayer(nn.Module):
    def __init__(self, dim, hidden_dim=64, dropout=0.):
        super(FeedForwardLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, hid_dim=64, dropout=0., use_sdpa=True):
        super(Attention, self).__init__()
        self.heads = heads
        assert hid_dim % heads == 0
        dim_head = hid_dim // heads
        self.scale = dim_head ** -0.5
        self.use_sdpa = use_sdpa  # use SDPA or not
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, hid_dim)
        self.to_k = nn.Linear(dim, hid_dim)
        self.to_v = nn.Linear(dim, hid_dim)
        self.to_out = nn.Sequential(nn.Linear(hid_dim, dim), nn.Dropout(dropout))

    def forward(self, q, k, v):
        # q: fore/back feature
        # k: reference feature
        # v: target feature
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'n b (h d) -> b h n d', h=self.heads), (q, k, v))
        
        if self.use_sdpa:
            # q = q * self.scale
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> n b (h d)')
        return self.to_out(out)


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim, dropout=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, hid_dim=hidden_dim, dropout=dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForwardLayer(dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, masked, reference, target):
        patch_size = target.shape[2]
        target = rearrange(target, 'b c h w -> (h w) b c')
        reference = rearrange(reference, 'b c h w -> (h w) b c')
        masked = rearrange(masked, 'b c h w -> (h w) b c')

        target = target + self.attn(masked, reference, target)
        target = self.attn_norm(target)
        target = target + self.ffn(target)
        target = self.ffn_norm(target)

        target = rearrange(target, '(h w) b c -> b c h w', h=patch_size)

        return target
# ======================================== CrossAttention ========================================

# ======================================== Model ========================================
class ICLCamo(nn.Module):
    def __init__(self, n_layers=[2, 5, 8, 11], hid_dim=64, dropout=0.1):
        super(ICLCamo, self).__init__()

        self.target_net = torch.hub.load('/root/autodl-tmp/dinov2', 'dinov2_vitb14',source='local', pretrained=False)
        self.target_net.load_state_dict(torch.load('/root/autodl-tmp/dinov2/dinov2_vitb14_pretrain.pth'))
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        self.reference_net = torch.hub.load('/root/autodl-tmp/dinov2', 'dinov2_vitb14', source='local', pretrained=False)
        self.reference_net.load_state_dict(torch.load('/root/autodl-tmp/dinov2/dinov2_vitb14_pretrain.pth'))

        self.n_layers = n_layers
        self.hid_dim = hid_dim

        self.detail_net = TinyNet(in_channel=3, mid_channel=32, out_channel=hid_dim)

        self.tgt_conv1 = BasicConvBlock(in_channel=768, out_channel=hid_dim)
        self.tgt_conv2 = BasicConvBlock(in_channel=768, out_channel=hid_dim)
        self.tgt_conv3 = BasicConvBlock(in_channel=768, out_channel=hid_dim)
        self.tgt_conv4 = BasicConvBlock(in_channel=768, out_channel=hid_dim)
        self.ref_conv1 = BasicConvBlock(in_channel=768, out_channel=hid_dim)
        self.ref_conv2 = BasicConvBlock(in_channel=768, out_channel=hid_dim)
        self.ref_conv3 = BasicConvBlock(in_channel=768, out_channel=hid_dim)
        self.ref_conv4 = BasicConvBlock(in_channel=768, out_channel=hid_dim)

        self.fore_cross_attn1 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.back_cross_attn1 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.fore_cross_attn2 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.back_cross_attn2 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.fore_cross_attn3 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.back_cross_attn3 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.fore_cross_attn4 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.back_cross_attn4 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)

        self.tgt_attn1 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.tgt_attn2 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.tgt_attn3 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)
        self.tgt_attn4 = CrossTransformerBlock(dim=hid_dim, heads=8, hidden_dim=hid_dim, dropout=dropout)

        self.encode_conv1 = BasicConvBlock(in_channel=hid_dim*2, out_channel=hid_dim)
        self.encode_conv2 = BasicConvBlock(in_channel=hid_dim*2, out_channel=hid_dim)
        self.encode_conv3 = BasicConvBlock(in_channel=hid_dim*2, out_channel=hid_dim)
        self.encode_conv4 = BasicConvBlock(in_channel=hid_dim*2, out_channel=hid_dim)

        self.decode_conv1 = BasicConvBlock(in_channel=hid_dim*4, out_channel=hid_dim)

        self.decoder = Decoder(channel=hid_dim, scale_list=[112, 224, 392])

        self.mask_conv1 = nn.Conv2d(hid_dim, 1, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, target, reference, masks):
        tgt_feats = self.target_net.get_intermediate_layers(x=target, n=self.n_layers,
                                                            reshape=True,
                                                            return_class_token=False,
                                                            norm=True)
        tgt_f1, tgt_f2, tgt_f3, tgt_f4 = tgt_feats
        tgt_f1 = self.tgt_conv1(tgt_f1)
        tgt_f2 = self.tgt_conv2(tgt_f2)
        tgt_f3 = self.tgt_conv3(tgt_f3)
        tgt_f4 = self.tgt_conv4(tgt_f4)
        ref_feats = self.reference_net.get_intermediate_layers(x=reference, n=self.n_layers,
                                                               reshape=True,
                                                               return_class_token=False,
                                                               norm=True)
        ref_f1, ref_f2, ref_f3, ref_f4 = ref_feats
        ref_f1 = self.ref_conv1(ref_f1)
        ref_f2 = self.ref_conv2(ref_f2)
        ref_f3 = self.ref_conv3(ref_f3)
        ref_f4 = self.ref_conv4(ref_f4)

        detail = self.detail_net(x=target)

        ref_fore1, ref_back1 = get_masked_features(ref_f1, masks)
        ref_fore2, ref_back2 = get_masked_features(ref_f2, masks)
        ref_fore3, ref_back3 = get_masked_features(ref_f3, masks)
        ref_fore4, ref_back4 = get_masked_features(ref_f4, masks)

        tgt_fore1 = self.fore_cross_attn1(masked=ref_fore1, reference=ref_f1, target=ref_f1)
        tgt_back1 = self.back_cross_attn1(masked=ref_back1, reference=ref_f1, target=ref_f1)
        tgt_fore2 = self.fore_cross_attn2(masked=ref_fore2, reference=ref_f2, target=ref_f2)
        tgt_back2 = self.back_cross_attn2(masked=ref_back2, reference=ref_f2, target=ref_f2)
        tgt_fore3 = self.fore_cross_attn3(masked=ref_fore3, reference=ref_f3, target=ref_f3)
        tgt_back3 = self.back_cross_attn3(masked=ref_back3, reference=ref_f3, target=ref_f3)
        tgt_fore4 = self.fore_cross_attn4(masked=ref_fore4, reference=ref_f4, target=ref_f4)
        tgt_back4 = self.back_cross_attn4(masked=ref_back4, reference=ref_f4, target=ref_f4)

        ref_en_f1 = self.encode_conv1(torch.cat([tgt_fore1, tgt_back1], dim=1))
        ref_en_f2 = self.encode_conv2(torch.cat([tgt_fore2, tgt_back2], dim=1))
        ref_en_f3 = self.encode_conv3(torch.cat([tgt_fore3, tgt_back3], dim=1))
        ref_en_f4 = self.encode_conv4(torch.cat([tgt_fore4, tgt_back4], dim=1))

        tgt_en_f1 = self.tgt_attn1(masked=tgt_f1, reference=ref_en_f1, target=tgt_f1)
        tgt_en_f2 = self.tgt_attn2(masked=tgt_f2, reference=ref_en_f2, target=tgt_f2)
        tgt_en_f3 = self.tgt_attn3(masked=tgt_f3, reference=ref_en_f3, target=tgt_f3)
        tgt_en_f4 = self.tgt_attn4(masked=tgt_f4, reference=ref_en_f4, target=tgt_f4)

        context = self.decode_conv1(torch.cat([ref_en_f1, ref_en_f2, ref_en_f3, ref_en_f4], dim=1))

        de_f1, _, _, _ = self.decoder(x=[tgt_en_f1, tgt_en_f2, tgt_en_f3, tgt_en_f4],
                                      context=context, detail=detail)

        mask1 = self.mask_conv1(de_f1)

        return mask1

    def inference(self, target, reference, masks):
        self.eval()
        with torch.no_grad():
            prediction = self.forward(target, reference, masks)
        return prediction
    
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
# ======================================== Model ========================================    
        
if __name__ == '__main__':
    # input = torch.rand(8, 3, 392, 392).to('cuda')
    # reference = torch.rand(8, 3, 392, 392).to('cuda')
    # masks = torch.rand(8, 1, 392, 392).to('cuda')
    
    # model = ICLCamo(n_layers=[2, 5, 8, 11], hid_dim=64, dropout=0.1).to('cuda')
    # model.eval()

    # output = model(input, reference, masks)
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(output[3].shape)

    # Use torchinfo to print model summary
    from torchinfo import summary
    model = ICLCamo()
    summary(model, input_size=[(1, 3, 392, 392), (1, 3, 392, 392), (1, 1, 392, 392)])
    