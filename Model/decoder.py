import torch.nn as nn
import torch.nn.functional as F


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

                                              
class Decoder(nn.Module):
    def __init__(self, channel, scale_list=[112, 224, 392]):
        super(Decoder, self).__init__()
        self.scale_list = scale_list
        self.de_conv_4 = BasicConvBlock(in_channel=channel, out_channel=channel)
        self.de_conv_3 = BasicConvBlock(in_channel=channel, out_channel=channel)
        self.de_conv_2 = BasicConvBlock(in_channel=channel, out_channel=channel)
        self.de_conv_1 = BasicConvBlock(in_channel=channel, out_channel=channel)
    def forward(self, x, context, detail):
        x1, x2, x3, x4 = x
        c1, c2, c3, c4 = context, context, context, context
        d1, d2, d3, d4 = detail, detail, detail, detail
        o4 = self.de_conv_4(F.interpolate(x4, size=self.scale_list[0], mode='bilinear', align_corners=False) +
                            F.interpolate(c4, size=self.scale_list[0], mode='bilinear', align_corners=False) +
                            F.interpolate(d4, size=self.scale_list[0], mode='bilinear', align_corners=False))
        o3 = self.de_conv_3(F.interpolate(x3, size=self.scale_list[1], mode='bilinear', align_corners=False) +
                            F.interpolate(c3, size=self.scale_list[1], mode='bilinear', align_corners=False) +
                            F.interpolate(d3, size=self.scale_list[1], mode='bilinear', align_corners=False) +
                            F.interpolate(o4, size=self.scale_list[1], mode='bilinear', align_corners=False))
        o2 = self.de_conv_2(F.interpolate(x2, size=self.scale_list[2], mode='bilinear', align_corners=False) +
                            F.interpolate(c2, size=self.scale_list[2], mode='bilinear', align_corners=False) +
                            F.interpolate(d2, size=self.scale_list[2], mode='bilinear', align_corners=False) +
                            F.interpolate(o3, size=self.scale_list[2], mode='bilinear', align_corners=False))
        o1 = self.de_conv_1(F.interpolate(x1, size=self.scale_list[2], mode='bilinear', align_corners=False) +
                            F.interpolate(c1, size=self.scale_list[2], mode='bilinear', align_corners=False) +
                            F.interpolate(d1, size=self.scale_list[2], mode='bilinear', align_corners=False) +
                            o2)
        return o1, o2, o3, o4
