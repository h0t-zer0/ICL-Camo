import torch
import torch.nn.functional as F
import numpy as np
from utils.dataloader import TestDataset
from tqdm import tqdm
import os
import cv2


def inference(dataset, model, cfg):
    model.eval()
    save_path = f'{cfg.save_dir}/{dataset}'
    os.makedirs(save_path, exist_ok=True)

    test_dataset = TestDataset(image_root=getattr(cfg.dp, f'test_{dataset}_imgs'),
                               gt_root=getattr(cfg.dp, f'test_{dataset}_masks'),
                               ref_image_root=cfg.dp.train_imgs,
                               ref_gt_root=cfg.dp.train_masks,
                               test_size=cfg.trainsize)

    for _, ori_gt, name, img, _, ref_img, ref_mask in tqdm(test_dataset):
        img = img.unsqueeze(0).cuda()
        ref_img = ref_img.unsqueeze(0).cuda()
        ref_mask = ref_mask.unsqueeze(0).cuda()
        p = model.inference(img, ref_img, ref_mask)
        p = F.interpolate(p, size=ori_gt.shape[1:], mode='bilinear', align_corners=False)
        p = torch.sigmoid(p) * 255
        p = p.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        # save preds
        cv2.imwrite(os.path.join(save_path, name), p)


if __name__ == '__main__':
    from utils.config import Config
    from Model.ICLCamo import ICLCamo

    cfg = Config()

    pth_path = 'path to your trained model checkpoint'

    model = ICLCamo(n_layers=[2, 5, 8, 11], hid_dim=64, dropout=0.1)
    model.load_checkpoint(path=pth_path)
    model.to('cuda')

    datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
    for dataset in datasets:
        inference(dataset=dataset, model=model, cfg=cfg)
