import torch
import random
import numpy as np
import torch.nn.functional as F
from utils.dataloader import TrainDataset
from utils.utils import CosineDecay
from tqdm import tqdm
import os


def structure_loss(logits, mask):
    """
    loss function (ref: F3Net-AAAI-2020)

    pred: logits without activation
    mask: binary mask {0, 1}
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class Trainer:
    def __init__(self, cfg, model, train_datald, optimizer, scheduler=None):
        self.cfg = cfg
        self.model = model
        self.train_datald = train_datald
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.if_scheduler = True if scheduler is not None else False
        self.current_epoch = 1
        self.total_steps = len(train_datald)

    def train_epoch(self):
        self.model.train()
        steps = 0
        print(f'[Train] Epoch: {self.current_epoch}, lr: {self.scheduler.get_lr()}')
        for _, img, mask, ref_img, ref_mask in tqdm(self.train_datald):
            self.optimizer.zero_grad()

            img = img.to(self.cfg.device)
            mask = mask.to(self.cfg.device)
            ref_img = ref_img.to(self.cfg.device)
            ref_mask = ref_mask.to(self.cfg.device)

            p1 = self.model(img, ref_img, ref_mask)

            loss = structure_loss(p1, mask)
            loss.backward()
            self.optimizer.step()

            steps += 1
            if steps % self.cfg.log_interval == 0:
                print(f'[Train {steps}/{self.total_steps}] Epoch: {self.current_epoch}, Loss: {loss.item()}')
        
        if self.if_scheduler:
            self.scheduler.step()
            
        if self.current_epoch >= cfg.epochs - 10:
            torch.save(self.model.state_dict(), f'{self.cfg.save_dir}/epoch_{self.current_epoch}.pth')
        
        self.current_epoch += 1
    
    def start_training(self):
        print('Start training...')
        for epoch in range(self.cfg.epochs):
            self.train_epoch()
        return self.model


if __name__ == '__main__':
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    from utils.config import Config
    from Model.ICLCamo import ICLCamo

    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    train_dataset = TrainDataset(image_root=cfg.dp.train_imgs,
                                 gt_root=cfg.dp.train_masks,
                                 train_size=cfg.trainsize,
                                 rVFlip=cfg.rVFlip,
                                 rCrop=cfg.rCrop,
                                 rRotate=cfg.rRotate,
                                 colorEnhance=cfg.colorEnhance,
                                 rPeper=cfg.rPeper)
    train_datald = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True)

    model = ICLCamo(n_layers=[2, 5, 8, 11], hid_dim=64, dropout=0.1).to(cfg.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineDecay(optimizer, max_lr=cfg.learning_rate, min_lr=cfg.min_lr, max_epoch=cfg.epochs)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_datald=train_datald,
    )

    trained_model = trainer.start_training()
