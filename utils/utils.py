import math


def normalize(img):
   """
   img: torch.tensor
   normalize img to [0, 1]
   """
   img = (img - img.min()) / (img.max() - img.min() + 1e-8)
   return img


class CosineDecay:
    def __init__(self,
                 optimizer,
                 max_lr,
                 min_lr,
                 max_epoch,
                 test_mode=False):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_epoch = max_epoch
        self.test_mode = test_mode

        self.current_lr = max_lr
        self.cnt = 0
        if self.max_epoch > 1:
            self.scale = (max_lr - min_lr) / 2
            self.shift = (max_lr + min_lr) / 2
            self.alpha = math.pi / (max_epoch - 1)

    def step(self):
        self.cnt += 1
        self.current_lr = self.scale * math.cos(self.alpha * self.cnt) + self.shift

        if not self.test_mode:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr

    def get_lr(self):
        return self.current_lr
