import os
import torch


class Config():
    def __init__(self):
        self.dp = DataPath()

        self.num_workers = 8

        self.trainsize = 392
        self.hid_dim = 64

        ''' Save Paths '''
        self.result_path = './results'
        self.dir_name = ''
        self.save_dir = os.path.join(self.result_path, self.dir_name)

        self.CUDA = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if self.CUDA else 'cpu')

        self.start_epoch = 1
        self.epochs = 100
        self.batch_size = 8

        ''' Data Augmentation '''
        self.rVFlip = True
        self.rCrop = True
        self.rRotate = False
        self.colorEnhance = True
        self.rPeper = False

        ''' Optimizer '''
        self.weight_decay = 0

        ''' LR Scheduler '''
        self.learning_rate = 1e-4
        self.min_lr = 1e-6

        ''' logging '''
        self.log_interval = 50


class DataPath():
    def __init__(self):
        self.dataset_dir = './Dataset'
        
        ''' Train Dataset: CAMO-Train + COD10K-Train '''
        self.train_imgs = os.path.join(self.dataset_dir, 'TrainDataset', 'Imgs')
        self.train_masks = os.path.join(self.dataset_dir, 'TrainDataset', 'GT')
        
        ''' Test Dataset '''
        # CHAMELEON
        self.test_CHAMELEON_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'Imgs')
        self.test_CHAMELEON_masks = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'GT')
        # CAMO-Test
        self.test_CAMO_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'Imgs')
        self.test_CAMO_masks = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'GT')
        # COD10K-Test
        self.test_COD10K_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'Imgs')
        self.test_COD10K_masks = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'GT')
        # NC4K
        self.test_NC4K_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'Imgs')
        self.test_NC4K_masks = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'GT')  


if __name__ == '__main__':
    config = Config()
    print(config)
