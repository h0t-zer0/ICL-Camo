import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.data_augmentation import cv_random_flip, randomCrop, randomRotation, randomPeper, colorEnhance
import cv2
import numpy as np


class TrainDataset(Dataset):
    '''
    DataLoader for COD tasks
    '''

    def __init__(self, image_root,
                       gt_root,
                       train_size,
                       rVFlip=True,
                       rCrop=True,
                       rRotate=True,
                       colorEnhance=False,
                       rPeper=False):
        self.train_size = train_size
        self.rVFlip = rVFlip
        self.rCrop = rCrop
        self.rRotate = rRotate
        self.colorEnhance = colorEnhance
        self.rPeper = rPeper

        self.images, self.gts = [], []
        self.images.extend([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts.extend([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')])

        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor()])

        self.size = len(self.images)
        print(f'>>> training/validing with {self.size} samples')

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        random_index = np.random.randint(0, self.size)

        ref_image = self.rgb_loader(self.images[random_index])
        ref_gt = self.binary_loader(self.gts[random_index])

        # Data Augmentation
        if self.rVFlip:
            image, gt = cv_random_flip([image, gt])
            ref_image, ref_gt = cv_random_flip([ref_image, ref_gt])
        if self.rCrop:
            image, gt = randomCrop([image, gt])
            ref_image, ref_gt = randomCrop([ref_image, ref_gt])
        if self.rRotate:
            image, gt = randomRotation([image, gt])
            ref_image, ref_gt = randomRotation([ref_image, ref_gt])
        if self.colorEnhance:
            image = colorEnhance(image)
            ref_image = colorEnhance(ref_image)
        if self.rPeper:
            gt = randomPeper(gt)
            ref_gt = randomPeper(ref_gt)

        ori_img = self.gt_transforms(image)

        image = self.img_transforms(image)
        gt = self.gt_transforms(gt)
        ref_image = self.img_transforms(ref_image)
        ref_gt = self.gt_transforms(ref_gt)

        return ori_img, image, gt, ref_image, ref_gt

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img, mode='L')
        return img


class TestDataset(Dataset):
    def __init__(self, image_root,
                       gt_root,
                       ref_image_root,
                       ref_gt_root,
                       test_size):
        self.test_size = test_size

        self.images, self.gts = [], []
        self.images.extend([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts.extend([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')])
            
        self.ref_images, self.ref_gts = [], []
        self.ref_images.extend([os.path.join(ref_image_root, f) for f in os.listdir(ref_image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.ref_gts.extend([os.path.join(ref_gt_root, f) for f in os.listdir(ref_gt_root) if f.endswith('.jpg') or f.endswith('.png')])
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.ref_images = sorted(self.ref_images)
        self.ref_gts = sorted(self.ref_gts)

        # transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor()])
    
        self.size = len(self.images)
        self.ref_size = len(self.ref_images)
        
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # load reference from training set
        random_index = np.random.randint(0, self.ref_size)
        assert self.ref_images[random_index].split('/')[-1][:-4] == self.ref_gts[random_index].split('/')[-1][:-4], "Reference Image and Mask Do Not Match!"
        ref_image = self.rgb_loader(self.ref_images[random_index])
        ref_gt = self.binary_loader(self.ref_gts[random_index])

        ori_img = self.gt_transforms(image)
        image = self.img_transforms(image)
        ori_gt = transforms.PILToTensor()(gt)
        gt = self.gt_transforms(gt)

        ref_image = self.img_transforms(ref_image)
        ref_gt = self.gt_transforms(ref_gt)

        name = self.images[index].split('/')[-1]
        if '\\' in name:
            name = name.split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return ori_img, ori_gt, name, image, gt, ref_image, ref_gt

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img, mode='L')
        return img
    