import os
import cv2
import pdb
import numpy as np
import random
from skimage import io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import datasets_catalog as dc
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    RandomBrightnessContrast,
    HueSaturationValue,
    RGBShift
)


# from ever by Zhuo Zheng
class ToTensor(ToTensorV2):
    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, 'masks': self.apply_to_masks}

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(m, **params) for m in masks]


class MM2Dataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.datasets_name = cfg.TRAIN.DATASETS
        self.mode = mode
        assert dc.contains(self.datasets_name), 'Unknown dataset_name: {}'.format(self.datasets_name)
        self.metas = []
        if mode == 'test' and 'BATCH_SIZE' in cfg.TEST:
            self.batch_size = cfg.TEST.BATCH_SIZE
        else:
            self.batch_size = cfg.TRAIN.BATCH_SIZE

        source_file = dc.get_source_index(self.datasets_name)[self.mode]
        print(source_file)
        prefix = dc.get_prefix(self.datasets_name)
        with open(source_file, 'r') as f:
            lines = f.readlines()
            self.info_num = len(lines[0].strip().split(' '))
            for line in lines:
                pkg = line.strip().split(' ')
                for i in range(self.info_num):
                    pkg[i] = os.path.join(prefix, pkg[i])
                self.metas.append(tuple(pkg))
        self.num = len(lines)

        transforms = []
        self.color_aug = None
        transforms.append(A.Resize(cfg.AUG.INPUT_SIZE[0], cfg.AUG.INPUT_SIZE[1]))
        if self.mode == 'train':
            if self.cfg.AUG.RANDOM_ROTATION is True:
                transforms.append(A.RandomRotate90(always_apply=True))
            if self.cfg.AUG.RANDOM_HFLIP is True:
                transforms.append(A.HorizontalFlip())
            if 'RANDOM_COLOR' in self.cfg.AUG and self.cfg.AUG.RANDOM_COLOR:
                self.color_aug = Compose(
                    [RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1),
                     HueSaturationValue(hue_shift_limit=20,
                                        sat_shift_limit=30,
                                        val_shift_limit=20,
                                        p=1),
                     RGBShift(r_shift_limit=10,
                              g_shift_limit=10,
                              b_shift_limit=10,
                              p=1.0)
                     ])

        transforms.append(A.Normalize(mean=dc.get_mean(self.datasets_name),
                                      std=dc.get_std(self.datasets_name),
                                      max_pixel_value=1.0),
                          )
        transforms.extend([
            ToTensor()
        ])
        self.transforms = A.Compose(transforms)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        pkg = []
        file_num = len(self.metas[idx])
        for i in range(file_num - 1):
            # for filename in self.metas[idx]:
            filename = self.metas[idx][i]
            img_m = io.imread(filename)
            if len(img_m.shape) == 2:
                img_m = img_m[:, :, np.newaxis]
            pkg.append(img_m)
        # label
        label_img = io.imread(self.metas[idx][-1])
        pkg.append(label_img)

        assert pkg[0].shape[2] == self.cfg.MODEL.opt_in_channels
        assert pkg[1].shape[2] == self.cfg.MODEL.sar_in_channels

        if self.color_aug is not None and self.mode == 'train':
            pkg[0] = self.color_aug(image=pkg[0])['image']
            pkg[1] = self.color_aug(image=pkg[1])['image']

        # transform
        transformed = self.transforms(image=np.concatenate([pkg[0], pkg[1]], axis=2), mask=pkg[2])
        pkg[0], pkg[1] = torch.split(
            transformed["image"],
            [self.cfg.MODEL.opt_in_channels, self.cfg.MODEL.sar_in_channels],
            dim=0
        )
        pkg[2] = transformed["mask"]
        if torch.isnan(pkg[0].sum()):
            print(self.metas[idx][0])
        if torch.isnan(pkg[1].sum()):
            print(self.metas[idx][1])
        if torch.isnan(pkg[2].sum()):
            print(self.metas[idx][2])

        return tuple(pkg)
