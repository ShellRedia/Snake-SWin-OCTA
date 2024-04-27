from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
import torchio as tio
from scipy.ndimage import zoom

import os
import cv2
import torch
import numpy as np
from collections import *
from bisect import bisect_left
from itertools import accumulate
import matplotlib.pyplot as plt
import albumentations as alb
from tqdm import tqdm
from random import randint

class OCTA500_2d_Dataset(Dataset):
    def __init__(self, fov="3M", modal="OCTA", label_type="FAZ", layers = ["FULL", "ILM_OPL", "OPL_BM"], is_training=True, is_resize=False):
        self.is_training = is_training
        self.is_resize = is_resize
        
        data_dir = "datasets/OCTA-500/{}/ProjectionMaps".format(fov)
        label_dir = "datasets/OCTA-500/{}/GT_{}".format(fov, label_type)

        self.sample_ids = sorted([x[:-4] for x in os.listdir(label_dir)])

        self.samples, self.labels = [], []
        process = lambda x: np.array(x).transpose((1,2,0))

        for sample_id in self.sample_ids:
            sample = []
            for layer in layers:
                image_layer = cv2.imread("{}/{}({})/{}.bmp".format(data_dir,modal,layer,sample_id), cv2.IMREAD_GRAYSCALE)
                sample.append(image_layer)
            label = [cv2.imread("{}/{}.bmp".format(label_dir, sample_id), cv2.IMREAD_GRAYSCALE)]

            sample, label = map(process, [sample, label])

            self.samples.append(sample)
            self.labels.append(label)
        
        probability = 0.2
        self.transform = alb.Compose([
            # level 1
            alb.RandomBrightnessContrast(p=probability),
            alb.CLAHE(p=probability), 
            # level 2
            alb.Rotate(limit=15, p=probability),
            alb.VerticalFlip(p=probability),
            alb.HorizontalFlip(p=probability),
            # level 3
            alb.AdvancedBlur(p=probability),
            alb.PiecewiseAffine(p=probability),
            alb.CoarseDropout(40,10,10,p=probability),
        ])
        self.resize = alb.Compose([alb.Resize(height=512, width=512, always_apply=True, p=1)])
            
    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample, label = self.samples[index], self.labels[index]
        process = lambda x: x.transpose((2,0,1)) / 255
        if self.is_training:
            transformed = self.transform(**{"image": sample, "mask": label})
            sample, label = transformed["image"], transformed["mask"]
    
        if self.is_resize:
            transformed = self.resize(**{"image": sample, "mask": label})
            sample, label = transformed["image"], transformed["mask"]
            
        return process(sample), process(label), self.sample_ids[index]

class OCTA_25K_Dataset(Dataset):
    def __init__(self):
        self.data_dir = "datasets/OCTA-25K"
        self.sample_ids = sorted([x[:-4] for x in os.listdir(self.data_dir)])[:2000]

        self.sample = []
        for sample_id in self.sample_ids:
            sample = cv2.imread("{}/{}.png".format(self.data_dir, sample_id), cv2.IMREAD_GRAYSCALE)
            self.sample.append(sample)
        
        probability = 0.1
        self.transform = alb.Compose([
            alb.RandomBrightnessContrast(p=probability),
            alb.CLAHE(p=probability), 
            alb.AdvancedBlur(p=probability),
        ])
        self.resize = alb.Compose([alb.Resize(height=256, width=256, always_apply=True, p=1)])
        self.mask = alb.Compose(
            [alb.CoarseDropout(max_holes=40, max_height=32, max_width=32, always_apply=True, p=1)])
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        resized = self.resize(**{"image": self.sample[index], "mask": self.sample[index]})
        sample, label = resized["image"], resized["mask"]
        #
        masked = self.mask(**{"image": sample, "mask": sample})
        sample = masked["image"]
        #
        transformed = self.transform(**{"image": sample, "mask": label})
        sample, label = transformed["image"], transformed["mask"]
        sample, label = np.array([sample]), np.array([label])
        return sample / 255, label / 255, self.sample_ids[index]

class OCTA500_3d_Dataset(Dataset):
    def __init__(self, fov="3M", modal="OCTA", label="FAZ", is_training=True):
        self.data_dir = "datasets/OCTA-500/data_3D/{}_{}/".format(fov, modal)
        self.label_dir = "datasets/OCTA-500/gt_3D/{}_{}/".format(fov, label)
        self.sample_ids = sorted(set(x[:-4] for x in os.listdir(self.label_dir+"original/")))

        self.target_shape = (128, 256, 128)
        # self.target_shape = None

        self.transform = alb.Compose([
            alb.Rotate(limit=30, p=1.0),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        target_shape_str = "original"
        if self.target_shape:
            target_shape_str = "-".join(list(map(str, self.target_shape)))
        get_origin_path = lambda x:"{}/original/{}.npy".format(x, self.sample_ids[index])
        get_resized_dir = lambda x:"{}/resized/{}/".format(x, target_shape_str)
        get_resized_path = lambda x:"{}/resized/{}/{}.npy".format(x, target_shape_str, self.sample_ids[index])

        sample_path, label_path = map(get_origin_path, (self.data_dir, self.label_dir))
        sample, label = map(lambda x:np.load(x), (sample_path, label_path))

        if self.target_shape:
            sample_path, label_path = map(get_resized_path, (self.data_dir, self.label_dir))

            if os.path.exists(sample_path):
                sample, label = map(lambda x:np.load(x), (sample_path, label_path))
            else:
                sample_dir, label_dir = map(get_resized_dir, (self.data_dir, self.label_dir))
                if not os.path.exists(sample_dir): os.makedirs(sample_dir)
                if not os.path.exists(label_dir): os.makedirs(label_dir)
                zoom_factor = [y / x for x, y in zip(sample.shape, self.target_shape)]
                sample, label = map(lambda x:zoom(x, zoom_factor, order=3, mode='nearest'), (sample, label))
                [np.save(y, x) for x, y in zip([sample, label], [sample_path, label_path])]
        
        sample_id = self.sample_ids[index]
        sample, label = map(lambda x:x[np.newaxis,:,100:228,:], (sample, label))
        return sample, label, sample_id

class DataLoader_Producer:
    def __init__(self, k_fold=10, fov="3M", modal="OCTA", label_type="FAZ", layers = ["FULL", "ILM_OPL", "OPL_BM"], batch_size=1, dim=2, is_resize=False):

        self.k_fold = k_fold
        self.batch_size = batch_size
        self.fov = fov

        self.dataset_train = {
            2 : OCTA500_2d_Dataset(fov=fov, modal=modal, label_type=label_type, layers=layers, is_training=True, is_resize=is_resize),
            # 3 : OCTA500_3d_Dataset(fov=fov, modal=modal, label=label, is_training=True)
        }[dim]
        self.dataset_val = {
            2 : OCTA500_2d_Dataset(fov=fov, modal=modal, label_type=label_type, layers=layers, is_training=False, is_resize=is_resize),
            # 3 : OCTA500_3d_Dataset(fov=fov, modal=modal, label=label, is_training=False)
        }[dim]
    
    def get_data_loader(self, fold_index):
        num_samples = len(self.dataset_train)   
        indices = list(range(num_samples))
        split = self.num_samples // self.k_fold
        train_indices = indices[:fold_index * split] + indices[(fold_index + 1) * split:]
        val_indices = indices[fold_index * split:(fold_index + 1) * split]    
        train_sampler, val_sampler = map(SubsetRandomSampler, (train_indices, val_indices))
        train_loader = DataLoader(self.dataset_train, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(self.dataset_val, batch_size=1, sampler=val_sampler)

        return train_loader, val_loader, None
    
    def get_data_loader_ipn_v2(self):
        # use the validation set to select the best model
        train_indices, val_indices, test_indice = {
            "3M" : (list(range(140)), list(range(140, 150)), list(range(150, 200))),
            "6M" : (list(range(180)), list(range(180, 200)), list(range(200, 300)))
        }[self.fov]
        
        train_sampler, val_sampler, test_sampler = map(SubsetRandomSampler, (train_indices, val_indices, test_indice))

        train_loader = DataLoader(self.dataset_train, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(self.dataset_val, batch_size=1, sampler=val_sampler)
        test_loader = DataLoader(self.dataset_val, batch_size=1, sampler=test_sampler)

        return train_loader, val_loader, test_loader

if __name__=="__main__":
    dataset = OCTA500_2d_Dataset(layers = ["OPL_BM"], is_resize=True)
    sample, label, sample_ids = dataset[0]
    print(sample.max(), sample.shape)
    cv2.imwrite("temp.png", sample[0] * 255)