import json
import os
import torch
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from Preprocessor.Transforms import RandomWarp, TupleToTensor, TupleResize, LowResTuple, HistTuple
from PIL.Image import BICUBIC

from configuration.general_config import A, B, PREPROCESSED, LANDMARKS_JSON, ANNOTATIONS


class ImageDatesetCombined(Dataset):
    def __init__(self, root_folder: Path, size_multiplicator=1, img_size=(64, 64)):
        self.size_multiplicator = size_multiplicator

        self.random_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=(-5, 5), translate=(0.03, 0.03), scale=(0.95, 1.05), shear=(-5, 5),
                                    resample=BICUBIC),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05),
            transforms.RandomHorizontalFlip(),
        ])
        self.transforms = transforms.Compose([
            self.random_transforms,
            RandomWarp(),
            TupleResize(img_size),
            TupleToTensor(),
        ])

        self.dataset_a = ImageFolder(str(root_folder / PREPROCESSED / A), transform=self.transforms)
        self.dataset_b = ImageFolder(str(root_folder / PREPROCESSED / B), transform=self.transforms)

        print(f"Number of items in datasets:\n"
              f"A:\t{len(self.dataset_a)}\n"
              f"B:\t{len(self.dataset_b)}\n"
              f"AB:\t{len(self)}")

    def __len__(self):
        return min(len(self.dataset_a), len(self.dataset_b)) * self.size_multiplicator

    def __getitem__(self, i):
        i %= min(len(self.dataset_a), len(self.dataset_b))
        return self.dataset_a[i][0], self.dataset_b[i][0]


class LandmarksDataset(Dataset):
    def __init__(self, root_folder: Path, size_multiplicator=10, img_size=(64, 64)):
        self.img_size = img_size
        self.size_multiplicator = size_multiplicator

        self.transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=BICUBIC),
            transforms.ToTensor()
        ])

        self.dataset_a = ImageFolder(str(root_folder / PREPROCESSED), transform=self.transforms)
        with open(root_folder / LANDMARKS_JSON) as f:
            self.landmarks = json.load(f)

        print(f"Number of items in datasets:\n"
              f"A:\t{len(self.dataset_a)}\n"
              f"AB:\t{len(self)}")

    def __len__(self):
        return len(self.dataset_a) * self.size_multiplicator

    def __getitem__(self, i):
        i %= len(self.dataset_a)
        file_name_a = self.dataset_a.classes[self.dataset_a.samples[i][1]] + "/" + os.path.basename(
            self.dataset_a.samples[i][0])
        landmarks_a = np.reshape(self.landmarks[file_name_a], -1).astype(np.float32)
        return landmarks_a, self.dataset_a[i][0]


class LandmarksLowResDataset(LandmarksDataset):
    def __init__(self, root_folder: Path, size_multiplicator=10, target_img_size=(64, 64), lowres_img_size=(8, 8)):
        # todo self.dataset_a could be different
        super().__init__(root_folder, size_multiplicator=size_multiplicator, img_size=target_img_size)
        self.transforms = transforms.Compose([
            transforms.Resize(target_img_size, interpolation=BICUBIC),
            LowResTuple(lowres_img_size),
            TupleToTensor()
        ])
        self.dataset_a.transform = self.transforms

    def __getitem__(self, i):
        landmarks, img_list = super().__getitem__(i)
        low_res = img_list[0].numpy().flatten()
        latent = np.append(landmarks, low_res)
        img = img_list[1]
        return latent, img


class LandmarksHistDataset(LandmarksDataset):
    def __init__(self, root_folder: Path, size_multiplicator=10, img_size=(64, 64), bins=50):
        super().__init__(root_folder=root_folder, size_multiplicator=size_multiplicator, img_size=img_size)
        self.transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=BICUBIC),
            HistTuple(bins=bins)
        ])
        self.dataset_a.transform = self.transforms

    def __getitem__(self, i):
        landmarks, img_list = super().__getitem__(i)
        # todo find better method to scale to [0..1] like softmax?
        hist = img_list[0].flatten()
        latent = np.append(landmarks, hist).astype(np.float32)
        img = img_list[1]
        return latent, img


class LandmarksHistAnnotationsDataset(LandmarksHistDataset):
    def __init__(self, root_folder: Path, size_multiplicator=10, img_size=(64, 64)):
        super().__init__(root_folder=root_folder, size_multiplicator=size_multiplicator, img_size=img_size)
        with open(root_folder / ANNOTATIONS) as f:
            self.annotations = json.load(f)

    def __getitem__(self, i):
        latent, img = super().__getitem__(i)
        file_name_a = self.dataset_a.classes[self.dataset_a.samples[i][1]] + "/" + os.path.basename(
            self.dataset_a.samples[i][0])
        latent = np.append(latent, self.annotations[file_name_a]).astype(np.float32)
        return latent, img


class LandmarksLowResAnnotationsDataset(LandmarksLowResDataset):
    def __init__(self, root_folder: Path, size_multiplicator=10, target_img_size=(64, 64), lowres_img_size=(8, 8)):
        super().__init__(root_folder=root_folder, size_multiplicator=size_multiplicator,
                         target_img_size=target_img_size, lowres_img_size=lowres_img_size)
        with open(root_folder / ANNOTATIONS) as f:
            self.annotations = json.load(f)

    def __getitem__(self, i):
        latent, img = super().__getitem__(i)
        file_name_a = self.dataset_a.classes[self.dataset_a.samples[i][1]] + "/" + os.path.basename(
            self.dataset_a.samples[i][0])
        latent = np.append(latent, self.annotations[file_name_a]).astype(np.float32)
        return latent, img


class StaticLandmarks32x32Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.data_X = np.load('/nfs/students/summer-term-2018/project_2/data/CelebA/data64.npy')
        self.data_Y = np.load('/nfs/students/summer-term-2018/project_2/data/CelebA/landmarks5.npy')
        StaticLandmarks32x32Dataset.y_mean = np.mean(self.data_Y, axis=0)
        StaticLandmarks32x32Dataset.y_cov = np.cov(self.data_Y, rowvar=0)
        self.data_X = torch.from_numpy(self.data_X).type(torch.FloatTensor)
        self.data_Y = torch.from_numpy(self.data_Y).type(torch.FloatTensor)[:, :, None, None]
        self.size_multiplicator = 1

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, i):
        return self.data_X[i], self.data_Y[i]
