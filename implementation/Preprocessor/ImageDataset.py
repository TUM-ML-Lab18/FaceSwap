import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from Preprocessor.Transforms import RandomWarp, TupleToTensor, TupleResize, LowResTuple
from PIL.Image import BICUBIC

from configuration.general_config import A, B, PREPROCESSED, LANDMARKS_JSON


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


class LandmarkDataset(Dataset):
    def __init__(self, root_folder: Path, size_multiplicator=10, img_size=(64, 64)):
        self.size_multiplicator = size_multiplicator

        self.transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=BICUBIC),
            transforms.ToTensor()
        ])

        self.dataset_a = ImageFolder(str(root_folder / PREPROCESSED / A), transform=self.transforms)
        with open(root_folder / LANDMARKS_JSON) as f:
            self.landmarks = json.load(f)

        print(f"Number of items in datasets:\n"
              f"A:\t{len(self.dataset_a)}\n"
              f"AB:\t{len(self)}")

    def __len__(self):
        return len(self.dataset_a) * self.size_multiplicator

    def __getitem__(self, i):
        i %= self.dataset_a
        file_name_a = os.path.basename(self.dataset_a.samples[i][0])
        landmarks_a = np.reshape(self.landmarks[file_name_a], -1).astype(np.float32)
        return landmarks_a, self.dataset_a[i][0]


class CelebA_Landmarks_LowRes(LandmarkDataset):
    def __init__(self, root_folder: Path, size_multiplicator=10, target_img_size=(64, 64), lowres_img_size=(8, 8)):
        super().__init__(root_folder, size_multiplicator=size_multiplicator, img_size=target_img_size)
        # todo self.dataset_a could be different
        self.transforms = transforms.Compose([
            transforms.Resize(target_img_size, interpolation=BICUBIC),
            LowResTuple(lowres_img_size),
            TupleToTensor()
        ])
