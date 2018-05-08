import json
import os

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from Preprocessor.Transforms import RandomWarp, TupleToTensor, TupleResize
from PIL.Image import BICUBIC

from configuration.gerneral_config import A, B, PREPROCESSED, LANDMARKS


class ImageDatesetCombined(Dataset):
    def __init__(self, dataset, size_multiplicator=10, img_size=(64, 64)):
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

        self.dataset_a = ImageFolder(str(dataset / PREPROCESSED / A), transform=self.transforms)
        self.dataset_b = ImageFolder(str(dataset / PREPROCESSED / B), transform=self.transforms)

    def __len__(self):
        return min(len(self.dataset_a), len(self.dataset_b)) * self.size_multiplicator

    def __getitem__(self, i):
        i %= min(len(self.dataset_a), len(self.dataset_b))
        return self.dataset_a[i][0], self.dataset_b[i][0]


class LandmarkDataset(Dataset):
    def __init__(self, dataset, size_multiplicator=10, img_size=(64, 64)):
        self.size_multiplicator = size_multiplicator

        self.transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=BICUBIC),
            transforms.ToTensor()
        ])

        self.dataset_a = ImageFolder(str(dataset / PREPROCESSED / A), transform=self.transforms)
        self.dataset_b = ImageFolder(str(dataset / PREPROCESSED / B), transform=self.transforms)
        with open(dataset / LANDMARKS) as f:
            self.landmarks = json.load(f)

        print(f"Number of items in datasets:\n"
              f"A:\t{len(self.dataset_a)}\n"
              f"B:\t{len(self.dataset_b)}\n"
              f"AB:\t{len(self)}")

    def __len__(self):
        return min(len(self.dataset_a), len(self.dataset_b)) * self.size_multiplicator

    def __getitem__(self, i):
        i %= min(len(self.dataset_a), len(self.dataset_b))
        file_name_a = os.path.basename(self.dataset_a.samples[i][0])
        landmarks_a = self.landmarks[file_name_a]
        file_name_b = os.path.basename(self.dataset_b.samples[i][0])
        landmarks_b = self.landmarks[file_name_b]
        return (self.dataset_a[i][0], landmarks_a), (self.dataset_b[i][0], landmarks_b)
