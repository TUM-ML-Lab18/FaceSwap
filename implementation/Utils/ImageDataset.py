from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL.Image import BICUBIC
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from Configuration.config_general import A, B, PREPROCESSED
from Preprocessor.Transforms import RandomWarp, TupleToTensor, TupleResize


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


class ImageFeatureDataset(Dataset):
    """
    Generic data set class to load
    * images with corresponding features
    * only images
    * only features
    from stored NumPy array
    """

    def __init__(self, path_to_image_array, paths_to_feature_arrays):
        """
        Initialize a dataset
        :param path_to_image_array: None if no images should be loaded
        :param paths_to_feature_arrays: List of feature arrays to be loaded
                                        None if no features should be loaded
        """
        print('Loading data... This may take some time')
        if path_to_image_array is not None:
            self.images = np.load(path_to_image_array)
            self.images = torch.from_numpy(self.images).type(torch.float32)
            self.images /= 255.
            self.images -= 0.5
            self.images /= 0.5
        else:
            self.images = None
        if paths_to_feature_arrays is not None:
            # convert to list if a single item
            if type(paths_to_feature_arrays) is not list:
                paths_to_feature_arrays = [paths_to_feature_arrays]
            features = []
            for path in paths_to_feature_arrays:
                feature = np.load(path)
                feature = feature.reshape((len(feature), -1))
                features.append(feature)
            self.features = np.hstack(features)
            self.features = torch.from_numpy(self.features).type(torch.float32)
            self.features -= 0.5
            self.features /= 0.5
        else:
            self.features = None
        print('Data loaded.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        items = []
        if self.images is not None:
            items.append(self.images[index])
        if self.features is not None:
            items.append(self.features[index])

        return items[0] if len(items) == 1 else tuple(items)
