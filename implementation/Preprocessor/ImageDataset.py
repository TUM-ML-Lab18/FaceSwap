import os
import cv2

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from Preprocessor.Transforms import FromPIL, ToPIL, Resize, ResizeTuple, RandomWarp, RandomTransform, ToTensor


class ImageDataset(ImageFolder):

    def __init__(self, root_dir, size_multiplicator=10):
        """
        :param root_dir:
        :param transform:
        :param target_transform:
        :param size_multiplicator:
        """
        super().__init__(root_dir)
        self.size_multiplicator = size_multiplicator
        self.transform = Compose([
            FromPIL(),
            Resize(),
            RandomTransform(),
            RandomWarp(),
            ResizeTuple((64, 64)),
            ToPIL(),
            ToTensor()
        ])

    def __len__(self):
        return super().__len__() * self.size_multiplicator

    def __getitem__(self, i):
        """
        :param i: index
        :return: a tuple of (a tuple of warped, target) and (class label)
        """
        return super().__getitem__(i % self.size_multiplicator)