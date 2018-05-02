from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from Preprocessor.Transforms import RandomWarp, TupleToTensor, TupleResize
from PIL.Image import BICUBIC

class ImageDatesetCombined(Dataset):
    def __init__(self, dataset_a, dataset_b, size_multiplicator=10):
        """
        :param root_dir:
        :param size_multiplicator:
        """
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
            TupleResize((64, 64)),
            TupleToTensor(),
        ])

        self.dataset_a = ImageFolder(dataset_a, transform=self.transforms)
        self.dataset_b = ImageFolder(dataset_b, transform=self.transforms)

    def __len__(self):
        return min(len(self.dataset_a), len(self.dataset_b)) * self.size_multiplicator

    def __getitem__(self, i):
        i %= self.size_multiplicator
        return self.dataset_a[i][0], self.dataset_b[i][0]
