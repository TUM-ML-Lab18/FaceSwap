import torch

import cv2
import os

import face_recognition
from torch.utils.data import Dataset
from torchvision.transforms import transforms

"""inspired by http://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""

TRUMP_CAGE_BASE = "/nfs/students/summer-term-2018/project_2/data/FaceSwap/data"
TRUMP = TRUMP_CAGE_BASE + "/trump"
CAGE = TRUMP_CAGE_BASE + "/cage"


class DatasetPerson(Dataset):
    """Dataset containing images from only one person without face detection"""

    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Directory with the images.
        :param transform: Transformations applied to the images.
        """
        self.transform = transform
        self.root_dir = root_dir
        self.file_names = os.listdir(self.root_dir)
        self.images = []

        # load all images into ram
        for img_name in self.file_names:
            path2img = os.path.join(self.root_dir, img_name)
            img = cv2.imread(path2img)
            if self.transform:
                img = self.transform(img)
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class DatasetPersonDetection(DatasetPerson):
    """Dataset containing images from only one person with face detection"""

    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Directory with the images.
        :param transform: Transformations applied to the images.
        """
        # don't transform the images -> transform=None
        super().__init__(root_dir, transform=None)
        self.transform = transform
        self.detected_faces = []

        # run face detection on images
        for img in self.images:
            face_location = face_recognition.face_locations(img, model='hog')

            # ignore if 2 faces detected because in most cases they originate not form the same person
            if face_location and len(face_location) == 1:
                top, right, bottom, left = face_location[0]
                cropped_img = img[top:bottom, left:right]

                if self.transform:
                    cropped_img = self.transform(cropped_img)

                self.detected_faces.append(cropped_img)
        self.images.clear()
        self.images = self.detected_faces


class Resize:
    """Resize the image to a predefined resolution"""

    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, img):
        return cv2.resize(img, self.resolution)


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # switch dimensions
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)


if __name__ == '__main__':
    dataset = DatasetPerson(TRUMP, transform=transforms.Compose([Resize((64, 64)), ToTensor()]))
    zero = dataset.__getitem__(0)
    print(zero)

    dataset = DatasetPersonDetection(CAGE, transform=transforms.Compose([Resize((64, 64)), ToTensor()]))
    zero = dataset.__getitem__(0)
    print(zero)
