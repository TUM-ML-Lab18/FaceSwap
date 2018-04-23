import torch

import cv2
import os
import numpy as np

import face_recognition
from torch.utils.data import Dataset
from torchvision.transforms import transforms

"""inspired by http://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""

TRUMP_CAGE_BASE = "/nfs/students/summer-term-2018/project_2/data/FaceSwap/data"
TRUMP = TRUMP_CAGE_BASE + "/trump"
CAGE = TRUMP_CAGE_BASE + "/cage"

PROCESSED_IMAGES_FOLDER = "/nfs/students/summer-term-2018/project_2/projects/faceswap/processed_images"


class DatasetPerson(Dataset):
    """Dataset containing images from only one person without face detection"""

    def __init__(self, root_dir, transform=None, detect_faces=False):
        """
        :param root_dir: Directory with the images.
        :param transform: Transformations applied to the images.
        :param detect_faces: Detect images with the face_locations module.
        """
        self.transform = transform
        self.root_dir = root_dir
        self.file_names = os.listdir(self.root_dir)
        self.images = []

        l = len(self.file_names)
        printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)

        # load all images into ram
        for idx, img_name in enumerate(self.file_names):
            path2img = os.path.join(self.root_dir, img_name)
            img = cv2.imread(path2img, cv2.COLOR_RGB2BGR).astype(np.float32)
            if detect_faces:
                face_location = face_recognition.face_locations(img, model='hog')

                # ignore if 2 faces detected because in most cases they originate not form the same person
                if face_location and len(face_location) == 1:
                    top, right, bottom, left = face_location[0]
                    img = img[top:bottom, left:right]
                else:
                    continue
            if self.transform:
                img = self.transform(img)
            self.images.append(img)

            printProgressBar(idx + 1, l, prefix='Progress:', suffix='Complete', length=50)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def save_processed_images(self, path):
        for idx, img in enumerate(self.images):
            img = img.numpy().transpose((1, 2, 0))
            cv2.imwrite(os.path.join(path, str(idx) + ".jpg"), img)


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


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    dataset = DatasetPerson(TRUMP, transform=transforms.Compose([Resize((64, 64)), ToTensor()]), detect_faces=True)
    dataset.save_processed_images(PROCESSED_IMAGES_FOLDER)

    # dataset = DatasetPerson(CAGE, transform=transforms.Compose([Resize((64, 64)), ToTensor()]), detect_faces=True)
    # zero = dataset.__getitem__(0)
    # print(zero)
