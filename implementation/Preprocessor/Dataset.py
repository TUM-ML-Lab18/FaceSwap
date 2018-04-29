import cv2
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from implementation.Logging.LoggingUtils import printProgressBar

"""inspired by http://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""


class DatasetPerson(Dataset):
    """Dataset containing images from only one person"""

    def __init__(self, root_dir, preprocessor,
                 transform=True, detect_faces=False, warp_faces=True, size_multiplicator=10, convertion_dataset=False):
        """
        :param root_dir: Directory with the images.
        :param preprocessor: Preprocessor object
        :param transform: Transformations applied to the images.
        :param detect_faces: Detect images with the face_locations module.
        :param warp_faces: Warp faces before supplying them to a DataLoader
        :param size_multiplicator: Enlarges the dataset virtually times this factor
        """
        self.preprocessor = preprocessor
        self.transform = transform  # TODO: Unused member
        self.warp_faces = warp_faces
        self.size_multiplicator = size_multiplicator
        self.root_dir = root_dir
        self.file_names = os.listdir(self.root_dir)
        self.images = []
        self.borders = []
        self.real_images = []

        l = len(self.file_names)
        printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)

        # load all images into ram
        for idx, img_name in enumerate(self.file_names):
            if not img_name.__contains__(".jpg"):
                continue
            path2img = os.path.join(self.root_dir, img_name)
            try:
                img = cv2.cvtColor(cv2.imread(path2img), cv2.COLOR_BGR2RGB).astype(np.float32)
            except Exception as e:
                print(path2img, '\n', e)
            if detect_faces or convertion_dataset:
                if convertion_dataset:
                    #img = img[:, :, ::-1]
                    self.real_images.append(img)
                    img, face_detected, borders = self.preprocessor.extract_faces(img, return_borders=True)
                else:
                    img, face_detected = self.preprocessor.extract_faces(img)
                if not face_detected:
                    if convertion_dataset:
                        self.real_images.pop(len(self.real_images) - 1)
                    continue
            self.images.append(img)
            if convertion_dataset:
                self.borders.append(borders)

            printProgressBar(idx + 1, l, prefix='Progress:', suffix='Complete', length=50)

    def __len__(self):
        return len(self.images) * self.size_multiplicator

    def __getitem__(self, idx):
        # Resize on 256x256 for faster processing
        image = cv2.resize(self.images[idx % self.size_multiplicator], (256, 256))
        # Apply random transformations to augment dataset
        if self.transform:
            image = self.preprocessor.random_transform(image)
        # Warp faces
        if self.warp_faces:
            warped_image, target_image = self.preprocessor.warp(image)
        else:
            warped_image, target_image = image, image
        # Resize input and output for training
        warped_image = self.preprocessor.resize(warped_image, (64, 64))
        target_image = self.preprocessor.resize(target_image, (64, 64))
        # Transform images into Tensors
        warped_image = ToTensor()(warped_image) / 255.0
        target_image = ToTensor()(target_image) / 255.0
        return warped_image, target_image

    def save_processed_images(self, path):
        for idx, img in enumerate(self.images):
            cv2.imwrite(os.path.join(path, str(idx) + ".jpg"), img)
