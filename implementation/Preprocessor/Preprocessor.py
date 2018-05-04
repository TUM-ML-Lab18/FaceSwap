import os
from pathlib import Path

import numpy as np
from PIL import Image

from Preprocessor.FaceExtractor import FaceExtractor
from Logging.LoggingUtils import print_progress_bar
from Preprocessor.ImageDataset import ImageDatesetCombined
from config import *


class Preprocessor:
    def __init__(self, root_folder: str, config=None):
        self.config = config
        self.root_folder = Path(root_folder)
        self.raw_folder = self.root_folder / RAW
        self.processed_folder = self.root_folder / PREPROCESSED
        self.extractor = FaceExtractor(mask=np.bool, margin=0)

    def process_images(self):
        """
        Processes all image classes in the raw folder and saves the results to the processed folder.
        """
        if not self.processed_folder.exists():
            self.processed_folder.mkdir()

        # dataset A
        dataset_a = self.raw_folder / A

        # should be only one
        for person_dir in dataset_a.iterdir():
            target_dir = self.processed_folder / A / person_dir.parts[-1]
            if not target_dir.exists():
                self.process_person_folder(person_dir, target_dir)

        # and all the other person in dataset B
        dataset_b = self.raw_folder / B

        # should be only one
        for person_dir in dataset_b.iterdir():
            target_dir = self.processed_folder / B / person_dir.parts[-1]
            if not target_dir.exists():
                self.process_person_folder(person_dir, target_dir)

    def process_person_folder(self, source, target):
        if source.is_dir():
            # log progress
            images_count = len([f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))])
            print_progress_bar(0, images_count)

            target.mkdir(parents=True)
            for idx, image_path in enumerate(source.iterdir()):
                # open image and extract facial region
                try:
                    img = Image.open(image_path)
                except OSError:
                    continue
                extracted_information = self.extractor(img)  # np.asarray(img).astype(np.uint8))
                # if there was an face save the extracted part now in the processed folder
                if extracted_information.image is not None:
                    extracted_image = extracted_information.image
                    extracted_image.save(target / image_path.parts[-1], format='JPEG')

                print_progress_bar(idx, images_count)
        print("\n")

    @property
    def dataset(self):
        """
        This function returns the preprocessed images for a dataset. If the images of this dataset are not preprocessed
        it will apply the necessary step.
        :return: ImageDataset containing the image classes from the dataset.
        """
        self.process_images()
        return ImageDatesetCombined((self.processed_folder / A).__str__(),
                                    (self.processed_folder / B).__str__(),
                                    **self.config['dataset_arguments'])
