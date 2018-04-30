import os
from pathlib import Path

import numpy as np
from PIL import Image

from FaceExtractor import FaceExtractor
from Logging.LoggingUtils import print_progress_bar
from Preprocessor.ImageDataset import ImageDataset


class Preprocessor:
    RAW = "raw"
    PREPROCESSED = "preprocessed"

    def __init__(self, root_folder: str):
        self.root_folder = Path(root_folder)
        self.raw_folder = self.root_folder / Preprocessor.RAW
        self.processed_folder = self.root_folder / Preprocessor.PREPROCESSED
        self.extractor = FaceExtractor()

    @property
    def processed_folder_exists(self):
        """
        Checks if the processed folder exists.
        :return: Bool.
        """
        return self.processed_folder.exists()

    def process_images(self):
        """
        Processes all image classes in the raw folder and saves the results to the processed folder.
        """
        self.processed_folder.mkdir()
        for person_dir in self.raw_folder.iterdir():
            if person_dir.is_dir():
                # log progress
                images_count = len([f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))])
                print_progress_bar(0, images_count)

                processed_person_dir = self.processed_folder / person_dir.parts[-1]
                processed_person_dir.mkdir()
                # iterate over every image of the person
                for idx, image_path in enumerate(person_dir.iterdir()):
                    # open image and extract facial region
                    img = Image.open(image_path)
                    extracted_information = self.extractor(np.asarray(img).astype(np.uint8))
                    # if there was an face save the extracted part now in the processed folder
                    if extracted_information is not None:
                        extracted_image = Image.fromarray(extracted_information.image)
                        extracted_image.save(processed_person_dir / image_path.parts[-1])

                    print_progress_bar(idx, images_count)
                print()

    @property
    def get_dataset(self):
        """
        This function returns the preprocessed images for a dataset. If the images of this dataset are not preprocessed
        it will apply the necessary step.
        :return: ImageDataset containing the image classes from the dataset.
        """
        if not self.processed_folder_exists:
            self.process_images()
        return ImageDataset(self.processed_folder.__str__())
