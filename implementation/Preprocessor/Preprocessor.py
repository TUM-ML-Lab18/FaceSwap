import json
import ast
import os
from pathlib import Path

import numpy as np
from PIL import Image

from Logging.LoggingUtils import print_progress_bar
from configuration.general_config import RAW, PREPROCESSED, LANDMARKS_BUFFER, LANDMARKS_JSON
from Preprocessor.FaceExtractor import FaceExtractor

img_file_extensions = ['.jpg', '.JPG', '.png', '.PNG']
separator = '        '

class Preprocessor:
    """
    Preprocesses all images in the given directory
    Directory must have the following structure:
    root/raw: Place, where the unprocessed images are stored
    root/preprocessed: Place, where the processed images get stored
    """
    def __init__(self, face_extractor):
        """
        :param face_extractor: Initialized FaceExtractor
        """
        # Used extractor
        self.extractor = face_extractor

    def __call__(self, root_folder):
        """
        Processes all image classes in the raw folder and saves the results to the preprocessed folder.
        :param root_folder: Path to the dataset that should be processed
        """
        # ============================ Initializations
        # Initialize paths
        raw_folder = root_folder / RAW
        preprocessed_folder = root_folder / PREPROCESSED
        landmarks_buffer = root_folder / LANDMARKS_BUFFER
        landmarks_json = root_folder / LANDMARKS_JSON
        # Check root path
        if not raw_folder.exists():
            raise FileNotFoundError('No RAW folder with images to process: ' + str(raw_folder))
        # Check preprocessed path
        if not preprocessed_folder.exists():
            preprocessed_folder.mkdir()

        # Count files recursively for logging
        total_files_count = sum([len(files) for r, d, files in os.walk(raw_folder)])
        files_count = 0

        # ============================ Process images
        # Iterate recursively over RAW directory
        for root, dirs, files in os.walk(raw_folder):
            root = Path(root)

            # Create subdirectories
            for subdir in dirs:
                # Create paths for subdirectory
                subdir_raw = root / subdir
                relative_path = subdir_raw.relative_to(raw_folder)
                subdir_processed = preprocessed_folder / relative_path
                if not subdir_processed.exists():
                    subdir_processed.mkdir()

            # Convert images
            for file in files:
                # Logging
                files_count += 1
                print_progress_bar(files_count, total_files_count)
                # Create paths for image file
                file_path_raw = root / file
                relative_path = file_path_raw.relative_to(raw_folder)
                # Check if image is already preprocessed
                if (preprocessed_folder / relative_path).exists():
                    continue
                if file_path_raw.suffix in img_file_extensions:
                    # open image and extract facial region
                    try:
                        image = Image.open(file_path_raw)
                    except OSError:
                        continue
                    # Convert image to RGB
                    if image.mode is not 'RGB':
                        image = image.convert('RGB')
                    # Extract facial region in image
                    extracted_image, extracted_information = self.extractor(image)
                    # Check if extraction found a face
                    if extracted_image is None:
                        continue
                    # Convert landmarks into normalized list
                    landmarks = (np.array(extracted_information.landmarks) / extracted_information.size_fine).tolist()
                    # Buffer landmarks in CSV file
                    with open(landmarks_buffer, 'a') as lm_buffer:
                        lm_buffer.write(str(relative_path) + separator + str(landmarks) + '\n')
                    # Save extraction result in PROCESSED folder
                    extracted_image.save(preprocessed_folder / relative_path, format='JPEG')

        # Convert landmarks to json
        landmarks_storage = convert_buffer_to_dict(landmarks_buffer)

        # Save json file
        with open(landmarks_json, 'w') as lm_json:
            json.dump(landmarks_storage, lm_json)


def convert_buffer_to_dict(landmarks_buffer_file):
    """
    Converts the landmarks from the buffer file  to a dict
    :param landmarks_buffer_file: Path to the landmarks buffer file
    :return: Landmarks stored in a dict
    """
    # Extract landmarks from file and remove separator
    with open(landmarks_buffer_file) as lm_buffer:
        buffered_landmarks = lm_buffer.readlines()
    # Extract separators
    buffered_landmarks = [line.strip() for line in buffered_landmarks]
    buffered_landmarks = [line.split(separator) for line in buffered_landmarks]
    # Save landmarks in dict
    landmarks_storage = {}
    for filename, landmarks in buffered_landmarks:
        landmarks_storage[filename] = ast.literal_eval(landmarks)
    return landmarks_storage

    @property
    def dataset(self):
        """
        This function returns the preprocessed images for a dataset. If the images of this dataset are not preprocessed
        it will apply the necessary step.
        :return: ImageDataset containing the image classes from the dataset.
        """
        self.process_images()
        return self.image_dataset(self.root_folder / PREPROCESSED)

