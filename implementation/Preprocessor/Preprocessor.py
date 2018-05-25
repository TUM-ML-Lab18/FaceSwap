import json
import ast
import os
import cv2
from pathlib import Path

import numpy as np
from PIL import Image

from Utils.Logging.LoggingUtils import print_progress_bar
from Configuration.config_general import *

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

    def __call__(self, root_folder:Path):
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
        histo_buffer = root_folder / HISTO_BUFFER
        histo_json = root_folder / HISTO_JSON

        # Check root path
        if not raw_folder.exists():
            raise FileNotFoundError('No RAW folder with images to process: ' + str(raw_folder))
        # Check preprocessed folders
        preprocessed_folder.mkdir(exist_ok=True)
        for size in RESOLUTIONS:
            resolution_folder = root_folder / ('preprocessed' + str(size))
            resolution_folder.mkdir(exist_ok=True)

        # Count files recursively for logging
        total_files_count = sum([len(files) for r, d, files in os.walk(raw_folder)])
        files_count = 0

        # ============================ Process images
        # Iterate recursively over RAW directory
        for root, dirs, files in os.walk(raw_folder):
            root = Path(root)

            # Create subdirectories
            for subdir in dirs:
                # Create paths for subdirectories
                subdir_raw = root / subdir
                relative_path = subdir_raw.relative_to(raw_folder)
                subdir_processed = preprocessed_folder / relative_path
                subdir_processed.mkdir(exist_ok=True)
                for size in RESOLUTIONS:
                    subdir_res = root_folder / ('preprocessed' + str(size)) / relative_path
                    subdir_res.mkdir(exist_ok=True)

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
                    # Normalized histogram as list
                    histo_r = cv2.calcHist([np.array(extracted_image)], [0], None, [256], [0, 256])
                    histo_g = cv2.calcHist([np.array(extracted_image)], [1], None, [256], [0, 256])
                    histo_b = cv2.calcHist([np.array(extracted_image)], [2], None, [256], [0, 256])
                    histo = np.vstack((histo_r, histo_g, histo_b)) / extracted_information.size_fine**2
                    histo = histo.reshape(-1).tolist()
                    # Normalized landmarks as list
                    landmarks = np.array(extracted_information.landmarks) / extracted_information.size_fine
                    landmarks = landmarks.reshape(-1).tolist()
                    # Buffer histogram in CSV file
                    with open(histo_buffer, 'a') as h_buffer:
                        h_buffer.write(str(relative_path) + separator + str(histo) + '\n')
                    # Buffer landmarks in CSV file
                    with open(landmarks_buffer, 'a') as lm_buffer:
                        lm_buffer.write(str(relative_path) + separator + str(landmarks) + '\n')
                    # Store different resolutions
                    for size in RESOLUTIONS:
                        resized_img = extracted_image.resize((size, size))
                        resized_img.save(root_folder / ('preprocessed' + str(size)) / relative_path,
                                         format='JPEG')
                    # Save extraction result in PROCESSED folder
                    extracted_image.save(preprocessed_folder / relative_path, format='JPEG')

        # Convert landmarks to json
        landmarks_storage = convert_buffer_to_dict(landmarks_buffer)
        # Save json file
        with open(landmarks_json, 'w') as lm_json:
            json.dump(landmarks_storage, lm_json)

        # Convert histo to json
        histo_storage = convert_buffer_to_dict(histo_buffer)
        # Save json file
        with open(histo_json, 'w') as h_json:
            json.dump(histo_storage, h_json)


def convert_buffer_to_dict(buffer_file):
    """
    Converts the landmarks from the buffer file  to a dict
    :param buffer_file: Path to the buffer file
    :return: Landmarks stored in a dict
    """
    # Extract lines from file and remove separator
    with open(buffer_file) as buffer:
        buffered_lines = buffer.readlines()
    # Extract separators
    buffered_lines = [line.strip() for line in buffered_lines]
    buffered_lines = [line.split(separator) for line in buffered_lines]
    # Save landmarks in dict
    storage = {}
    for filename, data in buffered_lines:
        storage[filename] = ast.literal_eval(data)
    return storage

