import ast
import json
import os
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from PIL import Image

from Configuration.config_general import *
from Utils.Logging.LoggingUtils import print_progress_bar

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

    def __call__(self, root_folder: Path):
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
        landmarks_npy = root_folder / LANDMARKS_NPY
        histo_buffer = root_folder / HISTO_BUFFER
        histo_json = root_folder / HISTO_JSON
        histo_npy = root_folder / HISTO_NPY
        embeddings_buffer = root_folder / FACE_EMBEDDINGS_BUFFER
        embeddings_json = root_folder / FACE_EMBEDDINGS_JSON
        embeddings_npy = root_folder / FACE_EMBEDDINGS_NPY

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
                    # Calculate histogram
                    histo = self.calculate_masked_histogram(extracted_image)
                    # Normalized landmarks as list
                    landmarks = np.array(extracted_information.landmarks) / extracted_information.size_fine
                    landmarks = landmarks.reshape(-1).tolist()

                    # calculate embeddings
                    # (top, right, bottom, left)
                    embedding = face_recognition.face_encodings(np.array(extracted_image), known_face_locations=[(
                        0, extracted_information.size_fine, extracted_information.size_fine, 0)])[0].tolist()

                    # Buffer histogram in CSV file
                    with open(histo_buffer, 'a') as h_buffer:
                        h_buffer.write(str(relative_path) + separator + str(histo) + '\n')
                    # Buffer landmarks in CSV file
                    with open(landmarks_buffer, 'a') as lm_buffer:
                        lm_buffer.write(str(relative_path) + separator + str(landmarks) + '\n')
                    # Buffer embeddings in CSV file
                    with open(embeddings_buffer, 'a') as em_buffer:
                        em_buffer.write(str(relative_path) + separator + str(embedding) + '\n')
                    # Store different resolutions
                    for size in RESOLUTIONS:
                        resized_img = extracted_image.resize((size, size))
                        resized_img.save(root_folder / ('preprocessed' + str(size)) / relative_path,
                                         format='JPEG')
                    # Save extraction result in PROCESSED folder
                    extracted_image.save(preprocessed_folder / relative_path, format='JPEG')

        # ============= Conversions / Storage

        # Convert landmarks to json
        landmarks_storage = convert_buffer_to_dict(landmarks_buffer)
        # Save json file
        with open(landmarks_json, 'w') as lm_json:
            json.dump(landmarks_storage, lm_json)
        # Save npy file
        landmarks_array = np.array(list(landmarks_storage.values()))
        np.save(landmarks_npy, landmarks_array)

        # Convert histo to json
        histo_storage = convert_buffer_to_dict(histo_buffer)
        # Save json file
        with open(histo_json, 'w') as h_json:
            json.dump(histo_storage, h_json)
        histo_array = np.array(list(histo_storage.values()))
        np.save(histo_npy, histo_array)

        # Convert embeddings to json
        embeddings_storage = convert_buffer_to_dict(embeddings_buffer)
        # Save json file
        with open(embeddings_json, 'w') as em_json:
            json.dump(embeddings_storage, em_json)
        embeddings_array = np.array(list(embeddings_storage.values()))
        np.save(embeddings_npy, embeddings_array)

        # Convert images folder into arrays
        for size in RESOLUTIONS:
            subdir_res = root_folder / ('preprocessed' + str(size) + '/images')
            data = np.array([np.array(Image.open(fname)) for fname in subdir_res.iterdir()])
            data = data.transpose((0, 3, 1, 2))
            np.save(root_folder / ('data' + str(size) + '.npy'), data)

    @staticmethod
    def calculate_masked_histogram(image):
        """
        Calculates the histogram of the masked region
        :param image: PIL image
        :return:
        """
        image = np.array(image)
        # Calculate histogram of the whole image
        histo_r = cv2.calcHist([image], [0], None, [256], [0, 256])
        histo_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        histo_b = cv2.calcHist([image], [2], None, [256], [0, 256])
        # Remove masked pixels from histogram
        histo_r[0] = 0
        histo_g[0] = 0
        histo_b[0] = 0
        # Normalize histogram
        histo_r /= np.sum(histo_r)
        histo_g /= np.sum(histo_g)
        histo_b /= np.sum(histo_b)

        histo = np.vstack((histo_r, histo_g, histo_b))
        histo = histo.reshape(-1).tolist()

        return histo


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
