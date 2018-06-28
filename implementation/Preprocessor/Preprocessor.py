import ast
import os

import numpy as np
from PIL import Image

from Configuration.config_general import *
from Preprocessor.FaceExtractor import normalize_landmarks, extract_landmarks
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

    def __call__(self):
        """
        Processes all image classes in the raw folder and saves the results to the preprocessed folder.
        """
        # Check root path
        if not RAW_FOLDER.exists():
            raise FileNotFoundError('No RAW folder with images to process: ' + str(RAW_FOLDER))
        # Check preprocessed folders
        PREPROCESSED_FOLDER.mkdir(exist_ok=True)
        for size in RESOLUTIONS:
            resolution_folder = ROOT / (PREPROCESSED + str(size))
            resolution_folder.mkdir(exist_ok=True)

        # Count files recursively for logging
        total_files_count = sum([len(files) for r, d, files in os.walk(RAW_FOLDER)])
        files_count = 0

        # ========== Process images
        # Iterate recursively over RAW directory
        for root, dirs, files in os.walk(RAW_FOLDER):
            root = Path(root)

            # Create subdirectories
            for subdir in dirs:
                # Create paths for subdirectories
                subdir_raw = root / subdir
                relative_path = subdir_raw.relative_to(RAW_FOLDER)
                subdir_processed = PREPROCESSED_FOLDER / relative_path
                subdir_processed.mkdir(exist_ok=True)
                for size in RESOLUTIONS:
                    subdir_res = ROOT / (PREPROCESSED + str(size)) / relative_path
                    subdir_res.mkdir(exist_ok=True)

            # Convert images
            for file in files:
                # Logging
                files_count += 1
                print_progress_bar(files_count, total_files_count)
                # Create paths for image file
                file_path_raw = root / file
                relative_path = file_path_raw.relative_to(RAW_FOLDER)
                # Check if image is already preprocessed
                if (PREPROCESSED_FOLDER / relative_path).exists():
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
                    # ===== Extract facial region in image
                    extracted_image, extracted_information = self.extractor(image)
                    # Check if extraction found a face
                    if extracted_image is None:
                        continue
                    # ===== Calculate features & store them in lists -> format to buffer
                    # Calculate landmarks
                    normalized_landmarks = normalize_landmarks(extracted_information)
                    landmarks = normalized_landmarks.tolist()

                    # ===== Save different resolutions of the extracted image
                    for size in RESOLUTIONS:
                        resized_img = extracted_image.resize((size, size), resample=Image.BILINEAR)
                        resized_img.save(ROOT / (PREPROCESSED + str(size)) / relative_path,
                                         format='JPEG')

                    # ===== Save extracted image in original resolution
                    extracted_image.save(PREPROCESSED_FOLDER / relative_path, format='JPEG')

                    # ===== Buffer features in CSV files
                    # Buffer landmarks in CSV file
                    with open(LANDMARKS_BUFFER_PATH, 'a') as lm_buffer:
                        lm_buffer.write(str(relative_path) + separator + str(landmarks) + '\n')

        print('All images processed. Storing results in NumPy arrays...')
        # ========== Conversions / Storage
        # ===== Save all extracted features as numpy array
        # ===== Landmarks
        landmarks_storage = convert_buffer_to_dict(LANDMARKS_BUFFER_PATH)
        # All landmarks
        landmarks = np.array(list(landmarks_storage.values())).astype(np.float32)
        landmarks_mean = np.mean(landmarks, axis=0)
        landmarks_cov = np.cov(landmarks, rowvar=False)
        np.save(ARRAY_LANDMARKS, landmarks)
        np.save(ARRAY_LANDMARKS_MEAN, landmarks_mean)
        np.save(ARRAY_LANDMARKS_COV, landmarks_cov)
        # 5 Landmarks
        landmarks_5 = extract_landmarks(landmarks, n=5)
        landmarks_5_mean = np.mean(landmarks_5, axis=0)
        landmarks_5_cov = np.cov(landmarks_5, rowvar=False)
        np.save(ARRAY_LANDMARKS_5, landmarks_5)
        np.save(ARRAY_LANDMARKS_5_MEAN, landmarks_5_mean)
        np.save(ARRAY_LANDMARKS_5_COV, landmarks_5_cov)
        # 10 Landmarks
        landmarks_10 = extract_landmarks(landmarks, n=10)
        landmarks_10_mean = np.mean(landmarks_10, axis=0)
        landmarks_10_cov = np.cov(landmarks_10, rowvar=False)
        np.save(ARRAY_LANDMARKS_10, landmarks_10)
        np.save(ARRAY_LANDMARKS_10_MEAN, landmarks_10_mean)
        np.save(ARRAY_LANDMARKS_10_COV, landmarks_10_cov)
        # 28 Landmarks
        landmarks_28 = extract_landmarks(landmarks, n=28)
        landmarks_28_mean = np.mean(landmarks_28, axis=0)
        landmarks_28_cov = np.cov(landmarks_28, rowvar=False)
        np.save(ARRAY_LANDMARKS_28, landmarks_28)
        np.save(ARRAY_LANDMARKS_28_MEAN, landmarks_28_mean)
        np.save(ARRAY_LANDMARKS_28_COV, landmarks_28_cov)

        # ===== Images in different resolutions and lowres pixel maps
        for size in RESOLUTIONS:
            subdir_res = ROOT / (PREPROCESSED + str(size))
            data = np.array([np.array(Image.open(fname)) for fname in subdir_res.iterdir()])
            data = data.transpose((0, 3, 1, 2))
            np.save(ROOT / ('data' + str(size) + '.npy'), data)
            if size in LOW_RESOLUTIONS:
                lowres = data.reshape((-1, 3 * size * size)) / 255.0
                lowres = lowres.astype(np.float32)
                lowres_mean = np.mean(lowres, axis=0)
                lowres_cov = np.cov(lowres, rowvar=False)
                np.save(ROOT / ('lowres' + str(size) + '.npy'), lowres)
                np.save(ROOT / ('lowres' + str(size) + '_mean.npy'), lowres_mean)
                np.save(ROOT / ('lowres' + str(size) + '_cov.npy'), lowres_cov)

        print('Preprocessing finished! You can now start to train your models!')


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
