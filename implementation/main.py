from FaceAnonymizer.Anonymizer import Anonymizer
from Preprocessor.Dataset import DatasetPerson, ToTensor
from config import PROCESSED_IMAGES_FOLDER, TRUMP, CAGE, TRUMP_CAGE_BASE

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def process_trump_cage_images():
    dataset = DatasetPerson(TRUMP_CAGE_BASE + CAGE, detect_faces=True)
    dataset.save_processed_images(PROCESSED_IMAGES_FOLDER + CAGE)
    dataset = DatasetPerson(TRUMP_CAGE_BASE + TRUMP, detect_faces=True)
    dataset.save_processed_images(PROCESSED_IMAGES_FOLDER + TRUMP)


if __name__ == '__main__':
    trump = DatasetPerson(PROCESSED_IMAGES_FOLDER + TRUMP, detect_faces=False)
    cage = DatasetPerson(PROCESSED_IMAGES_FOLDER + CAGE, detect_faces=False)

    a = Anonymizer(trump, cage)
    a.train()
