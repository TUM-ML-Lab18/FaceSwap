from FaceAnonymizer.Anonymizer import Anonymizer
from Preprocessor.Dataset import DatasetPerson, ToTensor, PROCESSED_IMAGES_FOLDER

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

if __name__ == '__main__':
    trump = DatasetPerson(PROCESSED_IMAGES_FOLDER + "/trump", transform=ToTensor(), detect_faces=False)
    cage = DatasetPerson(PROCESSED_IMAGES_FOLDER + "/cage", transform=ToTensor(), detect_faces=False)

    a = Anonymizer(trump, cage)
    a.train()
