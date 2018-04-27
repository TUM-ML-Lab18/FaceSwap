from FaceAnonymizer.Anonymizer import Anonymizer
from Preprocessor.Dataset import DatasetPerson, ToTensor, TRUMP, CAGE

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    trump = DatasetPerson(TRUMP, transform=ToTensor(), detect_faces=True)
    cage = DatasetPerson(CAGE, transform=ToTensor(), detect_faces=True)

    a = Anonymizer(trump, cage)
    a.train()
