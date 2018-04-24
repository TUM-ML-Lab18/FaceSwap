from torchvision.transforms import transforms

from FaceAnonymizer.Anonymizer import Anonymizer
from Preprocessor.Dataset import DatasetPerson, Resize, ToTensor, PROCESSED_IMAGES_FOLDER

if __name__ == '__main__':
    trump = DatasetPerson(PROCESSED_IMAGES_FOLDER + "/trump",
                          transform=transforms.Compose([Resize((64, 64)), ToTensor()]), detect_faces=False)
    cage = DatasetPerson(PROCESSED_IMAGES_FOLDER + "/cage",
                         transform=transforms.Compose([Resize((64, 64)), ToTensor()]), detect_faces=False)

    a = Anonymizer(trump, cage)
    a.train()
