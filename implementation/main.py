from FaceAnonymizer.Anonymizer import Anonymizer
from Preprocessor.Dataset import DatasetPerson
from Preprocessor.Preprocessor import Preprocessor
from config import PROCESSED_IMAGES_FOLDER, TRUMP, CAGE, TRUMP_CAGE_BASE


def process_trump_cage_images():
    dataset = DatasetPerson(TRUMP_CAGE_BASE + CAGE, detect_faces=True)
    dataset.save_processed_images(PROCESSED_IMAGES_FOLDER + CAGE)
    dataset = DatasetPerson(TRUMP_CAGE_BASE + TRUMP, detect_faces=True)
    dataset.save_processed_images(PROCESSED_IMAGES_FOLDER + TRUMP)


if __name__ == '__main__':
    #process_trump_cage_images()
    preprocessor = Preprocessor(rotation_range=10, zoom_range=0.05, shift_range=0.05,
                                hue_range=7, saturation_range=0.2, brightness_range=80,
                                flip_probability=0.5)

    trump = DatasetPerson(PROCESSED_IMAGES_FOLDER + TRUMP, preprocessor, detect_faces=False)
    cage = DatasetPerson(PROCESSED_IMAGES_FOLDER + CAGE, preprocessor, detect_faces=False)

    a = Anonymizer(trump, cage)
    a.train()
