from FaceAnonymizer.Trainer import Trainer
#from Preprocessor.Dataset import DatasetPerson
from Preprocessor.Preprocessor import Preprocessor
from config import PROCESSED_IMAGES, TRUMP, CAGE, TRUMP_CAGE_BASE


def process_trump_cage_images():
    p = Preprocessor(TRUMP_CAGE_BASE)
    return p.get_dataset


if __name__ == '__main__':
    dataset = process_trump_cage_images()
# process_trump_cage_images()


# preprocessor = Preprocessor(rotation_range=10, zoom_range=0.05, shift_range=0.05,
#                            hue_range=7, saturation_range=0.2, brightness_range=80,
#                            flip_probability=0.5)

# trump = DatasetPerson(PROCESSED_IMAGES + TRUMP, preprocessor, detect_faces=False)
# cage = DatasetPerson(PROCESSED_IMAGES + CAGE, preprocessor, detect_faces=False)

# a = Trainer(trump, cage)
# a.train()
