from FaceAnonymizer.Anonymizer import Anonymizer
from config import CONVERTER_INPUT, SAMPLE_MODEL, CONVERTER_OUTPUT, CONVERTER_BASE

if __name__ == '__main__':
    anonymizer = Anonymizer(images_folder=CONVERTER_BASE + CONVERTER_INPUT,
                            output_folder=CONVERTER_BASE + CONVERTER_OUTPUT,
                            model_folder=SAMPLE_MODEL)
    anonymizer.convert_images()
