from FaceAnonymizer.Anonymizer import Anonymizer
from config import CONVERTER_INPUT, SAMPLE_MODEL, CONVERTER_OUTPUT, CONVERTER_BASE
from PIL import Image

if __name__ == '__main__':
    anonymizer = Anonymizer(model_folder='/nfs/students/summer-term-2018/project_2/test/model')
    image = Image.open('/nfs/students/summer-term-2018/project_2/test/input.jpg')
    new_image = anonymizer(image)
    new_image.save('/nfs/students/summer-term-2018/project_2/test/result.jpg')
