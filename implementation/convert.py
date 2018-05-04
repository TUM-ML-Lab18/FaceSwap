from FaceAnonymizer.Anonymizer import Anonymizer
from PIL import Image
from pathlib import Path

from config import deep_fake_config

if __name__ == '__main__':
    anonymizer = Anonymizer(model_folder='./model/', config=deep_fake_config)
    path = Path('/nfs/students/summer-term-2018/project_2/test/')
    path_sebi = Path('/nfs/students/summer-term-2018/project_2/test_sebi/')
    for image_file in path.iterdir():
        if image_file.is_dir():
            continue
        print('Processing image:', image_file.name)
        image = Image.open(image_file)
        new_image = anonymizer(image)
        new_image.save(path_sebi / ("sebi_" + image_file.name.__str__()))
