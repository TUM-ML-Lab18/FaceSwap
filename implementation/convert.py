from FaceAnonymizer.Anonymizer import Anonymizer
from PIL import Image
from pathlib import Path

if __name__ == '__main__':
    anonymizer = Anonymizer(model_folder='/nfs/students/summer-term-2018/project_2/models/1_gpu_64_bs/')
    path = Path('/nfs/students/summer-term-2018/project_2/test/')
    for image_file in path.iterdir():
        if image_file.is_dir():
            continue
        print('Processing image:', image_file.name)
        image = Image.open(image_file)
        new_image = anonymizer(image)
        new_image.save(path / 'results' / image_file.name )
