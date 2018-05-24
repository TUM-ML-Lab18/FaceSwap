import cv2
import numpy as np

from FaceAnonymizer.Anonymizer import Anonymizer
from PIL import Image
from pathlib import Path

from Logging.LoggingUtils import print_progress_bar
from configuration.run_config import current_config


def convert_images():
    anonymizer = Anonymizer(model_folder='logs/2018-05-21 21:09:02.004979/model',
                            model=current_config['model'],
                            config=current_config)
    path = Path('/nfs/students/summer-term-2018/project_2/test/')
    path_sebi = Path('/nfs/students/summer-term-2018/project_2/test_sebi/')
    for image_file in path.iterdir():
        if image_file.is_dir():
            continue
        print('Processing image:', image_file.name)
        image = Image.open(image_file)
        new_image = anonymizer(image)
        new_image.save(path_sebi / (image_file.name.__str__()))


def convert_video():
    anonymizer = Anonymizer(model_folder='/nfs/students/summer-term-2018/project_2/models/celebA', model=current_config['model'],
                            config=current_config)
    path = Path('/nfs/students/summer-term-2018/project_2/test_obama/')
    path_sebi = Path('/nfs/students/summer-term-2018/project_2/test_sebi/')
    for video_file in path.iterdir():
        if video_file.is_dir():
            continue
        print(f'Processing video:{video_file.name}')
        cap = cv2.VideoCapture(video_file.__str__())
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter((path_sebi / (video_file.name.__str__())).__str__(),
                              cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps,
                              (frame_width, frame_height))
        curr = 0
        print_progress_bar(curr, length)
        while True:
            ret, frame = cap.read()
            curr += 1
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                new_image = anonymizer(image)
                if not new_image:
                    new_image = image
                new_image = np.array(new_image)
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                out.write(new_image)
                print_progress_bar(curr, length)
            else:
                break
        cap.release()
        out.release()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    convert_images()
