from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from Configuration.config_model import current_config
from Utils.Anonymizer import Anonymizer
from Utils.Logging.LoggingUtils import print_progress_bar


def convert_images():
    anonymizer = Anonymizer(
        model_folder='/home/stromaxi/ml-lab-summer-18-project-2/implementation/logs/CGAN_InstanceNoise/model',
        config=current_config)
    path = Path('/nfs/students/summer-term-2018/project_2/test_max')
    result_path = path / 'result'
    result_path.mkdir(exist_ok=True)

    for image_file in path.iterdir():
        if image_file.is_dir():
            print('Skipping image:', image_file.name)
            continue
        print('Processing image:', image_file.name)
        image = Image.open(image_file)
        image = image.convert('RGB')
        new_image = anonymizer(image)
        if new_image is not None:
            new_image.save(result_path / image_file.name.__str__())


def convert_video():
    anonymizer = Anonymizer(model_folder='/nfs/students/summer-term-2018/project_2/models/latent_model/model',
                            config=current_config, video_mode=True)
    path = Path('/nfs/students/summer-term-2018/project_2/test_simone_short/')
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
    convert_video()
