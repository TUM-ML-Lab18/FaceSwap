import json
from pathlib import Path
from Evaluator.Evaluator import Evaluator

import numpy as np
from PIL import Image

if __name__ == '__main__':

    img1 = Image.open(Path('/nfs/students/summer-term-2018/project_2/test_alex/merkel1.jpg'))
    img2 = Image.open(Path('/nfs/students/summer-term-2018/project_2/test_alex/merkel2.jpg'))

    score_list = []

    # for folder in MERKEL_DATASET.iterdir():
    #    print(f'Processing {folder}.')
    #    for img in folder.iterdir():
    #        img = Image.open(img)
    #        score = Evaluator.evaluate_image_pair(reference_image, img)
    #        score_list.append(score)

    # with open('./score_list.json', 'w') as h_json:
    #    json.dump(score_list, h_json)

    score = Evaluator.evaluate_image_pair(img1, img2)
    print(score)