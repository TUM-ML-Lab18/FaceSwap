import json
from pathlib import Path

import numpy as np
from PIL import Image

if __name__ == '__main__':
    MERKEL_DATASET = Path('/nfs/students/summer-term-2018/project_2/data/MEGA_Merkel_Trump/preprocessed/B')
    reference_image = Image.open('/nfs/students/summer-term-2018/project_2/test/70. bundeskanzlerin-angela-merkel.jpg')

    score_list = []

    # for folder in MERKEL_DATASET.iterdir():
    #    print(f'Processing {folder}.')
    #    for img in folder.iterdir():
    #        img = Image.open(img)
    #        score = Evaluator.evaluate_image_pair(reference_image, img)
    #        score_list.append(score)

    # with open('./score_list.json', 'w') as h_json:
    #    json.dump(score_list, h_json)
    with open('./score_list.json', 'r') as h_json:
        data = json.load(h_json)
    data = np.array(data)
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    hist = np.histogram(data)
    print(f'mean: {mean}, var: {var}, hist: {hist}')
    # Evaluator.evaluate_model()
