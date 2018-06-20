import os, json
from pathlib import Path

import numpy as np
from PIL import Image

from Evaluator.Evaluator import Evaluator

if __name__ == '__main__':

    path1 = Path('/nfs/students/summer-term-2018/project_2/test_alex/original.jpg')
    path2 = Path('/nfs/students/summer-term-2018/project_2/test_alex/original.jpg')
    path3 = Path('/nfs/students/summer-term-2018/project_2/test_alex/original.jpg')
    path4 = Path('/nfs/students/summer-term-2018/project_2/test_alex/original.jpg')

    img1 = Image.open(path1)
    img2 = Image.open(path2)
    img3 = Image.open(path3)
    img4 = Image.open(path4)

    score, sim, emo = Evaluator.evaluate_image_pair(img1, img1, 6, 1)

    print("1-1")
    print(score)

    score, sim, emo = Evaluator.evaluate_image_pair(img1, img2, 6, 1)

    print("1-2")
    print(score)

    score, sim, emo = Evaluator.evaluate_image_pair(img1, img3, 6, 1)

    print("1-3")
    print(score)

    score, sim, emo = Evaluator.evaluate_image_pair(img1, img4, 6, 1)

    print("1-4")
    print(score)