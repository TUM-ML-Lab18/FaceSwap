import json
from pathlib import Path
from Evaluator.Evaluator import Evaluator

import numpy as np
from PIL import Image

if __name__ == '__main__':

    img1 = Image.open(Path('/nfs/students/summer-term-2018/project_2/test_alex/merkel1.jpg'))
    img2 = Image.open(Path('/nfs/students/summer-term-2018/project_2/test_alex/merkel2.jpg'))

    score = Evaluator.evaluate_image_pair(img1, img2)
    print(score)