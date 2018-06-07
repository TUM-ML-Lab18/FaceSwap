import json
from pathlib import Path
from Evaluator.Evaluator import Evaluator

import numpy as np
from PIL import Image

if __name__ == '__main__':

    path = '/nfs/students/summer-term-2018/project_2/test_alex/'
    sims = []
    emos = []

    for image_a in path.iterdir():
        if image_a.is_dir():
            continue
        for image_b in path.iterdir():
            if image_b.is_dir():
                continue
            img1 = Image.open(Path(image_a))
            img2 = Image.open(Path(image_b))

            _, sim, emo = Evaluator.evaluate_image_pair(img1, img2)
            sims.append(sim)
            emos.append(emo)

    sims = np.array(sims)
    emos = np.array(emos)
    
    print('Mean sim score: ' + sims.mean())
    print('sim score std: ' + sims.std())

    print('Mean emo score: ' + sims.mean())
    print('emo score std: ' + sims.std())

    alpha = 1
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emos - beta * (sims - 0.6)))

    print("1-1")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")

    alpha = 2
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emos - beta * (sims - 0.6)))

    print("2-1")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")

    alpha = 4
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emos - beta * (sims - 0.6)))

    print("4-1")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")