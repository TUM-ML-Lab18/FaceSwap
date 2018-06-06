import numpy as np
from pathlib import Path
from PIL import Image
from Evaluator.Evaluator import Evaluator


if __name__ == '__main__':
    scores = Evaluator.evaluate_model(model_folder='model', image_folder='/nfs/students/summer-term-2018/project_2/test/')

    sim = [s['sim'] for _,s in scores.items()]
    emo = [s['emo'] for _,s in scores.items()]

    sim = np.array(sim)
    emo = np.array(emo)

    alpha = 1
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emo - beta * (sim - 0.6)))

    print("1-1")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")

    alpha = 2
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emo - beta * (sim - 0.6)))

    print("2-1")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")

    alpha = 4
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emo - beta * (sim - 0.6)))

    print("2-1")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")