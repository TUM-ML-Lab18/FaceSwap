import numpy as np
from pathlib import Path
from PIL import Image
from Evaluator.Evaluator import Evaluator


if __name__ == '__main__':
    scores = Evaluator.evaluate_model(model_folder='model', image_folder='/nfs/students/summer-term-2018/project_2/test/')
    scores = np.array(scores)

    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")