import json
import numpy as np
from pathlib import Path
from PIL import Image
from Evaluator import Evaluator


if __name__ == '__main__':
    Evaluator.evaluate_model(model_folder='model', image_folder='/nfs/students/summer-term-2018/project_2/test/')