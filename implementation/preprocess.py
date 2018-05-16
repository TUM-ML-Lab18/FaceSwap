import json

from Preprocessor.Preprocessor import Preprocessor
from Preprocessor.FaceExtractor import FaceExtractor
from pathlib import Path
import numpy as np

from configuration.general_config import ANNOTATIONS, CELEBA


def process_annoations(root_folder):
    root_folder = Path(root_folder)
    annotations = {}
    with open(root_folder / "Anno/" / ANNOTATIONS) as f:
        line_count = int(f.readline())
        f.readline()  # read label names
        for i in range(line_count):
            line_parts = f.readline().split()
            img_key = f"images/{line_parts[0]}"
            values = [max(int(x), 0) for x in line_parts[1:]]
            annotations[img_key] = values
    # Save json file
    with open(root_folder / ANNOTATIONS, 'w') as lm_json:
        json.dump(annotations, lm_json)


if __name__ == '__main__':
    # face_extractor = FaceExtractor(margin=0.05, mask_type=np.bool, mask_factor=10)
    # preprocessor = Preprocessor(face_extractor)
    # root = Path('folder to process')
    # preprocessor(root)
    process_annoations(CELEBA)
