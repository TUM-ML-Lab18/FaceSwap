import json
from pathlib import Path

from Configuration.config_general import ANNOTATIONS
from Preprocessor.FaceExtractor import FaceExtractor
from Preprocessor.Preprocessor import Preprocessor


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
    face_extractor = FaceExtractor(margin=0.05, sharp_edge=True, mask_factor=10)
    preprocessor = Preprocessor(face_extractor)
    root = Path('/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/')
    preprocessor(root)
