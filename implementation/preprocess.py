from Preprocessor.Preprocessor import Preprocessor
from Preprocessor.FaceExtractor import FaceExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    face_extractor = FaceExtractor(margin=0.05, mask_type=np.bool, mask_factor=10)
    preprocessor = Preprocessor(face_extractor)
    root = Path('folder to process')
    preprocessor(root)