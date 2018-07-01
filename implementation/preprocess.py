from Preprocessor.FaceExtractor import FaceExtractor
from Preprocessor.Preprocessor import Preprocessor

# Script to preprocess the data
# Root-path of the data set has to be set in Configuration/config_general.py as
# ROOT = Path(/path_to_your_data/)

if __name__ == '__main__':
    face_extractor = FaceExtractor(margin=0.05, sharp_edge=True, mask_factor=10)
    preprocessor = Preprocessor(face_extractor)
    preprocessor()
