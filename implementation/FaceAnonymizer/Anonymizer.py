from pathlib import Path

import numpy as np
from PIL.Image import BICUBIC, LANCZOS
from torchvision.transforms import ToTensor, ToPILImage

from FaceAnonymizer.models.DeepFakeOriginal import DeepFakeOriginal
from Preprocessor.FaceExtractor import FaceExtractor
from Preprocessor.FaceReconstructor import FaceReconstructor
from config import deep_fake_config


class Anonymizer:
    def __init__(self, model_folder: str, config=deep_fake_config) -> None:
        """
        :param model_folder: Path to models folder.
        """
        self.config = config
        self.model_folder = Path(model_folder)
        self.model = DeepFakeOriginal(None, **config['model_arguments'])
        self.model.load_model(self.model_folder)

        # use extractor and transform later get correct input for network
        self.extractor = FaceExtractor(mask_type=np.float, margin=0.05, mask_factor=10)
        self.reconstructor = FaceReconstructor(mask_factor=-20)

    def __call__(self, image):
        """
        Merges an anonymized face on the scene
        :param image: PIL image
        :return: PIL image
        """
        # Extract face
        extracted_face, extracted_information = self.extractor(image)

        # Resize to 64x64
        face_in = extracted_face.resize(self.config['dataset_arguments']['img_size'], resample=LANCZOS)
        # Transform into tensor
        face_in = ToTensor()(face_in)
        # feed into network
        face_out = self.model.anonymize(face_in.unsqueeze(0).cuda()).squeeze(0)
        # get it back to the cpu and get the data
        face_out = ToPILImage()(face_out.cpu().detach())
        # scale to original resolution
        face_out = face_out.resize(extracted_face.size, resample=BICUBIC)

        # Constructed scene with new face
        constructed_image = self.reconstructor(face_out, extracted_information)

        return constructed_image
