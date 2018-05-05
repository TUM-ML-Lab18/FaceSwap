import cv2
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import BICUBIC, LANCZOS
from torchvision.transforms import ToTensor, ToPILImage

from FaceAnonymizer.models.DeepFakeOriginal import DeepFakeOriginal
from Preprocessor.FaceExtractor import FaceExtractor
from config import deep_fake_config


class Anonymizer:
    def __init__(self, model_folder: str, config=deep_fake_config) -> None:
        """
        :param model_folder: Path to models folder.
        """
        self.model_folder = Path(model_folder)
        self.model = DeepFakeOriginal(None, **config['model_arguments'])
        self.model.load_model(self.model_folder)

        # use extractor and transform later get correct input for network
        self.extractor = FaceExtractor(padding=True, alignment=True, mask=np.float, margin=5)

    def __call__(self, image):
        """
        Merges an anonymized face on the scene
        TODO: merge into scene - currently only on head
        :param image: PIL image
        :return: PIL image
        """
        # Cut face
        extracted_information = self.extractor(image)
        # Resize to 64x64
        face_in = extracted_information.image.resize((128,128), resample=LANCZOS)
        # Transform into tensor
        face_in = ToTensor()(face_in)
        # feed into network
        face_out = self.model.anonymize_2(face_in.unsqueeze(0).cuda()).squeeze(0)
        # get it back to the cpu and get the data
        face_out = ToPILImage()(face_out.cpu().detach())
        # scale to original resolution
        face_out = face_out.resize(extracted_information.image.size, resample=BICUBIC)
        # TODO: Clean solution -> no hacks
        # reduce mask size
        H, W = face_out.size
        kernel = np.ones((np.int(H*10/100), np.int(W*10/100)), np.uint8)
        mask_reduced = cv2.erode(extracted_information.mask, kernel, iterations=2)
        mask_reduced = cv2.GaussianBlur(mask_reduced, (25,25), 0)
        # merge face in original image
        image = merge_face_image(face_out, extracted_information.image_unmasked, mask_reduced)

        return image

def merge_face_image(face, image, mask):
    """
    Merges a face on an image corresponding to the mask
    :param face: PIL image of the generated face
    :param image: PIL image with the original face
    :param mask: np.array
    :return: Image with merged face
    """
    # Convert PIL into np.array
    face = np.array(face)
    image = np.array(image)
    # Check type of mask
    if np.bool == mask.dtype:
        masked_image = np.where(mask[:,:,None], face, image)
    if np.float == mask.dtype:
        masked_image = mask[:,:,None] * face + (1-mask[:,:,None]) * image
        masked_image = masked_image.astype(np.uint8)
    # Reconvert np.array into PIL
    masked_image = Image.fromarray(masked_image)
    return masked_image