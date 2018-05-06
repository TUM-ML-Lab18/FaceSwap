import numpy as np
import cv2
from PIL import Image

class FaceReconstructor(object):
    """
    Inverts the extraction steps of the FaceExtractor
    1. Invert fine cropping of the ROI
    2. Invert alignment of eyes
    3. Invert masking of background
    4. Invert coarse cropping of the ROI
    """
    def __init__(self, mask_factor=-10):
        self.face_decropper_fine = FaceDecropperFine()
        self.face_dealigner = FaceDealigner()
        self.face_demasker = FaceDemasker(mask_factor)
        self.face_decropper_coarse = FaceDecropperCoarse()

    def __call__(self, processed_image, extraction_information):
        """
        :param processed_image: PIL image
        :param extraction_information: namedtuple with the following elements
                    * image_original: The original scene (PIL image)
                    * image_cropped: The cropped region from the original image (PIL image)
                    * bounding_box: Namedtuple with the coordinates of the cropped region in
                                    the original image
                    * mask: Mask applied to filter background
                    * rotation: Namedtuple with rotation and rotation center to align eyes
        :return: reconstructed image (PIL image)
        """
        # Convert PIL image into np.array
        processed_image = np.array(processed_image)
        original_image = np.array(extraction_information.image_original)
        coarse_cropped_image = np.array(extraction_information.image_cropped)

        decropped_image = self.face_decropper_fine(processed_image,
                                                   extraction_information.bounding_box_fine,
                                                   extraction_information.bounding_box_coarse)
        dealigned_image = self.face_dealigner(decropped_image,
                                              extraction_information.rotation)
        demasked_image = self.face_demasker(dealigned_image,
                                            coarse_cropped_image,
                                            extraction_information.mask)
        decropped_image = self.face_decropper_coarse(demasked_image,
                                                     original_image,
                                                     extraction_information.bounding_box_coarse)
        # Convert np.array into PIL image
        reconstructed_image = Image.fromarray(decropped_image)
        return reconstructed_image


class FaceDecropperFine(object):
    """
    Invert the fine cropping of the aligned and masked image
    """
    def __call__(self, fine_cropped_image, bounding_box_fine, bounding_box_coarse):
        """
        :param fine_cropped_image: The constructed image
        :param bounding_box_fine: named tuple with absolute coordinates of the fine crop
        :param bounding_box_coarse: named tuple with absolute coordinates of the coarse crop
        :return: The aligned face only coarsely cropped
        """
        H = bounding_box_coarse.bottom - bounding_box_coarse.top
        W = bounding_box_coarse.right - bounding_box_coarse.left
        # Determine number of channels
        C = fine_cropped_image.shape[2] if len(fine_cropped_image.shape) == 3 else 1
        top = bounding_box_fine.top
        bottom = bounding_box_fine.bottom
        left = bounding_box_fine.left
        right = bounding_box_fine.right
        decropped_image = np.zeros((H, W, C))
        decropped_image[top:bottom, left:right] = fine_cropped_image

        return decropped_image

class FaceDealigner(object):
    """
    Invert the alignment of the face with the position of the eyes
    """
    def __call__(self, aligned_image, rotation):
        """
        :param aligned_image: The uncropped constructed image
        :param rotation: The rotation applied to align the image
        :return: The constructed image in original pose
        """
        H, W = aligned_image.shape[:2]
        R = cv2.getRotationMatrix2D(rotation.center, -rotation.angle, 1.0)
        dealigned_image = cv2.warpAffine(aligned_image, R, (W, H))
        return dealigned_image

class FaceDemasker(object):
    """
    Invert the masking of the image
    The mask can be additionally accessed via an morphological operation
    Recommended is an erosion to fit only the center of the face
    """
    def __init__(self, morphing=-10):
        """
        :param morphing: Size of the morphological kernel in percent of
                 the image size
                 * morphing > 0: dilation -> increase mask
                 * morphing < 0: erosion -> decrease mask (recommended)
        """
        self.morphing = morphing

    def __call__(self, masked_image, cropped_image, mask):
        """
        :param masked_image: The masked constructed image
        :param cropped_image: The cropped original image
        :param mask: The mask applied to the image
        :return: The reconstructed image in the cropped scene
        """
        dtype = mask.dtype
        H, W = masked_image.shape[:2]

        # Make boolean mask accessible for morphological operations
        if np.bool == dtype:
            mask = np.where(mask, 1.0, 0.0)

        # Calculate image resolution dependent kernel (H==W) (odd size)
        k_size = int(abs(self.morphing)/100 * H)
        k_size = k_size if (k_size % 2 == 1) else k_size+1
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
        # Execute morphological operations
        # Dilation -> increase masked region
        # Erosion -> decrease masked region
        operation = cv2.MORPH_ERODE if self.morphing < 0 else cv2.MORPH_DILATE
        mask = cv2.morphologyEx(mask, op=operation, kernel=kernel)

        # Check type of mask
        if np.bool == dtype:
            demasked_image = np.where(mask[:,:,None], masked_image, cropped_image)
        if np.float == dtype:
            demasked_image = mask[:,:,None] * masked_image + \
                             (1-mask[:,:,None]) * cropped_image
            demasked_image = demasked_image.astype(np.uint8)
        return demasked_image

class FaceDecropperCoarse(object):
    """
    Invert the coarse cropping of the image
    """
    def __call__(self, cropped_image, original_image, bounding_box):
        """
        :param cropped_image: The cropped constructed image
        :param original_image: The original image
        :param bounding_box: Indicator where the cropped region was in the image
        :return: The reconstructed image in the original scene
        """
        decropped_image = original_image.copy()
        top = bounding_box.top
        bottom = bounding_box.bottom
        left = bounding_box.left
        right = bounding_box.right
        decropped_image[top:bottom, left:right] = cropped_image
        return decropped_image
