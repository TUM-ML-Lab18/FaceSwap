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
        self.face_sharpener = FaceSharpener()
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
                    * bounding_box_coarse: Namedtuple with the coordinates of the cropped ROI
                                           in the original image
                    * offsets_coarse: index shift to pad image and prevent indices out of image
                    * size_coarse: size (quadratic) of the coarse cropped image
                    * mask: Mask applied to filter background
                    * rotation: Namedtuple with rotation and rotation center to align eyes
                    * bounding_box_fine: Namedtuple with the coordinates of the fine cropped
                                         ROI in the coarse cropped region
                    * offsets_fine: index shift to pad image and prevent indices out of image
                    * size_fine: size (quadratic) of the fine cropped image
                    * landmarks: coordinates of facial regions (x,y)
        :return: reconstructed image (PIL image)
        """
        # Convert PIL image into np.array
        processed_image = np.array(processed_image)
        original_image = np.array(extraction_information.image_original)
        coarse_cropped_image = np.array(extraction_information.image_cropped)

        sharpened_image = self.face_sharpener(processed_image)
        decropped_image = self.face_decropper_fine(sharpened_image,
                                                   extraction_information.bounding_box_fine,
                                                   extraction_information.offsets_fine,
                                                   extraction_information.size_coarse)
        dealigned_image = self.face_dealigner(decropped_image,
                                              extraction_information.rotation)
        demasked_image = self.face_demasker(dealigned_image,
                                            coarse_cropped_image,
                                            extraction_information.mask)
        decropped_image = self.face_decropper_coarse(demasked_image,
                                                     original_image,
                                                     extraction_information.bounding_box_coarse,
                                                     extraction_information.offsets_coarse)
        # Convert np.array into PIL image
        reconstructed_image = Image.fromarray(decropped_image)
        return reconstructed_image

class FaceSharpener(object):
    """
    Sharpen the given image
    Sharpening via inverse gaussian filtering on the
    L channel of the image in the CIELab color space
    """
    def __init__(self, sharp_factor=5):
        """
        :param sharp_factor: Sharpening degree
        """
        self.sharp_factor = sharp_factor

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        # Extract L channel
        L = image[:,:,0]
        # Inverse filtering
        L_blur = cv2.GaussianBlur(L, (0,0), self.sharp_factor)
        L_sharp = cv2.addWeighted(L_blur, -1, L, 2, 0)
        # Substitute L channel with sharpened L channel
        image[:,:,0] = L_sharp
        image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)

        return image

class FaceDecropperFine(object):
    """
    Invert the fine cropping of the aligned and masked image
    """
    def __call__(self, cropped_image, bounding_box, offsets, size_coarse):
        """
        :param cropped_image: The constructed image
        :param bounding_box: named tuple with absolute coordinates of the fine crop
        :param offsets: named tuple with the offsets (padding + image out of range) of the fine
                        crop for every bounding box side
        :param size_coarse: size of the coarse cropped image
        :return: The aligned face only coarsely cropped
        """
        decropped_image = np.zeros((size_coarse, size_coarse, cropped_image.shape[2]), dtype=np.uint8)
        # Invert the crop
        decropped_image[bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right] = \
            cropped_image[offsets.top:offsets.bottom,offsets.left:offsets.right]
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
    def __call__(self, cropped_image, original_image, bounding_box, offsets):
        """
        :param cropped_image: The cropped constructed image
        :param original_image: The original image
        :param bounding_box: Indicator where the cropped region was in the image
        :param offsets: named tuple with the offsets (padding + image out of range) of the
                        crop for every bounding box side
        :return: The reconstructed image in the original scene
        """
        decropped_image = original_image.copy()
        decropped_image[bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right] = \
            cropped_image[offsets.top:offsets.bottom, offsets.left:offsets.right]
        return decropped_image
