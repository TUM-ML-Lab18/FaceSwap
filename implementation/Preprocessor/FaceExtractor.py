import numpy as np
import cv2
import face_recognition
from recordclass import recordclass
from PIL import Image, ImageDraw


ExtractionInformation = recordclass('ExtractionInformation',
                           ('image_original', 'image_cropped',
                            'bounding_box_coarse', 'offsets_coarse', 'size_coarse',
                            'mask', 'rotation',
                            'bounding_box_fine', 'offsets_fine', 'size_fine',
                            'landmarks'))
BoundingBox = recordclass('BoundingBox', ('left', 'right', 'top', 'bottom'))
Rotation = recordclass('Rotation', ('angle', 'center'))

facial_features = [
    'chin',
    'left_eyebrow',
    'right_eyebrow',
    'nose_bridge',
    'nose_tip',
    'left_eye',
    'right_eye',
    'top_lip',
    'bottom_lip'
    ]

class FaceExtractor(object):
    """
    Extracts a face with the following sub steps:
    1. Extract landmarks of facial features
    2. Crop an axis aligned region based on those landmarks
    3. Mask the face to face out background
    4. Align face horizontally with the eyes position
    5. Crop image fine to center face
    """
    def __init__(self, margin=0.05, mask_factor=10, mask_type=np.float):
        """
        Initializer for a FaceExtractor object
        :param margin: Factor to adapt size of the cropped region
                        * margin > 0: increase bounding box
                        * margin < 0: decrease bounding box
        :param mask_factor: Strength of increasing or decreasing the mask
                          * mask_factor > 0: increase mask
                          * mask_factor < 0: decrease mask
        :param mask_type: Datatype of the created mask:
                          * np.bool: hard, sharp mask
                          * np.float: soft, blurred mask
        """
        self.landmarks_extractor = LandmarksExtractor()
        # 1.05: Crop coarse face with additional safety margin
        # => no facial landmarks can get lost during rotation
        self.face_cropper_coarse = FaceCropperCoarse(margin=margin*1.05)
        self.face_masker = FaceMasker(dtype=mask_type, morphing=mask_factor)
        self.face_aligner = FaceAligner()
        self.face_cropper_fine = FaceCropperFine(margin=margin)

    def __call__(self, image):
        """
        Extracts a image with the given configuration
        :param image: PIL image
        :return: extracted_face: Extracted face
                 extraction_information: namedtuple with the following elements
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
        """
        extracted_face = None
        original_image = None
        cropped_image = None
        bounding_box_coarse = None
        offsets_coarse = None
        size_coarse = None
        mask = None
        rotation = None
        bounding_box_fine = None
        offsets_fine = None
        size_fine = None
        landmarks = None

        # Convert PIL image into np.array
        image = np.array(image)
        landmarks = self.landmarks_extractor(image)
        if landmarks is not None:
            original_image = image
            cropped_image, bounding_box_coarse, offsets_coarse, size_coarse= \
                self.face_cropper_coarse(original_image, landmarks)
            masked_image, mask = self.face_masker(cropped_image, landmarks)
            aligned_image, rotation = self.face_aligner(masked_image, landmarks)
            extracted_face, bounding_box_fine, offsets_fine, size_fine = \
                self.face_cropper_fine(aligned_image, landmarks)
            # Convert np.array into PIL image
            extracted_face = Image.fromarray(extracted_face)
            original_image = Image.fromarray(original_image)
            cropped_image = Image.fromarray(cropped_image)
            # Save landmarks in list
            landmarks = list_landmarks(landmarks)

        extraction_information = ExtractionInformation(image_original=original_image,
                                                       image_cropped=cropped_image,
                                                       bounding_box_coarse=bounding_box_coarse,
                                                       offsets_coarse=offsets_coarse,
                                                       size_coarse=size_coarse,
                                                       mask=mask, rotation=rotation,
                                                       bounding_box_fine=bounding_box_fine,
                                                       offsets_fine=offsets_fine,
                                                       size_fine=size_fine,
                                                       landmarks=landmarks)

        return extracted_face, extraction_information

def list_landmarks(landmarks_dict):
    """
    Converts the coordinates of the landmarks from the landmarks dictionary
    to a list of coordinates
    :param landmarks_dict: Dict of facial landmarks
    :return: List with tuples that represent the coordinates
    """
    # Extract coordinates from landmarks dict via list comprehension
    landmarks_list = [coordinate for feature in list(landmarks_dict.values())
                      for coordinate in feature]
    return landmarks_list

def dict_landmarks(landmarks_list):
    """
    Converts the coordinates of the landmarks from a list of landmarks
    to a landmarks dictionary
    :param landmarks_list: List of facial landmarks
    :return: Dict with tuples that represent the coordinates
    """
    landmarks_dict = {}
    landmarks_dict['chin'] = landmarks_list[0:17]
    landmarks_dict['left_eyebrow'] = landmarks_list[17:22]
    landmarks_dict['right_eyebrow'] = landmarks_list[22:27]
    landmarks_dict['nose_bridge'] = landmarks_list[27:31]
    landmarks_dict['nose_tip'] = landmarks_list[31:36]
    landmarks_dict['left_eye'] = landmarks_list[36:42]
    landmarks_dict['right_eye'] = landmarks_list[42:48]
    landmarks_dict['top_lip'] = landmarks_list[48:60]
    landmarks_dict['bottom_lip'] = landmarks_list[60:72]
    return landmarks_dict

def update_landmarks(landmarks_dict, transformation):
    """
    Updates all landmarks based on the given transformation
    :param landmarks_dict: Dict of facial landmarks
    :param transformation: Lambda function with the transformation
    :return:
    """
    for feature in landmarks_dict:
        landmarks = []
        for landmark in landmarks_dict[feature]:
            landmark = transformation(landmark)
            # Landmarks have to be integers
            #landmark = np.round(landmark).astype(int)
            landmarks.append(tuple(landmark))
        landmarks_dict[feature] = landmarks

def draw_landmarks(image, landmarks_dict):
    """
    Draw landmarks into image
    :param image: PIL image
    :param landmarks_dict: Dict of facial landmarks
    :return: PIL image with the landmarks drawn
    """
    landmarks_face = image.copy()
    d = ImageDraw.Draw(landmarks_face)
    r = 3
    for facial_feature in facial_features:
        for x, y in landmarks_dict[facial_feature]:
            d.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0))
        d.line(landmarks_dict[facial_feature], width=r, fill=(0, 255, 0))
    return landmarks_face

class LandmarksExtractor(object):
    """
    Extract facial landmarks of the first detected face in the image with the
    external face_recognition module
    """
    def __call__(self, image):
        """
        :param image: np.array / cv2 image
        :return: face_landmarks or None if no face detected
        """
        landmarks = face_recognition.face_landmarks(image)
        return landmarks[0] if landmarks else None


class FaceCropperCoarse(object):
    """
    Crops face coarsely to further preprocess it
    Center of the bounding box is determined by the center of all landmarks
    Size of the cropped region is determined by the largest distance between
    any landmark and additionally a parametrized margin
    """
    def __init__(self, margin=0.05):
        """
        :param margin: Factor to adapt size of the cropped region
                        * margin > 0: increase bounding box
                        * margin < 0: decrease bounding box
        """
        self.margin = margin

    def __call__(self, image, landmarks_dict):
        """
        Crops face coarsely and updates the landmarks accordingly
        :param image: cv2 image / np.array
        :param landmarks_dict: Dict of facial landmarks
        :return: (cropped_image, bounding_box, offsets):
                    * cropped_image: The cropped image
                    * bounding_box: named tuple with absolute coordinates of the ROI
                                    in the given image
                    * offsets: named tuple with the offsets (padding + image out of range)
                               for every bounding box side
        """
        bounding_box, offsets, size = self.calculate_coarse_bounding_box(image,
                                                                         landmarks_dict)
        cropped_image = self.apply_coarse_crop(image, bounding_box, offsets, size)

        # Update landmarks: Linear coordinate shift
        transformation = lambda x: x - np.array([bounding_box.left-offsets.left,
                                                 bounding_box.top-offsets.top])
        update_landmarks(landmarks_dict, transformation)

        return cropped_image, bounding_box, offsets, size

    def calculate_coarse_bounding_box(self, image, landmarks_dict):
        """
        Calculates a bounding box centered at the face
        :param image: cv2 image / np.array
        :param landmarks_dict: Dict of facial landmarks
        :return: (bounding_box, offsets, size):
                    * bounding_box: named tuple with absolute coordinates of the ROI
                                    in the given image
                    * offsets: named tuple with the offsets (padding + image out of range)
                               for every bounding box side
                    * size: size of the cropped region
        """
        H, W = image.shape[:2]

        # ===================== Bounding box size
        # Transform landmarks into list of coordinates for calculations
        landmarks_list = list_landmarks(landmarks_dict)
        # Calculate the center (W,H) of the face via center of landmarks
        center = np.mean(landmarks_list, axis=0)
        # Calculate the distances of the landmarks to the center
        dist = np.linalg.norm(landmarks_list-center, axis=1)
        # Select the maximum distance as dimension for the bounding box
        # and add a safety margin
        size = int(max(dist) * (1+self.margin))*2

        # ===================== Bounding box + Offset (Indices outside the image)
        # Calculate bounding box and limit it to points inside the image
        # Additionally add offset inside the cropped image if bounding box is limited
        # => Center of the face should be center of the cropped image
        # ATTENTION 1: Coordinates of the landmarks in format W,H
        # ATTENTION 2: floor function because integer rounds towards zero (neg values!)
        top = int(np.floor(center[1] - size/2))
        top, dtop = (top, 0) if (top >= 0) else (0, -top)
        bottom = int(np.floor(center[1] + size/2))
        bottom, dbottom = (bottom, size) if (H >= bottom) else (H, (H+size-bottom))
        left = int(np.floor(center[0] - size/2))
        left, dleft = (left, 0) if (left >= 0) else (0, -left)
        right = int(np.floor(center[0] + size/2))
        right, dright = (right, size) if (W >= right) else (W, (W+size-right))

        # Store values in named tuple
        bounding_box = BoundingBox(left=left, right=right, top=top, bottom=bottom)
        offsets = BoundingBox(left=dleft, right=dright, top=dtop, bottom=dbottom)

        return bounding_box, offsets, size

    def apply_coarse_crop(self, image, bounding_box, offsets, size):
        """
        Apply the crop
        :param image: np.array / cv2 image
        :param bounding_box: named tuple with absolute coordinates of the ROI
                             in the given image
        :param offsets: named tuple with the offsets (padding + image out of range)
                        for every bounding box side
        :param size: size of the cropped region
        :return: The cropped image
        """
        cropped_image = np.zeros((size, size, image.shape[2]), dtype=np.uint8)
        # If bounding box touches original images limits the cropped image is
        # padded such that the face is centered
        cropped_image[offsets.top:offsets.bottom, offsets.left:offsets.right] = \
            image[bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right]
        return cropped_image


class FaceMasker(object):
    """
    Masks image to suppress background details
    Mask is determined by the convex hull of all landmarks and an optional
    morphological operation to increase or decrease the mask.
    """
    def __init__(self, dtype=np.bool, morphing=10):
        """
        :param dtype: Datatype of the created mask:
                      * np.bool: hard, sharp mask
                      * np.float: soft, blurred mask
        :param morphing: Size of the morphological kernel in percent of
                         the image size
                         * morphing > 0: dilation -> increase mask (recommended)
                         * morphing < 0: erosion -> decrease mask
        """
        self.dtype = dtype
        self.morphing = morphing

    def __call__(self, image, landmarks_dict):
        """
        Masks face
        :param image: cv2 image / np.array
        :param landmarks_dict: Dict of facial landmarks
        :return: (masked_image, mask):
                    * masked_image: The masked image
                    * mask: The mask applied to the image
        """
        mask = self.calculate_mask(image, landmarks_dict)
        masked_image = self.apply_mask(image, mask)

        return masked_image, mask

    def calculate_mask(self, image, landmarks_dict):
        """
        Calculates a mask where in the image the face is located
        :param image: cv2 image / np.array
        :param landmarks_dict: Dict of facial landmarks
        :return: mask
        """
        H, W = image.shape[:2]
        mask = np.zeros((H, W))
        # Transform landmarks into list of coordinates for calculations
        landmarks_list = list_landmarks(landmarks_dict)
        # Create convex hull that includes all landmarks
        convex_hull = cv2.convexHull(np.array(landmarks_list))
        # Fill convex hull with ones
        mask = cv2.fillConvexPoly(mask, convex_hull, 1)
        # Calculate image resolution dependent kernel (H==W) (odd size)
        k_size = int(abs(self.morphing)/100 * H)
        k_size = k_size if (k_size % 2 == 1) else k_size+1
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
        # Execute morphological operations
        # Dilation -> increase masked region
        # Erosion -> decrease masked region
        operation = cv2.MORPH_DILATE if self.morphing > 0 else cv2.MORPH_ERODE
        mask = cv2.morphologyEx(mask, op=operation, kernel=kernel)
        # Blur mask if floating mask -> softer transition
        if np.float == self.dtype:
            mask = cv2.GaussianBlur(mask, (k_size,k_size), 0)
        # Convert data type
        mask = mask.astype(self.dtype)

        return mask

    def apply_mask(self, image, mask):
        """
        Apply the given mask to the image
        :param image: np.array / cv2 image
        :param mask: np.array of type boolean or float
        :return: The masked image
        """
        if np.bool == self.dtype:
            masked_image = np.where(mask[:, :, None], image, 0)
        if np.float == self.dtype:
            masked_image = mask[:, :, None] * image
            masked_image = masked_image.astype(np.uint8)
        return masked_image


class FaceAligner(object):
    """
    Aligns face by the position of the eyes
    Center of the rotation is determined by the center of all landmarks
    Rotation angle is the angle of the vector between the eyes
    """
    def __call__(self, image, landmarks_dict):
        """
        Aligns face such that eyes are horizontal and updates the landmarks
        accordingly
        :param image: np.array / cv2 image
        :param landmarks_dict: Dict of facial landmarks
        :return: (aligned_image, rotation):
                    * aligned_image: The eyes aligned image
                    * rotation: The rotation applied to align the image
        """
        rotation = self.calculcate_rotation(landmarks_dict)
        R = cv2.getRotationMatrix2D(rotation.center, rotation.angle, 1.0)
        aligned_image = self.apply_rotation(image, R)

        # Update landmarks: Apply affine transformation to homogeneous coordinate
        transformation = lambda x: np.dot(R, np.array(x+(1,)))
        update_landmarks(landmarks_dict, transformation)

        return aligned_image, rotation

    def calculcate_rotation(self, landmarks_dict):
        """
        Calculates the rotation matrix to align the face from eye coordinates
        :param landmarks_dict: Dict of facial landmarks
        :return: rotation: named tuple with rotation angle and rotation center
        """
        # Calculate centers of the eyes
        center_right_eye = np.mean(landmarks_dict['right_eye'], axis=0)
        center_left_eye = np.mean(landmarks_dict['left_eye'], axis=0)
        # Components of the vector between both eyes
        dx, dy = center_right_eye - center_left_eye
        # Calculate rotation angle with x & y component of the eyes vector
        angle = np.rad2deg(np.arctan2(dy, dx))
        # Center of landmarks as rotation center (transform to list for calculations)
        center = np.mean(list_landmarks(landmarks_dict), axis=0)
        return Rotation(angle=angle, center=tuple(center))

    def apply_rotation(self, image, R):
        """
        Apply the given rotation to the image
        :param image: np.array / cv2 image
        :param R: cv2 rotation matrix
        :return: The rotated image
        """
        H, W = image.shape[:2]
        return cv2.warpAffine(image, R, (W, H))

class FaceCropperFine(object):
    """
    Crops face fine
    Bounding box is determined by the minimum rectangle containing all
    landmarks and additionally a parametrized margin
    """
    def __init__(self, margin=0.05):
        """
        :param margin: Factor to adapt size of the cropped region
                        * margin > 0: increase bounding box
                        * margin < 0: decrease bounding box
        """
        self.margin = margin

    def __call__(self, image, landmarks_dict):
        """
        Crops face fine and updates the landmarks accordingly
        If image is not squared a padding is added
        :param image: cv2 image / np.array
        :param landmarks_dict: Dict of facial landmarks
        :return: (cropped_image, bounding_box, offsets):
                    * cropped_image: The cropped image
                    * bounding_box: Indicator where the cropped region was in
                                    the given image
                    * offsets: named tuple with the offsets (padding + image out of range)
                               for every bounding box side
        """
        bounding_box, offsets, size = self.calculate_fine_bounding_box(image,
                                                                       landmarks_dict)
        cropped_image = self.apply_fine_crop(image, bounding_box, offsets, size)

        # Update landmarks: Linear coordinate shift
        transformation = lambda x: x - np.array([bounding_box.left, bounding_box.top])
        update_landmarks(landmarks_dict, transformation)

        return cropped_image, bounding_box, offsets, size

    def calculate_fine_bounding_box(self, image, landmarks_dict):
        """
        Calculates a bounding box containing all landmarks
        :param image: cv2 image / np.array
        :param landmarks_dict: Dict of facial landmarks
        :return: (bounding_box, offsets, size):
                    * bounding_box: named tuple with absolute coordinates of the ROI
                                    in the given image
                    * offsets: named tuple with the offsets (padding + image out of range)
                               for every bounding box side
                    * size: size of the cropped region
        """
        H, W = image.shape[:2]


        # ===================== Bounding box
        # Transform landmarks into list of coordinates for calculations
        landmarks_list = list_landmarks(landmarks_dict)
        # Determine smallest bounding box
        left, top = np.min(landmarks_list, axis=0)
        right, bottom = np.max(landmarks_list, axis=0)
        # Resize bounding box according to margin
        bbH = bottom - top
        bbW = right - left
        # ATTENTION: floor function because integer rounds towards zero (neg values!)
        left = int(np.floor(left - W * self.margin))
        top = int(np.floor(top - H * self.margin))
        right = int(np.floor(right + W * self.margin))
        bottom = int(np.floor(bottom + H * self.margin))

        # ===================== Padding + bounding box size
        # Dimensions of the bounding box
        bbH = bottom - top
        bbW = right - left
        size = max(bbH,bbW)
        # Calculate the padding
        dH, dW = 0, 0
        if bbH < bbW:
            dH = (bbW - bbH) // 2
        elif bbH < bbW:
            dW = (bbH - bbW) // 2

        # ===================== Offset (Indices outside the image)
        # Limit bounding box to image interior
        # Difference will be subtracted to match dimensions
        top, dtop = (top, 0) if (top >= 0) else (0, -top)
        bottom, dbottom = (bottom, 0) if (H >= bottom) else (H, (H-bottom))
        left, dleft = (left, 0) if (left >= 0) else (0, -left)
        right, dright = (right, 0) if (W >= right) else (W, (W-right))

        bounding_box = BoundingBox(left=left, right=right, top=top, bottom=bottom)
        offsets = BoundingBox(left=dW+dleft, right=dW+bbW+dright,
                              top=dH+dtop, bottom=dH+bbH+dbottom)

        return bounding_box, offsets, size

    def apply_fine_crop(self, image, bounding_box, offsets, size):
        """
        Apply the crop
        :param image: np.array / cv2 image
        :param bounding_box: named tuple with absolute coordinates of the
                             ROI in the given image
        :param offsets: named tuple with the offsets (padding + image out of range)
                        for every bounding box side
        :param size: size of the cropped region
        :return: The cropped image
        """
        cropped_image = np.zeros((size, size, image.shape[2]), dtype=np.uint8)
        # If bounding box touches original images limits the cropped image is
        # padded such that the face is centered
        cropped_image[offsets.top:offsets.bottom,offsets.left:offsets.right] = \
            image[bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right]

        return cropped_image
