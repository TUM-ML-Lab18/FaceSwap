import numpy as np
import cv2
import face_recognition
from collections import namedtuple
from PIL import Image

BoundingBox = namedtuple('BoundingBox', ('left', 'right', 'top', 'bottom'))
ExtractedFace = namedtuple('ExtractedFace',
                           ('image', 'image_unmasked', 'bounding_box', 'face_landmarks', 'rotation', 'mask'))
Rotation = namedtuple('Rotation', ('angle', 'center'))


class FaceExtractor(object):
    def __init__(self, padding=True, alignment=True, mask=np.float, margin=5):
        """
        Initializer for a FaceExtractor object
        :param padding: Boolean flag if image is padded to square image
        :param alignment: Boolean flag if image is aligned with the positions of the eyes
        :param mask: type of mask that is used for face masking {None, 'np.bool', 'np.float'}
        :param margin: Specify the margin between the bounding box and the landmarks in percent
        """
        self.padding = padding
        self.alignment = alignment
        self.mask = mask
        self.margin = margin

    def __call__(self, image):
        """
        Extracts a image with the given configuration
        :param image: PIL image
        :return: Extracted PIL image, bounding box, face landmarks as named tuple
                 None if no face is detected
        """
        image_face = None
        image_unmasked = None
        bounding_box = None
        face_landmarks = None
        rotation = None
        mask = None

        # Convert PIL image into np.array
        image = np.array(image)
        face_landmarks = extract_face_landmarks(image)
        if face_landmarks is not None:
            if self.alignment:
                # Align image with the position of the eyes
                rotation = calculcate_rotation(face_landmarks)
                R = cv2.getRotationMatrix2D(rotation.center, rotation.angle, 1.0)
                image = rotate_image(image, R)
                rotated_landmarks = rotate_landmarks(face_landmarks, R)
                # Recalculate face landmarks on aligned image
                face_landmarks = extract_face_landmarks(image)
                # If no face landmarks could be extracted from the aligned image => use transformed landmarks
                if face_landmarks is None:
                    face_landmarks = rotated_landmarks
            bounding_box = calculate_bounding_box(face_landmarks, self.margin)
            bounding_box = limit_bounding_box(image.shape[:2], bounding_box)
            image_face = crop(image, bounding_box)
            image_face = pad_image(image_face) if self.padding else image_face
            image_unmasked = image_face
            if self.mask:
                # Mask image
                mask = calculate_face_mask(face_landmarks, bounding_box, self.mask)
                mask = pad_mask(mask) if self.padding else mask
                image_face = apply_face_mask(image_face, mask)
            image_face = Image.fromarray(image_face)

        return ExtractedFace(image=image_face, image_unmasked=image_unmasked, bounding_box=bounding_box,
                             face_landmarks=face_landmarks, rotation=rotation, mask=mask)


def extract_face_landmarks(image):
    """
    Extract facial landmarks of the first detected face in the image
    :param image: np.array / cv2 image
    :return: face_landmarks or None if no face detected
    """
    face_landmarks = face_recognition.face_landmarks(image)
    return face_landmarks[0] if face_landmarks else None


def extract_landmark_coordinates(face_landmarks):
    """
    Extractes the coordinates of the landmarks from the landmarks dictionary
    :param face_landmarks: Coordinates of the facial landmarks (dict)
    :return: List with tuples that represent the coordinates
    """
    # Extract coordinates from landmarks dict via list comprehension
    face_landmarks_coordinates = [coordinate for feature in list(face_landmarks.values()) for
                                  coordinate in feature]
    return face_landmarks_coordinates


def calculate_bounding_box(face_landmarks, margin=5):
    """
    Calculate the bounding box with a margin for the given landmarks
    :param face_landmarks: Coordinates of the facial landmarks (dict)
    :param margin: Margon of the created bounding box
    :return: BoundingBox as named tuple with left, right, bottom, top
    """
    face_landmarks_coordinates = extract_landmark_coordinates(face_landmarks)
    # Determine smallest bounding box
    left, top = np.min(face_landmarks_coordinates, axis=0)
    right, bottom = np.max(face_landmarks_coordinates, axis=0)
    # Enlarge bounding box
    height = bottom - top
    width = right - left
    left -= (width * margin) // 100
    top -= (height * margin) // 100
    right += (width * margin) // 100
    bottom += (height * margin) // 100

    return BoundingBox(left=left, right=right, top=top, bottom=bottom)


def limit_bounding_box(image_shape, bounding_box):
    """
    Limits the bounding box to the size of the image
    :param image_shape: Shape of the image (H,W)
    :param bounding_box: named_tuple
    :return: BoundingBox as named tuple with left, right, bottom, top
    """
    H, W = image_shape

    left = bounding_box.left
    right = bounding_box.right
    top = bounding_box.top
    bottom = bounding_box.bottom

    left = 0 if left < 0 else left
    top = 0 if top < 0 else top
    right = W - 1 if right >= W else right
    bottom = H - 1 if bottom >= H else bottom

    return BoundingBox(left=left, right=right, top=top, bottom=bottom)


def crop(image, bounding_box):
    """
    Crops region from image by defined bounding box
    Bounding box is limited to the size of the image
    :param image: np.array / cv2 image
    :param bounding_box: named tuple with bounding box coordinates
    :return: Cropped region
    """
    return image[bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right]


def pad_image(image, color=[0, 0, 0]):
    """
    Pads image with zeros to be squared
    inspired by https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    :param image: np.array / cv2 image
    :param color: list with RGB channels
    :return: Padded image
    """
    H, W = image.shape[:2]

    output_resolution = max(H, W)
    dH = output_resolution - H
    dW = output_resolution - W

    top, bottom = dH // 2, dH - (dH // 2)
    left, right = dW // 2, dW - (dW // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=color)

    return image


def calculcate_rotation(face_landmarks):
    """
    Calculates the rotation matrix to align the face from eye coordinates
    :param face_landmarks: Coordinates of the facial landmarks (dict)
    :return: rotation: named tuple with rotation angle and rotation center
    """
    """ DEPRECATED: Alignment with PCA
    # Extract the coordinates of the eyes
    coords_eyes = np.array(face_landmarks['left_eye'] + face_landmarks['right_eye'])
    # Center of eye coordinates as rotation center
    center = np.mean(coords_eyes, axis=0)
    # PCA
    cov = np.cov((coords_eyes-center).T)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    # Select eigenvector corresponding to largest eigenvalue
    x,y = eig_vecs[:, np.argmax(eig_vals)]
    """
    # Calculcate centers of the eyes
    center_right_eye = np.mean(face_landmarks['right_eye'], axis=0)
    center_left_eye = np.mean(face_landmarks['left_eye'], axis=0)
    # Components of the vector between both eyes
    x, y = center_right_eye - center_left_eye
    # Calculate rotation angle with x & y component of the vector
    angle = np.rad2deg(np.arctan2(y, x))
    return Rotation(angle=angle, center=tuple(center_left_eye))


def rotate_image(image, R):
    """
    Rotates an image with the given rotation matrix
    :param image: np.array / cv2 image
    :param R: cv2 rotation matrix
    :return: Rotated image
    """
    H, W = image.shape[:2]
    return cv2.warpAffine(image, R, (W, H))


def rotate_landmarks(face_landmarks, R):
    """
    :param face_landmarks: Coordinates of the facial landmarks (dict)
    :param R: cv2 rotation matrix
    :return: Dict with rotated landmarks
    """
    rotated_landmarks = {}
    for feature in face_landmarks:
        n = len(face_landmarks[feature])
        # Stack coordinates into an array
        coords = np.array(face_landmarks[feature]).T
        # Augment coordinates with ones -> homogenous coordinates
        coords = np.vstack((coords, np.ones((1, n))))
        # Transform via rotation matrix
        coords = np.dot(R, coords)
        # Landmarks have to be integers
        coords = np.round(coords).astype(int)
        # Resort it again as a list of tuples and store it in the dict
        rotated_landmarks[feature] = [tuple(coordinate) for coordinate in coords.T]

    return rotated_landmarks


def calculate_face_mask(face_landmarks, bounding_box, dtype=np.float, margin=15):
    """
    Calculates a mask where in the image the face is located
    :param face_landmarks: Coordinates of the facial landmarks (dict)
    :param bounding_box: named tuple with bounding box coordinates
    :param dtype: Datatype of the created mask
    :param margin: Margin to increase mask size in percent
    :return: Mask
    """
    H = bounding_box.bottom - bounding_box.top
    W = bounding_box.right - bounding_box.left
    mask = np.zeros((H, W))
    face_landmarks_coordinates = np.array(extract_landmark_coordinates(face_landmarks))
    # Calculate relative landmark coordinates inside the bounding box
    face_landmarks_coordinates = face_landmarks_coordinates - [bounding_box.left, bounding_box.top]
    # Create convex hull that includes all landmarks
    convex_hull = cv2.convexHull(face_landmarks_coordinates)
    # Fill convex hull -> our mask
    mask = cv2.fillConvexPoly(mask, convex_hull, 1)
    # Dilate our mask -> make it bigger; Kernel size is (H,W) * margin [%]
    kernel = np.ones((np.int(H * margin / 100), np.int(W * margin / 100)), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if np.bool == dtype:
        # Boolean mask for indexing
        mask = mask.astype(np.bool)
    if np.float == dtype:
        # Float mask for indexing allows soft transition
        mask = mask.astype(np.float)
        # Blur image to make transitions smooth
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return mask


def apply_face_mask(image, mask):
    """
    Applys the given mask to the image
    :param image: np.array / cv2 image
    :param mask: np.array of type boolean or float
    :return: The masked image
    """
    if np.bool == mask.dtype:
        masked_image = np.where(mask[:, :, None], image, 0)
    if np.float == mask.dtype:
        masked_image = mask[:, :, None] * image
        masked_image = masked_image.astype(np.uint8)
    return masked_image


def pad_mask(mask):
    """
    Pads mask to be squared
    :param mask: np.array
    :return: Padded mask
    """
    H, W = mask.shape

    output_resolution = max(H, W)
    dH = output_resolution - H
    dW = output_resolution - W

    top, bottom = dH // 2, dH - (dH // 2)
    left, right = dW // 2, dW - (dW // 2)

    # Save the datatype -> 'bool' or 'float'
    dtype = mask.dtype
    mask = mask.astype(np.float)
    # Pad mask manually with zeros
    mask = np.vstack((np.zeros((top, W)), mask, np.zeros((bottom, W))))
    mask = np.hstack((np.zeros((H + dH, left)), mask, np.zeros((H + dH, right))))
    # Restore old datatype
    mask = mask.astype(dtype)

    return mask
