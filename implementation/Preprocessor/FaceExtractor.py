import numpy as np
import cv2
import face_recognition
from collections import namedtuple
from PIL import Image

BoundingBox = namedtuple('BoundingBox', ('left', 'right', 'top', 'bottom'))
ExtractedFace = namedtuple('ExtractedFace', ('image', 'bounding_box', 'face_landmarks'))


class FaceExtractor(object):
    def __init__(self, padding=True, margin=5):
        """
        Initializer for a FaceExtractor object
        :param padding: Boolean flag if image is padded to square image
        :param margin: Specify the margin between the bounding box and the landmarks in percent
        """
        self.padding = padding
        self.margin = margin

    def __call__(self, image):
        """
        Extracts a image with the given configuration
        :param image: PIL image
        :return: Extracted PIL image, bounding box, face landmarks as named tuple
                 None if no face is detected
        """
        # Convert PIL image into np.array
        image = np.array(image)
        extracted_face = None
        face_landmarks = self.extract_face_landmarks(image)
        if face_landmarks is not None:
            bounding_box = self.calculate_bounding_box(face_landmarks)
            image_face = self.crop(image, bounding_box)
            if self.padding:
                image_face = self.pad(image_face)
            image_face = Image.fromarray(image_face)
            extracted_face = ExtractedFace(image=image_face, bounding_box=bounding_box, face_landmarks=face_landmarks)

        return extracted_face

    def extract_face_landmarks(self, image):
        """
        Extract facial landmarks of the first detected face in the image
        :param image: np.array / cv2 image
        :return: face_landmarks or None if no face detected
        """
        face_landmarks = face_recognition.face_landmarks(image)
        return face_landmarks[0] if face_landmarks else None

    def calculate_bounding_box(self, face_landmarks):
        """
        Calculate the bounding box with a margin for the given landmarks
        :param face_landmarks: Coordinates of the facial landmarks
        :return: BoundingBox as named tuple with left, right, bottom, top
        """
        # Extract coordinates from landmarks dict via list comprehension
        face_landmarks_coordinates = [coordinate for feature in list(face_landmarks.values()) for
                                      coordinate in feature]
        # Determine smallest bounding box
        left, top = np.min(face_landmarks_coordinates, axis=0)
        right, bottom = np.max(face_landmarks_coordinates, axis=0)
        # Enlarge bounding box
        height = bottom - top
        width = right - left
        left -= (width * self.margin) // 100
        top -= (height * self.margin) // 100
        right += (width * self.margin) // 100
        bottom += (height * self.margin) // 100

        return BoundingBox(left=left, right=right, top=top, bottom=bottom)

    def limit_bounding_box(self, image, bounding_box):
        """
        Limits the bounding box to the size of the image
        :param image: np.array / cv2 image
        :param bounding_box: named_tuple
        :return: BoundingBox as named tuple with left, right, bottom, top
        """
        left = bounding_box.left
        right = bounding_box.right
        top = bounding_box.top
        bottom = bounding_box.bottom

        left = 0 if left < 0 else left
        top = 0 if top < 0 else top
        right = image.shape[1] - 1 if right >= image.shape[1] else right
        bottom = image.shape[0] - 1 if bottom >= image.shape[0] else bottom

        return BoundingBox(left=left, right=right, top=top, bottom=bottom)

    def crop(self, image, bounding_box):
        """
        Crops region from image by defined bounding box
        Bounding box is limited to the size of the image
        :param image: np.array / cv2 image
        :param bounding_box: named_tuple
        :return: Cropped region
        """
        bounding_box = self.limit_bounding_box(image, bounding_box)
        return image[bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right]

    def pad(self, image, color=[0,0,0]):
        """
        Pads image with zeros to be squared
        inspired by https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
        :param image: np.array / cv2 image
        :param color: list with RGB channels
        :return: Padded image
        """
        H, W, C = image.shape

        output_resolution = max(H,W)
        dH = output_resolution - H
        dW = output_resolution - W

        top, bottom = dH // 2, dH - (dH // 2)
        left, right = dW // 2, dW - (dW // 2)

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)

        return image
