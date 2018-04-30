import numpy as np
import cv2
import face_recognition
from collections import namedtuple

BoundingBox = namedtuple('BoundingBox', ('left', 'right', 'top', 'bottom'))
ExtractedFace = namedtuple('ExtractedFace', ('image', 'bounding_box', 'face_landmarks'))

class FaceExtractor(object):
    def __init__(self, output_resolution=(64,64), margin=5):
        """
        Initializer for a FaceExtractor object
        :param output_resolution: Specify the resolution of the extracted face
        :param margin: Specify the margin between the bounding box and the landmarks in percent
        """
        self.output_resolution = output_resolution
        self.margin = margin

    def __call__(self, image):
        """
        Extracts a image with the given configuration
        :param image:
        :return: Extracted image, bounding box, face landmarks as named tuple
                 None if no face is detected
        """
        extracted_face = None
        face_landmarks = self.extract_face_landmarks(image)
        if face_landmarks is not None:
            bounding_box = self.calculate_bounding_box(face_landmarks)
            image_face = self.crop(image, bounding_box)
            if self.output_resolution:
                cv2.resize(image_face, self.output_resolution)
            extracted_face = ExtractedFace(image=image_face, bounding_box=bounding_box, face_landmarks=face_landmarks)

        return extracted_face

    def extract_face_landmarks(self, image):
        """
        Extract facial landmarks of the first detected face in the image
        :param image: Image with a face
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
        :param image:
        :param bounding_box:
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
        :param image:
        :param bounding_box:
        :return: Cropped region
        """
        bounding_box = self.limit_bounding_box(image, bounding_box)
        return image[bounding_box.top:bounding_box.bottom, bounding_box.left:bounding_box.right]
