import numpy as np
import cv2
import face_recognition
from collections import namedtuple
from PIL import Image

BoundingBox = namedtuple('BoundingBox', ('left', 'right', 'top', 'bottom'))
ExtractedFace = namedtuple('ExtractedFace', ('image', 'bounding_box', 'face_landmarks', 'rotation_matrix'))


class FaceExtractor(object):
    def __init__(self, padding=True, alignment=True, margin=5):
        """
        Initializer for a FaceExtractor object
        :param padding: Boolean flag if image is padded to square image
        :param alignment: Boolean flag if image is aligned with the positions of the eyes
        :param margin: Specify the margin between the bounding box and the landmarks in percent
        """
        self.padding = padding
        self.alignment = alignment
        self.margin = margin

    def __call__(self, image):
        """
        Extracts a image with the given configuration
        :param image: PIL image
        :return: Extracted PIL image, bounding box, face landmarks as named tuple
                 None if no face is detected
        """
        image_face = None
        bounding_box = None
        face_landmarks = None
        R = None

        # Convert PIL image into np.array
        image = np.array(image)
        face_landmarks = self.extract_face_landmarks(image)
        if face_landmarks is not None:
            if self.alignment:
                R = self.calculcate_rotation(face_landmarks)
                image = self.rotate_image(image, R)
                rotated_landmarks = self.rotate_landmarks(face_landmarks, R)
                # Recalculate face landmarks on aligned image
                face_landmarks = self.extract_face_landmarks(image)
                # If no face landmarks could be extracted from the aligned image => use transformed landmarks
                if face_landmarks is None:
                    face_landmarks = rotated_landmarks
            bounding_box = self.calculate_bounding_box(face_landmarks)
            image_face = self.crop(image, bounding_box)
            if self.padding:
                image_face = self.pad(image_face)
            image_face = Image.fromarray(image_face)

        return ExtractedFace(image=image_face, bounding_box=bounding_box, face_landmarks=face_landmarks,
                                           rotation_matrix=R)

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
        :param face_landmarks: Coordinates of the facial landmarks (dict)
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

    def calculcate_rotation(self, face_landmarks):
        """
        Calculates the rotation matrix to align the face from eye coordinates
        :param face_landmarks: Coordinates of the facial landmarks (dict)
        :return: cv2 rotation matrix
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
        return cv2.getRotationMatrix2D(tuple(center_left_eye), angle, 1.0)

    def rotate_image(self, image, R):
        """
        Rotates an image with the given rotation matrix
        :param image: np.array / cv2 image
        :param R: cv2 rotation matrix
        :return: Rotated image
        """
        H, W, C = image.shape
        return cv2.warpAffine(image, R, (W,H))

    def rotate_landmarks(self, face_landmarks, R):
        """
        :param landmarks: Coordinates of the facial landmarks (dict)
        :param R: cv2 rotation matrix
        :return: Dict with rotated landmarks
        """
        rotated_landmarks = {}
        for feature in face_landmarks:
            n = len(face_landmarks[feature])
            # Stack coordinates into an array
            coords = np.array(face_landmarks[feature]).T
            # Augment coordinates with ones -> homogenous coordinates
            coords = np.vstack((coords, np.ones((1,n))))
            # Transform via rotation matrix
            coords = np.dot(R, coords)
            # Landmarks have to be integers
            coords = np.round(coords).astype(int)
            # Resort it again as a list of tuples and store it in the dict
            rotated_landmarks[feature] = [tuple(coordinate) for coordinate in coords.T]

        return rotated_landmarks