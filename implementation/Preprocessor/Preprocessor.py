import cv2
import numpy as np
import face_recognition


class Preprocessor(object):

    def __init__(self):
        pass

    def extract_faces(self, image, return_borders=False):
        # Cut face region via minimum bounding box of facial landmarks
        face_landmarks = face_recognition.face_landmarks(image.astype(np.uint8))
        left, top, right, bottom = 0, 0, 0, 0

        # ignore if 2 faces detected because in most cases they don't originate form the same person
        if face_landmarks and len(face_landmarks) == 1:
            # Extract coordinates from landmarks dict via list comprehension
            face_landmarks_coordinates = [coordinate for feature in list(face_landmarks[0].values()) for
                                          coordinate in feature]
            # Determine bounding box
            left, top = np.min(face_landmarks_coordinates, axis=0)
            right, bottom = np.max(face_landmarks_coordinates, axis=0)
            # Enlarge bounding box by 5 percent (// 20) on every side
            height = bottom - top
            width = right - left
            left -= width // 20
            top -= height // 20
            right += width // 20
            bottom += height // 20
            # => landmarks can lie outside of the image
            # Min & max values are the borders of an image (0,0) & img.shape
            left = 0 if left < 0 else left
            top = 0 if top < 0 else top
            right = image.shape[1] - 1 if right >= image.shape[1] else right
            bottom = image.shape[0] - 1 if bottom >= image.shape[0] else bottom
            # Extract face
            image = image[top:bottom, left:right]
            face_detected = True
        else:
            face_detected = False
        if return_borders:
            return image, face_detected, [left, top, right, bottom]
        return image, face_detected

    def BGR2RGB(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def resize(self, image, resolution):
        return cv2.resize(image, resolution)
