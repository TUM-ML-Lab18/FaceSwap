import cv2
import os
import numpy as np
import numpy.random as random
from Preprocessor.lib import umeyama

import face_recognition
from torch.utils.data import Dataset

"""inspired by http://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""


class DatasetPerson(Dataset):
    """Dataset containing images from only one person without face detection"""

    def __init__(self, root_dir, transform=None, detect_faces=False, warp_faces=True, rotation_range=10,
                 zoom_range=0.05, shift_range=0.05, size_multiplicator=10):
        """
        :param root_dir: Directory with the images.
        :param transform: Transformations applied to the images.
        :param detect_faces: Detect images with the face_locations module.
        :param warp_faces: Warp faces before supplying them to a DataLoader
        :param rotation_range: Range within the image gets rotated randomly
        :param zoom_range: Range within the image gets zoomed randomly
        :param shift_range: Range within the image gets shifted randomly
        :param size_multiplicator: Enlarges the dataset virtually times this factor
        """
        self.transform = transform
        self.warp_faces = warp_faces
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.size_multiplicator = size_multiplicator
        self.root_dir = root_dir
        self.file_names = os.listdir(self.root_dir)
        self.images = []

        l = len(self.file_names)
        printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)

        # load all images into ram
        for idx, img_name in enumerate(self.file_names):
            if not img_name.__contains__(".jpg"):
                continue
            path2img = os.path.join(self.root_dir, img_name)
            img = cv2.imread(path2img, cv2.COLOR_RGB2BGR).astype(np.float32)
            if detect_faces:
                # Cut face region via minimum bounding box of facial landmarks
                face_landmarks = face_recognition.face_landmarks(img.astype(np.uint8))

                # ignore if 2 faces detected because in most cases they don't originate form the same person
                if face_landmarks and len(face_landmarks) == 1:
                    # Extract coordinates from landmarks dict via list comprehension
                    face_landmarks_coordinates = [coordinate for feature in list(face_landmarks[0].values()) for
                                                  coordinate in feature]
                    # Determine bounding box
                    left, top = np.min(face_landmarks_coordinates, axis=0)
                    right, bottom = np.max(face_landmarks_coordinates, axis=0)
                    # => landmarks can lie outside of the image
                    # Min & max values are the borders of an image (0,0) & img.shape
                    left = 0 if left<0 else left
                    top = 0 if top<0 else top
                    right = img.shape[1]-1 if right>=img.shape[1] else right
                    bottom = img.shape[0]-1 if bottom>=img.shape[0] else bottom
                    # Extract face
                    img = img[top:bottom, left:right]

                else:
                    continue
            self.images.append(img)

            printProgressBar(idx + 1, l, prefix='Progress:', suffix='Complete', length=50)

    def __len__(self):
        return len(self.images) * self.size_multiplicator

    def __getitem__(self, idx):
        # perform random warp operation on face
        image = cv2.resize(self.images[idx % self.size_multiplicator], (256, 256))
        image = self.random_transform(image)
        warped_image, target_image = self.warp(image)
        warped_image = warped_image.transpose((2, 0, 1))
        target_image = target_image.transpose((2, 0, 1))
        return warped_image, target_image

    def random_transform(self, image):
        rotation = random.uniform(-self.rotation_range, self.rotation_range)
        scale = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        x_shift = random.uniform(-self.shift_range, self.shift_range) * 256
        y_shift = random.uniform(-self.shift_range, self.shift_range) * 256

        trans = cv2.getRotationMatrix2D((256 // 2, 256 // 2), rotation, scale)
        trans[:, 2] += (x_shift, y_shift)
        return cv2.warpAffine(image, trans, (256, 256), borderMode=cv2.BORDER_REPLICATE)

    def warp(self, image):
        # This function was taken from deepfakes/faceswap
        coverage = 160
        scale = 5
        zoom = 1
        range_ = np.linspace(128 - coverage // 2, 128 + coverage // 2, 5)
        mapx = np.broadcast_to(range_, (5, 5))
        mapy = mapx.T

        mapx = mapx + np.random.normal(size=(5, 5), scale=scale)
        mapy = mapy + np.random.normal(size=(5, 5), scale=scale)

        interp_mapx = cv2.resize(mapx, (80 * zoom, 80 * zoom))[8 * zoom:72 * zoom, 8 * zoom:72 * zoom].astype('float32')
        interp_mapy = cv2.resize(mapy, (80 * zoom, 80 * zoom))[8 * zoom:72 * zoom, 8 * zoom:72 * zoom].astype('float32')

        warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[0:65 * zoom:16 * zoom, 0:65 * zoom:16 * zoom].T.reshape(-1, 2)
        mat = umeyama(src_points, dst_points, True)[0:2]

        target_image = cv2.warpAffine(image, mat, (64 * zoom, 64 * zoom))

        return warped_image / 255.0, target_image / 255.0

    def save_processed_images(self, path):
        for idx, img in enumerate(self.images):
            cv2.imwrite(os.path.join(path, str(idx) + ".jpg"), img)


class Resize:
    """Resize the image to a predefined resolution"""

    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, img):
        return cv2.resize(img, self.resolution)


# todo make sure the input is dtype=np.float32
class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # switch dimensions
        img = img.transpose((2, 0, 1))
        return img#torch.from_numpy(img)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()