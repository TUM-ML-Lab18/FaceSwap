import cv2
import os
import numpy as np
import numpy.random as random
from Preprocessor.lib import umeyama

import face_recognition
from torch.utils.data import Dataset

"""inspired by http://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""


class DatasetPerson(Dataset):
    """Dataset containing images from only one person"""

    def __init__(self, root_dir, transform=None, detect_faces=False, warp_faces=True, rotation_range=10,
                 zoom_range=0.05, shift_range=0.05, hue_range=7, saturation_range=0.2, brightness_range=80,
                 flip_probability=0.4, size_multiplicator=10):
        """
        :param root_dir: Directory with the images.
        :param transform: Transformations applied to the images.
        :param detect_faces: Detect images with the face_locations module.
        :param warp_faces: Warp faces before supplying them to a DataLoader
        :param rotation_range: Range within the image gets rotated randomly
        :param zoom_range: Range within the image gets zoomed randomly
        :param shift_range: Range within the image gets shifted randomly
        :param hue_range: Range within the image's hue gets shifted randomly
        :param saturation_range: Range within the image's saturation gets shifted randomly
        :param brightness_range: Range within the image's brightness gets shifted randomly
        :param flip_probability: Probability of a random flip of the image
        :param size_multiplicator: Enlarges the dataset virtually times this factor
        """
        self.transform = transform
        self.warp_faces = warp_faces
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.brightness_range = brightness_range
        self.flip_probability = flip_probability
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
            try:
                img = img = cv2.cvtColor(cv2.imread(path2img), cv2.COLOR_BGR2RGB).astype(np.float32)
            except Exception as e:
                print(path2img, '\n', e)
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
                    left = 0 if left < 0 else left
                    top = 0 if top < 0 else top
                    right = img.shape[1] - 1 if right >= img.shape[1] else right
                    bottom = img.shape[0] - 1 if bottom >= img.shape[0] else bottom
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
        warped_image, target_image = None, None
        if self.warp_faces:
            warped_image, target_image = self.warp(image)
        else:
            warped_image, target_image = image, image
        warped_image = cv2.resize(warped_image, (64,64))
        target_image = cv2.resize(target_image, (64,64))
        warped_image = warped_image.transpose((2, 0, 1))
        target_image = target_image.transpose((2, 0, 1))
        return self.img_normalize(warped_image), self.img_normalize(target_image)

    def random_transform(self, image):
        rotation = random.uniform(-self.rotation_range, self.rotation_range)
        scale = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        x_shift = random.uniform(-self.shift_range, self.shift_range) * 256
        y_shift = random.uniform(-self.shift_range, self.shift_range) * 256

        trans = cv2.getRotationMatrix2D((256 // 2, 256 // 2), rotation, scale)
        trans[:, 2] += (x_shift, y_shift)
        image = cv2.warpAffine(image, trans, (256, 256), borderMode=cv2.BORDER_REPLICATE)

        if random.random() < self.flip_probability:
            image = image[:, ::-1]

        h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        h += random.uniform(-self.hue_range, self.hue_range)
        s += random.uniform(-self.saturation_range, self.saturation_range)
        v += random.uniform(-self.brightness_range, self.brightness_range)

        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    def warp(self, image, warp_factor=5):
        H, W, C = image.shape

        # Generate warped image
        # Create coarse mapping with the size of the image
        grid_size = 5
        grid = (grid_size, grid_size)
        range_x = np.linspace(0, W, grid_size)
        range_y = np.linspace(0, H, grid_size)
        mapx, mapy = np.meshgrid(range_x, range_y)
        # Add randomness to the mapping -> warps image
        warp_mapx = mapx + np.random.normal(size=grid, scale=warp_factor)
        warp_mapy = mapy + np.random.normal(size=grid, scale=warp_factor)
        # Scale coarse mapping to fine mapping
        interp_mapx = cv2.resize(warp_mapx, (H, W)).astype('float32')
        interp_mapy = cv2.resize(warp_mapy, (H, W)).astype('float32')
        # Apply warping
        warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

        # Generate target image
        # Define points correspondences between warped source map and linear destination map
        src_points = np.vstack([warp_mapx.ravel(), warp_mapy.ravel()]).T
        dst_points = np.vstack([mapx.ravel(), mapy.ravel()]).T
        # Umeyama transformation: Calculate affine mapping between two sets of point correspondences
        mat = umeyama(src_points, dst_points, True)[0:2]
        # Apply Umeyama transformation
        target_image = cv2.warpAffine(image, mat, (W, H))

        # Warping makes border regions almost completely black
        # Create mapping larger than image and cut border regions (outer 10% of the image)
        warped_image = warped_image[W // 10:W // 10 * 9, H // 10:H // 10 * 9]
        #target_image = target_image[W // 10:W // 10 * 9, H // 10:H // 10 * 9]

        return warped_image, target_image

    def img_normalize(self, image):
        return image / 255.0

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
        return img  # torch.from_numpy(img)


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