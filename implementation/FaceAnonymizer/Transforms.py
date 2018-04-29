import cv2
import numpy as np
import lib.umeyama as umeyama


class RandomTransform(object):

    def __init__(self, rotation_range=10, zoom_range=0.05, shift_range=0.05, hue_range=7, saturation_range=0.2,
                 brightness_range=80, flip_probability=0.5):
        """
        :param rotation_range: Range within the image gets rotated randomly
        :param zoom_range: Range within the image gets zoomed randomly
        :param shift_range: Range within the image gets shifted randomly
        :param hue_range: Range within the image's hue gets shifted randomly
        :param saturation_range: Range within the image's saturation gets shifted randomly
        :param brightness_range: Range within the image's brightness gets shifted randomly
        :param flip_probability: Probability of a random flip of the image
        """
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.brightness_range = brightness_range
        self.flip_probability = flip_probability

    def __call__(self, image):
        rotation = np.random.uniform(-self.rotation_range, self.rotation_range)
        scale = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        x_shift = np.random.uniform(-self.shift_range, self.shift_range) * 256
        y_shift = np.random.uniform(-self.shift_range, self.shift_range) * 256

        trans = cv2.getRotationMatrix2D((256 // 2, 256 // 2), rotation, scale)
        trans[:, 2] += (x_shift, y_shift)
        image = cv2.warpAffine(image, trans, (256, 256), borderMode=cv2.BORDER_REPLICATE)

        if np.random.random() < self.flip_probability:
                image = image[:, ::-1]

        h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        h += np.random.uniform(-self.hue_range, self.hue_range)
        s += np.random.uniform(-self.saturation_range, self.saturation_range)
        v += np.random.uniform(-self.brightness_range, self.brightness_range)

        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


class RandomWarp(object):

    def __init__(self, warp_factor=5):
        """
        :param warp_factor: The warping factor
        """
        self.warp_factor = warp_factor

    def __call__(self, image):
        H, W, C = image.shape

        # Generate warped image
        # Create coarse mapping with the size of the image
        grid_size = 5
        grid = (grid_size, grid_size)
        range_x = np.linspace(0, W, grid_size)
        range_y = np.linspace(0, H, grid_size)
        mapx, mapy = np.meshgrid(range_x, range_y)
        # Add randomness to the mapping -> warps image
        warp_mapx = mapx + np.random.normal(size=grid, scale=self.warp_factor)
        warp_mapy = mapy + np.random.normal(size=grid, scale=self.warp_factor)
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

        return warped_image, target_image