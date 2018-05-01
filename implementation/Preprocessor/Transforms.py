import cv2
import numpy as np
from lib.umeyama import umeyama
from PIL import Image
import torchvision.transforms as transforms

class RandomWarp(object):
    """
    Class to apply random warps on PIL images
    """

    def __init__(self, warp_factor=1):
        """
        :param warp_factor: The warping factor
        """
        self.warp_factor = warp_factor

    def __call__(self, image):
        """
        Warps an PIL Image
        :param image: Image to be warped
        :return: PIL Image: Randomly warped image
                 PIL Image: Affine transformed target (Umeyama)
        """

        # Convert PIL image into np.array
        image = np.array(image)

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
        interp_mapx = cv2.resize(warp_mapx, (W, H)).astype('float32')
        interp_mapy = cv2.resize(warp_mapy, (W, H)).astype('float32')
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
        warped_image = warped_image[H // 10:H // 10 * 9, W // 10:W // 10 * 9]

        # Resize target on same size as cutted warped image
        target_image = cv2.resize(target_image, warped_image.shape[:2])

        # Convert np.arrays into PIL images
        warped_image = Image.fromarray(warped_image.astype(np.uint8))
        target_image = Image.fromarray(target_image.astype(np.uint8))

        return warped_image, target_image

class TupleResize(object):
    """
    Class to resize a tuple of images
    """
    def __init__(self, resolution=(256, 256)):
        """
        :param resolution: Resolution to scale the tuple of images to
        """
        self.resolution = resolution
        self.resize = transforms.Resize(self.resolution)

    def __call__(self, image_tuple):
        """

        :param image_tuple: Images to be resized
        :return: Tuple of resized images
        """
        return self.resize(image_tuple[0]), self.resize(image_tuple[1])


class TupleToTensor(object):
    """
    Class to convert tuples of images into tensors
    """
    def __init__(self):
        """
        Initializer for TupleToTensor class
        """
        self.toTensor = transforms.ToTensor()

    def __call__(self, image_tuple):
        """

        :param image_tuple: Images to be converted in tensors
        :return: Tuple of tensors
        """
        return self.toTensor(image_tuple[0]), self.toTensor(image_tuple[1])
