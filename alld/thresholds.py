"""Module to create a thresholded binary image"""

from alld import colorspace

import abc
import cv2
import numpy as np


class Colorspace:
    RGB = 0
    HLS = 1
    GRAY = 2


class Threshold(abc.ABC):

    COLORSPACE = Colorspace.RGB

    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    @abc.abstractclassmethod
    def core(self, image):
        pass

    def filter_(self, candidate):
        """Create binary image by 'candidate'"""
        binary_output = np.zeros_like(candidate)
        binary_output[(candidate >= self.min) & (candidate <= self.max)] = 1
        return binary_output

    def __call__(self, image):
        candidate = self.core(image)
        return self.filter_(candidate)


class Direction:
    X = 1
    Y = 2


def scaled_sobel(gray_image, direction):
    """Calculate absolute scaled cv2.Sobel
    
    Applies (HEIGHT, WIDTH, CHANNELS) image.
    
    Returns (HEIGHT, WIDTH) matrix.    
    """
    if direction == Direction.X:
        abs_sobel = np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    return np.uint8(255 * abs_sobel / np.max(abs_sobel))


class GrayscaleThreshold(Threshold):

    COLORSPACE = Colorspace.GRAY


class AbsSobelXThreshold(GrayscaleThreshold):
    """Applies grayscaled image and returns binary image (Direction.X)"""

    NAME = 'sobelx'

    def core(self, gray_image):
        return scaled_sobel(gray_image, Direction.X)


class AbsSobelYThreshold(GrayscaleThreshold):
    """Applies grayscaled image and returns binary image (Direction.Y)"""

    NAME = 'sobely'

    def core(self, gray_image):
        return scaled_sobel(gray_image, Direction.Y)


class MagSobelThreshold(GrayscaleThreshold):
    """Applies grayscaled image and returns binary image (magnitude)"""

    NAME = 'mag'

    def __init__(self, min, max, kernel_size):
        super().__init__(min, max)
        self.kernel_size = kernel_size

    def core(self, gray_image):
        sobelx = np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=self.kernel_size))
        sobely = np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.kernel_size))
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        return gradmag


class DirectionThreshold(GrayscaleThreshold):

    NAME = 'dir'

    def __init__(self, min, max, kernel_size):
        super().__init__(min, max)
        self.kernel_size = kernel_size

    def core(self, gray_image):
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        return absgraddir


class HLSThreshold(Threshold):
    """Select color using one channel"""

    H = 0
    L = 1
    S = 2

    COLORSPACE = Colorspace.HLS

    def __init__(self, name, min, max, channel):
        super().__init__(min, max)
        self.NAME = name
        self.channel = channel

    def core(self, image):
        return image[:, :, self.channel]


class Thresholds:

    def __init__(self, *filters):
        self._filters = []
        self._filters.extend(filters)

    def cores(self, image):
        gray = colorspace.bgr2gray(image)
        hls = colorspace.bgr2hls(image)
        binaries = {}
        for filter_ in self._filters:
            if filter_.COLORSPACE == Colorspace.GRAY:
                binary = filter_.core(gray)
            elif filter_.COLORSPACE == Colorspace.HLS:
                binary = filter_.core(hls)
            else:
                binary = filter_.core(image)
            binaries[filter_.NAME] = binary
        return binaries

    def __call__(self, image):
        gray = colorspace.bgr2gray(image)
        hls = colorspace.bgr2hls(image)
        binaries = {}
        for filter_ in self._filters:
            if filter_.COLORSPACE == Colorspace.GRAY:
                binary = filter_(gray)
            elif filter_.COLORSPACE == Colorspace.HLS:
                binary = filter_(hls)
            else:
                binary = filter_(image)
            binaries[filter_.NAME] = binary
        return binaries


