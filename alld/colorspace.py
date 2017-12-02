import collections

import cv2


# type to describe colors in HLS color space
HLS = collections.namedtuple('HLS', ['h', 'l', 's'])
# type to describe colors in RGB color space
RGB = collections.namedtuple('RGB', ['r', 'g', 'b'])


def rgb2hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def bgr2hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)


def bgr2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def hls2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_HLS2BGR)