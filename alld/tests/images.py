import os

import cv2


_cwd = os.path.dirname(__file__)
_test_images = os.path.dirname(_cwd)
_test_images = os.path.dirname(_test_images)
_test_images = os.path.join(_test_images, 'test_images')


def imread(file_name):
    path = os.path.join(_test_images, file_name)
    return cv2.imread(path)
