import numpy as np

def height(binary_image):
    """Return a height of binary_image"""
    return binary_image.shape[0]


class BottomHalfHistogram:
    """Find X coords of left and right peaks"""

    def __init__(self, binary_image):
        self._binary_image = binary_image
        h = height(self._binary_image)
        bottom_half = self._binary_image[h // 2:, :]
        self._histogram = np.sum(bottom_half, axis=0)
        self._midpoint = np.int(self._histogram.shape[0] / 2)

    @property
    def left_peak_x(self):
        return np.argmax(self._histogram[:self._midpoint])

    @property
    def right_peak_x(self):
        return np.argmax(self._histogram[self._midpoint:]) + self._midpoint
