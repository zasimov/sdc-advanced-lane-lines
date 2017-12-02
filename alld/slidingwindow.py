import cv2
import numpy as np


def height(binary_image):
    """Return a height of binary_image"""
    return binary_image.shape[0]


class SlidingWindow:
    """Describes a rectangle on binary image
    
    SlidingWindow has following attributes:
      - high_y - 'y' coord of top of rectangle
      - low_y = 'y' coord of bottom of rectangle
      - high_x - see high_y
      - low_x - see low_y
      
    """

    def __init__(self, x_current, y_current, height, margin, color=(0, 255, 0)):
        self.height = height
        self.margin = margin
        self._y_high = y_current
        self._x_current = x_current
        self._color = color

    @property
    def y_high(self):
        return self._y_high

    @property
    def y_low(self):
        return self._y_high - self.height

    @property
    def x_low(self):
        return self._x_current - self.margin

    @property
    def x_high(self):
        return self._x_current + self.margin

    def move_down(self):
        """Move sliding window down"""
        self._y_high -= self.height

    def move_x(self, new_x):
        self._x_current = new_x

    def draw(self, outimg, thinkness=2):
        cv2.rectangle(outimg, (self.x_low, self.y_low), (self.x_high, self.y_high), self._color, thinkness)


class BinaryImage:
    """Wrapper for cv2 binary image"""

    def __init__(self, binary_image):
        self._binary_image = binary_image
        nonzero = self._binary_image.nonzero()
        self._x = np.array(nonzero[1])
        self._y = np.array(nonzero[0])

    def slice(self, sliding_window):
        """Return X coords of nonzero points from sliding window"""
        inds = ((self._x >= sliding_window.x_low) & (self._x <= sliding_window.x_high) &
                (self._y >= sliding_window.y_low) & (self._y <= sliding_window.y_high))
        return inds.nonzero()[0]

    def slice_poly2(self, poly2, margin):
        xs = poly2(self._y)
        inds = ((self._x > xs - margin) & (self._x < xs + margin))
        return inds

    def x(self, slice):
        """Return x-coords for slice"""
        return self._x[slice]

    def y(self, slice):
        """Return y-coords for slice"""
        return self._y[slice]


