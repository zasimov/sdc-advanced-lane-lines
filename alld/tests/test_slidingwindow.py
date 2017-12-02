import unittest

from alld import slidingwindow

import numpy


class TestBinaryImage(unittest.TestCase):

    def test_x_set(self):

        # binary image mock (shape 2x3)
        binary = numpy.array([
            [1, 0, 2],
            [3, 0, 4],
        ])

        binary_image = slidingwindow.BinaryImage(binary)

        self.assertEqual(binary_image._x.shape, (4,))
        self.assertListEqual(list(binary_image._x), [0, 2, 0, 2])

    def test_y_set(self):

        # binary image mock (shape 2x3)
        binary = numpy.array([
            [1, 0, 2],
            [3, 0, 4],
        ])

        binary_image = slidingwindow.BinaryImage(binary)

        self.assertEqual(binary_image._y.shape, (4,))
        self.assertListEqual(list(binary_image._y), [0, 0, 1, 1])

    def test_slice(self):

        # binary image mock (shape 3x4)
        binary = numpy.array([
            [1, 0, 3, 4],
            [5, 0, 7, 8],
            [9, 10, 11, 12],
        ])

        binary_image = slidingwindow.BinaryImage(binary)

        sliding_window = slidingwindow.SlidingWindow(
            x_current=1, y_current=2,
            height=3, margin=1)

        slice = binary_image.slice(sliding_window)

        x = binary_image.x(slice)
        y = binary_image.y(slice)

        binary[y, x] = 50

        self.assertListEqual(list(map(list, list(binary))), [
            [50, 0, 50, 4],
            [50, 0, 50, 8],
            [50, 50, 50, 12]
        ])
