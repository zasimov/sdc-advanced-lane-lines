import unittest

from alld import histogram

import numpy


class TestBottomHalfHistogram(unittest.TestCase):

    def test_peaks_test1_jpg(self):
        binary = numpy.array([
            [0, 0, 0],
            [10, 0, 20],
            [10, 3, 20],
        ])

        hist = histogram.BottomHalfHistogram(binary)

        self.assertEqual(hist.left_peak_x, 0)
        self.assertEqual(hist.right_peak_x, 2)
