from alld import histogram
from alld import polynom2
from alld import slidingwindow

import cv2
import numpy as np


def height(binary_image):
    """Return a height of binary_image"""
    return binary_image.shape[0]


def search(binary, nwindows=9, window_margin=100, minpix=50, outimg=None):
    """Search left and right lines using sliding window
    
    Return tuple (left, right) where left and right are polynom2.Points
    """

    window_height = height(binary) // nwindows

    hist = histogram.BottomHalfHistogram(binary)
    binary_image = slidingwindow.BinaryImage(binary)

    left_window = slidingwindow.SlidingWindow(hist.left_peak_x, height(binary), window_height, window_margin)
    right_window = slidingwindow.SlidingWindow(hist.right_peak_x, height(binary), window_height, window_margin)

    left_lane = []
    right_lane = []

    for _ in range(nwindows):
        if outimg is not None:
            left_window.draw(outimg)
            right_window.draw(outimg)

        left_slice = binary_image.slice(left_window)
        right_slice = binary_image.slice(right_window)

        left_lane.append(left_slice)
        right_lane.append(right_slice)

        if len(left_slice) > minpix:
            new_x = np.int(np.mean(binary_image.x(left_slice)))
            left_window.move_x(new_x)

        if len(right_slice) > minpix:
            new_x = np.int(np.mean(binary_image.x(right_slice)))
            right_window.move_x(new_x)

        left_window.move_down()
        right_window.move_down()

    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    left = polynom2.Points(binary_image.x(left_lane), binary_image.y(left_lane))
    right = polynom2.Points(binary_image.x(right_lane), binary_image.y(right_lane))

    return left, right


def marginsearch(binary, left_poly2, right_poly2, margin):
    """Search in a margin around the previous line position"""

    binary_image = slidingwindow.BinaryImage(binary)

    left_slice = binary_image.slice_poly2(left_poly2, margin)
    right_slice = binary_image.slice_poly2(right_poly2, margin)

    left = polynom2.Points(binary_image.x(left_slice), binary_image.y(left_slice))
    right = polynom2.Points(binary_image.x(right_slice), binary_image.y(right_slice))

    return left, right





