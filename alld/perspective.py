""""Module performs persperctive transform

"""

import collections
import operator

import cv2
import numpy as np


# Pair and Map4 are structures for Perspective Transform
# Pair describes a pair (src, dst)
Pair = collections.namedtuple('Pair', ['src', 'dst'])


class _Map4(collections.namedtuple('_Map4', ['p1', 'p2', 'p3', 'p4'])):

    @property
    def src(self):
        return np.array(
            list(map(operator.attrgetter('src'), self)),
            dtype=np.float32,
        )

    @property
    def dst(self):
        return np.array(
            list(map(operator.attrgetter('dst'), self)),
            dtype=np.float32,
        )


class Perspective:

    def __init__(self, p1, p2, p3, p4):
        map4 = _Map4(p1, p2, p3, p4)
        src = map4.src
        dst = map4.dst
        self.mtx = cv2.getPerspectiveTransform(src, dst)
        self.backmtx = np.linalg.inv(self.mtx)

    def warp(self, undistorted_image):
        """Warp Perspective with Linear interpolation"""
        shape = (undistorted_image.shape[1], undistorted_image.shape[0])
        return cv2.warpPerspective(undistorted_image, self.mtx, shape, flags=cv2.INTER_LINEAR)

    def unwarp(self, image):
        return cv2.warpPerspective(image, self.backmtx, (image.shape[1], image.shape[0]))
