import collections

from alld import polynom2

import numpy as np


class Line():
    """Define a class to receive the characteristics of each line detection"""

    def __init__(self, maxlen=10):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients for the most recent fit
        # self.current_poly contains alld.polynom2.Polynom2
        self.current_poly2 = None
        # poly2 of the last n fits
        self.history = collections.deque(maxlen=maxlen)
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = []
        # y values for detected line pixels
        self.ally = []

    def collect_points(self, points):
        self.allx.extend(points.xs)
        self.ally.extend(points.ys)

    def fit(self, line):
        self.detected = line.is_fitted
        if not self.detected:
            if not self.current_poly2:
                self.current_poly2 = line
            return

        self.current_poly2 = line

        self.history.append(line)

    @property
    def smoothed(self):
        coeffs = [poly2.coefficients for poly2 in self.history]
        mean_poly2 = np.mean(coeffs, axis=0)
        return polynom2.Polynom2(mean_poly2)

