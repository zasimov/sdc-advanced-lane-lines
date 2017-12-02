"""Module contains functions to work with Polynom of Degree Two"""

import numpy as np


class Points:
    """(x, y) points"""

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        assert len(self.xs) == len(self.ys)

    def fit_poly2(self):
        """Construct Polynom2 (fitted or unfitted)"""
        if len(self.xs) and len(self.ys):
            return Polynom2(np.polyfit(self.ys, self.xs, 2))
        return Polynom2(None)

    def draw(self, outimg, color):
        outimg[self.ys, self.xs] = color

    def __len__(self):
        return len(self.xs)


class Polynom2:

    def __init__(self, polynom):
        super().__init__()
        self._polynom = polynom

    @property
    def is_fitted(self):
        return self._polynom is not None

    @property
    def coefficients(self):
        return self._polynom

    @property
    def a(self):
        return self._polynom[0]

    @property
    def b(self):
        return self._polynom[1]

    @property
    def c(self):
        return self._polynom[2]

    def curvature(self, y):
        """Curvature in y"""
        double_a = 2 * self.a
        return np.power(1 + np.square(double_a * y + self.b), 1.5) / (np.absolute(double_a))

    def __call__(self, y):
        return self.a * y ** 2 + self.b * y + self.c
