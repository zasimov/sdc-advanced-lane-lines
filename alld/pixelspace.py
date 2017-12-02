"""Module contains functions to convert from pixel space to real world space"""

from alld import polynom2


# meters per pixel in y dimension
ym_per_pix = 30 / 720
# meters per pixel in x dimension
xm_per_pix = 3.7 / 700


def to_real_world_space(xs, ys):
    """Convert `xs` and `ys` from pixel space to real world space and fit `Polynom2`
    
    Return new polynom2.Polynom2
    """
    real_world_xs = xs * xm_per_pix
    real_world_ys = ys * ym_per_pix
    points = polynom2.Points(real_world_xs, real_world_ys)
    return points.fit_poly2()


def x_pix2m(pixels):
    """Convert X-pixels to meters"""
    return pixels * xm_per_pix


def y_pix2m(pixels):
    """Convert Y-pixels to meters"""
    return pixels * ym_per_pix
