"""calibrate.py calculates distortion coefficients and saves them to file

Usage:

  python calibrate.py --chessboards photos --rows 8 --cols 6

"""

import glob
import os
import sys

import cv2
import numpy

from alld import camera


class Chessboard:

    def __init__(self, rows, cols):
        """Constructs Chessboard
        
        rows and cols are a numbers of INNER inner corners
        
        """
        self.rows = rows
        self.cols = cols

    @property
    def shape(self):
        return (self.rows, self.cols)

    def obj_points(self, z=0, dtype=numpy.float32):
        """obj_points calculates real-world inner corners coords"""
        shape = (self.rows * self.cols, 3)
        obj_points = numpy.zeros(shape, dtype)
        obj_points[:, :2] = numpy.mgrid[0:self.rows, 0:self.cols].T.reshape(-1, 2)
        return obj_points

    def find_corners(self, gray_img):
        """find_corners uses cv2 function to find chessboard corners
        
        Returns found flag and corners array
        """
        return cv2.findChessboardCorners(gray_img, self.shape, None)


def die(message):
    sys.stderr.write(message)
    sys.stderr.write('\n')
    exit(1)


def shots(folder, mask='*.jpg'):
    """shots reads images from folder (by mask) and converts them to gray"""
    for file_name in glob.glob(os.path.join(folder, mask)):
        bgr = cv2.imread(file_name)
        yield file_name, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('python calibrate.py')
    parser.add_argument('--chessboards', required=True, help='path to folder with chessboard images')
    parser.add_argument('--rows', required=True, type=int, help='number of chessboard rows')
    parser.add_argument('--cols', required=True, type=int, help='number of chessboard cols')
    parser.add_argument('--stop-on-fail', default=False, action='store_true', help='die if corners cannot be found at least for one image')
    parser.add_argument('-o', '--output', required=True, help='an output file name (camera file)')

    args = parser.parse_args()

    if not os.path.isdir(args.chessboards):
        die('shots folder doesn\'t exist: %s' % args.chessboards)

    chessboard = Chessboard(args.rows, args.cols)

    chess_points = chessboard.obj_points()

    obj_points = []
    img_points = []

    for file_name, shot in shots(args.chessboards):
        found, corners = chessboard.find_corners(shot)
        if not found:
            if args.stop_on_fail:
                die('cannot find corners: %s' % file_name)
            continue
        obj_points.append(chess_points)
        img_points.append(corners)

    ret, cmx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, shot.shape[::-1], None, None)

    cam = camera.Camera(cmx, dist)
    cam.save(args.output)
