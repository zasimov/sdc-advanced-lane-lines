"""Undistort and show image

Usage:

    python undistort.py --input image --camera cam.pickle

"""

from alld import camera

import cv2
from matplotlib import pyplot


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('python undistort.py')
    parser.add_argument('--image', required=True)
    parser.add_argument('--camera', required=True)
    parser.add_argument('--output', required=False)

    args = parser.parse_args()

    cam = camera.fromfile(args.camera)

    image = cv2.imread(args.image)

    undistorted = cam.undistort(image)

    if not args.output:
        pyplot.imshow(undistorted)
        pyplot.show()
    else:
        cv2.imwrite(args.output, undistorted)


