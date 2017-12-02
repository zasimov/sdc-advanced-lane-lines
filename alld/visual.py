import cv2
import numpy as np

from matplotlib import pyplot


def create_outimg(binary):
    return np.dstack((binary, binary, binary)) * 255


def draw_lanes(binary, left, right):
    ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])

    left_fitx = left(ploty)
    right_fitx = right(ploty)

    if left_fitx is not None:
        pyplot.plot(left_fitx, ploty, color='yellow')
    if right_fitx is not None:
        pyplot.plot(right_fitx, ploty, color='yellow')


def lanepoly(ploty, left, right):
    leftx = left(ploty)
    rightx = right(ploty)
    left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    return np.hstack([left, right])


def draw_text(img, curvature, offset):
    cv2.putText(img, 'Radius of curvature = ' + str("{0:.2f}".format(curvature)) + 'm', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, 'Vehicle offset = ' + str("{0:.2f}".format(offset)) + 'm', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)