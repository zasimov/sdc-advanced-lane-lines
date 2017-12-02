"""SDC Advanced Lane Line Detection pipeline

NOTE: if collect_pixels is True script collects a lot of data and stores it in memory (~ 4Gb)
"""

from alld import camera
from alld import line
from alld import perspective
from alld import pixelspace
from alld import slidingwindowsearch
from alld import thresholds
from alld import visual

import cv2
import numpy
import scipy.io


def calc_curvature(bin, poly2, ploty):
    """Calculate curvature of poly2 in `y_closest_to_vehicle` point
    
    Return value in meters.
    """
    y_closest_to_vehicle = bin.shape[0]
    poly2_meters = pixelspace.to_real_world_space(poly2(ploty), ploty)
    y_closest_to_vehicle_in_meters = pixelspace.y_pix2m(y_closest_to_vehicle)
    return poly2_meters.curvature(y_closest_to_vehicle_in_meters)


class Lane:

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def width_m(self, y):
        """Lane width in meters
        
        `y` should be in pixel space
        """
        left_base = self.left(y)
        right_base = self.right(y)
        lane_width_m = pixelspace.x_pix2m(right_base - left_base)
        return lane_width_m

    def curvature(self, bin, ploty):
        """Calculate lane curvature (in meters)
        
        `ploty` should be in pixel space
        """
        left_roc = calc_curvature(bin, self.left, ploty)
        right_roc = calc_curvature(bin, self.right, ploty)
        return (left_roc + right_roc) / 2

    def base(self, y):
        """Return left base and right base in pixel space"""
        left_base = self.left(y)
        right_base = self.right(y)
        return left_base, right_base

    def center(self, y):
        """Return lane center X in pixel space"""
        left_base, right_base = self.base(y)
        return (left_base + right_base) / 2


class Pipeline:

    ETALON_LINE_WIDTH_M = 3.7  # "etalon" lane width in meters

    ALLOWED_MISSES = 5  # max misses in a row

    # Sanity Check parameters
    LANE_WIDTH_PRECISION = 1  # meter
    ROC_DIFF = 1000  # meters


    def __init__(self, margin=30, history_length=5, collect_points=False):
        """Construct Pipeline
        
        Set `collect_points` to True to save coords of detected pixels.        
        """
        self.collect_points = collect_points

        # tune warp perspective
        a1 = perspective.Pair(src=(580, 460), dst=(260, 0))
        a2 = perspective.Pair(src=(700, 460), dst=(1040, 0))
        a3 = perspective.Pair(src=(1040, 680), dst=(1040, 780))
        a4 = perspective.Pair(src=(260, 680), dst=(260, 780))
        self.persp = perspective.Perspective(a1, a2, a3, a4)

        # load camera from file (camera.pickle was created by calibrate.py)
        self.cam = camera.fromfile('camera.pickle')

        self.yellow_h_op = thresholds.HLSThreshold('yellow_h', 20, 40, thresholds.HLSThreshold.H)
        self.yellow_s_op = thresholds.HLSThreshold('yellow_s', 120, 255, thresholds.HLSThreshold.S)
        self.white_l_op = thresholds.HLSThreshold('white_l', 220, 255, thresholds.HLSThreshold.L)
        self.sobelx_op = thresholds.AbsSobelXThreshold(10, 120)
        self.sobely_op = thresholds.AbsSobelYThreshold(10, 120)
        self.mag_op = thresholds.MagSobelThreshold(5, 150, 3)
        self.dir_op = thresholds.DirectionThreshold(numpy.pi / 8, numpy.pi / 2 - numpy.pi / 8, 5)

        # construct a list of thresholds
        # each *_op object has NAME attribute
        # self.th_op(frame) returns dictionary {op_name => binary_image}
        self.th_op = thresholds.Thresholds(self.yellow_s_op, self.yellow_h_op, self.white_l_op,
                                           self.sobelx_op, self.sobely_op, self.mag_op, self.dir_op)

        # self.left represents "left line" object
        self.left = line.Line(maxlen=history_length)
        # self.right represnets "right line" object
        self.right = line.Line(maxlen=history_length)

        self.margin = margin

        self.misses = 0

        # metrics
        self.left_points_n = []
        self.right_points_n = []
        self.curvature = []
        self.offset = []
        self.left_base = []
        self.right_base = []
        self.lane_width_m = []  # detected lane width in meters
        self.left_roc = []  # roc means 'radius of curvature'
        self.right_roc = []
        self.sc = []  # stores "sanity check" flags for each frame
        self.sliding_window = []  # True if sliding windows was used for frame
        self.miss = []  # miss count

    def _collect_points(self, left_points, right_points):
        # add points and poly2 to Line
        if self.collect_points:
            self.left.collect_points(left_points)
            self.right.collect_points(right_points)

    def save_metrics(self, output_file_name):
        """Save collected metrics to MAT file"""
        scipy.io.savemat(output_file_name, dict(
            curvature=self.curvature,
            offset=self.offset,
            left_base=self.left_base,
            right_base=self.right_base,
            lane_width_m=self.lane_width_m,
            left_roc=self.left_roc,
            right_roc=self.right_roc,
            sc=self.sc,
            sliding_window=self.sliding_window,
            miss=self.miss,
        ))

    def binarize(self, frame):
        """Return binary image"""
        frame = self.cam.undistort(frame)
        frame = self.persp.warp(frame)

        # use thresholds: see `__init__` to understand which thresholds will be calculated
        binaries = self.th_op(frame)

        select_yellow = (binaries['yellow_s'] == 1) & (binaries['yellow_h'] == 1)
        select_white = binaries['white_l'] == 1
        select_sobel = (binaries['sobelx'] == 1) | (binaries['mag'] == 1) & (binaries['dir'] == 1)

        # combine
        combined = numpy.zeros_like(binaries['sobelx'])
        combined[select_sobel | select_yellow | select_white] = 1

        return combined

    def _sanity_check(self, bin, ploty, y, left_candidate, right_candidate):
        roc_diff = numpy.absolute(calc_curvature(bin, left_candidate, ploty) -
                                  calc_curvature(bin, right_candidate, ploty))
        if (roc_diff > self.ROC_DIFF):
            return False

        lane_candidate = Lane(left_candidate, right_candidate)

        if numpy.absolute(self.ETALON_LINE_WIDTH_M - lane_candidate.width_m(y) > self.LANE_WIDTH_PRECISION):
            return False

        return True

    @property
    def should_run_sliding_window(self):
        """Return True if a program should use sliding window for current frame"""
        if self.misses >= self.ALLOWED_MISSES:
            return True
        return not self.left.detected or not self.right.detected

    def process(self, frame):
        """Process `frame` and return processed image and `outimg`
        
        I used `outimg` for debug purposes.
        
        """
        bin = self.binarize(frame)

        ploty = numpy.linspace(0, bin.shape[0] - 1, bin.shape[0])
        y_closest_to_vehicle = bin.shape[0]

        outimg = visual.create_outimg(bin)

        # find points for each line of the lane
        if self.should_run_sliding_window:
            left_points, right_points = slidingwindowsearch.search(bin, outimg=outimg)
            sliding_window_was_used = True
        else:
            left_points, right_points = slidingwindowsearch.marginsearch(
                bin, self.left.current_poly2, self.right.current_poly2, self.margin)
            sliding_window_was_used = False

        left_points.draw(outimg, (255, 0, 0))
        right_points.draw(outimg, (0, 0, 255))
        self._collect_points(left_points, right_points)

        # fit polynom2 for each line of the lane
        left_candidate = left_points.fit_poly2()
        right_candidate = right_points.fit_poly2()

        sc = self._sanity_check(bin, ploty, y_closest_to_vehicle,
                                left_candidate, right_candidate)

        if sc:
            # sanity check is passed
            self.left.fit(left_candidate)
            self.right.fit(right_candidate)
            self.misses = 0
        else:
            self.left.detected = False
            self.right.detected = False
            self.misses += 1

        # calculate smoothed line
        left = self.left.smoothed
        right = self.right.smoothed

        lane = Lane(left, right)

        # calculate curvature
        curvature = lane.curvature(bin, ploty)

        # calculate offset
        vehicle_center = bin.shape[1] / 2
        offset = pixelspace.x_pix2m(vehicle_center - lane.center(y_closest_to_vehicle))

        # draw lane
        zero = numpy.zeros_like(bin).astype(numpy.uint8)
        laneimg = numpy.dstack((zero, zero, zero))
        lanepoly = visual.lanepoly(ploty, left, right)
        cv2.fillPoly(laneimg, numpy.int_([lanepoly]), (0, 255, 0))

        # draw text
        visual.draw_text(frame, curvature, offset)

        #
        # collect metrics
        #
        left_base, right_base = lane.base(y_closest_to_vehicle)
        left_roc = calc_curvature(bin, left, ploty)
        right_roc = calc_curvature(bin, right, ploty)

        self.left_points_n.append(len(left_points))
        self.right_points_n.append(len(right_points))
        self.curvature.append(curvature)
        self.offset.append(offset)
        self.left_base.append(left_base)
        self.right_base.append(right_base)
        self.lane_width_m.append(lane.width_m(y_closest_to_vehicle))
        self.left_roc.append(left_roc)
        self.right_roc.append(right_roc)
        self.sc.append(sc)
        self.sliding_window.append(sliding_window_was_used)
        self.miss.append(self.misses)

        # TODO: return outimg or use dashboard
        return (cv2.addWeighted(frame, 1, self.persp.unwarp(laneimg), 0.3, 0),
                outimg)

    def __call__(self, frame):
        processed_frame, _ = self.process(frame)
        return processed_frame


if __name__ == '__main__':
    pipeline = Pipeline()
    from udacitylib import video
    video.convert('project_video.mp4', pipeline, 'output.avi')
    pipeline.save_metrics('metrics.mat')
