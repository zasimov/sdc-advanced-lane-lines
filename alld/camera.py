""""Camera model
"""

import pickle

import cv2


class Camera:

    def __init__(self, cmx, dist):
        """Constructs camera
        
        Arguments:
          - cmx - camera matrix (result of cv2.calibrateCamera)
          - dist - distortion coefficients (result of cv2.calibrateCamera)
        """
        self.cmx = cmx
        self.dist = dist

    def save(self, filename):
        camera = {
            # TODO: rename cmx to mtx (verify mtx using lessons)
            'cmx': self.cmx,
            'dist': self.dist,
        }
        with open(filename, 'wb') as f:
            pickle.dump(camera, f)

    @classmethod
    def fromfile(cls, filename):
        with open(filename, 'rb') as f:
            camera = pickle.load(f)
            return cls(camera['cmx'], camera['dist'])

    def undistort(self, img):
        """undistort undistorts source image and returns undistorted image"""
        return cv2.undistort(img, self.cmx, self.dist, None, self.cmx)


def fromfile(filename):
    """syntax sugar: load Camera from file
    
     camera.Camera.fromfile it the same
    """
    return Camera.fromfile(filename)
