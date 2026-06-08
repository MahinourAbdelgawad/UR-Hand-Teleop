import numpy as np
from dt_apriltags import Detector
import cv2 as cv

class DepthEstimator:
    def __init__(self, fx = 800.0, fy = 800.0, cx = 960.0, cy = 540.0, tag_size = 0.05):
        """
        Depth estimator using an AprilTag for webcam/camera without depth
        """

        #TODO: CALIBRATE CAMERA AND FIX THESE INTRINSICS
        self.fx = fx
        self.fy = fy 
        self.cy = cy
        self.cx = cx 

        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.dist_coeffs = np.zeros(5)

        self.tag_size = tag_size # in meters

        self.detector = Detector(families = "tag36h11")

    def estimate(self, frame):
        try:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # preprocessing
            detections = self.detector.detect(gray)

            if not detections:
                return None
            
            corners = detections[0].corners.astype(np.float32)

            object_points = np.array([
                [-self.tag_size / 2, self.tag_size / 2, 0],
                [self.tag_size / 2, self.tag_size / 2, 0],
                [self.tag_size / 2, -self.tag_size / 2, 0],
                [-self.tag_size / 2, -self.tag_size / 2, 0]
            ], dtype=np.float32)

            ok, rvec, tvec = cv.solvePnP(object_points, corners, self.camera_matrix, self.dist_coeffs)

            if ok:
                return float(tvec[2][0]) # return z in meters
            return None
            
        except Exception as e:
            print(f"Error estimating depth: {e}")
            return None
