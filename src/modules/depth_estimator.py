import numpy as np
from dt_apriltags import Detector
import cv2 as cv

class DepthEstimator:
    def __init__(self, fx = 800.0, fy = 800.0, cx = 960.0, cy = 540.0, tag_size = 0.05, alpha = 0.3, max_missing = 10):
        """
        Depth estimator using an AprilTag for webcam/camera without depth
        alpha: EMA weight
        max_missing: max frames to hold last value if no tag detected
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

        self.alpha = alpha

        self.max_missing = max_missing

        self._filtered_z = None # running EMA
        self._missing = 0 # consecutive frames without tag detected

        self._object_points = np.array([
                [-self.tag_size / 2, self.tag_size / 2, 0],
                [self.tag_size / 2, self.tag_size / 2, 0],
                [self.tag_size / 2, -self.tag_size / 2, 0],
                [-self.tag_size / 2, -self.tag_size / 2, 0]
            ], dtype=np.float32)


    def estimate(self, frame):
        try:
            raw_z = self._detect_raw(frame)
 
            if raw_z is not None:
                self._missing = 0
                if self._filtered_z is None:
                    self._filtered_z = raw_z #cold start

                else:
                    # EMA: new = alpha*raw + (1-alpha)*old
                    self._filtered_z = self.alpha * raw_z + (1.0 - self.alpha) * self._filtered_z
            else:
                self._missing += 1
                if self._missing > self.max_missing:
                    self._filtered_z = None
    
            return self._filtered_z
        
        except Exception as e:
            print(f"Error estimating depth: {e}")


    def reset(self):
        self._filtered_z = None 
        self._missing = 0
        

    def _detect_raw(self, frame):
        try:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # preprocessing
            detections = self.detector.detect(gray)

            if not detections:
                return None
            
            corners = detections[0].corners.astype(np.float32)


            ok, rvec, tvec = cv.solvePnP(self._object_points, corners, self.camera_matrix, self.dist_coeffs)

            if ok:
                return float(tvec[2][0]) # return z in meters
            return None
            
        except Exception as e:
            print(f"Error estimating depth: {e}")
            return None
