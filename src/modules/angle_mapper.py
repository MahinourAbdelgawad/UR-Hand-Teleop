import numpy as np

class AngleMapper:
    def __init__(self):
        # TODO: allow other robot joint limits

        self.HUMAN_ELBOW_MIN = 30.0 
        self.HUMAN_ELBOW_MAX = 175.0

        self.HUMAN_SHOULDER_MIN = 0.0
        self.HUMAN_SHOULDER_MAX = 180.0


        self.UR5_ELBOW_MIN = np.radians(-160)
        self.UR5_ELBOW_MAX = np.radians(0)

        self.UR5_SHOULDER_MIN = np.radians(-180)
        self.UR5_SHOULDER_MAX = np.radians(0)

        self.smoothing_factor = 0.15
        self.last_smoothed_value = None


    def map_(self, angle, joint_type, smoothing_factor = 0.15):
        try:
            self.smoothing_factor = smoothing_factor

            human_min = None
            human_max = None
            robot_min = None
            robot_max = None

            if joint_type == "SHOULDER":
                human_min = self.HUMAN_SHOULDER_MIN
                human_max = self.HUMAN_SHOULDER_MAX

                robot_min = self.UR5_SHOULDER_MIN
                robot_max = self.UR5_SHOULDER_MAX

            elif joint_type == "ELBOW":
                human_min = self.HUMAN_ELBOW_MIN
                human_max = self.HUMAN_ELBOW_MAX

                robot_min = self.UR5_ELBOW_MIN
                robot_max = self.UR5_ELBOW_MAX
            
            else:
                print(f"Invalid joint type. Cannot map angle")
                return
            
            factor = np.clip((angle - human_min) / (human_max - human_min), 0.0, 1.0)

            mapped_angle = robot_min + factor * (robot_max - robot_min)
            
            # smooth first
            return self._smooth(mapped_angle)
            

        except Exception as e:
            print(f"Error mapping angle: {e}")


    def _smooth(self, value):
        """
        Uses exponential moving average filter
        """
        try:
            # 0.0 is same as saying no smoothing
            if self.smoothing_factor == 0.0:
                return
            
            if self.last_smoothed_value is None:
                self.last_smoothed_value = value 
            
            else:
                self.last_smoothed_value = self.smoothing_factor * value + (1 - self.smoothing_factor) * self.last_smoothed_value

            return self.last_smoothed_value
            
        except Exception as e:
            print(f"Error smoothing angle: {e}")

