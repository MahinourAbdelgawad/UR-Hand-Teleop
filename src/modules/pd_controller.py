import numpy as np
import time

class PDController:
    """
    PD Controller for EE position (PID without the I)
    Maintains a desired EE position, updated by hand tracking changes
    """
    def __init__(self, kp = 5.0, kd = 0.01, max_step = 0.05, dt = None):
        self.kp = kp
        self.kd = kd 
        self.max_step = max_step
        self.dt = dt

        self.target = None 
        self._last_error = np.zeros(3)
        self._last_time = time.monotonic()


    def reset(self, current_ee_pos = None):
        """
        Initialize the controller to the current arm position
        Called every time the system is armed (aka thumbs up)
        """
        self.target = np.asarray(current_ee_pos, dtype=float).copy()
        self._last_error = np.zeros(3)
        self._last_time = time.monotonic()


    def settarget(self, target = None):
        self.target = np.asarray(target, dtype=float).copy()


    def compute(self, current_pos):
        """
        compute delta between current and target
        """
        try:
            if self.target is None:
                return np.zeros(3)
            
            now = time.monotonic()
            dt = self.dt if self.dt is not None else max(now - self._last_time, 1e-4)

            self._last_time = now 

            error = self.target - np.asarray(current_pos, dtype=float)
            error_dt = (error - self._last_error) / dt 
            
            command = (self.kp * error) #+ (self.kd * error_dt)# 

            # cap for safety
            norm = np.linalg.norm(command)
            if norm > self.max_step:
                command = command / norm * self.max_step

            self._last_error = error 

            return command
   

        except Exception as e:
            print(f"Error computing in PD controller: {e}")
        




