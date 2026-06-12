import cv2 as cv
import mediapipe
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import numpy as np

class HandTracker:
    """
    Outputs:
        Palm centroid position
        Hand opened or closed

        Hand landmarker gives 21 landmarks
        Palm centroid: averages of the MCPs aka 5, 9, 13, 17
        For gesture detection: finger tips and pips 

        Thumbs up to arm, thumbs down to disarm
        Thumb tip = 4, thumb IP = 3, thumb MCP = 2, thumb CMC = 1
    """
    def __init__(self, model_path = "mp_models/hand_landmarker.task", alpha = 0.2):
        try:
            self.finger_tips = [8,12,16,20] # index - middle - ring - pinky
            self.finger_pips = [6, 10, 14, 18]
            self.MCPs = [0, 5, 9, 13, 17]

            self.thumb_tip = 4
            self.thumb_IP = 3
            self.thumb_MCP = 2
            self.thumb_CMC = 1 

            self.wrist = 0

            base_options = python.BaseOptions(model_asset_path = model_path)
            options = vision.HandLandmarkerOptions(    
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                result_callback=self._result_callback
            )

            self.landmarker = vision.HandLandmarker.create_from_options(options)

            self._latest_result = None
            self._latest_frame  = None
            self._timestamp_ms  = 0

            self._reference_pos = None 

            self.alpha = alpha 
            self._smoothed_palm = None # fr EMA

        except Exception as e:
            print(f"Error initializating Hand Tracker: {e}")
    
    def _is_closed(self, landmarks):
        try:
            closed_count = sum(
                landmarks[tip].y > landmarks[pip].y
                for tip, pip in zip(self.finger_tips, self.finger_pips)
                )
            
            return closed_count >= 3  # 3 of 4 fingers folded = closed
        
        except Exception as e:
            print(f"Error detecting hand state (closed or open): {e}")
    
    def _get_palm_center(self, landmarks):
        try:
            # # average of the four MCPs
            # x = sum(landmarks[i].x for i in self.MCPs) / len(self.MCPs)
            # y = sum(landmarks[i].y for i in self.MCPs) / len(self.MCPs)
            # return x, y
            raw_x = sum(landmarks[i].x for i in self.MCPs) / len(self.MCPs)
            raw_y = sum(landmarks[i].y for i in self.MCPs) / len(self.MCPs)

            if self._smoothed_palm is None:
                self._smoothed_palm = np.array([raw_x, raw_y])
            else:
                self._smoothed_palm = (
                    self.alpha * np.array([raw_x, raw_y])
                    + (1 - self.alpha) * self._smoothed_palm
                )

            return float(self._smoothed_palm[0]), float(self._smoothed_palm[1])
        
        except Exception as e:
            print(f"Error finding palm center: {e}")


    def _thumb_extended(self, landmarks):
        """
        True when the thumb tip is farther from the wrist than the thumb MCP
        """
        wrist = np.array([landmarks[self.wrist].x, landmarks[self.wrist].y])
        tip = np.array([landmarks[self.thumb_tip].x, landmarks[self.thumb_tip].y])
        mcp = np.array([landmarks[self.thumb_MCP].x, landmarks[self.thumb_MCP].y])
        return np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * 1.1
 
    def _thumb_pointing_up(self, landmarks):
        """
        True when thumb tip is above the thumb MCP (in image y, so smaller = higher)
        """
        return landmarks[self.thumb_tip].y < landmarks[self.thumb_MCP].y - 0.05
 
    def _thumb_pointing_down(self, landmarks):
        """
        True when thumb tip is below the thumb MCP
        """
        return landmarks[self.thumb_tip].y > landmarks[self.thumb_MCP].y + 0.05
    
    
    def get_hand_state(self):
        """
        Returns: dict with keys: palm_x, palm_y, is_closed, gesture, landmarks
        """
        try:
            if not self._latest_result or not self._latest_result.hand_landmarks:
                self._smoothed_palm = None # reset
                return None

            landmarks = self._latest_result.hand_landmarks[0]

            palm_x, palm_y = self._get_palm_center(landmarks)
            closed = self._is_closed(landmarks)
            gesture  = self._detect_gesture(landmarks)

            return {
                "palm_x": palm_x,
                "palm_y": palm_y,
                "is_closed": closed,
                "gesture": gesture,
                "landmarks": landmarks
            }
        
        except Exception as e:
            print(f"Error retrieving hand state: {e}")

    def _detect_gesture(self, landmarks):
        """
        Returns one of: thumb_up, thumb_down, closed, open, None
        """
        fingers_closed = self._is_closed(landmarks)
        thumb_ext = self._thumb_extended(landmarks)
 
        if fingers_closed and thumb_ext:
            if self._thumb_pointing_up(landmarks):
                return "thumb_up"
            
            if self._thumb_pointing_down(landmarks):
                return "thumb_down"
 
        if fingers_closed and not thumb_ext:
            return "closed"
 
        if not fingers_closed:
            return "open"
 
        return None
    
    def set_reference(self, palm_x, palm_y):
        """
        Callwd when the tracking gesture is first detected
        """
        self._reference_pos = np.array([palm_x, palm_y])

    def get_delta(self, palm_x, palm_y):
        """
        Return coords relative to the stored reference
        Returns None if no reference has been set
        """
        if self._reference_pos is None:
            return None
        
        return np.array([palm_x, palm_y]) - self._reference_pos
    
    def clear_reference(self):
        self._reference_pos = None 
    
    def _result_callback(self, result, output_image, timestamp_ms):
        self._latest_result = result

    def _get_landmark_xyz(self, landmarks, idx):
        landmark = landmarks[idx]

        return np.array([landmark.x, landmark.y, landmark.z])
    

    def _draw_landmarks(self, frame, result):
        try:
            annotated = np.copy(frame)

            for hand_landmarks in result.hand_landmarks:
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=hand_landmarks,
                    connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
                )

            return annotated
        
        except Exception as e:
            print(f"Error drawing landmarks: {e}")
        

    def process_frame(self, frame):
        try:
            self._timestamp_ms += 1
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mediapipe.Image(image_format= mediapipe.ImageFormat.SRGB, data=rgb)
            self.landmarker.detect_async(mp_image, self._timestamp_ms)

            if self._latest_result and self._latest_result.hand_landmarks:
                frame = self._draw_landmarks(frame, self._latest_result)

            return frame 

        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame 
       
    def close(self):
        self.landmarker.close()