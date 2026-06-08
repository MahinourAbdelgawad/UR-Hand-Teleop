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
    """
    def __init__(self, model_path = "mp_models/hand_landmarker.task"):
        try:
            self.finger_tips = [8,12,16,20] # index - middle - ring - pinky
            self.finger_pips = [6, 10, 14, 18]
            self.MCPs = [0, 5, 9, 13, 17]

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
            # average of the four MCPs
            x = sum(landmarks[i].x for i in self.MCPs) / len(self.MCPs)
            y = sum(landmarks[i].y for i in self.MCPs) / len(self.MCPs)
            return x, y
        
        except Exception as e:
            print(f"Error finding palm center: {e}")
    
    
    def get_hand_state(self):
        try:
            if not self._latest_result or not self._latest_result.hand_landmarks:
                return None

            landmarks = self._latest_result.hand_landmarks[0]

            palm_x, palm_y = self._get_palm_center(landmarks)
            closed = self._is_closed(landmarks)

            return palm_x, palm_y, closed
        
        except Exception as e:
            print(f"Error retrieving hand state: {e}")
    
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