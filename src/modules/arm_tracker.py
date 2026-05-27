# Arm tracker built on MediaPipe to extract arm poses from webcam


import cv2 as cv
import mediapipe
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import numpy as np

class ArmTracker:
    def __init__(self, model_path = "mp_models/pose_landmarker_full.task"):
        try:
            base_options = python.BaseOptions(model_asset_path = model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                min_pose_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                result_callback=self._result_callback
            )

            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self._latest_result = None
            self._latest_frame  = None
            self._timestamp_ms  = 0


            self.RIGHT_SHOULDER_ID = 12
            self.RIGHT_ELBOW_ID = 14
            self.RIGHT_WRIST_ID = 16

        except Exception as e:
            print(f"Error initializing Arm Tracker: {e}")

    def _result_callback(self, result, output_image, timestamp_ms):
        self._latest_result = result

    def _angle_between(self, a, b, c):
        """
        Angle at point b, formed by vectors b->a and b->c.
        a, b, c are numpy arrays of shape (3,).
        Returns angle in degrees.
        """

        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    

    def _get_landmark_xyz(self, landmarks, idx):
        landmark = landmarks[idx]

        return np.array([landmark.x, landmark.y, landmark.z])
    

    # def _draw_landmarks(self, frame, result):
    #     RIGHT_ARM_CONNECTIONS = frozenset([
    #             (12, 14),  
    #             (14, 16),
    #         ])
        
    #     annotated_image = np.copy(frame)
    #     h, w = frame.shape[:2]

    #     for pose_landmarks in result.pose_landmarks:
    #         for start_idx, end_idx in RIGHT_ARM_CONNECTIONS:
    #             p1 = pose_landmarks[start_idx]
    #             p2 = pose_landmarks[end_idx]
    #             if p1.visibility > 0.5 and p2.visibility > 0.5:
    #                 x1, y1 = int(p1.x * w), int(p1.y * h)
    #                 x2, y2 = int(p2.x * w), int(p2.y * h)
    #                 cv.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #         for idx in [12, 14, 16]:
    #             lm = pose_landmarks[idx]
    #             if lm.visibility > 0.5:
    #                 cx, cy = int(lm.x * w), int(lm.y * h)
    #                 cv.circle(annotated_image, (cx, cy), 6, (0, 0, 255), -1)
    #                 cv.circle(annotated_image, (cx, cy), 6, (255, 255, 255), 1)

    #     return annotated_image
    
    def _draw_landmarks(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style)

        return annotated_image

    def track(self):
        try:
            cap = cv.VideoCapture(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self._timestamp_ms += 1
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mediapipe.Image(image_format= mediapipe.ImageFormat.SRGB, data=rgb)
                self.landmarker.detect_async(mp_image, self._timestamp_ms)

                if self._latest_result and self._latest_result.pose_landmarks:
                    lms = self._latest_result.pose_landmarks[0]

                    # check viz first
                    if lms[self.RIGHT_ELBOW_ID].visibility > 0.5 and lms[self.RIGHT_SHOULDER_ID].visibility > 0.5 and lms[self.RIGHT_WRIST_ID].visibility > 0.5:
                        shoulder = self._get_landmark_xyz(lms, self.RIGHT_SHOULDER_ID)
                        elbow = self._get_landmark_xyz(lms, self.RIGHT_ELBOW_ID)
                        wrist = self._get_landmark_xyz(lms, self.RIGHT_WRIST_ID)

                        elbow_angle = self._angle_between(shoulder, elbow, wrist)
                        shoulder_angle = self._angle_between(elbow, shoulder, np.array([shoulder[0], shoulder[1] - 1, shoulder[2]]))

                        print(f"Shoulder: {shoulder_angle:.1f}°  Elbow: {elbow_angle:.1f}°")

                    frame = self._draw_landmarks(frame, self._latest_result)

                cv.imshow("Arm Tracker", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv.destroyAllWindows()
            self.landmarker.close()

        except Exception as e:
            print(f"Error running track: {e}")
                        




