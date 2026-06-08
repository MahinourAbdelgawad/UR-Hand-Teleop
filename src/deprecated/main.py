from src.deprecated.arm_tracker import ArmTracker
from src.deprecated.angle_mapper import AngleMapper
from src.modules.mujoco_wrapper import MujocoWrapper
import cv2 as cv
import numpy as np


def main():
    tracker = ArmTracker()
    mapper = AngleMapper()
    sim = MujocoWrapper()

    sim.launch()
    capture = cv.VideoCapture(0)


    while capture.isOpened() and sim.is_running():
        ret, frame = capture.read()

        if not ret: break

        frame = cv.flip(frame, 1)
        frame = tracker.process_frame(frame)

        shoulder_angle, elbow_angle = tracker.get_angles()

        if shoulder_angle is not None:
            sim.set_joint(1, mapper.map_angle(shoulder_angle, "SHOULDER", 0.0))

        if elbow_angle is not None:
            mapped = mapper.map_angle(elbow_angle, "ELBOW", 0.0)
            print(f"Elbow raw: {elbow_angle:.1f} deg  ->  mapped: {np.degrees(mapped):.1f} deg")
            sim.set_joint(2, mapped)

        sim.step()

        cv.imshow("Arm Tracker", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break 

    sim.close()
    capture.release()
    tracker.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running program: {e}")