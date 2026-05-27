from modules.arm_tracker import ArmTracker
from modules.angle_mapper import AngleMapper
from modules.mujoco_wrapper import MujocoWrapper
import cv2 as cv


def main():
    tracker = ArmTracker()
    mapper = AngleMapper()
    sim = MujocoWrapper()

    sim.launch()
    capture = cv.VideoCapture(0)


    while capture.isOpened() and sim.is_running():
        ret, frame = capture.read()

        if not ret: break

        tracker.process_frame(frame)

        shoulder_angle, elbow_angle = tracker.get_angles()

        if shoulder_angle is not None:
            sim.set_joint(1, mapper.map_(shoulder_angle, "SHOULDER"))

        if elbow_angle is not None:
            sim.set_joint(2, mapper.map_(elbow_angle, "ELBOW"))

        sim.step()

        cv.imshow("Arm Tracker", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break 

    sim.close()
    capture.release()
    cv.destroyAllWindows()
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running program: {e}")