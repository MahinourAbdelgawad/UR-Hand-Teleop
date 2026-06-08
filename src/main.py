import cv2 as cv
from src.modules.hand_tracker import HandTracker
from src.modules.mujoco_wrapper import MujocoWrapper


def main():
    tracker = HandTracker()
    sim = MujocoWrapper()
    sim.launch()

    cap = cv.VideoCapture(0)

    while cap.isOpened() and sim.is_running():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv.flip(frame, 1)
        frame = tracker.process_frame(frame)

        state = tracker.get_hand_state()

        if state is not None:
            palm_x, palm_y, is_closed = state

            sim.set_gripper(is_closed)

            status = "CLOSED" if is_closed else "OPEN"

            color  = (0, 0, 255) if is_closed else (0, 255, 0)
            cv.putText(frame, f"Gripper: {status}", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv.putText(frame, "No hand detected", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        sim.step()

        cv.imshow("URHandTeleop", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    sim.close()
    cap.release()
    tracker.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()