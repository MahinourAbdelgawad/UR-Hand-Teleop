import cv2 as cv
import numpy as np
from src.modules.hand_tracker import HandTracker
from src.modules.mujoco_wrapper import MujocoWrapper
from src.modules.ik_solver import IKSolver
from src.modules.depth_estimator import DepthEstimator

X_MIN, X_MAX = -0.4, 0.4
Y_MIN, Y_MAX = -0.4, 0.4
Z_MIN, Z_MAX =  0.2, 0.7

def assemble_target(palm_x, palm_y, depth):
    # palm_x and palm_y are normalized 0-1 
    # depth is in meters from AprilTag, None if tag not visible
    x = X_MIN + palm_x * (X_MAX - X_MIN)
    y = Y_MIN + palm_y * (Y_MAX - Y_MIN)
    z = float(np.clip(depth, Z_MIN, Z_MAX)) if depth is not None else 0.4

    return np.array([x, y, z])

def main():
    tracker = HandTracker()
    sim = MujocoWrapper()
    estimator = DepthEstimator()
    IK = IKSolver(sim.model, sim.data)

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
            depth = estimator.estimate(frame)

            target = assemble_target(palm_x, palm_y, depth)  # your coord mapping function
            q = IK.solve(target)

            for i in range(6):
                sim.set_joint(i, q[i])

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