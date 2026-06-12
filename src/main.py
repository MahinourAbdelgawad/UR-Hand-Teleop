import cv2 as cv
import numpy as np
import threading
import time
from src.modules.hand_tracker import HandTracker
from src.modules.mujoco_wrapper import MujocoWrapper
from src.modules.ik_solver import IKSolver
from src.modules.depth_estimator import DepthEstimator
from src.modules.pd_controller import PDController

FPS = 30
CONTROL_RATE = 50
CONTROL_PERIOD = 1.0 / CONTROL_RATE

JITTER_THRESHOLD = 0.008
HAND_EE_SCALE_XY = 0.6
HAND_EE_SCALE_Z  = 0.4

LIMITS = {
    "x": (-0.7, 0.7),
    "y": (-0.7, 0.7),
    "z": (0.05, 0.9),
}

HAND_STATE = None
DEPTH = None
FRAME = None

# Thread safety
STATE_LOCK = threading.Lock()
MUJOCO_LOCK = threading.Lock()  # Protects all MuJoCo operations


def clamp(position):
    return np.array([
        np.clip(position[0], LIMITS["x"][0], LIMITS["x"][1]),
        np.clip(position[1], LIMITS["y"][0], LIMITS["y"][1]),
        np.clip(position[2], LIMITS["z"][0], LIMITS["z"][1]),
    ])


def draw(frame, armed, gesture, is_closed, ee_pos):
    arm_text  = "ARMED"  if armed else "DISARMED"
    arm_color = (0, 220, 0) if armed  else (0, 80, 220)
    closed_text  = "CLOSED" if is_closed else "OPEN"
    closed_color = (0, 0, 220) if is_closed else (0, 200, 80)

    cv.putText(frame, f"{arm_text}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, arm_color, 2)
    cv.putText(frame, f"{closed_text}", (20, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, closed_color, 2)

    if gesture:
        cv.putText(frame, f"{gesture}", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)


def camera_thread(tracker, estimator, cap, stop_event):
    global HAND_STATE, DEPTH, FRAME  
    period = 1.0 / FPS

    while not stop_event.is_set():
        t0 = time.monotonic()

        ret, raw = cap.read()
        if not ret:
            continue

        try:
            annotated = tracker.process_frame(raw)
            hand_state = tracker.get_hand_state()
            depth = estimator.estimate(raw)
            
            with STATE_LOCK:
                HAND_STATE = hand_state
                DEPTH = depth
                FRAME = annotated
        except Exception as e:
            print(f"Error in camera thread: {e}")
            continue

        elapsed = time.monotonic() - t0
        sleep = period - elapsed
        if sleep > 0:
            time.sleep(sleep)


def control_thread(tracker, sim, solver, pd, stop_event):
    global HAND_STATE, DEPTH, MUJOCO_LOCK

    armed = False
    ref_palm_x = 0.0
    ref_palm_y = 0.0
    ref_depth = None
    ref_ee_pos = None
    gripper_closed = False

    while not stop_event.is_set() and sim.is_running():
        t0 = time.monotonic()

        with STATE_LOCK:
            state = HAND_STATE
            dep = DEPTH

        if state is not None:
            try:
                gesture = state["gesture"]
                is_closed = state["is_closed"]
                px, py = state["palm_x"], state["palm_y"]

                gripper_closed = is_closed
                
                # if gesture:
                #     print(f"Detected gesture: {gesture}, armed: {armed}")

                if gesture == "thumb_up" and not armed:
                    armed = True
                    ref_palm_x = px
                    ref_palm_y = py
                    ref_depth = dep
                    print("ARM ACTIVATED")
                    
                    with MUJOCO_LOCK:
                        ref_ee_pos = solver.get_end_effector_pos().copy()
                        pd.reset(ref_ee_pos)

                elif gesture == "thumb_down" and armed:
                    armed = False
                    print("ARM DEACTIVATED")

                if armed:
                    dx_img = px - ref_palm_x
                    dy_img = py - ref_palm_y
                    movement = np.sqrt(dx_img**2 + dy_img**2)

                    if movement > JITTER_THRESHOLD:
                        # dx_world = dx_img  *  HAND_EE_SCALE_XY
                        # dy_world = -dy_img *  HAND_EE_SCALE_XY
                        robot_dy = -dx_img * HAND_EE_SCALE_XY
                        robot_dz = -dy_img * HAND_EE_SCALE_XY

                        if dep is not None and ref_depth is not None:
                            # dz_world = (ref_depth - dep) * HAND_EE_SCALE_Z
                            robot_dx = (ref_depth - dep) * HAND_EE_SCALE_Z

                        else:
                            # dz_world = 0.0
                            robot_dx = 0.0

                        # desired = ref_ee_pos + np.array([dx_world, dy_world, dz_world])
                        desired = ref_ee_pos + np.array([robot_dx, robot_dy, robot_dz])
                        desired = clamp(desired)
                        pd.settarget(desired)

                    with MUJOCO_LOCK:
                        current_ee = solver.get_end_effector_pos()
                        delta = pd.compute(current_ee)
                        new_q = solver.solve_incremental(delta)
                        sim.set_joints(new_q)
            except Exception as e:
                print(f"Error in control thread: {e}")
                armed = False

        with MUJOCO_LOCK:
            sim.set_gripper(gripper_closed)
            if not sim.step():
                print("Simulation step failed, exiting control loop")
                break

        elapsed = time.monotonic() - t0
        sleep = CONTROL_PERIOD - elapsed
        if sleep > 0:
            time.sleep(sleep)


def main():
    try:
        tracker = HandTracker()
        sim = MujocoWrapper()
        
        if sim.model is None or sim.data is None:
            print("Error: Failed to initialize MuJoCo. Exiting.")
            return
        
        estimator = DepthEstimator()
        IK = IKSolver(sim.model, sim.data, LIMITS)
        pd = PDController()

        if not sim.launch():
            print("Error: Failed to launch MuJoCo viewer. Exiting.")
            return

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Error: cannot open camera.")
            sim.close()
            return

        stop_event = threading.Event()

        camera = threading.Thread(
            target=camera_thread,
            args=(tracker, estimator, cap, stop_event),
            daemon=True, name="camera_thread",
        )
        control = threading.Thread(
            target=control_thread,
            args=(tracker, sim, IK, pd, stop_event),
            daemon=True, name="control_thread",
        )
        camera.start()
        control.start()

        try:
            while sim.is_running():
                with STATE_LOCK:
                    fr = FRAME
                    state = HAND_STATE
                
                if fr is None:
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue             

                gesture = state["gesture"]   if state else None
                closed = state["is_closed"] if state else False
                armed  = pd.target is not None

                ee_pos = None
                try:
                    with MUJOCO_LOCK:
                        ee_pos = IK.get_end_effector_pos()
                except Exception:
                    pass

                draw(fr, armed, gesture, closed, ee_pos) 

                cv.imshow("URHandTeleop", fr) 
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            stop_event.set()
            camera.join(timeout=2)
            control.join(timeout=2)
            sim.close()
            cap.release()
            tracker.close()
    
    except Exception as e:
        print(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()