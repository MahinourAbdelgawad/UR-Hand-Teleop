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
CONTROL_RATE = 50 # hz
CONTROL_PERIOD = 1.0 / CONTROL_RATE

#TODO: TWEAK THESE
JITTER_THRESHOLD = 0.008 
HAND_EE_SCALE_XY = 0.6 #scaling palm to ee movement
HAND_EE_SCALE_Z = 0.4

X_MIN, X_MAX = -0.4, 0.4
Y_MIN, Y_MAX = -0.4, 0.4
Z_MIN, Z_MAX =  0.2, 0.7

LIMITS = { # workspace limits
            "x": (-0.7, 0.7),
            "y": (-0.7, 0.7),
            "z": (0.05, 0.9),
        }

def assemble_target(palm_x, palm_y, depth):
    # palm_x and palm_y are normalized 0-1 
    # depth is in meters from AprilTag, None if tag not visible
    x = X_MIN + palm_x * (X_MAX - X_MIN)
    y = Y_MIN + palm_y * (Y_MAX - Y_MIN)
    z = float(np.clip(depth, Z_MIN, Z_MAX)) if depth is not None else 0.4

    return np.array([x, y, z])

def draw(frame, armed, gesture, is_closed, ee_pos):
    arm_text = "ARMED" if armed else "DISARMED"
    arm_color = (0, 220, 0) if armed else (0, 80, 220)

    closed_text = "CLOSED" if is_closed else "OPEN"
    closed_color = (0, 0, 220) if is_closed else (0, 200, 80)

    cv.putText(frame, f"Arm: {arm_text}", (20, 40),  cv.FONT_HERSHEY_SIMPLEX, 0.8, arm_color,  2)
    cv.putText(frame, f"Grip: {closed_text}", (20, 75),  cv.FONT_HERSHEY_SIMPLEX, 0.7, closed_color, 2)

    if gesture:
        cv.putText(frame, f"Gesture: {gesture}", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # if ee_pos is not None:
    #     pos_str = f"EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
    #     cv.putText(frame, pos_str, (20, 145), cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

def clamp(position):
    return np.array([
        np.clip(position[0], LIMITS["x"]),
        np.clip(position[1], LIMITS["y"]),
        np.clip(position[2], LIMITS["z"]),
    ])

HAND_STATE = None 
DEPTH = None 
FRAME = None 
def camera_thread(tracker, estimator, cap, stop_event):
    period = 1.0 / FPS

    while not stop_event.is_set():
        t0 = time.monotonic()

        ret, raw = cap.read() 
        if not ret:
            continue 

        annotated = tracker.process_frame(raw)
        HAND_STATE = tracker.get_hand_state() 
        DEPTH = estimator.estimate(raw)
        FRAME = annotated 

        elapsed = time.monotonic() - t0 
        sleep = period - elapsed 

        if sleep > 0:
            time.sleep(sleep)


def control_thread(tracker, sim, solver, pd, stop_event):
    # 50 hz loop
    armed = False 
    ref_palm_x = 0.0
    ref_palm_y = 0.0 
    ref_depth = None 
    ref_ee_pos = None 
    curr_joints = sim.get_qpos().copy()
    gripper_closed = False 

    while not stop_event.is_set() and sim.is_running():
        t0 = time.monotonic()

        state = HAND_STATE 
        dep = DEPTH

        if state is not None:
            gesture = state["gesture"]
            is_closed = state["is_closed"]
            px, py = state["palm_x"], state["palm_y"]

            gripper_closed = is_closed 

            if gesture == "thumbs_up" and not armed:
                armed = True 
                ref_palm_x  = px
                ref_palm_y  = py
                ref_depth   = dep
                ref_ee_pos  = solver.get_end_effector_pos().copy()
                pd.reset(ref_ee_pos)

            elif gesture == "thumbs_down" and armed:
                armed = False


            if armed:
                dx_img = px - ref_palm_x
                dy_img = py - ref_palm_y
 
                # Only update if movement exceeds jitter threshold
                movement = np.sqrt(dx_img**2 + dy_img**2)
                if movement > JITTER_THRESHOLD:
                    dx_world = dx_img * HAND_EE_SCALE_XY
                    dy_world = -dy_img * HAND_EE_SCALE_XY 
 
                    if dep is not None and ref_depth is not None:
                        dz_world = (ref_depth - dep) * HAND_EE_SCALE_Z  # closer = extend
                    else:
                        dz_world = 0.0
 
                    desired = ref_ee_pos + np.array([dx_world, dy_world, dz_world])
                    desired = clamp(desired)
                    pd.set_desired(desired)


                current_ee = solver.get_end_effector_pos()
                delta = pd.compute(current_ee)

                new_q  = solver.solve_incremental(delta)
                sim.set_joints(new_q)
                current_joints = new_q
        
                sim.set_gripper(gripper_closed)
                sim.step()
        
                elapsed = time.monotonic() - t0
                sleep = CONTROL_PERIOD - elapsed

                if sleep > 0:
                    time.sleep(sleep)

def main():
    tracker = HandTracker()
    sim = MujocoWrapper()
    estimator = DepthEstimator()
    IK = IKSolver(sim.model, sim.data, LIMITS)
    pd = PDController()

    sim.launch()

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        ret, frame = cap.read()
        print("Error: cannot open camera.")
        return

    stop_event = threading.Event()

    # CAMERA THREAD
    camera = threading.Thread(target=camera_thread, args =(tracker, estimator, cap, stop_event), daemon=True, name="camera_thread",)
    camera.start()


    # CONTROL THREAD
    control = threading.Thread(target=control_thread, args = (tracker, sim, IK, pd, stop_event), daemon= True, name = "control_thread",)
    control.start()

    try:
        while sim.is_running():
            fr = FRAME
            if fr is not None:
                state = HAND_STATE
                gesture = state["gesture"] if state else None
                closed = state["is_closed"] if state else False 

                ee_pos = None
                try:
                    ee_pos = IK.get_end_effector_pos()
                except Exception:
                    pass 

                armed = pd.desired is not None 

                fr = draw(fr, armed, gesture, closed, ee_pos)

        # target = assemble_target(palm_x, palm_y, depth)  # coord mapping function
        # sim.step()

            cv.imshow("URHandTeleop", fr)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error running main loop: {e}")

    finally:
        stop_event.set()
        camera.join(timeout=2)
        control.join(timeout=2)

        sim.close()
        cap.release()
        tracker.close()
        cv.destroyAllWindows()



if __name__ == "__main__":
    try: 
        main()

    except Exception as e:
        print(f"Error running program: {e}")
