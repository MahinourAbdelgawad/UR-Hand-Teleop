"""
Tests the full control pipeline using keyboard input instead of hand tracking

Controls:
    W / S   →  move target +Y / -Y  (forward / back)
    A / D   →  move target -X / +X  (left / right)
    Q / E   →  move target +Z / -Z  (up / down)
    G       →  toggle gripper open / closed
    R       →  reset target to home position
    ESC     →  quit
"""

import time
import threading

import cv2 as cv
import numpy as np

from src.modules.mujoco_wrapper import MujocoWrapper
from src.modules.ik_solver import IKSolver
from src.modules.pd_controller import PDController
from src.modules.scene import Scene

HOME = np.array([0.3, 0.0, 0.4])  

STEP = 0.01  
CTRL_HZ = 50
CTRL_DT = 1.0 / CTRL_HZ

WORKSPACE = {
    "x": (-0.7, 0.7),
    "y": (-0.7, 0.7),
    "z": (0.05, 0.9),
}

PD_KP = 5.0
PD_KD = 0.0 
PD_MAX_STEP = 0.05

IK_DAMPING = 0.05
IK_STEP_SIZE = 0.3

_target = HOME.copy()
_gripper_closed = False
_running = True
_lock = threading.Lock()
_ee_pos = HOME.copy() 


def clamp(pos):
    return np.array([
        np.clip(pos[0], WORKSPACE["x"][0], WORKSPACE["x"][1]),
        np.clip(pos[1], WORKSPACE["y"][0], WORKSPACE["y"][1]),
        np.clip(pos[2], WORKSPACE["z"][0], WORKSPACE["z"][1]),
    ])


def control_thread(sim, ik, pd, stop_event):
    global _running, _ee_pos

    pd.reset(HOME)

    while not stop_event.is_set() and sim.is_running():
        t0 = time.monotonic()

        with _lock:
            target = _target.copy()
            gc = _gripper_closed

        pd.settarget(target)
        current_ee = ik.get_end_effector_pos()
        delta = pd.compute(current_ee)
        new_q = ik.solve_incremental(delta)

        sim.set_joints(new_q)
        sim.set_gripper(gc)
        sim.step()

        elapsed = time.monotonic() - t0
        sleep = CTRL_DT - elapsed
        if sleep > 0:
            time.sleep(sleep)

    _running = False


def make_hud(target, ee, gripper_closed):
    """Return a small OpenCV image showing current state."""
    h, w = 260, 420
    img  = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    def txt(text, y, color=(200, 200, 200), scale=0.55, thickness=1):
        cv.putText(img, text, (16, y), cv.FONT_HERSHEY_SIMPLEX,
                   scale, color, thickness, cv.LINE_AA)

    txt("Keyboard Pipeline Test", 28, (255, 255, 255), 0.65, 2)
    cv.line(img, (16, 38), (w - 16, 38), (80, 80, 80), 1)

    txt(f"Target  X: {target[0]:+.3f}  Y: {target[1]:+.3f}  Z: {target[2]:+.3f}", 65)
    txt(f"EE pos  X: {ee[0]:+.3f}  Y: {ee[1]:+.3f}  Z: {ee[2]:+.3f}", 90)

    err = float(np.linalg.norm(target - ee))
    col = (0, 220, 0) if err < 0.01 else (0, 180, 255) if err < 0.05 else (0, 80, 220)
    txt(f"Error:  {err:.4f} m", 115, col)

    grip_txt = "CLOSED" if gripper_closed else "OPEN"
    grip_col = (0, 80, 220) if gripper_closed else (0, 200, 80)
    txt(f"Gripper: {grip_txt}", 140, grip_col)

    cv.line(img, (16, 158), (w - 16, 158), (60, 60, 60), 1)
    txt("W/S  forward/back    A/D  left/right", 178, (120, 120, 120), 0.45)
    txt("Q/E  up/down         G    gripper",    198, (120, 120, 120), 0.45)
    txt("R    reset home      ESC  quit",       218, (120, 120, 120), 0.45)

    return img


def main():
    global _target, _gripper_closed, _running

    print("=" * 55)
    print("  Keyboard Pipeline Test")
    print("  Home position:", HOME)
    print("=" * 55)

    scene = Scene().build_scene()
    sim = MujocoWrapper(scene=scene)
    ik  = IKSolver(
        sim.model, sim.data,
        WORKSPACE,
        arm_qpos_idx=sim.arm_qpos_idx,
        damping=IK_DAMPING,
        step_size=IK_STEP_SIZE,
    )
    print("Init IK solver")
    pd  = PDController(kp=PD_KP, kd=PD_KD, max_step=PD_MAX_STEP)
    print("Init pd controller")

    sim.launch()
    time.sleep(0.5)
    print("launched sim")

    stop_event = threading.Event()
    ctrl = threading.Thread(
        target=control_thread,
        args=(sim, ik, pd, stop_event),
        daemon=True,
        name="ctrl",
    )
    ctrl.start()
    print("started control thread")


    # Key → (axis, direction)
    KEY_MAP = {
        ord('w'): (1,  1),   # +Y
        ord('s'): (1, -1),   # -Y
        ord('a'): (0, -1),   # -X
        ord('d'): (0,  1),   # +X
        ord('q'): (2,  1),   # +Z
        ord('e'): (2, -1),   # -Z
    }

    try:
        while _running and sim.is_running():
            # print("in while loop")
            with _lock:
                target = _target.copy()
                gc     = _gripper_closed

            # ee  = ik.get_end_effector_pos()
            ee = _ee_pos.copy()
            hud = make_hud(target, ee, gc)

            cv.imshow("Keyboard Test", hud)
            key = cv.waitKey(20) & 0xFF
            # print("opened cv window")

            if key == 27:# ESC
                break

            elif key in KEY_MAP:
                axis, sign = KEY_MAP[key]
                with _lock:
                    _target[axis] = np.clip(
                        _target[axis] + sign * STEP,
                        WORKSPACE[["x","y","z"][axis]][0],
                        WORKSPACE[["x","y","z"][axis]][1],
                    )
                # print(f"  Target → {_target}")


            elif key == ord('g'):
                with _lock:
                    _gripper_closed = not _gripper_closed
                print(f"  Gripper → {'CLOSED' if _gripper_closed else 'OPEN'}")

            elif key == ord('r'):
                with _lock:
                    _target = HOME.copy()
                pd.reset(ik.get_end_effector_pos())
                print(f"  Reset → {HOME}")

    finally:
        print("in finally")
        stop_event.set()
        ctrl.join(timeout=2)
        sim.close()
        cv.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()