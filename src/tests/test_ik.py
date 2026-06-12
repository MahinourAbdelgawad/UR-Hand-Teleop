import time
import numpy as np
import mujoco
import mujoco.viewer

from src.modules.mujoco_wrapper import MujocoWrapper
from src.modules.ik_solver import IKSolver


TARGET = np.array([0.3, 0.0, 0.4]) # (x, y, z) in metres

DAMPING    = 0.05
STEP_SIZE  = 0.3
MAX_ITER   = 1
TOL        = 1e-3

WORKSPACE = {
    "x": (-0.7, 0.7),
    "y": (-0.7, 0.7),
    "z": (0.05, 0.9),
}


def main():
    print("=" * 55)
    print("  IK Isolation Test")
    print("  Target:", TARGET)
    print("  Damping:", DAMPING, "  Step size:", STEP_SIZE)
    print("=" * 55)

    sim = MujocoWrapper()
    ik  = IKSolver(
        sim.model, sim.data,
        WORKSPACE,
        damping=DAMPING,
        step_size=STEP_SIZE,
        max_iter=MAX_ITER,
        tol=TOL,
    )

    sim.launch()
    time.sleep(0.5)  

    start = time.monotonic()
    prev_q = sim.get_qpos().copy()
    max_dq = 0.0
    converged = False

    while sim.is_running():
        t0 = time.monotonic()
        elapsed = t0 - start

        q = ik.solve(TARGET, q_init=sim.get_qpos().copy())
        sim.set_joints(q)
        sim.step()

        ee  = ik.get_end_effector_pos()
        err = float(np.linalg.norm(TARGET - ee))
        dq  = float(np.linalg.norm(q - prev_q))
        max_dq = max(max_dq, dq)
        prev_q = q.copy()

        if int(elapsed * 2) != int((elapsed - 0.02) * 2):
            print(f"  t={elapsed:5.1f}s  EE error={err:.4f}m  joint_delta={dq:.5f} rad")

        if elapsed > 1.0 and err < TOL * 2:
            if not converged:
                print(f"\nConverged at t={elapsed:.2f}s  (error={err:.4f}m)")
                converged = True

        if elapsed > 3.0:
            if not hasattr(main, "_phase2_start"):
                main._phase2_start = elapsed
                main._phase2_dqs   = []

            main._phase2_dqs.append(dq)

        if elapsed > 8.0:
            break

        sleep = 0.02 - (time.monotonic() - t0)   # 50 Hz
        if sleep > 0:
            time.sleep(sleep)

    print("\n" + "=" * 55)
    print("  Results")
    print("=" * 55)

    ee  = ik.get_end_effector_pos()
    err = float(np.linalg.norm(TARGET - ee))
    print(f"  Final EE error : {err:.4f} m")
    print(f"  Max joint delta: {max_dq:.5f} rad")

    if hasattr(main, "_phase2_dqs") and main._phase2_dqs:
        hold_dq = np.array(main._phase2_dqs)
        print(f"  Hold phase — mean joint_delta : {hold_dq.mean():.5f} rad")
        print(f"  Hold phase — max  joint_delta : {hold_dq.max():.5f} rad")

        if hold_dq.max() < 0.001:
            print("\nPASS! IK is stable.")
        elif hold_dq.max() < 0.01:
            print("\nslight drift.")
        else:
            print("\nFAIL!")

    sim.close()


if __name__ == "__main__":
    main()