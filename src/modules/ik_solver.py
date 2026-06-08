import mujoco
import numpy as np

#TODO: go over this and verify the vibe coded logic

class IKSolver:
    def __init__(self, model, data, site_name="attachment_site",
        max_iter=50, tol=1e-3, step_size=0.5, damping=0.01):

        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.damping = damping

        if self.site_id == -1:
            raise ValueError(f"Site '{site_name}' not found in model. "
                             f"Check the site name using mj_id2name.")

    def solve(self, target_pos, q_init=None):
        """
        Iterative Jacobian pseudoinverse IK solver.

        target_pos: np.array (3,) — desired end effector position in world frame (meters)
        q_init:     np.array (6,) — initial joint angles to warm start from (optional)

        Returns:    np.array (6,) — joint angles that reach target_pos within tolerance,
                    or best effort if max_iter reached without convergence
        """
        # warm start from provided config or current sim state
        if q_init is not None:
            self.data.qpos[:6] = q_init.copy()
        
        for i in range(self.max_iter):
            # forward kinematics — updates site positions
            mujoco.mj_forward(self.model, self.data)

            # current end effector position
            current_pos = self.data.site_xpos[self.site_id].copy()

            # error vector
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)

            if error_norm < self.tol:
                return self.data.qpos[:6].copy()

            # compute jacobian — mj_jacSite fills a (3, nv) matrix
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
            J = jacp[:, :6]  # first 6 cols = arm joints only, ignore gripper

            # check for singularity
            manipulability = np.linalg.det(J @ J.T)
            # if manipulability < 1e-4:
            #     print(f"Warning: near singularity at iter {i} "
            #           f"(manipulability={manipulability:.2e}) — holding position")
            #     break

            # damped least squares pseudoinverse
            # J_pinv = J^T * (J * J^T + lambda^2 * I)^-1
            J_pinv = J.T @ np.linalg.inv(J @ J.T + self.damping ** 2 * np.eye(3))

            # joint update
            dq = self.step_size * J_pinv @ error
            self.data.qpos[:6] += dq

            # clamp to joint limits
            for j in range(6):
                lo = self.model.jnt_range[j, 0]
                hi = self.model.jnt_range[j, 1]
                self.data.qpos[j] = np.clip(self.data.qpos[j], lo, hi)

        return self.data.qpos[:6].copy()

    def get_end_effector_pos(self):
        """Returns current end effector position in world frame."""
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    def get_error(self, target_pos):
        """Returns distance between current EE position and target in meters."""
        return float(np.linalg.norm(target_pos - self.get_end_effector_pos()))