import mujoco
import numpy as np

#TODO: go over this and verify the vibe coded logic

class IKSolver:
    def __init__(self, model, data, limits, site_name="attachment_site",
        max_iter=50, tol=1e-3, step_size=0.5, damping=0.01, adaptive_damping=True):

        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.damping = damping
        self.adaptive_damping = adaptive_damping

        if self.site_id == -1:
            raise ValueError(f"Site '{site_name}' not found in model. "
                             f"Check the site name using mj_id2name.")
        
        self.limits = limits

        self.joint_limit_margin = 0.02


    def _clamp(self, position):
        """
        Return position clipped to the workspace limits
        """
        return np.array([
            np.clip(position[0], self.limits["x"][0], self.limits["x"][1]),
            np.clip(position[1], self.limits["y"][0], self.limits["y"][1]),
            np.clip(position[2], self.limits["z"][0], self.limits["z"][1]),
        ])

    def solve(self, target_pos, q_init=None):
        if self.model is None or self.data is None:
            raise RuntimeError("Model or data is None")
        
        target_pos = self._clamp(np.asarray(target_pos, dtype=float))
 
        if q_init is not None:
            if len(q_init) != 6:
                raise ValueError(f"q_init must be 6-dimensional, got {len(q_init)}")
            self.data.qpos[:6] = q_init.copy()
 
        for i in range(self.max_iter):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[self.site_id].copy()
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
 
            if error_norm < self.tol:
                break
 
            J = self._get_jacobian()
            lam = self._compute_damping(J)
            J_dls = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(3))
            dq = self.step_size * J_dls @ error
            self.data.qpos[:6] += dq
            self._apply_joint_limits()
 
        return self.data.qpos[:6].copy()
    
    def solve_incremental(self, delta_pos, max_delta_norm = 0.05):
        """
        Apply an incremental cartesian delta to the current EE position.
 
        This is the method used by the PD control loop
        """
        if self.model is None or self.data is None:
            raise RuntimeError("Model or data is None")
        
        # Safety cap
        norm = np.linalg.norm(delta_pos)
        if norm > max_delta_norm:
            delta_pos = delta_pos / norm * max_delta_norm
 
        current_ee = self.get_end_effector_pos()
        target_pos = current_ee + delta_pos
        return self.solve(target_pos, q_init=self.data.qpos[:6].copy())
    

    def get_end_effector_pos(self):
        """Returns current end effector position in world frame."""
        if self.model is None or self.data is None:
            raise RuntimeError("Model or data is None")
        if self.site_id < 0 or self.site_id >= len(self.data.site_xpos):
            raise RuntimeError(f"Invalid site_id: {self.site_id}")
        
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    def get_error(self, target_pos):
        """Returns distance between current EE position and target in meters."""
        return float(np.linalg.norm(target_pos - self.get_end_effector_pos()))
    

    def _get_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
        return jacp[:, :6] # first 6 columns only
    

    def _compute_damping(self, J):
        if not self.adaptive_damping:
            return self.damping
        
        # Scale lam with inverse of manipulability measure
        manip = float(np.linalg.det(J @ J.T))
        if manip < 1e-4:
            # Very near singularity...use large damping
            return self.damping * 10.0
        
        elif manip < 1e-2:
            scale = 1.0 + (1e-2 - manip) / (1e-2 - 1e-4) * 9.0   # 1× → 10×
            return self.damping * scale
        
        return self.damping
    

    def _apply_joint_limits(self):
        m = self.joint_limit_margin

        for j in range(6):
            lo = self.model.jnt_range[j, 0] + m
            hi = self.model.jnt_range[j, 1] - m

            self.data.qpos[j] = np.clip(self.data.qpos[j], lo, hi)