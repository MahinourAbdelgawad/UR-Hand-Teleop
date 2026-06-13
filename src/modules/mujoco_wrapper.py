import mujoco
import mujoco.viewer
import numpy as np

class MujocoWrapper:
    def __init__(self, model = "arm_models/ur5e_model/ur5e.xml", gripper = "arm_models/robotiq_2f85/2f85.xml", scene = None):
        self.model = None
        self.data = None
        self.viewer = None
        self.arm_qpos_idx = None
        
        try:
            # self.model = mujoco.MjModel.from_xml_path(model)
            self.model = self._build_combined_model(
                model,
                gripper,
                scene
            )
            if self.model is None:
                raise RuntimeError("Failed to build combined model")
            
            self.data = mujoco.MjData(self.model)

            # set default pose
            # # TODO: tweak this default pose
            # self.data.qpos[:6] = [0, -np.pi/4, -np.pi/2, -np.pi/4, np.pi/2, 0]
            # self.data.ctrl[:6] = [0, -np.pi/4, -np.pi/2, -np.pi/4, np.pi/2, 0]

            # self.data.ctrl[6] = 0 # gripper actuator open

            mujoco.mj_forward(self.model, self.data)
            self.arm_qpos_idx = self.get_arm_joint_indices()

            # print('DOFs:', self.model.nv, '  Actuators:', self.model.nu)

            default_pose = [0, -np.pi/4, -np.pi/2, -np.pi/4, np.pi/2, 0]
            for i, qi in enumerate(self.arm_qpos_idx):
                self.data.qpos[qi] = default_pose[i]
                self.data.ctrl[i]  = default_pose[i]
            mujoco.mj_forward(self.model, self.data)


        except Exception as e:
            print(f"Error initializing MuJoCo: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.data = None
            raise


    def launch(self):
        if self.model is None or self.data is None:
            print("Error: Model or data not initialized. Cannot launch viewer.")
            return False
        
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            return True
            
        except Exception as e:
            print(f"Error launching MuJoCo Viewer: {e}")
            self.viewer = None
            return False

        
    def set_joint(self, joint_index, angle_rad):
        """
        Set a single joint to a target angle in radians
        """
        try:
            if self.data is None or joint_index < 0 or joint_index >= len(self.data.ctrl):
                raise IndexError(f"Invalid joint index {joint_index}, available: {len(self.data.ctrl) if self.data else 0}")
            self.data.ctrl[joint_index] = angle_rad

        except Exception as e:
            print(f"Error setting joint {joint_index} to angle {angle_rad}: {e}")

    
    def set_joints(self, angles):
        """
        Set multiple joints to target angles in radians
        """
        if self.data is None:
            return
        
        try:
            if len(angles) > 6:
                print(f"Warning: trying to set {len(angles)} joints but only 6 arm joints available")
                angles = angles[:6]
            
            for i, idx in enumerate(self.arm_qpos_idx):
                self.data.ctrl[i] = float(angles[i])

        except Exception as e:
            print(f"Error setting joints: {e}")

    
    def set_gripper(self, status):
        """
        Set gripper command. Expects gripper actuator at index 6.
        """
        if self.data is None or len(self.data.ctrl) <= 6:
            if self.data is not None:
                print(f"Warning: Gripper index 6 not available. Available: {len(self.data.ctrl)} controls")
            return
        
        try:
            value = 255 if status else 0
            self.data.ctrl[6] = np.clip(value, 0, 255)
        except Exception as e:
            print(f"Error setting gripper: {e}")

        
    def step(self):
        """
        Advance the simulation by one step and sync
        """
        if self.model is None or self.data is None:
            return False
            
        try:
            mujoco.mj_step(self.model, self.data)

            if self.viewer:
                self.viewer.sync()
            
            return True
        except Exception as e:
            print(f"Error stepping simulation: {e}")
            return False


    def is_running(self):
        return self.viewer is not None and self.viewer.is_running()
    

    def close(self):
        try:
            if self.viewer:
                self.viewer.close()

        except Exception as e:
            print(f"Error closing Mujoco Viewer: {e}")
            

    def _get_body(self, parent_body, target_name):
        if parent_body.name == target_name:
            return parent_body
        
        child = parent_body.first_body()
        while child is not None:
            found = self._get_body(child, target_name)
            if found:
                return found
            
            # MjSpec uses this pattern to iterate through siblings
            child = parent_body.next_body(child)
            
        return None

    def _build_combined_model(self, ur5e_path, gripper_path, scene = None):
        arm = mujoco.MjSpec.from_file(ur5e_path)
        gripper = mujoco.MjSpec.from_file(gripper_path)

        wrist = self._get_body(arm.worldbody, 'wrist_3_link')
        if wrist is None:
            raise ValueError("Could not find 'wrist_3_link' in the arm model.")

        frame = wrist.add_frame()
        frame.name = 'attachment_frame'
        
        arm.attach(gripper, frame=frame, prefix='gripper_')

        if scene is not None:
            # Attach arm into the scene
            mount = scene.worldbody.add_frame()
            mount.name = "arm_mount"
            mount.pos = (0.0, 0.0, 0.04) # lift base a bit

            scene.attach(arm, frame=mount, prefix="arm_")
            
            return scene.compile()
        else:
            return arm.compile()

        
    def get_qpos(self):
        return self.data.qpos[self.arm_qpos_idx].copy()
    
    def get_arm_joint_indices(self):
        arm_joint_names = [
            "arm_shoulder_pan_joint",
            "arm_shoulder_lift_joint", 
            "arm_elbow_joint",
            "arm_wrist_1_joint",
            "arm_wrist_2_joint",
            "arm_wrist_3_joint",
        ]
        
        qpos_indices = []
        for name in arm_joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id == -1:
                bare = name.replace("arm_", "", 1)
                jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, bare)
            if jnt_id == -1:
                raise ValueError(f"Could not find joint '{name}' in model")
            
            qpos_indices.append(self.model.jnt_qposadr[jnt_id])
        
        return np.array(qpos_indices) 

        
