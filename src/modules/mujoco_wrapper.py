import mujoco
import mujoco.viewer
import numpy as np

class MujocoWrapper:
    def __init__(self, model = "arm_models/ur5e_model/ur5e.xml", gripper = "arm_models/robotiq_2f85/2f85.xml"):
        try:
            # self.model = mujoco.MjModel.from_xml_path(model)
            self.model = self._build_combined_model(
                model,
                gripper
            )
            self.data = mujoco.MjData(self.model)

            # set default pose
            # # TODO: tweak this default pose
            # self.data.qpos[:6] = [0, -np.pi/4, -np.pi/2, -np.pi/4, np.pi/2, 0]
            # self.data.ctrl[:6] = [0, -np.pi/4, -np.pi/2, -np.pi/4, np.pi/2, 0]

            # self.data.ctrl[6] = 0 # gripper actuator open

            mujoco.mj_forward(self.model, self.data)

            self.viewer = None
            # print('DOFs:', self.model.nv, '  Actuators:', self.model.nu)


        except Exception as e:
            print(f"Error initializing Mujoco Viewer: {e}")


    def launch(self):
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
        except Exception as e:
            print(f"Error launching Mujoco Viewer: {e}")

        
    def set_joint(self, joint_index, angle_rad):
        """
        Set a single joint to a target angle in radians
        """
        try:
            self.data.ctrl[joint_index] = angle_rad


        except Exception as e:
            print(f"Error setting joint {joint_index} to angle {angle_rad}: {e}")

        
    def step(self):
        """
        Advance the simulation by one step and sync
        """
        try:
            mujoco.mj_step(self.model, self.data)

            if self.viewer:
                self.viewer.sync()
            
        except Exception as e:
            print(f"Error stepping simulation: {e}")


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

    def _build_combined_model(self, ur5e_path, gripper_path):
        arm = mujoco.MjSpec.from_file(ur5e_path)
        gripper = mujoco.MjSpec.from_file(gripper_path)

        wrist = self._get_body(arm.worldbody, 'wrist_3_link')
        if wrist is None:
            raise ValueError("Could not find 'wrist_3_link' in the arm model.")

        frame = wrist.add_frame()
        frame.name = 'attachment_frame'
        
        arm.attach(gripper, frame=frame, prefix='gripper_')

        return arm.compile()


     


        
