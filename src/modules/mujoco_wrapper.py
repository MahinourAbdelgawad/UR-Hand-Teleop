import mujoco
import mujoco.viewer
import numpy as np

class MujocoWrapper:
    def __init__(self, model = "arm_models/ur5e_model/ur5e.xml", gripper = "arm_models/robotiq_2f85/2f85.xml", scene = None):
        self.model = None
        self.data = None
        self.viewer = None
        self.site_name = "attachment_site"
        
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

            # print('DOFs:', self.model.nv, '  Actuators:', self.model.nu)


        except Exception as e:
            print(f"Error initializing MuJoCo: {e}")
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
            if len(angles) > len(self.data.ctrl):
                print(f"Warning: trying to set {len(angles)} joints but only {len(self.data.ctrl)} available")
                angles = angles[:len(self.data.ctrl)]
            
            self.data.ctrl[:len(angles)] = angles

        except Exception as e:
            print(f"Error setting joints: {e}")

    
    def set_gripper(self, status):
        """
        Set gripper command. Expects gripper actuator at index 6.
        """
        if self.data is None or len(self.data.ctrl) <= 6:
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

    def _build_combined_model(self, ur5e_path, gripper_path, scene_path):
        arm = mujoco.MjSpec.from_file(ur5e_path)
        gripper = mujoco.MjSpec.from_file(gripper_path)

        wrist = self._get_body(arm.worldbody, 'wrist_3_link')
        if wrist is None:
            raise ValueError("Could not find 'wrist_3_link' in the arm model.")

        frame = wrist.add_frame()
        frame.name = 'attachment_frame'
        
        arm.attach(gripper, frame=frame, prefix='gripper_')

        if scene_path is not None:
            self._merge_scene(arm, scene_path)

        return arm.compile()

        
    def get_qpos(self):
        return self.data.qpos[:6].copy()

        
    def _merge_scene(self, arm_spec, scene_path):
            scene = mujoco.MjSpec.from_file(scene_path)
    
            # Copy lights
            light = scene.worldbody.first_light()
            while light is not None:
                arm_spec.worldbody.add_light(
                    name=light.name,
                    pos=list(light.pos),
                    dir=list(light.dir),
                    diffuse=list(light.diffuse),
                    specular=list(light.specular),
                    castshadow=light.castshadow,
                )
                light = scene.worldbody.next_light(light)
    
            # Copy geoms
            geom = scene.worldbody.first_geom()
            while geom is not None:
                arm_spec.worldbody.add_geom(
                    name=geom.name,
                    type=geom.type,
                    size=list(geom.size),
                    pos=list(geom.pos),
                    rgba=list(geom.rgba),
                    friction=list(geom.friction),
                )
                geom = scene.worldbody.next_geom(geom)
    
            # Copy bodies
            body = scene.worldbody.first_body()
            while body is not None:
                self._copy_body(arm_spec.worldbody, body)
                body = scene.worldbody.next_body(body)


    def _copy_body(self, parent_spec_body, src_body):
        new_body = parent_spec_body.add_body(name=src_body.name)
        new_body.pos = list(src_body.pos)
        new_body.quat = list(src_body.quat)
 
        # Copy geoms
        geom = src_body.first_geom()
        while geom is not None:
            kwargs = dict(
                name=geom.name,
                type=geom.type,
                size=list(geom.size),
                pos=list(geom.pos),
                rgba=list(geom.rgba),
            )
            if geom.mass > 0:
                kwargs["mass"] = geom.mass

            if list(geom.friction) != [0.0, 0.0, 0.0]:
                kwargs["friction"] = list(geom.friction)

            if geom.contype == 0:
                kwargs["contype"] = 0
                kwargs["conaffinity"] = 0

            new_body.add_geom(**kwargs)
            geom = src_body.next_geom(geom)
 
        joint = src_body.first_joint()

        while joint is not None:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                new_body.add_freejoint(name=joint.name)

            joint = src_body.next_joint(joint)
 
        # Recurse into child bodies
        child = src_body.first_body()

        while child is not None:
            self._copy_body(new_body, child)
            child = src_body.next_body(child)
 
    def _find_body(self, parent, name):
        if parent.name == name:
            return parent
        
        child = parent.first_body()

        while child is not None:
            found = self._find_body(child, name)
            if found:
                return found
            
            child = parent.next_body(child)
            
        return None