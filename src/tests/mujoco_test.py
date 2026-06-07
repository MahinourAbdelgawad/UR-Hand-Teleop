import mujoco
import mujoco.viewer

def get_body(parent_body, target_name):
    if parent_body.name == target_name:
        return parent_body
    
    child = parent_body.first_body()
    while child is not None:
        found = get_body(child, target_name)
        if found:
            return found
        
        # MjSpec uses this pattern to iterate through siblings
        child = parent_body.next_body(child)
        
    return None

def build_combined_model(ur5e_path, gripper_path):
    arm = mujoco.MjSpec.from_file(ur5e_path)
    gripper = mujoco.MjSpec.from_file(gripper_path)

    wrist = get_body(arm.worldbody, 'wrist_3_link')
    if wrist is None:
        raise ValueError("Could not find 'wrist_3_link' in the arm model.")

    frame = wrist.add_frame()
    frame.name = 'attachment_frame'
    
    arm.attach(gripper, frame=frame, prefix='gripper_')

    return arm.compile()


# model = mujoco.MjModel.from_xml_path('arm_models/ur5e_model/ur5e.xml')
model = build_combined_model(
    'arm_models/ur5e_model/ur5e.xml',
    'arm_models/robotiq_2f85/2f85.xml'
)

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as v:
    import time

    while True:
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)


