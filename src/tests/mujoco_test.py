import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('arm_models/ur5e_model/ur5e.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as v:
    import time

    while True:
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)
