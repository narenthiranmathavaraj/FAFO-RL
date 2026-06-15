import mujoco
import mujoco.viewer
import os
os.environ["MUJOCO_GL"] = "egl"
model = mujoco.MjModel.from_xml_path(
    r"E:\personal projects\FAFO-RL\agnet_xml\reacher.xml"
)
data = mujoco.MjData(model)

print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Actuators:", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()