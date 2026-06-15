import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path(
    r"E:\personal projects\FAFO-RL\agnet_xml\three_pointer.xml"
)
data = mujoco.MjData(model)

grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_grip")
print("Grip id:", grip_id)

WINDUP = 0
FLIGHT = 1
LANDED = 2
phase  = WINDUP

data.eq_active[grip_id] = True
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0
    while viewer.is_running():
        step_start = time.time()

        if phase == WINDUP:
            data.ctrl[0] = 1.0
            mujoco.mj_step(model, data)
            shoulder_vel = abs(data.qvel[0])
            step_count += 1

            # debug every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count} | shoulder_vel={shoulder_vel:.3f} rad/s")

            if shoulder_vel > 3.0:   # lowered from 8.0
                data.eq_active[grip_id] = False
                phase = FLIGHT
                print(f"Released! vel={shoulder_vel:.2f} rad/s at step {step_count}")

        elif phase == FLIGHT:
            data.ctrl[0] = 0.0
            mujoco.mj_step(model, data)
            ball_x = data.qpos[1]
            ball_z = data.qpos[2]

            if ball_z < 0.15:
                dist = np.sqrt((ball_x - 6.0)**2 + (ball_z - 3.0)**2)
                print(f"Landed: x={ball_x:.2f} z={ball_z:.2f} | dist={dist:.2f}m")
                phase = LANDED

        elif phase == LANDED:
            pass

        viewer.sync()
        time_elapsed = time.time() - step_start
        if time_elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - time_elapsed)