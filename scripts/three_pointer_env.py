from pandas.io.formats import printing
__credits__ = ["Pixel_Dude"]

import numpy as np
from pathlib import Path

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import mujoco


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 10.04,
}



class ThreePointerEnv(MujocoEnv, utils.EzPickle):
    current_dir = Path.cwd()
    xml_path_str = str(current_dir.parent / "agnet_xml" / "three_pointer.xml")
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = xml_path_str,
        frame_skip: int = 2,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_control_weight,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.grip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_grip")
        self.hoop_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hoop_center")

        # MujocoEnv._set_action_space() already gave us a Box(shape=(1,)) from
        # the arm motor's ctrlrange. The weld release isn't a real actuator —
        # it's a constraint flag (eq_active) — so it can't be auto-derived.
        # Bolt on a second channel by hand and read it ourselves in step().
        low = np.append(self.action_space.low, -1.0).astype(np.float32)
        high = np.append(self.action_space.high, 1.0).astype(np.float32)
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        
    def step(self, action):
        self.reward = 0.0          # initialize here, not in _get_rew
        self.termination = False

        action = np.asarray(action, dtype=np.float64)
        motor_ctrl = action[:1]
        release_signal = action[1]

        if release_signal > 0.0 and self.data.eq_active[self.grip_id]:
            self.data.eq_active[self.grip_id] = False
            self.reward += 50.0     # small release bonus

        self.do_simulation(motor_ctrl, self.frame_skip)
        if self.data.eq_active[self.grip_id] == False and self.data.qpos[3] <0.15: 
            if(self.data.qpos[1] <2.8 or self.data.qpos[1]>4):
                self.termination  = True
        observation = self._get_obs()
        reward, reward_info = self._get_rew(motor_ctrl)
        info = reward_info

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, self.termination , False, info

    def _get_rew(self, action):
        ball_released = not bool(self.data.eq_active[self.grip_id])

        # dist reward only after release
        if ball_released:
            reward_no_release = 0
            vec = (self.get_body_com("basketball")
                - self.data.site_xpos[self.hoop_site_id])
            reward_dist = -np.linalg.norm(vec) * self._reward_dist_weight

            # reward horizontal velocity toward hoop
            ball_vx = self.data.qvel[1]
            reward_flight = 0.3 * max(ball_vx, 0)   # only reward moving toward hoop

            # success
            if np.linalg.norm(vec) < 0.25:
                self.reward += 500.0

            # bad landing
            if self.termination:
                self.reward += -100.0
        else:
            reward_no_release = -10
            vec = (self.get_body_com("basketball")
                - self.data.site_xpos[self.hoop_site_id])
            reward_dist = -np.linalg.norm(vec) * self._reward_dist_weight
            reward_flight = 0.0

        # control cost always
        reward_ctrl = -np.square(action).sum() * 0.01  # reduced weight

        self.reward += reward_dist + reward_flight + reward_ctrl + reward_no_release

        return self.reward, {
            "reward_dist":   reward_dist,
            "reward_ctrl":   reward_ctrl,
            "ball_released": ball_released,
        }

    def reset_model(self):
        self.data.qpos[1:4] = [-5.15, 0, 1.3] 
        self.data.qpos[4:8] = [1, 0, 0, 0]
        self.data.qvel[:] = np.zeros_like(self.init_qvel)
        self.set_state(self.data.qpos, self.data.qvel)
        self.data.eq_active[self.grip_id] = True
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos[0]
        return np.concatenate(
            [
                np.array([np.cos(theta)]),
                np.array([np.sin(theta)]),
                np.array([self.data.qvel[0]]),
                np.array([self.data.qpos[1]]),
                np.array([self.data.qpos[3]]),
                np.array([self.data.qvel[1]]),
                np.array([self.data.qvel[3]]),
                np.array([float(self.data.eq_active[self.grip_id])]),
                (self.get_body_com("basketball") - self.data.site_xpos[self.hoop_site_id])[[0, 2]],
            ]
        )