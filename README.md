# 🏀 Trick Shot Basketball Robot — RL from Scratch

> Teaching a robot arm to make a basketball trick shot using Reinforcement Learning — built as a learning journey from zero to a working sim.

---
<!-- Add your demo GIF here -->
> ## 📎 Demo
>
> ![Demo GIF](assets/gif/demo.gif)
>


## The Goal

The end goal of this project is to train a robotic arm in simulation to perform a **basketball trick shot** — throw a ball onto a **tilting surface** that deflects it into a hoop. The agent must learn not just *how* to throw, but *when* to throw, accounting for the moving surface and the delayed outcome of the shot.

This RL problem contains:
- Continuous action space (torque control)
- Contact-rich physics (ball, surface, rim)
- Delayed sparse reward (outcome only known at landing)
- Timing-dependent policy (surface angle changes mid-episode)

---

##  Current Progess 1-DOF Arm Successfully Makes the Hoop

A single-joint robot arm learns to throw a basketball into a fixed hoop using SAC. This is the foundation before adding the trick-shot surface.


---

## Roadmap


### Custom Environment
- Designed custom MuJoCo XML: 1-DOF arm + basketball + fixed hoop
- Implemented weld constraint (`<equality>`) for grab/release mechanic
- Built `ThreePointerEnv` as a `MujocoEnv` subclass with:
  - Custom observation space (arm angle, ball position, ball velocity, grip state, dist-to-hoop)
  - Augmented action space (motor torque + release signal)
  - Shaped reward function (dist-to-hoop, control cost, release bonus, success bonus)


### Next Stage: Trick Shot Platform
- Add tilting surface (motorized hinge joint) between arm and hoop
- Phase A: fixed tilt angle → agent observes and compensates
- Phase B: dynamic tilt (sinusoidal) → agent must time the throw
- Reward shaping for deflection trajectory
- Analysis of timing-dependent policy behaviour

---


## Key Notes

**Reward shaping is more important than algorithm choice**

**A fixed seed ensures reproducibility — the same seed will produce the same training run. Note that convergence is seed-sensitive:  while seed=3 did not converged but seed=33 did, which is why running multiple seeds matters.**

**Weld constraints for grab/release**
MuJoCo `<equality><weld>` constraints let you rigidly attach bodies and release them programmatically via `data.eq_active[id] = False`. This simulates a gripper without needing finger joints.

---

## Project Structure

```
FAFO-RL/
│
├── scripts/
│   ├── main.ipynb              # training runs — rough, needs cleanup
│   └── three_pointer_env.py   # custom Gymnasium env (ThreePointerEnv)
│
├── agent_xml/
│   └── three_pointer.xml      # MuJoCo XML — arm + ball + hoop scene
│
├── assets/
│   └── demo.gif               # demo recording (add yours here)
│
├── rl_roadmap.html               # interactive RL learning roadmap — track progress
│                              # stage-by-stage tasks, tips, hyperparameter guide
│                              # open in browse
├── three_pointer_log/         # TensorBoard logs
├── mujoco_xml_readme.md       # To learn about mujoco parameters in xml
└── README.md
```

> **Code note:** Comments in `scripts/main.ipynb` and `three_pointer_env.py` are minimal and rough — written fast during experimentation. Cleanup and proper docstrings are planned.

---

## Setup

```bash
# Clone
git clone https://github.com/yourusername/FAFO-RL.git
cd FAFO-RL

# Create venv
python -m venv fafo_rl_env
fafo_rl_env\Scripts\Activate.ps1  # Windows
source fafo_rl_env/bin/activate   # Linux

# Install
pip install -r requirements.txt
```

**Tested stack:**
```
Python      3.11
MuJoCo      3.9.0
Gymnasium   1.2.3
SB3         2.8.0
PyTorch     2.x
OS          Windows 11
GPU         RTX 4050 (training runs on CPU — MuJoCo physics is CPU-bound)
```

---

## Run Scripts

```bash
# Open training notebook
jupyter notebook scripts/main.ipynb

# Watch TensorBoard
tensorboard --logdir ./three_pointer_log/

# Visualise trained policy
python scripts/visualise.py   # coming soon
```

---

## References 

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium MuJoCo Envs](https://gymnasium.farama.org/environments/mujoco/)

