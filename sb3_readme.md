# Stable Baselines3 — Complete Reference
> Based on your actual training output. Every number explained, every parameter you'll tune.

---

## Reading Your Training Output

```
| rollout/           |           |
|    ep_len_mean     | 200       |  ← average episode length (steps)
|    ep_rew_mean     | -1.27e+03 |  ← average total reward per episode (YOUR GOAL: maximize this)
| time/              |           |
|    episodes        | 4         |  ← total episodes completed so far
|    fps             | 29        |  ← simulation steps per second (your CPU speed)
|    time_elapsed    | 27        |  ← wall clock seconds since training started
|    total_timesteps | 800       |  ← total env.step() calls so far
| train/             |           |
|    actor_loss      | 20        |  ← how wrong the policy is (should decrease over time)
|    critic_loss     | 0.28      |  ← how wrong the value estimate is (should decrease)
|    ent_coef        | 0.813     |  ← current entropy coefficient (auto-tuned by SAC)
|    ent_coef_loss   | -0.342    |  ← entropy tuning loss (negative = reducing entropy = more deterministic)
|    learning_rate   | 0.0003    |  ← current LR (constant unless you use a schedule)
|    n_updates       | 699       |  ← gradient update steps done so far
```

### What "healthy" training looks like
```
ep_rew_mean   → starts very negative, trends upward over time. Noisy is normal.
actor_loss    → can go up or down, not always meaningful alone
critic_loss   → should generally decrease and stabilize
ent_coef      → starts high (exploration), decreases as policy converges
fps           → your 29 fps is fine for MLP policies on CPU
```

### Red flags
```
critic_loss exploding (>1000)  → LR too high, reduce by 10x
ep_rew_mean flat for 100k+ steps → stuck, check reward function
ent_coef → 0 too fast          → policy collapsed, increase ent_coef_init
fps < 10                        → policy too large, reduce net_arch
NaN in any field                → LR too high or reward not normalized
```

---

## SAC — Soft Actor-Critic

Best algorithm for your throw env. Sample-efficient, handles continuous actions natively.

```python
from stable_baselines3 import SAC

model = SAC(
    policy           = "MlpPolicy",   # neural net type
    env              = env,
    learning_rate    = 3e-4,          # adam optimizer LR
    buffer_size      = 1_000_000,     # replay buffer capacity (transitions)
    learning_starts  = 10_000,        # random steps before training begins
    batch_size       = 256,           # transitions sampled per gradient update
    tau              = 0.005,         # target network soft update rate
    gamma            = 0.99,          # discount factor
    train_freq       = 1,             # update every N env steps
    gradient_steps   = 1,             # gradient updates per train_freq
    ent_coef         = "auto",        # entropy coefficient (auto-tunes)
    target_entropy   = "auto",        # target entropy for auto ent_coef
    use_sde          = False,         # state-dependent exploration
    policy_kwargs    = dict(net_arch=[256, 256]),  # network size
    verbose          = 1,
    tensorboard_log  = "./tb_logs/",
    seed             = 42,
    device           = "cpu",         # keep cpu for MuJoCo training
)

model.learn(total_timesteps=500_000)
model.save("sac_throw")
```

---

## SAC Parameter Deep Dive

### `learning_rate` — Most Impactful
```
Default:  3e-4
Range:    1e-5 to 1e-3

Too high → critic_loss explodes, NaNs
Too low  → trains correctly but very slowly
For throw env: start 3e-4, drop to 1e-4 if unstable contact physics
```

### `buffer_size` — Replay Buffer
```
Default:  1_000_000
Range:    50_000 to 2_000_000

Stores past (obs, action, reward, next_obs) tuples.
Larger = more diverse training data = more stable.
With 8GB RAM: safe at 1M. Reduce to 300_000 if RAM is tight.
For throw env: 500_000 is sufficient.
```

### `learning_starts` — Warmup
```
Default:  100 (SB3 default, too low)
Better:   10_000

Random actions fill the buffer first. Training on <1000 random
transitions = learning from noise. Always set to at least 5000.
For throw env: 10_000
```

### `batch_size` — Gradient Step Size
```
Default:  256
Options:  128, 256, 512, 1024

Larger = more stable gradients, slower per-step.
Always powers of 2.
For throw env: 256, increase to 512 if training is noisy.
```

### `tau` — Target Network Update
```
Default:  0.005
Range:    0.001 to 0.02

SAC uses two networks: online and target.
tau controls how fast target catches up to online.
Too high → unstable. Too low → slow convergence.
Rarely needs changing.
```

### `gamma` — Discount Factor
```
Default:  0.99
Range:    0.9 to 0.9999

How much future rewards matter.
gamma=0.99 → reward 100 steps ahead worth 0.99^100 = 0.37x
gamma=0.999 → same reward worth 0.90x (much more future-aware)

For throw env: use 0.995 or 0.999 — the hoop reward comes
at the END of the throw trajectory (delayed reward).
Low gamma causes the agent to ignore the landing outcome.
```

### `train_freq` and `gradient_steps`
```
train_freq=1, gradient_steps=1  → update after every env step (default)
train_freq=4, gradient_steps=4  → collect 4 steps, then 4 updates

Rule: keep them equal. More gradient_steps than train_freq
= overfitting on recent data.

With n_envs=4 (parallel): set train_freq=1, gradient_steps=4
```

### `ent_coef` — Entropy Coefficient
```
Default:  "auto"  ← always use this

Controls exploration-exploitation tradeoff.
High ent_coef → more random (explore)
Low ent_coef  → more deterministic (exploit)

"auto" uses a Lagrangian method to automatically tune this.
target_entropy="auto" sets target = -dim(action_space) = -1 for your arm.

Only set manually if auto is unstable:
  ent_coef=0.1  → moderately explorative
  ent_coef=0.01 → nearly deterministic
```

### `policy_kwargs` — Network Architecture
```python
# Default (usually fine)
policy_kwargs = dict(net_arch=[256, 256])

# Larger (if policy underfitting — reward plateaus early)
policy_kwargs = dict(net_arch=[512, 512])

# With custom activation
import torch as th
policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=th.nn.ReLU  # default is ReLU, can try Tanh
)

# For throw env: [256, 256] is sufficient for state-based obs
# Only go larger if eval reward stops improving before 500k steps
```

### `use_sde` — State-Dependent Exploration
```
Default:  False

Alternative exploration strategy — correlated noise based on state.
Can help for tasks requiring consistent throw trajectories.
Try: use_sde=True, sde_sample_freq=64
Not needed until basic SAC converges first.
```

---

## PPO — Proximal Policy Optimization

On-policy alternative. More stable but less sample-efficient than SAC.
Use if SAC diverges.

```python
from stable_baselines3 import PPO

model = PPO(
    policy         = "MlpPolicy",
    env            = env,
    learning_rate  = 3e-4,
    n_steps        = 2048,      # steps collected per update
    batch_size     = 64,        # minibatch size for gradient steps
    n_epochs       = 10,        # passes over collected data
    gamma          = 0.99,
    gae_lambda     = 0.95,      # GAE smoothing (bias-variance tradeoff)
    clip_range     = 0.2,       # PPO clip parameter (core mechanism)
    ent_coef       = 0.0,       # entropy bonus (start 0, try 0.01 if stuck)
    vf_coef        = 0.5,       # value function loss weight
    max_grad_norm  = 0.5,       # gradient clipping
    verbose        = 1,
)
```

### PPO vs SAC for your throw env
```
SAC:
  + Better sample efficiency (learns faster per env step)
  + Off-policy: reuses old experience
  - More hyperparameters
  - Can be unstable with contact-rich physics

PPO:
  + More stable, easier to debug
  + Better with parallel envs (designed for it)
  - Needs 5-10x more env steps to converge
  - On-policy: discards old data

Recommendation: Start with SAC. Switch to PPO if SAC diverges
after tuning LR and gamma.
```

---

## Parallel Environments — Major Speedup

```python
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Method 1: make_vec_env (easiest)
env = make_vec_env("HalfCheetah-v5", n_envs=4)

# Method 2: SubprocVecEnv with custom env (for your ThrowEnv)
def make_env():
    return ThrowEnv()

env = SubprocVecEnv([make_env for _ in range(4)])

# Always use SubprocVecEnv for MuJoCo (true multiprocessing)
# DummyVecEnv runs envs sequentially — no speedup, just batching
```

**Speedup on your laptop:** 4 envs → ~3x faster wall-clock training.

---

## Callbacks — Essential for Long Runs

```python
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
    CallbackList
)

# Auto-save best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10_000,       # evaluate every 10k steps
    n_eval_episodes=10,     # average over 10 episodes
    deterministic=True,
)

# Save checkpoint every 50k steps
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="sac_throw"
)

# Stop when reward threshold hit
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=-200,  # set your target
    verbose=1
)

# Combine
callbacks = CallbackList([eval_callback, checkpoint_callback])

model.learn(total_timesteps=1_000_000, callback=callbacks)
```

---

## Save, Load, Resume

```python
# Save
model.save("sac_throw_500k")

# Load for inference
model = SAC.load("sac_throw_500k", env=env)

# Resume training (load with replay buffer)
model = SAC.load("sac_throw_500k", env=env)
model.load_replay_buffer("sac_throw_500k_replay_buffer")
model.learn(total_timesteps=500_000, reset_num_timesteps=False)

# Save replay buffer (large file — only for resuming)
model.save_replay_buffer("sac_throw_500k_replay_buffer")
```

---

## Evaluate a Trained Model

```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=20,
    deterministic=True,    # no exploration noise during eval
)
print(f"Mean: {mean_reward:.1f} ± {std_reward:.1f}")
```

---

## TensorBoard

```powershell
# In a separate terminal
tensorboard --logdir ./tb_logs/
# Open http://localhost:6006
```

Key metrics to watch:
```
rollout/ep_rew_mean  → your main signal. Should trend up.
train/critic_loss    → should decrease and stabilize
train/actor_loss     → less meaningful, can fluctuate
train/ent_coef       → should decrease gradually (0.8 → 0.1)
time/fps             → if drops suddenly, check memory
```

---

## Hyperparameter Tuning with A

```python
from ax.service.ax_client import AxClient, ObjectiveProperties

ax = AxClient()
ax.create_experiment(
    name="throw_sac_tuning",
    parameters=[
        {"name": "lr",         "type": "range",  "bounds": [1e-5, 1e-3], "log_scale": True},
        {"name": "gamma",      "type": "range",  "bounds": [0.98, 0.999]},
        {"name": "batch_size", "type": "choice", "values": [128, 256, 512]},
        {"name": "net_arch",   "type": "choice", "values": ["small", "medium", "large"]},
    ],
    objectives={"mean_reward": ObjectiveProperties(minimize=False)},
)

def train_and_eval(lr, gamma, batch_size, net_arch):
    arch_map = {"small": [64,64], "medium": [256,256], "large": [512,512]}
    env = make_your_env()
    model = SAC(
        "MlpPolicy", env,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        policy_kwargs=dict(net_arch=arch_map[net_arch]),
        verbose=0,
    )
    model.learn(total_timesteps=100_000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    env.close()
    return mean_reward

for _ in range(20):  # 20 Bayesian trials
    params, trial_idx = ax.get_next_trial()
    reward = train_and_eval(**params)
    ax.complete_trial(trial_idx, raw_data={"mean_reward": reward})

best_params, values, _ = ax.get_best_parameters()
print("Best:", best_params)
print("Best reward:", values)
```

---

## Quick Reference — When Things Go Wrong

| Symptom | Likely Cause | Fix |
|---|---|---|
| `critic_loss` > 1000 | LR too high | Divide LR by 10 |
| `ep_rew_mean` flat after 200k steps | Bad reward shaping | Add intermediate rewards |
| NaN in losses | LR too high or reward not clipped | LR ÷10, clip reward to [-10, 10] |
| `ent_coef` → 0 immediately | Reward too easy, policy collapses | Add reward noise or ent_coef=0.1 |
| FPS < 5 | Network too large or n_envs too high | Reduce net_arch or n_envs |
| Episode always hits max length | Agent not learning to terminate | Check termination condition in env |
| actor_loss exploding | Critic giving bad gradients | Reduce gradient_steps, increase batch_size |

---

## Recommended Config for Your Throw Env

```python
model = SAC(
    "MlpPolicy",
    env,
    learning_rate    = 1e-4,           # conservative for contact physics
    buffer_size      = 500_000,
    learning_starts  = 10_000,
    batch_size       = 256,
    tau              = 0.005,
    gamma            = 0.995,          # delayed throw reward needs high gamma
    train_freq       = 1,
    gradient_steps   = 1,
    ent_coef         = "auto",
    policy_kwargs    = dict(net_arch=[256, 256]),
    verbose          = 1,
    tensorboard_log  = "./tb_logs/throw/",
    seed             = 42,
    device           = "cpu",
)
```
