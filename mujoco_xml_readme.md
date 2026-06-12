# MuJoCo XML — Complete Parameter Reference
> Based on HalfCheetah XML. Every tag and attribute you'll use when building your throw env.

---

## Top-Level Structure

```xml
<mujoco model="cheetah">        <!-- model name, arbitrary -->
  <compiler .../>               <!-- global compile settings -->
  <default .../>                <!-- default values inherited by all elements -->
  <size .../>                   <!-- memory allocation hints -->
  <option .../>                 <!-- physics simulation settings -->
  <asset .../>                  <!-- textures, materials, meshes -->
  <worldbody .../>              <!-- the actual scene: bodies, geoms, joints -->
  <actuator .../>               <!-- motors, servos — what your action vector controls -->
</mujoco>
```

---

## `<compiler>`

```xml
<compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
```

| Attribute | Value | Meaning |
|---|---|---|
| `angle` | `radian` / `degree` | Unit for all angles in this file. Use `radian`. |
| `coordinate` | `local` / `global` | Whether body positions are relative to parent or world. `local` is standard. |
| `inertiafromgeom` | `true` / `false` | Auto-compute inertia from geometry shape + mass. Set `true` unless you know exact inertia tensors. |
| `settotalmass` | float | Scales all body masses so total = this value (kg). Overrides individual mass values. |

---

## `<option>`

```xml
<option gravity="0 0 -9.81" timestep="0.01"/>
```

| Attribute | Value | Meaning |
|---|---|---|
| `gravity` | `x y z` | Gravity vector in m/s². Default earth = `0 0 -9.81`. |
| `timestep` | float (s) | Physics step size. Smaller = more accurate but slower. `0.01` = 10ms per step. For a throw env use `0.002`–`0.005`. |
| `integrator` | `Euler` / `RK4` | ODE solver. `RK4` is more accurate for fast-moving objects (ball). |

> **For your throw env:** Use `timestep="0.002"` — ball contact physics needs smaller steps or it tunnels through surfaces.

---

## `<default>`

```xml
<default>
  <joint armature=".1" damping=".01" limited="true" .../>
  <geom friction=".4 .1 .1" .../>
  <motor ctrllimited="true" ctrlrange="-1 1"/>
</default>
```

Anything set here is inherited by **all** joints/geoms/motors unless overridden locally. Think of it as a CSS class for physics.

### Joint defaults
| Attribute | Meaning |
|---|---|
| `armature` | Rotational inertia added to joint (kg·m²). Prevents jittery control. Start with `0.1`. |
| `damping` | Velocity-proportional resistance. Higher = more friction/sluggish. |
| `limited` | `true` = joint has range limits (set via `range` on the joint itself). |
| `stiffness` | Spring force pulling joint back to zero. 0 = free joint. |

### Geom defaults
| Attribute | Meaning |
|---|---|
| `friction` | Three values: `sliding torsional rolling`. Most important is sliding (first). `0.4` is rubber-on-floor. |
| `contype` / `conaffinity` | Collision group bitmasks. Two geoms collide only if `contype` of one ANDs with `conaffinity ` of other. Use `1` for both to collide with everything. `Collide = (contypeA & conaffinityB) or (contypeB & conaffinityA) `|
| `solimp` / `solref` | Constraint solver parameters. `solref="0.02 1"` = 20ms time constant. Controls how "hard" contacts are. Rarely need to change. |

### Motor defaults
| Attribute | Meaning |
|---|---|
| `ctrllimited` | `true` = clamp action to `ctrlrange`. Always use `true`. |
| `ctrlrange` | `min max` torque range. `-1 1` means your action space is [-1, 1]. |

---

## `<worldbody>`

The scene graph. Everything is a tree of `<body>` elements.

### `<body>`

```xml
<body name="torso" pos="0 0 .7">
```

| Attribute | Meaning |
|---|---|
| `name` | Identifier — used to get position/velocity in Python via `data.body("torso").xpos` |
| `pos` | `x y z` position **relative to parent body** (because `coordinate="local"`). |
| `quat` / `euler` | Orientation of this body relative to parent. |

Bodies are containers. They hold geoms (shapes), joints (DOF), and child bodies.

---

### `<joint>`

```xml
<joint name="bthigh" type="hinge" axis="0 1 0" range="-.52 1.05" damping="6" stiffness="240"/>
```

Defines a **degree of freedom** between this body and its parent.

| Attribute | Options / Meaning |
|---|---|
| `type` | `hinge` = rotation around axis (1 DOF). `slide` = translation along axis (1 DOF). `ball` = 3-DOF rotation. `free` = 6-DOF (position + orientation — use for ball/projectile). |
| `axis` | Direction of rotation/translation in local frame. `0 1 0` = rotate around Y axis. |
| `range` | `min max` in radians (for hinge) or meters (for slide). Only active if `limited="true"`. |
| `damping` | Velocity damping at this joint specifically (overrides default). |
| `stiffness` | Spring pulling back to zero. For a free-swinging arm, use `0`. |
| `armature` | Extra inertia. Helps stabilize training. |
| `pos` | Where the joint pivot is located within the body. |

> **For your throw env:**
> - Arm shoulder: `type="hinge" axis="0 1 0"` (rotate in XZ plane)
> - Ball: `type="free"` (full 6-DOF flying body)

---

### `<geom>`

```xml
<geom type="capsule" size="0.046 .145" pos=".1 0 -.13" axisangle="0 1 0 -3.8" name="bthigh"/>
```

The actual collision/visual shape attached to a body.

| Attribute | Meaning |
|---|---|
| `type` | `capsule`, `sphere`, `box`, `cylinder`, `plane`, `mesh` |
| `size` | Depends on type. Capsule: `radius length`. Sphere: `radius`. Box: `x y z` half-sizes. Cylinder: `radius height`. |
| `pos` | Position within the body (local frame). |
| `axisangle` | `ax ay az angle` — orientation of geom within body. |
| `fromto` | Alternative to pos+size for capsule: `x1 y1 z1 x2 y2 z2` (start and end points). |
| `mass` | Mass of this geom (kg). If `inertiafromgeom="true"`, inertia is computed from this. |
| `rgba` | Color: `r g b alpha` all in [0,1]. |
| `contype` / `conaffinity` | Collision groups. Set both to `0` for visual-only geoms (no collision). |
| `friction` | Override default friction for this specific geom. |

**Types cheat sheet:**
```
sphere:   size="0.1"              → radius 0.1m
capsule:  size="0.05 0.3"         → radius 0.05m, half-length 0.3m
box:      size="0.1 0.2 0.1"      → 20cm × 40cm × 20cm box
cylinder: size="0.23 0.02"        → radius 0.23m, half-height 0.02m (hoop rim)
plane:    size="10 10 0.1"        → infinite ground plane
```

---

### `<camera>`

```xml
<camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
```

| Attribute | Meaning |
|---|---|
| `mode` | `fixed` = static. `trackcom` = follows center of mass of parent body. `targetbody` = points at a body. |
| `pos` | Camera position relative to parent. |
| `xyaxes` | Defines camera orientation via two vectors (right, up). |

---

## `<actuator>`

```xml
<actuator>
  <motor gear="120" joint="bthigh" name="bthigh"/>
</actuator>
```

Maps your **action vector** to actual forces/torques in the sim.

| Attribute | Meaning |
|---|---|
| `joint` | Which joint this motor drives. |
| `gear` | Torque multiplier. Action value × gear = actual torque (N·m). `gear="120"` means action=1.0 → 120 N·m. |
| `name` | Used to identify in Python. |

Other actuator types:
```xml
<position joint="shoulder" kp="100"/>   <!-- position servo, kp = stiffness -->
<velocity joint="shoulder" kv="10"/>    <!-- velocity servo -->
<motor joint="shoulder" gear="50"/>     <!-- raw torque (what you'll use) -->
```

> **For your throw env:** Use `<motor>` — raw torque gives the policy full control. Position servos hide the dynamics.

---

## Accessing Data in Python

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("your_model.xml")
data  = mujoco.MjData(model)

mujoco.mj_step(model, data)

# Body state
data.body("torso").xpos        # world position [x, y, z]
data.body("torso").xquat       # world orientation quaternion

# Joint state
data.joint("shoulder").qpos    # angle (rad) or position (m)
data.joint("shoulder").qvel    # angular/linear velocity

# All joint positions and velocities (your observation)
data.qpos   # shape: (nq,) — all generalized positions
data.qvel   # shape: (nv,) — all generalized velocities

# Apply action
data.ctrl[0] = 0.5  # set first actuator to 0.5 (within ctrlrange)

# Sensor / geom position
data.geom("ball").xpos         # world position of a geom
```

---

## Your Throw Env XML Skeleton

```xml
<mujoco model="throw_env">
  <compiler angle="radian" inertiafromgeom="true"/>
  
  <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>
  
  <default>
    <joint armature="0.1" damping="0.5" limited="true"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
    <geom friction="0.8 0.1 0.01" rgba="0.7 0.7 0.7 1"/>
  </default>

  <worldbody>
    <!-- Ground -->
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"
          contype="1" conaffinity="1"/>

    <!-- Arm -->
    <body name="base" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0"
             range="-3.14 3.14" damping="1.0"/>
      <geom name="arm" type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"
            rgba="0.2 0.6 1.0 1"/>
      
      <!-- Ball attached at tip -->
      <body name="ball" pos="0.4 0 0">
        <joint name="ball_joint" type="free"/>   <!-- detach on release -->
        <geom name="ball_geom" type="sphere" size="0.12"
              mass="0.62" rgba="1.0 0.5 0.0 1"
              friction="0.8 0.1 0.01"/>
      </body>
    </body>

    <!-- Hoop (target) -->
    <body name="hoop" pos="3 0 1.5">
      <geom name="hoop_rim" type="cylinder" size="0.23 0.02"
            rgba="1.0 0.4 0.0 1" contype="0" conaffinity="0"/>
      <!-- contype=0: visual only, ball passes through -->
    </body>
  </worldbody>

  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="80"/>
  </actuator>
</mujoco>
```

---

## Quick Reference Card

```
timestep        → smaller = more accurate, slower. Use 0.002 for throw tasks.
gear            → scales your [-1,1] action to real torque. Start at 50-100.
damping         → joint friction. Too low = oscillation. Too high = sluggish.
armature        → inertia stabilizer. Always use 0.05-0.2.
friction        → "sliding torsional rolling". First value matters most.
contype=0       → disable collision (visual only geom)
type="free"     → 6-DOF body (ball, projectile)
type="hinge"    → 1-DOF rotation (arm joint)
data.qpos       → all joint angles — your observation source
data.ctrl       → your action vector writes here
```
