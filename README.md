# UAV Reinforcement Learning Control

Quadrotor hovering control using **Proximal Policy Optimization (PPO)** with **MuJoCo** physics simulation. A neural network policy learns to fly a drone to randomized target positions from randomized initial states.

## Repository Structure

```
.
├── train.py                  # Training script (PPO via Stable Baselines3)
├── evaluate.py               # Evaluation, visualization, and plotting
├── evaluate_althold.py       # Altitude-hold policy vs a PI baseline, with plots
├── export_policy.py          # Altitude-hold policy -> numpy bundle for skybridge
├── rc_sim.py                 # Fly the simulated Air65 with the real radio (optionally with altitude hold)
├── optimize.py               # Optuna hyperparameter optimization for PPO
├── debug_training.py         # Diagnostic tool for early episode termination
├── pid_controller.py            # Cascaded PID baseline controller
├── drones/                   # Drone registry: one folder per drone with its params.yaml
│   ├── __init__.py           # DroneSpec, load_drone()
│   ├── constraints.py        # Command limits read from skybridge's config
│   ├── small/                # The custom 150 g drone (default)
│   └── air65/                # BETAFPV Air65 II
├── envs/
│   ├── __init__.py           # make_env(drone): picks the environment for a drone
│   ├── hover_env.py          # HoverEnv: thrust + torque commands
│   ├── angle_env.py          # AngleModeEnv: Betaflight angle-mode commands (Air65)
│   ├── althold_env.py        # AltHoldEnv: pilot flies roll/pitch/yaw, the policy throttle (Air65)
│   ├── betaflight.py         # Simulated Betaflight angle-mode flight controller
│   ├── sim2real.py           # Delays, sensor rates, noise, randomization helpers
│   ├── wrappers.py           # Observation wrappers + registry
│   └── rate_wrapper.py       # CTBR action wrapper (body rates → torques)
├── sysid/                    # System identification from flight and blackbox logs
├── utils/
│   ├── __init__.py
│   ├── drone_config.py       # The default drone's parameters as constants
│   ├── state.py              # QuadState — 12D state with quaternion↔Euler conversion
│   └── normalization.py      # Observation/action normalization utilities
├── model/
│   ├── drone/                # 150 g drone: drone.xml, scene.xml, STL meshes
│   └── air65/                # Air65: drone.xml, scene.xml
├── ros2_ws/                  # ROS2 workspace for real-world deployment
│   └── src/rl_drone_control/ # Policy node, velocity estimator (see its README)
├── models_trained/           # Saved model checkpoints (per training run)
├── logs/                     # TensorBoard training logs
└── plots/                    # Generated evaluation plots
```

## Environment

**`HoverEnv`** defines the quadrotor hovering task:

| Property | Details |
|---|---|
| **Observation** | 12D normalized vector: relative position, attitude (roll/pitch/yaw), linear velocity, angular velocity |
| **Action** | 4D normalized vector: total thrust + 3-axis torques, mapped to 4 motor forces via a mixer matrix |
| **Reward** | `exp(-||position_error||²)` — Gaussian reward peaking at 1.0 when on target |
| **Episode length** | 512 steps (5.12 seconds at 100 Hz) |
| **Termination** | State out of bounds or NaN detected |
| **Randomization** | Both initial drone state and target position are randomized each episode |

The drone has 4 motors each producing 0–13 N of thrust. Motor commands are computed from thrust/torque demands using an inverse mixer matrix.

## Drones

The repo can train for more than one drone. Each drone is a folder in `drones/` with a `params.yaml`, plus a MuJoCo model under `model/`:

| Drone | `--drone` | Commanded as | Environment | Policy rate |
|---|---|---|---|---|
| Custom 150 g drone (default) | `small` | total thrust + torques, or body rates through `RateControlWrapper` | `HoverEnv` | 100 Hz |
| BETAFPV Air65 II | `air65` | Betaflight angle mode: throttle, roll, pitch, yaw rate | `AngleModeEnv` | 50 Hz |

```python
from drones import load_drone
from envs import make_env

spec = load_drone("air65")      # DroneSpec: mass, inertia, thrust, timestep, ...
env = make_env("air65")         # the environment that matches how the drone is commanded
```

`train.py`, `evaluate.py` and `optimize.py` take the drone from `--drone` (training) or from the run's `config.json` (evaluation). Without the flag everything behaves as before, for the 150 g drone.

`utils/drone_config.py` still exposes the default drone's parameters as constants (`MASS`, `MAX_MOTOR_THRUST`, ...) for `pid_controller.py` and `rate_wrapper.py`. For the 150 g drone, keep `model/drone/drone.xml` in sync with `drones/small/params.yaml` by hand (`ctrlrange` and the keyframe hover ctrl).

### Adding a drone

1. `drones/<name>/params.yaml`: copy an existing one. `interface` says how the drone is commanded and picks the environment (`ENV_FOR_INTERFACE` in `envs/__init__.py`).
2. `model/<name>/drone.xml`: the MuJoCo model.
3. Add the drone's default wrapper (or `None`) to `DEFAULT_WRAPPER` in `envs/__init__.py`.
4. A new way of commanding a drone needs a new environment class registered in `ENV_FOR_INTERFACE`.

## BETAFPV Air65 II

The Air65 is flown from a laptop through skybridge: CRSF commands at 50 Hz into Betaflight angle mode, position and velocity from a depth camera at 30 Hz, attitude from telemetry at about 17 Hz. `AngleModeEnv` (`envs/angle_env.py`) simulates that whole chain, so a policy trained in it sees what the laptop sees.

```bash
python train.py --drone air65
python evaluate.py --model ./models_trained/air65_<stamp>/best_model.zip --plot
```

**Action** (4, normalized to [-1, 1]), in skybridge's units and signs:

| | Range | Sign |
|---|---|---|
| throttle | 0.05 to 0.56 of the 1000 to 2000 us span (hover is about 0.305) | |
| roll angle | ± `max_angle` | + tilts right |
| pitch angle | ± `max_angle` | + tilts the nose down, flies forward |
| yaw rate | ± 120 deg/s | + turns right |

**Observation** (12 + 4 × `action_history`, 52 by default), all measurable on the laptop:

| | Scale | Source |
|---|---|---|
| target minus position: forward, left, up | 2 m | camera, rotated by the heading |
| velocity: forward, left, up | 3 m/s | camera tracker |
| roll, pitch (same signs as the action) | 60 deg | telemetry. Telemetry pitch has the opposite sign |
| target heading minus heading (+ left) | 180 deg | skybridge heading estimate |
| integral of the position error, clipped, rotated by the heading (reset to 0 when the policy takes over) | 1 m·s | computed on the laptop, once per step: summed in the world frame, an axis only while its error is under `int_zone` (0.3 m) |
| last `action_history` actions, newest first (10 = 200 ms) | | the commands the laptop sent |

There is no angular velocity: the laptop does not receive it. The integral lets the policy cancel what it cannot observe directly: the real drone is never trimmed exactly level (1 to 3.6 degrees in the logs) and its hover throttle moves with the battery.

The command history is what lets the policy cope with the delays (30-70 ms to the motors, attitude 100-200 ms old). With only the last command it cannot tell which commands are still on their way: the first Air65 policy (`air65_20261115_175353`, 16D observation) oscillated by about 10 cm around the target, at 3 cm with the delays removed. Runs record their drone parameters in `config.json`, and `evaluate.py` rebuilds the environment from them, so older runs still load.

**Reward**: `alive + exp(-(e/0.5)²) + exp(-(e/0.1)²)` on the true position error `e`, minus costs on the change of the action, the squared velocity and the heading error (weights in the `reward` section of `params.yaml`). The broad Gaussian finds the target, the narrow one makes holding it within a few centimetres worth something, and the velocity and action-change costs make oscillating around it cost reward.

**Command limits come from skybridge.** `drones/constraints.py` reads skybridge's `configs/drone.yaml` (`$SKYBRIDGE_CONFIG`, default `~/skybridge/configs/drone.yaml`). The policy's angle range is `min(betaflight.angle_limit, safety.max_angle)`. After raising the limit in Betaflight and in that YAML, retrain: the action scale changes with it. Every run stores the limits it used in `config.json` under `command_limits`. On a machine without skybridge the committed snapshot `drones/air65/skybridge_drone.yaml` is used, with a warning; refresh it with `python -m drones.constraints --sync air65`.

**What is simulated** (numbers and their sources are in `drones/air65/params.yaml`):

- Betaflight (`envs/betaflight.py`): RC smoothing, angle loop, rate loop, mixer with airmode, motor lag, and the throttle-to-thrust curve, at 250 Hz.
- The link (`envs/sim2real.py`): command delay, camera rate, latency and dropped frames, the tracker's velocity lag, slow telemetry with its own latency, sensor noise.
- Per-episode variation: mass, battery (thrust scale), drag, motor lag, delays, a level-trim error of up to 3 degrees, a heading error, and slow air disturbances.

### Altitude hold: the pilot flies, the policy holds altitude

The pilot flies roll, pitch and yaw with the radio. The policy only sets throttle, and holds the altitude the camera read when it was engaged. `AltHoldEnv` (`envs/althold_env.py`) is built on `AngleModeEnv`: it uses the same Betaflight model, link and camera, and a simulated pilot flies it. The aim is to go from sim straight to the real drone with no tuning on the real drone. To get there, everything the identification could not pin down is randomized widely, and the policy is only shown what the laptop can measure.

```bash
python train.py --drone air65 --task althold --timesteps 10000000 --n-envs 8
python evaluate_althold.py models_trained/air65_althold_<stamp>      # policy vs PI, plots in plots/
python rc_sim.py --policy models_trained/air65_althold_<stamp>        # fly it with the real radio
```

**Action** (1, normalized to [-1, 1]): throttle over `althold.throttle_range` (0.05 to 0.56). The midpoint 0.305 is the nominal hover.

**Observation** (4 + 4 × `stick_history` + `throttle_history`, 34 by default). It has no attitude telemetry, which is 100-200 ms old and noisy. The pilot's sticks stand in for it: in angle mode a stick asks for an angle, so the sticks say how far the drone is about to tilt and how much thrust will go sideways.

| | Scale | Source |
|---|---|---|
| target altitude minus altitude | 1 m | camera |
| climb rate | 2 m/s | camera tracker |
| integral of the altitude error, clipped (only within `int_zone`, 0.3 m, of the target) | 0.5 m·s | laptop, once per tick |
| age of the newest camera data | 0.5 s | laptop |
| last `stick_history` (5 = 100 ms) stick commands, newest first: roll, pitch (/ `angle_limit`), yaw rate (/ full-stick rate), tilt term `1/(cos r·cos p) − 1` (/ its value at full stick) | | radio |
| last `throttle_history` (10 = 200 ms) throttle commands, newest first | | what the laptop sent |

The integral covers the unknown hover throttle: mass, battery, and a thrust curve that is not quite right. The stick history covers the drone reaching a commanded angle only 100-200 ms later. The camera age tells the policy when its data is stale.

`AltHoldObserver` builds this observation and `AltHoldAssist` runs a policy on it. Neither depends on MuJoCo, so the real drone's code imports them and builds exactly the observation the policy was trained on. `AltHoldAssist` also holds the camera timeout: if the camera data gets older than `timeout` (1 s by default), the throttle goes to 0 and stays there until the policy is engaged again. It does not jump back when the camera returns.

**Reward**: `alive + exp(-(e/0.3)²) + exp(-(e/0.05)²)` on the true altitude error `e`, minus costs on the squared true climb rate and the squared change of the throttle (`althold.reward`). An episode ends at the floor (0.05 m), at the ceiling (2.2 m) or at 75° of tilt. Leaving the area sideways does not end it, because that is the pilot's job.

**Episodes**: 20 s. They start in flight at 0.4-1.4 m, with up to ±0.5 m/s of climb and 10° of tilt. The pilot's throttle at hand-over is ±10 % off hover. In half the episodes the target is offset by up to ±0.3 m, so the policy also learns to come back after a sag.

**The simulated pilot** (`SimPilot`, `althold.pilot`) pulls the drone back towards the centre of the view, reacting 0.15-0.35 s late. On top of that come wandering tilts, sharp punches of up to the full stick held for 0.2-1 s, yaw wandering, and spins of 90-200 deg/s. How much of each is set per episode by an aggressiveness from 0 to 1. In 20 % of episodes the pilot only keeps the drone in view.

**Randomized per episode** (`althold.randomize`, `link`, `noise`, `aero`, which override the drone's own ranges for this task):

| Group | What | Range |
|---|---|---|
| Drone | mass | ×0.85-1.2 |
| | battery thrust, and its sag during the episode | ×0.85-1.15, then 0-8 % lost over 20 s |
| | throttle curve: `thrust_expo`, `thrust_u0`, `max_thrust_ratio` | 1.3-1.8, 0.02-0.08, 2.6-3.8 |
| | motor lag, rising and falling drawn apart | ×0.6-1.5 each |
| | inertia (the FC's tuning stays nominal) | ×0.7-1.5 |
| | drag | ×0.5-2 |
| Air | slowly wandering gusts | 0-0.4 m/s² |
| | translational lift | +0-10 % thrust per m/s of horizontal speed, up to 1.5 m/s |
| | thrust loss when sinking faster than 0.5 m/s | 0-15 % per m/s |
| Response to the sticks | `level_kp`, `angle_ff`, `rc_tau`, `rate_tau` | ×0.7-1.3, 0.4-1.0, ×0.7-1.5, ×0.7-1.5 |
| | real angle per commanded angle | ×0.85-1.15 |
| | level trim error | ±4° |
| | command delay | 20-90 ms |
| Camera and link | altitude / climb-rate noise | 2-10 mm / 8-25 mm/s |
| | latency, dropped frames | 20-70 ms, 0-15 % |
| | outages | 0-0.2 per s, each 0.1-0.5 s long |
| | depth scale error | ±3 % |
| | laptop loop | ±3 ms jitter, 1 % of ticks missed |

Ground effect is not modelled, because hovering stays above 0.4 m. Neither is a thrust mismatch between motors: the simulated rate loop has no I-term, so the mismatch would hold the drone tilted, and the trim error already covers that.

**Curriculum**: early in training every draw is pulled towards nominal, and the pilot is kept calm. `train.py --curriculum START UNTIL` (default `0.2 0.4`) starts at 20 % of the ranges and reaches the full ranges after 40 % of the timesteps. TensorBoard shows the level as `curriculum/difficulty`. Evaluation always runs at the full ranges. `1 0` turns the curriculum off.

**Evaluation**: `evaluate_althold.py` scores the policy and a hand-tuned PI under three conditions: nominal, randomized, and hard (full ranges with the pilot at full aggressiveness). It reports survival, the RMS and 95th percentile of the true altitude error, and throttle chatter. It also plots altitude, throttle, camera age, and the sticks against the true attitude. The PI's scores (30 episodes) are the bar a policy has to clear:

| Condition | Survival | RMS error |
|---|---|---|
| nominal | 100 % | 3.4 cm |
| randomized | 100 % | 12 cm |
| hard | 80 % | 6 cm |

### Deploying the altitude hold: from a trained run to the real drone

The policy goes to the drone unchanged, with no tuning on the real drone. Each step below either catches a problem in simulation, or checks that the laptop feeds the policy exactly what it was trained on. Do them in order.

#### 1. Train

```bash
python train.py --drone air65 --task althold --timesteps 10000000 --n-envs 8
# cluster: DRONE=air65 TASK=althold bsub < train.sh
```

In TensorBoard (`tensorboard --logdir logs/`), `curriculum/difficulty` reaches 1.0 at 40 % of the run. Watch `eval/mean_reward` and `eval/mean_ep_length` after that point: evaluation always runs at the full ranges. A full episode is 1000 steps and scores about 2000 at most.

#### 2. Evaluate in simulation

**a. Score it against the PI baseline:**

```bash
python evaluate_althold.py models_trained/air65_althold_<stamp> --episodes 50 --plot 3
```

| Condition | What it tests |
|---|---|
| `nominal` | the identified drone, sensor noise on |
| `randomized` | the full training ranges: other drones, batteries, delays, camera faults |
| `hard` | full ranges, with the pilot at full aggressiveness every episode |

Only move on if, in **every** condition, the policy:
- survives at least as often as the PI (100 / 100 / 80 %);
- has a lower RMS error than the PI (3.4 / 12 / 6 cm);
- has a throttle chatter (mean |Δthrottle| per tick) of the same order as the PI's. A policy that wins on error by buzzing the throttle will heat the motors, and will look worse on the real drone than in the sim.

**b. Look at the plots** in `plots/althold_<run>/`:
- Altitude should settle on the target without a slow oscillation. A 1-2 s bounce means the delays are not handled.
- Throttle should be smooth, and should step up *before* the drone sinks when a stick punch starts. That shows it uses the stick history.
- Through camera outages (the age trace spikes), throttle should stay roughly steady, not drift or jump.

**c. Fly it with your radio** (Boxer plugged in, "USB Joystick" mode):

```bash
python rc_sim.py --policy models_trained/air65_althold_<stamp>
python rc_sim.py --policy models_trained/air65_althold_<stamp> --randomize   # a random drone each reset (R)
```

1. Arm, take off on your own throttle and hover.
2. Press Enter to engage, or use `--engage-axis N` for a radio switch, high = on. The HUD shows `ALT HOLD <target> (cam <z>, <age> ms)`. Your throttle stick is ignored until you disengage, and disarming also disengages.
3. Fly the way you will on the real drone: slow drifts, sharp punches to full stick, yaw spins, flying to the edge of the room and back. Altitude should hold within a few cm in a hover, and dip by no more than a few cm in a punch.
4. Do it again with `--randomize` a few times (R resets with a new draw), and with `--hover-throttle 0.33` for a sagging battery.

Options: `--camera-timeout` (default 1 s) and `--delay` for a fixed link delay. The thrust curve is `params.yaml`'s, as in training, unless you pass `--soft-cap`.

If anything here looks wrong, fix it in the sim: rewards, ranges, more training. Don't take it to the drone hoping it behaves better there.

**d. Replay it on recorded real flights against the PID** (after exporting, step 3, and enabling it in skybridge, step 4):

```bash
cd ~/skybridge
uv run python scripts/control_replay.py data/live/<run>_flight.rrd \
    --controller altitude_hold --controller altitude_policy --set altitude_policy.on_timeout=pilot
```

This runs both controllers on the same recorded flight: real camera heights, your real sticks, real noise and track dropouts. It prints, for each controller, how its throttle compares with what you flew and with each other, and opens Rerun with height, all throttles and their differences. It is **open loop**: neither output flew the drone, so this shows how each reacts to real data, not whether it would have held altitude. Look for a policy throttle that is smooth and close to the PID's, and doesn't jump when the track drops. Large or jerky differences point at a gap between the sim and the real sensor data.

#### 3. Export

```bash
python export_policy.py models_trained/air65_althold_<stamp> -o ~/skybridge/models/althold_policy.npz
```

This writes a numpy-only bundle: the actor's weights, the observer settings, and up to 900 recorded observer calls from three simulated episodes. Before writing anything, it checks two things:
- the observer this code builds is the one the run was trained with;
- the numpy forward pass matches SB3 (it prints the largest difference, normally around 1e-9).

#### 4. Install in skybridge

1. In `~/skybridge/configs/drone.yaml`, the `altitude_policy` entry is at the end of `control.controllers`, on **CH7 high**. The PID `altitude_hold` is on **CH7 middle**, so flipping up goes off → PID → policy, and flipping down from the policy lands on the PID first.
2. Check that `drone.yaml` still says `angle_limit: 20`, yaw `srate: 67` and `rc_hz: 50`, and that Betaflight agrees (`diff all`).

When `--control` starts, skybridge loads the bundle and refuses to start if:
- it is missing;
- replaying the recorded calls through skybridge's own copy of the observer and network gives anything else. **If you change `AltHoldObserver` in `envs/althold_env.py`, change `skybridge/control/policy.py` the same way, then re-export**;
- the stick scaling or RC rate in `drone.yaml` differs from training.

rc_fly forwards the sticks raw, so full stick is Betaflight's `angle_limit` (20°). That is the scale the policy was trained with, not skybridge's 15° `safety.max_angle`.

#### 5. Bench test, props off

```bash
cd ~/skybridge
uv run python scripts/rc_fly.py --dry-run --perception --control --replay data/realsense/<stem>.rrd
```

This uses a recorded flight instead of the camera, and nothing is transmitted. Arm, put CH7 in the middle (PID), then high (policy), and check:
- the status line shows `ALT-RL <z>><target>m T<throttle>%`, with a throttle near your hover;
- moving the throttle stick more than 30 % of its travel hands back (`throttle stick moved`);
- when the recording loses the drone, `ALT-RL NO CAMERA` appears after 1 s and throttle goes to 0 until CH7 is cycled.

Then do the same live with the camera and the drone on the desk, props off, still with `--dry-run`. Lift the drone by hand while engaged: the throttle should go down. Lower it: the throttle should go up.

#### 6. First flight

```bash
uv run python scripts/rc_fly.py --perception --control --max-throttle 0.6 --record
```

- Someone at the laptop, ready to press Ctrl-C (see skybridge's README: a frozen radio can't be detected yet).
- Fully charged battery. The policy adapts to sag, but the first flight shouldn't test that.
- Hover at about 1 m in the middle of the camera's view, with a `detected` track. Flip CH7 to the middle (PID hold) and let it settle, then flip CH7 high (policy). While the PID holds, the policy runs in **shadow mode**: the status line shows `[shadow ALT-RL ... T42%]`, the throttle it would send, without sending it. With `--record` it is logged next to the PID's. Stay on the PID until that throttle looks close to the PID's and smooth. Keep your hands off the sticks for 10 s, then make small roll and pitch moves, and only then bigger ones.
- Keep the throttle stick near hover the whole time. It is what you get back when you cycle CH7, or when a stick move or height error hands control back.
- **If the camera times out, throttle goes to 0.** Flip CH7 down to the middle at once: the PID takes over if the track is back, otherwise you get your stick. With `on_timeout: pilot` in `drone.yaml` you get the stick back automatically instead.

#### 7. After the flight

`--record` logs the controller's debug values (`plots/control/altitude_policy`: setpoint, z, climb rate, integral, data age, raw action, throttle) next to the sticks and the depth.

Compare them with step 2:
- **Oscillates where the sim didn't:** the delays are longer than trained. Widen `althold.link.action_delay` and `camera_latency`, and retrain.
- **Settles off target for long:** the hover throttle is outside the trained range. Check `mass` and `thrust`.
- **Jumps through outages:** they are longer or more frequent than `outage_rate` and `outage_duration` allow.

Feed what you measure back into `drones/air65/params.yaml` (the `sysid/` scripts help), retrain, and go through the steps again.

### System identification

The model is fitted to recorded data. See [`sysid/README.md`](sysid/README.md) for the pipeline and the current results.

## Installation

```bash
pip install gymnasium stable-baselines3 mujoco numpy scipy matplotlib pyyaml
```

## Training

```bash
python train.py
```

`--drone` picks the drone, and `--task althold` trains the Air65 altitude hold (see above). This trains a PPO policy with:

- **Policy network**: MLP with 2 hidden layers of 128 units (ReLU activation)
- **16 parallel environments** for experience collection
- **Checkpoints** saved every 50k steps to `models_trained/<timestamp>/`
- **TensorBoard logs** written to `logs/<timestamp>/`
- **Best model** tracked via periodic evaluation (5 episodes every 10k steps)
- **config.json** saved with reward function source, observation bounds, wrapper, and all hyperparameters

Key hyperparameters (configured in `train.py`):

| Parameter | Value |
|---|---|
| Total timesteps | 10,000,000 |
| Learning rate | 1.55e-4 |
| Rollout steps | 1024 per env |
| Batch size | 128 |
| PPO epochs | 20 |
| Gamma | 0.9906 |
| GAE lambda | 0.9079 |
| Clip range | 0.1915 |
| Entropy coeff | 9.1e-5 |

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

## Evaluation

```bash
# Evaluate with MuJoCo viewer and save plots
python evaluate.py --model ./models_trained/<run>/best_model.zip --episodes 5 --plot

# Evaluate without rendering
python evaluate.py --model ./models_trained/<run>/best_model.zip --no-render --episodes 10
```

| Flag | Description |
|---|---|
| `--model PATH` | Path to a trained `.zip` model (default: `best_model.zip`) |
| `--episodes N` | Number of evaluation episodes (default: 5) |
| `--no-render` | Disable MuJoCo viewer |
| `--plot` | Save performance plots to the model directory |

Plots show position tracking, position error, attitude, linear/angular velocity, motor commands, and policy actions over time. After evaluation, you are prompted to save a brief description of the policy behavior to `description.txt` in the model directory.

### Trajectory Evaluation

Evaluate the trained policy on a single-lap waypoint trajectory. The drone starts at the first waypoint with zero velocities, advances to the next waypoint when within 0.25 m (configurable), and stops after completing one full lap. Performance plots are always generated.

```bash
# Figure-8 trajectory
python evaluate.py --model ./models_trained/<run>/best_model.zip --trajectory eight

# Circle with tighter waypoint spacing
python evaluate.py --trajectory circle --spacing 0.3

# Square with custom reach threshold
python evaluate.py --trajectory square --spacing 0.4 --reach-radius 0.15
```

| Flag | Description |
|---|---|
| `--trajectory {eight,circle,square}` | Trajectory shape (enables trajectory mode) |
| `--spacing FLOAT` | Distance between waypoints in meters (default: 0.5) |
| `--reach-radius FLOAT` | Distance threshold to switch to next waypoint (default: 0.25) |
| `--max-steps N` | Max simulation steps as safety limit (default: 5000) |

Available trajectories:

| Trajectory | Description | Default parameters |
|---|---|---|
| `eight` | Figure-8 (lemniscate) in XY plane | radius=1.0 m, center=(0,0,1) |
| `circle` | Circle in XY plane | radius=1.0 m, center=(0,0,1) |
| `square` | Square in XY plane | side=1.5 m, center=(0,0,1) |

The viewer shows waypoints as spheres (yellow = current target, gray = upcoming), gray lines connecting the path, and a blue flight trail. The console reports each waypoint reached, and plots are saved automatically to `plots_trajectory/` in the model directory.

### Velocity Estimator Test

Test the velocity estimator used for sim-to-real deployment by comparing ground truth velocity (from MuJoCo `qvel`) against filtered differentiation of position for different low-pass filter alpha values.

```bash
# Default alphas [0.0, 0.3, 0.6, 0.8, 0.95]
python evaluate.py --model ./models_trained/<run>/best_model.zip --test-velocity

# Custom alphas
python evaluate.py --test-velocity 0.0 0.5 0.8 0.95

# Headless, longer episode
python evaluate.py --test-velocity 0.0 0.5 0.8 0.95 --no-render --max-steps 3000
```

| Flag | Description |
|---|---|
| `--test-velocity [ALPHA ...]` | Enable velocity estimator test mode with optional alpha values |
| `--max-steps N` | Maximum simulation steps (default: 5000) |

Generates two plots saved to `plots_velocity_estimator/` in the model directory:
- **`velocity_per_axis.png`** — GT vs estimated velocity for each alpha, per axis (vx, vy, vz)
- **`velocity_error.png`** — Euclidean error over time + RMSE bar chart per alpha

## PID Controller

`pid_controller.py` implements a **cascaded PID** baseline controller with four loops:

```
Position PID  -->  Desired acceleration (world frame)
Acceleration  -->  Desired attitude + thrust (gravity feedforward, tilt compensation)
Attitude P    -->  Desired body rates
Rate PID      -->  Torques (inertia-scaled P + integral in torque space)
```

The rate controller (innermost loop) is also reused by the `RateControlWrapper` to convert RL policy rate commands into torques.

| Parameter | Roll/Pitch | Yaw |
|---|---|---|
| Attitude Kp | 150.0 | 50.0 |
| Attitude Kd | 22.0 | 15.0 |
| Rate Ki (torque space) | 0.02 | 0.02 |
| Inertia | 6.44e-4 / 6.54e-4 kg m^2 | 8.31e-4 kg m^2 |

```bash
# Evaluate PID controller (5 episodes, rendered)
python pid_controller.py

# Headless with plots
python pid_controller.py --no-render --plot --episodes 20
```

## Action Wrapper (CTBR)

The `RateControlWrapper` (`envs/rate_wrapper.py`) changes the action semantics from **direct torques** to **Collective Thrust + Body Rates (CTBR)**, which is the standard interface for sim-to-real transfer via flight controllers like Betaflight in rate mode.

| | Base env (HoverEnv) | With RateControlWrapper |
|---|---|---|
| **Action[0]** | Thrust (normalized) | Thrust (normalized) — passthrough |
| **Action[1:4]** | Torques tau_x, tau_y, tau_z | Body rates: roll, pitch, yaw (deg/s) |
| **Range** | [-1, 1] | [-1, 1] maps to [-360, +360] deg/s |

The wrapper reads the current angular velocity from the simulation state, computes a rate error, and runs a PI controller (matching the PID's inner loop gains) to produce torques. These torques are normalized and passed to the base env, which handles the mixer matrix and physics stepping as usual.

```
Policy --> [thrust, roll_rate, pitch_rate, yaw_rate]  (normalized [-1, 1])
  |
  RateControlWrapper.action()
  |  denormalize rates --> compute rate error --> PI controller --> torques
  |
  HoverEnv.step([thrust, tau_x, tau_y, tau_z])
  |  denormalize --> mixer matrix --> motor forces --> MuJoCo physics
```

### Usage

```python
from envs import HoverEnv, RateControlWrapper

# Rate control only (12D observation)
env = RateControlWrapper(HoverEnv())

# Stacked with observation wrapper (7D observation)
from envs import RelPosActWrapper
env = RelPosActWrapper(RateControlWrapper(HoverEnv()))
```

In `train.py`, set `wrapper_cls = RateControlWrapper`.

### Configuration

All PID gains are configurable via constructor arguments:

```python
env = RateControlWrapper(
    HoverEnv(),
    max_rate=360.0,          # deg/s, symmetric for all axes
    kd=[22.0, 22.0, 15.0],  # P gains (inertia-scaled) [roll, pitch, yaw]
    ki_rate_torque=0.02,     # I gain in torque space
    integral_max=0.008,      # anti-windup clamp (N m)
)
```

## Observation Wrappers

The base `HoverEnv` returns a 12D normalized observation. Observation wrappers (`envs/wrappers.py`) allow experimenting with different observation spaces without modifying the base environment.

### Available wrappers

| Wrapper | Type | Effect |
|---|---|---|
| `None` (no wrapper) | — | 12D obs, direct torque actions |
| `RelPosActWrapper` | Observation | 7D obs: rel_pos + prev_action |
| `RateControlWrapper` | Action | Body rate commands (deg/s) via inner PID |

Observation and action wrappers can be stacked (e.g. `RelPosActWrapper(RateControlWrapper(HoverEnv()))`).

### Setting the wrapper for training

In `train.py`, set the `wrapper_cls` variable:

```python
# Use the 7D wrapper
wrapper_cls = RelPosActWrapper

# Or use no wrapper (original 12D obs)
wrapper_cls = None
```

The wrapper name is saved to `config.json` in the model directory alongside the wrapper source code.

### Automatic wrapper loading in evaluation

`evaluate.py` reads `config.json` from the model directory and automatically applies the correct wrapper. No manual changes needed when evaluating models trained with different wrappers.

### Adding a new wrapper

1. Define your wrapper class (as `gym.ObservationWrapper`, `gym.ActionWrapper`, or `gym.RewardWrapper`)
2. Register it: `WRAPPER_REGISTRY["MyWrapper"] = MyWrapper` in `envs/wrappers.py`
3. Export it in `envs/__init__.py`
4. Set `wrapper_cls = MyWrapper` in `train.py`

Wrappers can access base env attributes via `self.unwrapped` (e.g. `self.unwrapped._prev_action`, `self.unwrapped._state`, `self.unwrapped.model`).

## Model Directory Structure

Each training run saves everything to `models_trained/<timestamp>/`:

```
models_trained/20260212_170614/
├── best_model.zip       # Best policy (from EvalCallback)
├── hover_policy_final.zip
├── config.json          # Hyperparameters, reward function, wrapper, obs bounds
├── description.txt      # Policy behavior notes (written after evaluation)
└── plots/               # Evaluation plots
    ├── episode_1.png
    └── ...
```

## Hyperparameter Optimization

Uses [Optuna](https://optuna.readthedocs.io/) to automatically search for the best PPO hyperparameters.

```bash
pip install optuna
```

Each trial trains a PPO model for `--n-timesteps` steps (~1-2 hours per trial at 500k steps). Bad trials are pruned early.

```bash
# Run 50 trials (let it run overnight)
python optimize.py --n-trials 50

# Persistent storage — can stop and resume across sessions
python optimize.py --storage sqlite:///optuna.db --n-trials 100

# Quick smoke test (2 trials, very short)
python optimize.py --n-trials 2 --n-timesteps 5000 --timeout 120
```

| Flag | Description |
|---|---|
| `--n-trials N` | Maximum number of trials (default: 50) |
| `--timeout S` | Total time limit in seconds (default: no limit) |
| `--n-timesteps N` | Training steps per trial (default: 500,000) |
| `--n-envs N` | Parallel training envs per trial (default: 8) |
| `--n-jobs N` | Parallel Optuna workers (default: 1) |
| `--study-name NAME` | Study identifier (default: `ppo_hover`) |
| `--storage URL` | Optuna DB URL for persistence (e.g. `sqlite:///optuna.db`) |

The script tunes learning rate, rollout length, batch size, PPO epochs, discount factor, GAE lambda, clip range, entropy coefficient, network architecture, and activation function. Bad trials are pruned early via Optuna's `MedianPruner`.

After optimization, the best hyperparameters are printed as a ready-to-paste config for `train.py`. Results are saved to `study_results_ppo_hover.csv`.

### Reading Optuna results

```python
import optuna

study = optuna.load_study(study_name="ppo_hover", storage="sqlite:///optuna_full.db")

# Best trial
print(f"Best value: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")

# All trials as a DataFrame
df = study.trials_dataframe()
print(df.sort_values("value", ascending=False).head(10))
```

For a visual dashboard:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna_full.db
```

## Debugging

```bash
python debug_training.py
```

Runs diagnostic episodes to identify which state dimensions cause early termination, useful for tuning environment bounds and reward shaping.

## ROS2 Deployment

The `ros2_ws/` directory contains a ROS2 package (`rl_drone_control`) for deploying the trained policy on a real quadrotor with Betaflight. It subscribes to mocap, attitude, and IMU topics, estimates linear velocity via filtered position differentiation, builds the normalized observation, runs the PPO policy, and publishes CTBR commands.

See [ros2_ws/src/rl_drone_control/README.md](ros2_ws/src/rl_drone_control/README.md) for full details on topics, parameters, build instructions, and safety behavior.
