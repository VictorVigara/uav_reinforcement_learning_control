# UAV Reinforcement Learning Control

Quadrotor hovering control using **Proximal Policy Optimization (PPO)** with **MuJoCo** physics simulation. A neural network policy learns to fly a drone to randomized target positions from randomized initial states.

## Repository Structure

```
.
├── train.py                  # Training script (PPO via Stable Baselines3)
├── evaluate.py               # Evaluation, visualization, and plotting
├── optimize.py               # Optuna hyperparameter optimization for PPO
├── debug_training.py         # Diagnostic tool for early episode termination
├── pid_controller.py            # Cascaded PID baseline controller
├── envs/
│   ├── __init__.py
│   ├── hover_env.py          # Gymnasium environment (HoverEnv)
│   ├── wrappers.py           # Observation wrappers + registry
│   └── rate_wrapper.py       # CTBR action wrapper (body rates → torques)
├── utils/
│   ├── __init__.py
│   ├── drone_config.py       # Central drone physical parameters (single source of truth)
│   ├── state.py              # QuadState — 12D state with quaternion↔Euler conversion
│   └── normalization.py      # Observation/action normalization utilities
├── model/
│   └── drone/
│       ├── drone.xml         # MuJoCo drone model definition
│       ├── scene.xml         # Scene configuration
│       └── assets/           # STL mesh files
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

## Drone Configuration

All physical drone parameters are centralized in `utils/drone_config.py` — the single source of truth for the entire codebase. Every module (`hover_env.py`, `rate_wrapper.py`, `pid_controller.py`, `evaluate.py`) imports from here instead of hardcoding values.

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
| Max motor thrust | `MAX_MOTOR_THRUST` | 1.372 | N |
| Arm length | `ARM_LENGTH` | 0.039799 | m |
| Yaw torque coefficient | `YAW_TORQUE_COEFF` | 0.0201 | — |
| Mass | `MASS` | 0.3446 | kg |
| Timestep | `DT` | 0.01 | s |
| Roll inertia | `IXX` | 6.44e-4 | kg·m² |
| Pitch inertia | `IYY` | 6.54e-4 | kg·m² |
| Yaw inertia | `IZZ` | 8.31e-4 | kg·m² |

Derived constants (`MAX_TOTAL_THRUST`, `MAX_TORQUE`, `HOVER_THRUST_PER_MOTOR`) are computed automatically from the base parameters.

To adapt to a different drone, update the base parameters in `drone_config.py` and keep the MuJoCo XML (`model/drone/drone.xml`) in sync — specifically `ctrlrange` and the keyframe hover ctrl values.

## Installation

```bash
pip install gymnasium stable-baselines3 mujoco numpy scipy matplotlib
```

## Training

```bash
python train.py
```

This trains a PPO policy with:

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
