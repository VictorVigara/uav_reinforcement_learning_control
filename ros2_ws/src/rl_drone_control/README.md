# rl_drone_control

ROS2 node for deploying a trained PPO policy (CTBR: thrust + body rates) on a quadrotor with Betaflight.

## Architecture

```
Mocap (/mocap/pose) ──────┐
Attitude (/attitude) ─────┤
IMU (/imu) ───────────────┼──► rl_policy_node ──► /rl_control/cmd (Twist)
Target (/target_pose) ────┘
```

The node subscribes to sensor topics, estimates linear velocity from mocap position via filtered differentiation, builds a 12D normalized observation matching the training environment, runs the PPO policy, and publishes control commands as a Twist message.

## Topics

### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/mocap/pose` | `geometry_msgs/PoseStamped` | Drone position (x, y, z) from motion capture |
| `/attitude` | `geometry_msgs/Vector3Stamped` | Roll, pitch, yaw (rad) as x, y, z |
| `/imu` | `sensor_msgs/Imu` | Angular velocity from gyroscope (body frame) |
| `/target_pose` | `geometry_msgs/PoseStamped` | Dynamic target position (optional override) |

### Publications

| Topic | Type | Description |
|-------|------|-------------|
| `/rl_control/cmd` | `geometry_msgs/Twist` | Policy output (normalized [-1, 1]) |

### Twist Mapping

| Field | Meaning | Range |
|-------|---------|-------|
| `linear.x` | Thrust | [-1, 1] |
| `angular.x` | Roll rate | [-1, 1] |
| `angular.y` | Pitch rate | [-1, 1] |
| `angular.z` | Yaw rate | [-1, 1] |

To convert to Betaflight RC commands: `rc = 1500 + action * 500` (maps to [1000, 2000]).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | (required) | Path to trained `.zip` model |
| `control_rate` | 100.0 | Policy inference rate (Hz) |
| `target_x` | 0.0 | Default target X (m) |
| `target_y` | 0.0 | Default target Y (m) |
| `target_z` | 1.0 | Default target Z (m) |
| `velocity_alpha` | 0.8 | Low-pass filter coefficient for velocity estimation |
| `pose_timeout` | 0.1 | Max age of sensor data before safety cutoff (s) |

## Usage

### Build

```bash
cd ros2_ws
colcon build --packages-select rl_drone_control
source install/setup.bash
```

### Run with launch file

```bash
ros2 launch rl_drone_control policy_launch.py \
    model_path:=/path/to/best_model.zip \
    target_x:=0.0 target_y:=0.0 target_z:=1.0
```

### Run directly

```bash
ros2 run rl_drone_control policy_node \
    --ros-args -p model_path:=/path/to/best_model.zip
```

### Remap topics

```bash
ros2 run rl_drone_control policy_node \
    --ros-args \
    -p model_path:=/path/to/best_model.zip \
    -r /mocap/pose:=/vicon/drone/pose \
    -r /imu:=/drone/imu/data
```

## Observation Space

The policy expects a 12D normalized observation in [-1, 1]:

| Index | Component | Source | Physical Bounds |
|-------|-----------|--------|-----------------|
| 0-2 | Relative position (target - drone) | mocap + target | [-4, 4] m (xy), [-2, 2] m (z) |
| 3-5 | Attitude (roll, pitch, yaw) | /attitude topic | [-pi, pi] rad |
| 6-8 | Linear velocity | Estimated (filtered diff) | [-10, 10] m/s |
| 9-11 | Angular velocity | IMU gyroscope | [-6pi, 6pi] rad/s |

## Velocity Estimation

Linear velocity is estimated by differentiating mocap position with a first-order low-pass filter:

```
v_raw = (pos_current - pos_previous) / dt
v_filtered = alpha * v_previous + (1 - alpha) * v_raw
```

This is suitable for motion capture systems (100+ Hz, sub-mm accuracy). The `velocity_alpha` parameter controls smoothing (higher = smoother but more lag).

## Safety

- Publishes minimum thrust if any sensor data is older than `pose_timeout`
- Does not publish until all sensors (pose, attitude, IMU) have been received at least once
- Logs diagnostics at 1 Hz (position error, velocity estimate, actions)

## Dependencies

- ROS2 (Humble/Iron/Jazzy)
- `stable-baselines3` (for PPO model loading)
- `numpy`, `scipy`

## Notes

- The `RateControlWrapper`'s PD controller is NOT used here. Betaflight handles rate-to-motor conversion internally.
- The policy was trained at 100 Hz. Match this rate on the real system for best performance.
- Verify that the attitude convention from your `/attitude` topic matches training (intrinsic XYZ Euler: roll=X, pitch=Y, yaw=Z).
