"""Build and normalize the 12D observation vector for the RL policy.

Observation layout (matches HoverEnv._get_obs()):
    [0:3]  relative position  = target_pos - drone_pos
    [3:6]  attitude           = [roll, pitch, yaw] in radians
    [6:9]  linear velocity    = [vx, vy, vz] in m/s
    [9:12] angular velocity   = [wx, wy, wz] in rad/s (body frame)

All values are normalized to [-1, 1] using the same bounds as training.
"""

import numpy as np

# Training observation bounds — copied from envs/hover_env.py lines 33-36.
OBS_BOUNDS_LOW = np.array(
    [-4, -4, -2, -np.pi, -np.pi, -np.pi, -10, -10, -10,
     -6 * np.pi, -6 * np.pi, -6 * np.pi],
    dtype=np.float32,
)
OBS_BOUNDS_HIGH = np.array(
    [4, 4, 2, np.pi, np.pi, np.pi, 10, 10, 10,
     6 * np.pi, 6 * np.pi, 6 * np.pi],
    dtype=np.float32,
)


def normalize(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Normalize value from [low, high] to [-1, 1]."""
    return 2.0 * (value - low) / (high - low) - 1.0


def build_observation(
    drone_pos: np.ndarray,
    target_pos: np.ndarray,
    attitude: np.ndarray,
    linear_vel: np.ndarray,
    angular_vel: np.ndarray,
) -> np.ndarray:
    """Build the 12D normalized observation for the policy.

    Args:
        drone_pos: Drone position [x, y, z] in meters.
        target_pos: Target position [x, y, z] in meters.
        attitude: Euler angles [roll, pitch, yaw] in radians.
        linear_vel: Linear velocity [vx, vy, vz] in m/s.
        angular_vel: Angular velocity [wx, wy, wz] in rad/s (body frame).

    Returns:
        12D normalized observation in [-1, 1].
    """
    obs = np.zeros(12, dtype=np.float32)
    obs[0:3] = target_pos - drone_pos
    obs[3:6] = attitude
    obs[6:9] = linear_vel
    obs[9:12] = angular_vel

    obs_normalized = normalize(obs, OBS_BOUNDS_LOW, OBS_BOUNDS_HIGH)
    return np.clip(obs_normalized, -1.0, 1.0).astype(np.float32)
