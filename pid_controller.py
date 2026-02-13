"""Cascaded PID hover controller for the MuJoCo quadrotor.

Architecture:
    Position PID → desired acceleration (world frame)
    Acceleration → desired attitude + thrust (gravity feedforward + tilt compensation)
    Attitude P → desired body rates
    Rate PID → torques (inertia-scaled, I term in torque space for COM offset compensation)
    Normalize to [-1,1] for env.step()

Usage:
    python pid_controller.py                    # 5 episodes, rendered
    python pid_controller.py --no-render        # headless
    python pid_controller.py --plot             # save performance plots
    python pid_controller.py --episodes 20      # more episodes
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from envs import HoverEnv

# ── Physical constants (from MuJoCo model) ──
MASS = 0.3446       # kg (total drone mass from MuJoCo)
G = 9.81            # m/s²
DT = 0.01           # s (MuJoCo timestep)
MAX_THRUST = 52.0   # N (4 × 13 N max per motor)
MAX_TORQUE = 0.5    # N·m (env action bound)
L = 0.039799        # m (arm length)
IXX = 6.44e-4       # kg·m² (roll inertia from MuJoCo mass matrix)
IYY = 6.54e-4       # kg·m² (pitch inertia)
IZZ = 8.31e-4       # kg·m² (yaw inertia)

# ── Default PID gains (tuned through iterative experiments) ──
DEFAULT_GAINS = {
    "position_xy": {"kp": 2.0, "kd": 3.0, "ki": 0.2},
    "position_z":  {"kp": 4.0, "kd": 5.0, "ki": 0.6},
    "attitude":    {"kp": 150.0, "kd": 22.0},
    "yaw":         {"kp": 50.0, "kd": 15.0},
    "rate":        {"ki_torque": 0.02, "integral_max": 0.008},
    "limits": {
        "axy_max": 3.5,
        "az_min": -4.0,
        "az_max": 10.0,
        "tilt_max": 0.25,
        "z_integral_max": 1.5,
        "xy_integral_max": 0.3,
        "torque_motor_fraction": 0.7,
        "torque_abs_max": 0.05,
        "yaw_torque_scale": 0.5,
    },
}


class CascadedPIDController:
    """Cascaded PID controller: position → attitude → torques.

    Designed for the HoverEnv action space [thrust_norm, tau_x_norm, tau_y_norm, tau_z_norm]
    where each value is in [-1, 1].
    """

    def __init__(self, gains: dict | None = None):
        g = gains or DEFAULT_GAINS
        # Position XY gains
        self.kp_xy = g["position_xy"]["kp"]
        self.kd_xy = g["position_xy"]["kd"]
        self.ki_xy = g["position_xy"]["ki"]
        # Position Z gains
        self.kp_z = g["position_z"]["kp"]
        self.kd_z = g["position_z"]["kd"]
        self.ki_z = g["position_z"]["ki"]
        # Attitude gains
        self.kp_att = g["attitude"]["kp"]
        self.kd_att = g["attitude"]["kd"]
        # Yaw gains
        self.kp_yaw = g["yaw"]["kp"]
        self.kd_yaw = g["yaw"]["kd"]
        # Rate controller gains
        self.ki_rate_torque = g["rate"]["ki_torque"]
        self.rate_int_max = g["rate"]["integral_max"]
        # Limits
        lim = g["limits"]
        self.axy_max = lim["axy_max"]
        self.az_min = lim["az_min"]
        self.az_max = lim["az_max"]
        self.tilt_max = lim["tilt_max"]
        self.z_int_max = lim["z_integral_max"]
        self.xy_int_max = lim["xy_integral_max"]
        self.torque_motor_frac = lim["torque_motor_fraction"]
        self.torque_abs_max = lim["torque_abs_max"]
        self.yaw_torque_scale = lim["yaw_torque_scale"]
        # Integral states
        self.z_integral = 0.0
        self.xy_integral = np.zeros(2)
        self.rate_int_torque = np.zeros(3)  # accumulates in N·m (torque space)

    def reset(self):
        """Reset integral states (call on env.reset())."""
        self.z_integral = 0.0
        self.xy_integral = np.zeros(2)
        self.rate_int_torque = np.zeros(3)

    def compute(self, state: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
        """Compute normalized action from physical state and target position.

        Args:
            state: 12D state [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz]
            target: 3D target position [x,y,z]

        Returns:
            action: 4D normalized action [thrust, tau_x, tau_y, tau_z] in [-1,1]
            diag: Diagnostics dict with desired/actual rates and attitude setpoints
        """
        pos = state[0:3]
        roll, pitch, yaw = state[3:6]
        vel = state[6:9]
        wx, wy, wz = state[9:12]

        # ── 1. Position PID → desired acceleration (world frame) ──
        pos_err = target - pos

        # XY integral (compensates steady-state position bias from COM offset)
        self.xy_integral = np.clip(
            self.xy_integral + self.ki_xy * DT * pos_err[:2],
            -self.xy_int_max, self.xy_int_max,
        )
        ax = self.kp_xy * pos_err[0] - self.kd_xy * vel[0] + self.xy_integral[0]
        ay = self.kp_xy * pos_err[1] - self.kd_xy * vel[1] + self.xy_integral[1]
        az = self.kp_z * pos_err[2] - self.kd_z * vel[2]

        # Z integral (altitude hold)
        self.z_integral = np.clip(
            self.z_integral + self.ki_z * DT * pos_err[2],
            -self.z_int_max, self.z_int_max,
        )
        az += self.z_integral

        # Clamp accelerations
        ax = np.clip(ax, -self.axy_max, self.axy_max)
        ay = np.clip(ay, -self.axy_max, self.axy_max)
        az = np.clip(az, self.az_min, self.az_max)

        # ── 2. Acceleration → thrust + desired attitude ──
        cos_r, cos_p = np.cos(roll), np.cos(pitch)
        tilt = max(cos_r * cos_p, 0.5)  # prevent division blow-up
        thrust = MASS * (G + az) / tilt
        thrust = np.clip(thrust, 0.0, MAX_THRUST)

        # Motor-clipping-aware torque limit:
        # Each motor produces thrust/4. Max torque = motor_force × 2L.
        # Apply safety fraction to stay away from motor saturation.
        thrust_per_motor = thrust / 4.0
        max_tau = min(thrust_per_motor * 2.0 * L * self.torque_motor_frac,
                      self.torque_abs_max)

        # Rotate desired XY acceleration into body frame (yaw rotation)
        cy, sy = np.cos(yaw), np.sin(yaw)
        ax_b = cy * ax + sy * ay
        ay_b = -sy * ax + cy * ay

        # Desired roll/pitch from body-frame acceleration
        # NOTE: roll sign is NEGATED — positive roll → -Y force in XYZ Euler convention
        des_pitch = np.clip(np.arctan2(ax_b, G + az), -self.tilt_max, self.tilt_max)
        des_roll = np.clip(np.arctan2(-ay_b, G + az), -self.tilt_max, self.tilt_max)

        # ── 3. Attitude P → desired rates, Rate PID → torques ──
        # Outer attitude P: desired_rate = (Kp/Kd) * attitude_error
        des_wx = (self.kp_att / self.kd_att) * (des_roll - roll)
        des_wy = (self.kp_att / self.kd_att) * (des_pitch - pitch)
        des_wz = (self.kp_yaw / self.kd_yaw) * (0.0 - yaw)

        # Inner rate PID: P term (inertia-scaled) + I term (in torque space)
        rate_err = np.array([des_wx - wx, des_wy - wy, des_wz - wz])
        inertia = np.array([IXX, IYY, IZZ])
        kd = np.array([self.kd_att, self.kd_att, self.kd_yaw])
        tau_p = inertia * kd * rate_err

        # I term accumulates in torque space (N·m) to compensate COM offset bias
        self.rate_int_torque = np.clip(
            self.rate_int_torque + self.ki_rate_torque * DT * rate_err,
            -self.rate_int_max, self.rate_int_max,
        )
        tau = tau_p + self.rate_int_torque

        # Clamp torques (yaw gets less authority)
        tau[0] = np.clip(tau[0], -max_tau, max_tau)
        tau[1] = np.clip(tau[1], -max_tau, max_tau)
        tau[2] = np.clip(tau[2], -max_tau * self.yaw_torque_scale,
                         max_tau * self.yaw_torque_scale)

        # ── 4. Normalize to [-1, 1] ──
        thrust_norm = 2.0 * thrust / MAX_THRUST - 1.0
        action = np.array([
            thrust_norm,
            tau[0] / MAX_TORQUE,
            tau[1] / MAX_TORQUE,
            tau[2] / MAX_TORQUE,
        ], dtype=np.float32)

        diag = {
            "des_rate": np.array([des_wx, des_wy, des_wz]),
            "actual_rate": np.array([wx, wy, wz]),
            "des_att": np.array([des_roll, des_pitch, 0.0]),
        }

        return np.clip(action, -1.0, 1.0), diag


def plot_episode(data: dict, episode_num: int, save_dir: str = "./plots"):
    """Generate performance plots for a single PID evaluation episode."""
    os.makedirs(save_dir, exist_ok=True)

    t = np.array(data["times"])
    pos = np.array(data["positions"])
    tgt = np.array(data["targets"])
    att = np.rad2deg(np.array(data["attitudes"]))
    vel = np.array(data["velocities"])
    ang_vel = np.rad2deg(np.array(data["angular_velocities"]))
    motors = np.array(data["motor_commands"])
    actions = np.array(data["actions"])
    rewards = np.array(data["rewards"])
    pos_err = np.linalg.norm(pos - tgt, axis=1)
    des_rates = np.rad2deg(np.array(data["des_rates"]))
    actual_rates = np.rad2deg(np.array(data["actual_rates"]))
    rate_err = des_rates - actual_rates
    des_att = np.rad2deg(np.array(data["des_attitudes"]))

    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    fig.suptitle(f"PID Controller — Episode {episode_num}", fontsize=14)

    # Position tracking
    ax = axes[0, 0]
    for i, (label, color) in enumerate(zip(["x", "y", "z"], ["r", "g", "b"])):
        ax.plot(t, pos[:, i], color=color, label=label)
        ax.plot(t, tgt[:, i], color=color, linestyle="--", alpha=0.5)
    ax.set_ylabel("Position (m)")
    ax.set_title("Position (solid=UAV, dashed=target)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position error
    ax = axes[0, 1]
    ax.plot(t, pos_err, color="k")
    ax.set_ylabel("Error (m)")
    ax.set_title("Position Error (Euclidean)")
    ax.grid(True, alpha=0.3)

    # Attitude: actual vs desired
    ax = axes[1, 0]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, att[:, i], color=color, label=label)
        ax.plot(t, des_att[:, i], color=color, linestyle="--", alpha=0.5)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Attitude (solid=actual, dashed=desired)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rate tracking: commanded vs actual (per axis)
    ax = axes[1, 1]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, actual_rates[:, i], color=color, label=f"{label} actual")
        ax.plot(t, des_rates[:, i], color=color, linestyle="--", alpha=0.5,
                label=f"{label} cmd")
    ax.set_ylabel("Rate (deg/s)")
    ax.set_title("Rate Tracking (solid=actual, dashed=commanded)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Rate error per axis
    ax = axes[2, 0]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, rate_err[:, i], color=color, label=label, alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Rate error (deg/s)")
    ax.set_title("Rate Controller Error (cmd - actual)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rate error magnitude (Euclidean norm)
    ax = axes[2, 1]
    rate_err_norm = np.linalg.norm(rate_err, axis=1)
    ax.plot(t, rate_err_norm, color="k")
    ax.set_ylabel("|Rate error| (deg/s)")
    ax.set_title("Rate Error Magnitude")
    ax.grid(True, alpha=0.3)

    # Motor commands
    ax = axes[3, 0]
    for i in range(4):
        ax.plot(t, motors[:, i], label=f"M{i+1}")
    ax.set_ylabel("Thrust (N)")
    ax.set_title("Motor Commands")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Torque actions
    ax = axes[3, 1]
    for i, label in enumerate(["roll torque", "pitch torque", "yaw torque"]):
        ax.plot(t, actions[:, i + 1], label=label)
    ax.set_ylabel("Action (normalized)")
    ax.set_title("Actions: Torques")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rewards
    ax = axes[4, 0]
    ax.plot(t, rewards, color="k")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title("Step Reward")
    ax.grid(True, alpha=0.3)

    # Thrust action
    ax = axes[4, 1]
    ax.plot(t, actions[:, 0], label="thrust", color="k")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action (normalized)")
    ax.set_title("Action: Thrust")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(save_dir, f"pid_episode_{episode_num}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {filepath}")


def _update_visuals(viewer, drone_pos, target_pos, trail, pos_err):
    """Update custom visualization elements in the MuJoCo viewer.

    Draws:
        - Green translucent sphere at target position
        - Error line from drone to target (green=close, red=far)
        - Blue trajectory trail showing recent flight path
        - Ground shadow marker below the drone
    """
    scn = viewer.user_scn
    scn.ngeom = 0  # reset custom geoms each frame

    # 1. Target sphere (green, translucent)
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.05, 0, 0]),
            pos=np.asarray(target_pos, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.2, 1.0, 0.2, 0.5]),
        )
        scn.ngeom += 1

    # 2. Error line (drone → target), color-coded by distance
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            0.003,
            np.asarray(drone_pos, dtype=np.float64),
            np.asarray(target_pos, dtype=np.float64),
        )
        err_ratio = min(pos_err / 1.0, 1.0)
        scn.geoms[scn.ngeom].rgba[:] = [err_ratio, 1.0 - err_ratio, 0.0, 0.8]
        scn.ngeom += 1

    # 3. Trajectory trail (blue dots)
    for pt in trail:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.008, 0, 0]),
            pos=np.asarray(pt, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.2, 0.5, 1.0, 0.4]),
        )
        scn.ngeom += 1

    # 4. Ground shadow (small disc below drone)
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.array([0.04, 0.04, 0.001]),
            pos=np.array([drone_pos[0], drone_pos[1], 0.001]),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.3, 0.3, 0.3, 0.4]),
        )
        scn.ngeom += 1


def evaluate(num_episodes: int = 5, render: bool = True, plot: bool = False,
             gains_file: str | None = None):
    """Evaluate the PID controller on HoverEnv.

    Args:
        num_episodes: Number of episodes to run.
        render: Whether to render the MuJoCo viewer.
        plot: Whether to save performance plots.
        gains_file: Optional path to a JSON file with custom PID gains.
    """
    # Load gains
    gains = DEFAULT_GAINS
    if gains_file and os.path.exists(gains_file):
        with open(gains_file) as f:
            gains = json.load(f)
        print(f"Loaded gains from {gains_file}")

    controller = CascadedPIDController(gains)
    env = HoverEnv()

    episode_rewards = []
    episode_lengths = []
    episode_errors = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        controller.reset()
        total_reward = 0.0
        step_count = 0
        done = False
        errors = []
        trail = deque(maxlen=200)

        ep_data = {
            "times": [], "positions": [], "targets": [],
            "attitudes": [], "velocities": [], "angular_velocities": [],
            "motor_commands": [], "actions": [], "rewards": [],
            "des_rates": [], "actual_rates": [], "des_attitudes": [],
        }

        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            viewer.cam.distance = 7.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

        state = info["state"]
        target = info["target"]
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"  Start:  [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]")
        print(f"  Target: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")

        while not done:
            state = info["state"]
            target = info["target"]
            action, diag = controller.compute(state, target)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            drone_pos = info["state"][:3]
            pos_err = float(np.linalg.norm(drone_pos - info["target"]))
            errors.append(pos_err)

            # Record trail every 5 steps to avoid too many geoms
            if step_count % 5 == 0:
                trail.append(drone_pos.copy())

            if plot:
                s = info["state"]
                ep_data["times"].append(step_count * DT)
                ep_data["positions"].append(s[:3].copy())
                ep_data["targets"].append(info["target"].copy())
                ep_data["attitudes"].append(s[3:6].copy())
                ep_data["velocities"].append(s[6:9].copy())
                ep_data["angular_velocities"].append(s[9:12].copy())
                ep_data["motor_commands"].append(info["motor_commands"].copy())
                ep_data["actions"].append(action.copy())
                ep_data["rewards"].append(reward)
                ep_data["des_rates"].append(diag["des_rate"].copy())
                ep_data["actual_rates"].append(diag["actual_rate"].copy())
                ep_data["des_attitudes"].append(diag["des_att"].copy())

            if render and viewer is not None and viewer.is_running():
                _update_visuals(viewer, drone_pos, info["target"], trail, pos_err)
                viewer.sync()
                time.sleep(DT)
            elif render and viewer is not None and not viewer.is_running():
                break

            if step_count % 100 == 0:
                s = info["state"]
                print(f"  Step {step_count}: pos=[{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}], "
                      f"err={pos_err:.3f}m, reward={reward:.2f}")

        if viewer is not None:
            viewer.close()

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_errors.append(np.mean(errors))

        status = "TERMINATED" if terminated else "TRUNCATED (max steps)"
        print(f"  {status} after {step_count} steps")
        print(f"  Total reward: {total_reward:.2f}, mean error: {np.mean(errors):.3f}m")

        if plot and len(ep_data["times"]) > 0:
            plot_episode(ep_data, ep + 1, save_dir="./plots/pid")

    env.close()

    print("\n=== PID Evaluation Summary ===")
    print(f"Episodes:    {num_episodes}")
    print(f"Survival:    {sum(1 for l in episode_lengths if l >= 512)}/{num_episodes} "
          f"({100*sum(1 for l in episode_lengths if l >= 512)/num_episodes:.0f}%)")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Mean error:  {np.mean(episode_errors):.3f}m")


def main():
    parser = argparse.ArgumentParser(description="PID hover controller evaluation")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable MuJoCo viewer")
    parser.add_argument("--plot", action="store_true",
                        help="Save performance plots to ./plots/pid/")
    parser.add_argument("--gains", type=str, default=None,
                        help="Path to custom PID gains JSON file")
    args = parser.parse_args()

    evaluate(
        num_episodes=args.episodes,
        render=not args.no_render,
        plot=args.plot,
        gains_file=args.gains,
    )


if __name__ == "__main__":
    main()
