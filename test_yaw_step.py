"""Yaw step response test: body rate PID from pid_controller.py.

Spawns drone hovering, applies a yaw error, and uses a P controller
(output in deg/s) on top of the body rate PID to drive yaw to zero.

Usage:
    python test_yaw_step.py                            # defaults: 45° error, P=1
    python test_yaw_step.py --yaw 90 --kp 2.0          # 90° error, P=2
    python test_yaw_step.py --fixed --render --plot     # fixed position, with viewer
    python test_yaw_step.py --render --duration 10      # free drone, with viewer
"""

import argparse
import json
import os

import time as _time

import numpy as np
import mujoco
import mujoco.viewer
from gymnasium.spaces import Box

from envs import HoverEnv
from utils.drone_config import MASS, G, DT, MAX_TOTAL_THRUST, MAX_TORQUE, IXX, IYY, IZZ


_GAINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid_gains_x500.json")


class BodyRatePID:
    """Body rate PID: desired rates (rad/s) → normalized torques for env.step().

    Same logic as CascadedPIDController in pid_controller.py.
    """

    def __init__(self, kd, ki_torque, integral_max):
        self.inertia = np.array([IXX, IYY, IZZ])
        self.kd = np.array(kd)
        self.ki_torque = ki_torque
        self.int_max = integral_max
        self.rate_int_torque = np.zeros(3)

    def reset(self):
        self.rate_int_torque = np.zeros(3)

    def step(self, des_rates_rad, actual_rates_rad):
        """Compute normalized torques from desired and actual body rates.

        Args:
            des_rates_rad: desired [wx, wy, wz] in rad/s
            actual_rates_rad: actual [wx, wy, wz] in rad/s

        Returns:
            tau_norm: [tau_x, tau_y, tau_z] in [-1, 1] for env.step()
        """
        rate_err = des_rates_rad - actual_rates_rad
        tau_p = self.inertia * self.kd * rate_err
        self.rate_int_torque = np.clip(
            self.rate_int_torque + self.ki_torque * DT * rate_err,
            -self.int_max, self.int_max,
        )
        tau = tau_p + self.rate_int_torque
        return np.clip(tau / MAX_TORQUE, -1.0, 1.0)


def run_yaw_step(yaw_deg=45.0, kp_yaw=1.0, gains_path=_GAINS_PATH,
                 duration=5.0, plot=False, fixed=False, render=False):
    # Load gains
    with open(gains_path) as f:
        gains = json.load(f)

    kd_att = gains["attitude"]["kd"]
    kd_yaw = gains["yaw"]["kd"]
    ki = gains["rate"]["ki_torque"]
    imax = gains["rate"]["integral_max"]

    rate_pid = BodyRatePID(
        kd=[kd_att, kd_att, kd_yaw],
        ki_torque=ki,
        integral_max=imax,
    )

    # Create env with wide bounds to avoid termination
    env = HoverEnv(max_episode_steps=int(duration / DT) + 100)
    env._state_bounds = Box(
        low=np.array([-50, -50, -1, -np.pi, -np.pi, -np.pi,
                       -20, -20, -20, -20*np.pi, -20*np.pi, -20*np.pi], dtype=np.float32),
        high=np.array([50, 50, 50, np.pi, np.pi, np.pi,
                        20, 20, 20, 20*np.pi, 20*np.pi, 20*np.pi], dtype=np.float32),
    )
    obs, info = env.reset()
    base_env = env.unwrapped

    # If --fixed, constrain translation DOFs so drone only rotates in place
    if fixed:
        # Lock x, y, z translation by welding the base to world with a
        # free-to-rotate-only joint. We do this by resetting position each step.
        print("  [FIXED MODE] Drone position locked — rotation only.\n")

    # Set initial yaw and hover position
    hover_z = 1.0 if fixed else 10.0
    yaw_init = np.deg2rad(yaw_deg)
    base_env.data.qpos[0:3] = [0, 0, hover_z]
    base_env.data.qpos[3:7] = [np.cos(yaw_init / 2), 0, 0, np.sin(yaw_init / 2)]
    base_env.data.qvel[:] = 0
    base_env.data.ctrl[:] = MASS * G / 4.0
    mujoco.mj_forward(base_env.model, base_env.data)

    # Launch viewer
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data)
        viewer.cam.lookat[:] = [0.0, 0.0, hover_z]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

    n_steps = int(duration / DT)
    times = np.zeros(n_steps)
    yaws = np.zeros(n_steps)
    yaw_rates_cmd = np.zeros(n_steps)
    yaw_rates_actual = np.zeros(n_steps)

    fixed_pos = np.array([0.0, 0.0, hover_z])

    print(f"Yaw step: {yaw_deg}° → 0° | P = {kp_yaw} deg/s per deg")
    print(f"Rate PID: kd=[{kd_att}, {kd_att}, {kd_yaw}], ki={ki}, imax={imax}\n")

    for step in range(n_steps):
        # In fixed mode, lock position and linear velocity each step
        if fixed:
            base_env.data.qpos[0:3] = fixed_pos
            base_env.data.qvel[0:3] = 0.0
            mujoco.mj_forward(base_env.model, base_env.data)

        state = base_env._state
        yaw = state.attitude[2]
        roll, pitch = state.attitude[0], state.attitude[1]
        wx, wy, wz = state.angular_velocity

        # Tilt-compensated hover thrust
        cos_tilt = max(np.cos(roll) * np.cos(pitch), 0.5)
        thr_norm = np.clip(2.0 * (MASS * G / cos_tilt) / MAX_TOTAL_THRUST - 1.0, -1.0, 1.0)

        # Outer P: yaw error → desired yaw rate (deg/s)
        yaw_err = (0.0 - yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_rate_cmd_deg = kp_yaw * np.rad2deg(yaw_err)

        # Body rate PID → torques
        des_rates = np.array([0.0, 0.0, np.deg2rad(yaw_rate_cmd_deg)])
        tau_norm = rate_pid.step(des_rates, np.array([wx, wy, wz]))
        action = np.array([thr_norm, tau_norm[0], tau_norm[1], tau_norm[2]], dtype=np.float32)

        times[step] = step * DT
        yaws[step] = np.rad2deg(yaw)
        yaw_rates_cmd[step] = yaw_rate_cmd_deg
        yaw_rates_actual[step] = np.rad2deg(wz)

        obs, reward, terminated, truncated, info = env.step(action)

        if render and viewer is not None and viewer.is_running():
            viewer.sync()
            _time.sleep(DT)

        if terminated:
            print(f"TERMINATED at step {step}")
            times = times[:step]
            yaws = yaws[:step]
            yaw_rates_cmd = yaw_rates_cmd[:step]
            yaw_rates_actual = yaw_rates_actual[:step]
            break

    if viewer is not None:
        viewer.close()

    # Settling times
    for thresh in [10.0, 5.0, 2.0, 1.0, 0.5]:
        idx = np.where(np.abs(yaws[3:]) < thresh)[0]
        if len(idx) > 0:
            print(f"  Yaw < {thresh:>4}° at t = {times[idx[0] + 3]:.3f}s")

    # Table
    print(f"\n{'Time':>6}  {'Yaw':>8}  {'Rate cmd':>10}  {'Rate act':>10}  {'Rate err':>10}")
    print("-" * 52)
    skip = max(1, len(times) // 25)
    for i in range(0, len(times), skip):
        rc, ra = yaw_rates_cmd[i], yaw_rates_actual[i]
        print(f"{times[i]:6.2f}s  {yaws[i]:7.2f}°  {rc:8.1f}°/s  {ra:8.1f}°/s  {rc - ra:8.2f}°/s")

    if plot:
        import matplotlib.pyplot as plt
        os.makedirs("./plots/pid", exist_ok=True)

        t_th = np.linspace(0, duration, 500)
        yaw_th = yaw_deg * np.exp(-kp_yaw * t_th)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(times, yaws, "b-", lw=2, label="sim")
        axes[0].plot(t_th, yaw_th, "g--", lw=1.5, alpha=0.7, label=f"theory (τ={1/kp_yaw:.1f}s)")
        axes[0].axhline(0, color="k", ls="--", alpha=0.3)
        axes[0].set_ylabel("Yaw (deg)")
        axes[0].set_title(f"Yaw Step: {yaw_deg}° → 0° | P={kp_yaw} | kd_yaw={kd_yaw}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, yaw_rates_cmd, "r--", label="cmd", lw=1.5)
        axes[1].plot(times, yaw_rates_actual, "b-", label="actual", lw=1.5)
        axes[1].set_ylabel("Rate (deg/s)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(times, yaw_rates_cmd - yaw_rates_actual, "r-", lw=1)
        axes[2].axhline(0, color="k", ls="--", alpha=0.3)
        axes[2].set_ylabel("Rate err (deg/s)")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = "./plots/pid/yaw_step_response.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\nPlot saved to {path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yaw step response test")
    parser.add_argument("--yaw", type=float, default=45.0, help="Initial yaw error in degrees")
    parser.add_argument("--kp", type=float, default=1.0, help="P gain (deg/s per deg error)")
    parser.add_argument("--gains", type=str, default=_GAINS_PATH, help="Path to gains JSON")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds")
    parser.add_argument("--plot", action="store_true", help="Save plot")
    parser.add_argument("--fixed", action="store_true", help="Lock position, rotation only")
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer")
    args = parser.parse_args()

    run_yaw_step(
        yaw_deg=args.yaw,
        kp_yaw=args.kp,
        gains_path=args.gains,
        duration=args.duration,
        plot=args.plot,
        fixed=args.fixed,
        render=args.render,
    )
