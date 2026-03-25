"""Pure rate tracking test — no position or attitude control.

Sends random rate step commands through the rate PID and measures how well
the actual body rates follow. Thrust is held at hover to keep the drone
airborne.

Usage:
    python test_rate_tracking.py
    python test_rate_tracking.py --gains pid_gains_x500.json --plot
    python test_rate_tracking.py --max-rate 180   # limit commands to ±180 deg/s
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from envs import HoverEnv
from utils.drone_config import (
    MASS, G, DT, IXX, IYY, IZZ, MAX_TORQUE,
    MAX_TOTAL_THRUST, MAX_MOTOR_THRUST, ARM_LENGTH, HOVER_THRUST_PER_MOTOR,
)

# ── Load gains ──
_GAINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid_gains_x500.json")
with open(_GAINS_PATH) as _f:
    _DEFAULTS = json.load(_f).get("rate_wrapper", {})

_DEFAULT_KD = _DEFAULTS.get("kd", [50.0, 50.0, 35.0])
_DEFAULT_KI = _DEFAULTS.get("ki_rate_torque", 0.5)
_DEFAULT_IMAX = _DEFAULTS.get("integral_max", 0.10)


def rate_pid_step(des_rates_rad, actual_rates_rad, kd, inertia, ki, integral, dt,
                  thrust_phys):
    """One step of the inertia-scaled rate PID with thrust-aware integral anti-windup.

    The P term is unclamped (requests what physics needs). The integral is
    clamped to the actually available torque so it doesn't wind up against
    a motor saturation ceiling it can never overcome.

    Returns torques, updated integral, and available torque limit.
    """
    rate_err = des_rates_rad - actual_rates_rad

    # Thrust-aware torque budget (for integral anti-windup only)
    thrust_per_motor = thrust_phys / 4.0
    motor_margin = min(thrust_per_motor, MAX_MOTOR_THRUST - thrust_per_motor)
    available_tau = max(motor_margin * 2.0 * ARM_LENGTH * 0.85, 0.01)

    # P term (inertia-scaled) — unclamped, let it request what it needs
    tau_p = inertia * kd * rate_err

    # I term — clamp to available torque so integral doesn't wind up falsely
    effective_imax = min(_DEFAULT_IMAX, available_tau)
    integral = np.clip(integral + ki * dt * rate_err, -effective_imax, effective_imax)

    tau = tau_p + integral

    return tau, integral, available_tau


def run_test(gains_file=None, max_rate_deg=360.0, hold_steps=100,
             num_commands=30, plot=True, seed=42):
    """Run pure rate tracking test.

    Args:
        gains_file: Optional JSON gains file.
        max_rate_deg: Max commanded rate magnitude in deg/s.
        hold_steps: Steps to hold each rate command before switching.
        num_commands: Number of random rate commands to test.
        plot: Save plots.
        seed: Random seed.
    """
    # Load custom gains if provided
    kd = np.array(_DEFAULT_KD)
    ki = _DEFAULT_KI
    if gains_file and os.path.exists(gains_file):
        with open(gains_file) as f:
            g = json.load(f).get("rate_wrapper", {})
        kd = np.array(g.get("kd", _DEFAULT_KD))
        ki = g.get("ki_rate_torque", _DEFAULT_KI)
        print(f"Loaded gains from {gains_file}: kd={kd.tolist()}, ki={ki}")

    inertia = np.array([IXX, IYY, IZZ])
    max_rate_rad = np.deg2rad(max_rate_deg)

    # Hover thrust normalized to [-1, 1] action space
    hover_thrust_norm = 2.0 * (4 * HOVER_THRUST_PER_MOTOR) / MAX_TOTAL_THRUST - 1.0

    from gymnasium.spaces import Box
    # Wide state bounds so the drone doesn't terminate from drifting
    wide_state_bounds = Box(
        low=np.array([-50, -50, -1, -np.pi, -np.pi, -np.pi, -30, -30, -30, -20*np.pi, -20*np.pi, -20*np.pi], dtype=np.float32),
        high=np.array([50, 50, 50, np.pi, np.pi, np.pi, 30, 30, 30, 20*np.pi, 20*np.pi, 20*np.pi], dtype=np.float32),
    )
    env = HoverEnv(max_episode_steps=hold_steps + 10)
    env._state_bounds = wide_state_bounds
    rng = np.random.default_rng(seed)

    import mujoco

    # Generate random rate commands: single-axis, dual-axis, and multi-axis
    commands = []
    for _ in range(num_commands):
        cmd_deg = rng.uniform(-max_rate_deg, max_rate_deg, size=3)
        # 40% chance to zero out each axis → mostly single/dual-axis tests
        for ax in range(3):
            if rng.random() < 0.4:
                cmd_deg[ax] = 0.0
        commands.append(cmd_deg)

    total_steps = hold_steps * num_commands
    print(f"\nRunning {num_commands} rate commands, {hold_steps} steps each "
          f"({hold_steps * DT:.1f}s hold), total {total_steps} steps ({total_steps * DT:.1f}s)")
    print(f"Max rate: ±{max_rate_deg:.0f} deg/s, DT={DT}s")
    print(f"Drone reset to hover between each command.\n")

    # Data logging
    times = []
    des_rates_log = []
    act_rates_log = []
    torques_log = []
    cmd_idx_log = []
    global_step = 0

    for cmd_i, cmd_deg in enumerate(commands):
        # Reset drone to stable hover for each command
        env.reset(seed=seed + cmd_i)
        env.data.qpos[:3] = [0, 0, 10.0]
        env.data.qpos[3:7] = [1, 0, 0, 0]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        integral = np.zeros(3)

        cmd_rad = np.deg2rad(cmd_deg)

        for local_step in range(hold_steps):
            # Get actual rates
            env._state.set_from_mujoco(env.data.qpos[:7], env.data.qvel[:6])
            actual_rates = env._state.angular_velocity  # rad/s
            roll, pitch = env._state.attitude[0], env._state.attitude[1]

            # Tilt-compensated thrust to stay airborne
            cos_tilt = max(np.cos(roll) * np.cos(pitch), 0.3)
            thrust_phys = MASS * G / cos_tilt
            thrust_phys = np.clip(thrust_phys, 0.0, MAX_TOTAL_THRUST)
            thrust_norm = np.clip(2.0 * thrust_phys / MAX_TOTAL_THRUST - 1.0, -1.0, 1.0)

            # Rate PID → torques (thrust-aware limiting)
            tau, integral, avail_tau = rate_pid_step(
                cmd_rad, actual_rates, kd, inertia, ki, integral, DT, thrust_phys
            )

            # Normalize torques to action space
            tau_norm = np.clip(tau / MAX_TORQUE, -1.0, 1.0)
            action = np.array([thrust_norm, tau_norm[0], tau_norm[1], tau_norm[2]],
                              dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            # Log
            times.append(global_step * DT)
            des_rates_log.append(cmd_deg.copy())
            act_rates_log.append(np.rad2deg(actual_rates).copy())
            torques_log.append(tau.copy())
            cmd_idx_log.append(cmd_i)
            global_step += 1

            if terminated:
                # Pad remaining steps with last values so analysis works
                for _ in range(local_step + 1, hold_steps):
                    times.append(global_step * DT)
                    des_rates_log.append(cmd_deg.copy())
                    act_rates_log.append(act_rates_log[-1].copy())
                    torques_log.append(torques_log[-1].copy())
                    cmd_idx_log.append(cmd_i)
                    global_step += 1
                break

    env.close()

    # ── Numerical analysis ──
    des = np.array(des_rates_log)
    act = np.array(act_rates_log)
    err = des - act
    tau_arr = np.array(torques_log)
    t = np.array(times)
    axis_names = ["roll", "pitch", "yaw"]

    # Per-command analysis: skip first command (initial settle) and last
    print("=" * 80)
    print(f"{'Cmd':>4s} | {'Roll cmd':>9s} {'Pitch cmd':>10s} {'Yaw cmd':>9s} | "
          f"{'Roll err':>9s} {'Pitch err':>10s} {'Yaw err':>9s} | {'t90':>5s}")
    print("-" * 80)

    all_ss_errors = {ax: [] for ax in axis_names}
    all_t90 = []

    for ci in range(1, len(commands) - 1):
        mask = np.array(cmd_idx_log) == ci
        if mask.sum() < 10:
            continue

        cmd_err = err[mask]
        cmd_des = des[mask]

        # Steady-state: last 40% of hold period
        ss_start = int(len(cmd_err) * 0.6)
        ss_err = cmd_err[ss_start:]

        # t90: time for error magnitude to drop below 10% of initial
        err_norm = np.linalg.norm(cmd_err, axis=1)
        if err_norm[0] > 0.5:  # only compute if there's a meaningful step
            threshold = err_norm[0] * 0.1
            t90_idx = np.argmax(err_norm < threshold)
            t90_ms = t90_idx * DT * 1000 if t90_idx > 0 else float("inf")
        else:
            t90_ms = 0.0
        all_t90.append(t90_ms)

        ss_mean = np.mean(np.abs(ss_err), axis=0)
        for i, ax in enumerate(axis_names):
            all_ss_errors[ax].append(ss_mean[i])

        print(f"{ci:4d} | {cmd_des[0, 0]:+8.1f}° {cmd_des[0, 1]:+9.1f}° {cmd_des[0, 2]:+8.1f}° | "
              f"{ss_mean[0]:8.3f}° {ss_mean[1]:9.3f}° {ss_mean[2]:8.3f}° | {t90_ms:5.0f}ms")

    print("=" * 80)

    # Summary statistics
    print("\n=== Rate Tracking Summary ===")
    for ax in axis_names:
        errs = all_ss_errors[ax]
        if errs:
            print(f"  {ax:5s} SS error: mean={np.mean(errs):.3f} deg/s, "
                  f"max={np.max(errs):.3f} deg/s, p95={np.percentile(errs, 95):.3f} deg/s")

    valid_t90 = [t for t in all_t90 if t < 10000]
    if valid_t90:
        print(f"  t90:  mean={np.mean(valid_t90):.0f} ms, "
              f"max={np.max(valid_t90):.0f} ms, p95={np.percentile(valid_t90, 95):.0f} ms")

    # Overall RMS
    for i, ax in enumerate(axis_names):
        rms = np.sqrt(np.mean(err[:, i] ** 2))
        print(f"  {ax:5s} RMS error (full episode): {rms:.2f} deg/s")

    # ── Plot ──
    if plot:
        os.makedirs("./plots/pid", exist_ok=True)
        fig, axes_arr = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
        fig.suptitle("Pure Rate Tracking Test (no position control)", fontsize=14)

        colors = ["r", "g", "b"]

        # Desired vs actual rates per axis
        for i, (ax_name, color) in enumerate(zip(axis_names, colors)):
            ax = axes_arr[i]
            ax.plot(t, des[:, i], color=color, linestyle="--", alpha=0.7, label=f"{ax_name} cmd")
            ax.plot(t, act[:, i], color=color, alpha=0.9, label=f"{ax_name} actual")
            ax.set_ylabel(f"{ax_name} rate (deg/s)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{ax_name.capitalize()} Rate Tracking")

        # Rate error all axes
        ax = axes_arr[3]
        for i, (ax_name, color) in enumerate(zip(axis_names, colors)):
            ax.plot(t, err[:, i], color=color, alpha=0.7, label=ax_name)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel("Rate error (deg/s)")
        ax.set_title("Rate Error (cmd - actual)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Torques
        ax = axes_arr[4]
        for i, (ax_name, color) in enumerate(zip(axis_names, colors)):
            ax.plot(t, tau_arr[:, i], color=color, alpha=0.7, label=f"τ_{ax_name}")
        ax.set_ylabel("Torque (N·m)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Applied Torques")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        filepath = "./plots/pid/rate_tracking_test.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Pure rate tracking test")
    parser.add_argument("--gains", type=str, default=None,
                        help="Path to PID gains JSON file")
    parser.add_argument("--max-rate", type=float, default=200.0,
                        help="Max commanded rate in deg/s (default: 200)")
    parser.add_argument("--hold", type=int, default=100,
                        help="Steps to hold each command (default: 100 = 1s)")
    parser.add_argument("--commands", type=int, default=30,
                        help="Number of random rate commands (default: 30)")
    parser.add_argument("--plot", action="store_true",
                        help="Save plots to ./plots/pid/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_test(
        gains_file=args.gains,
        max_rate_deg=args.max_rate,
        hold_steps=args.hold,
        num_commands=args.commands,
        plot=args.plot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
