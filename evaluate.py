"""Evaluation and visualization script for trained hover policy."""

import argparse
import os
import json
import time
from collections import deque
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs import HoverEnv, get_wrapper
from utils.trajectories import TRAJECTORY_GENERATORS


def plot_episode(data: dict, episode_num: int, save_dir: str = "./plots",
                 max_rate: float = 360.0):
    """Generate performance plots for a single evaluation episode.

    Args:
        data: Dict with keys: times, positions, targets, attitudes,
              velocities, angular_velocities, motor_commands, rewards.
        episode_num: Episode number (for title and filename).
        save_dir: Directory to save plot images.
        max_rate: Max body rate in deg/s (for denormalizing rate actions).
    """
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

    # Commanded body rates (deg/s) from normalized actions
    cmd_rates = actions[:, 1:4] * max_rate
    rate_err = cmd_rates - ang_vel

    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    fig.suptitle(f"Episode {episode_num}", fontsize=14)

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

    # Attitude
    ax = axes[1, 0]
    for i, label in enumerate(["roll", "pitch", "yaw"]):
        ax.plot(t, att[:, i], label=label)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Attitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Linear velocity
    ax = axes[1, 1]
    for i, label in enumerate(["vx", "vy", "vz"]):
        ax.plot(t, vel[:, i], label=label)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Linear Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Angular velocity
    ax = axes[2, 0]
    for i, label in enumerate(["wx", "wy", "wz"]):
        ax.plot(t, ang_vel[:, i], label=label)
    ax.set_ylabel("Angular vel (deg/s)")
    ax.set_title("Angular Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Motor commands
    ax = axes[2, 1]
    for i in range(4):
        ax.plot(t, motors[:, i], label=f"M{i+1}")
    ax.set_ylabel("Thrust (N)")
    ax.set_title("Motor Commands")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Policy actions
    ax = axes[3, 0]
    ax.plot(t, actions[:, 0], label="thrust")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action (normalized)")
    ax.set_title("Policy Action: Thrust")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    for i, label in enumerate(["roll rate", "pitch rate", "yaw rate"]):
        ax.plot(t, actions[:, i + 1], label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action (normalized)")
    ax.set_title("Policy Actions: Body Rates")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rate tracking: commanded vs actual (per axis)
    ax = axes[4, 0]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, ang_vel[:, i], color=color, label=f"{label} actual")
        ax.plot(t, cmd_rates[:, i], color=color, linestyle="--", alpha=0.5,
                label=f"{label} cmd")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (deg/s)")
    ax.set_title("Rate Tracking (solid=actual, dashed=commanded)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Rate tracking error
    ax = axes[4, 1]
    for i, (label, color) in enumerate(zip(["roll", "pitch", "yaw"], ["r", "g", "b"])):
        ax.plot(t, rate_err[:, i], color=color, label=label, alpha=0.8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate error (deg/s)")
    ax.set_title("Rate Controller Error (cmd - actual)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(save_dir, f"episode_{episode_num}.png")
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


def _update_trajectory_visuals(viewer, drone_pos, waypoints, current_wp_idx,
                                trail, pos_err):
    """Update MuJoCo viewer with trajectory and waypoint visualization."""
    scn = viewer.user_scn
    scn.ngeom = 0
    num_wp = len(waypoints)

    # 1. Path lines connecting consecutive waypoints
    for i in range(num_wp):
        if scn.ngeom >= scn.maxgeom:
            break
        p1 = waypoints[i]
        p2 = waypoints[(i + 1) % num_wp]
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            0.002,
            np.asarray(p1, dtype=np.float64),
            np.asarray(p2, dtype=np.float64),
        )
        scn.geoms[scn.ngeom].rgba[:] = [0.7, 0.7, 0.7, 0.4]
        scn.ngeom += 1

    # 2. Waypoint spheres (current = larger yellow, others = small gray)
    for i in range(num_wp):
        if scn.ngeom >= scn.maxgeom:
            break
        is_current = (i == current_wp_idx)
        size = 0.06 if is_current else 0.03
        rgba = [1.0, 0.9, 0.0, 0.9] if is_current else [0.6, 0.6, 0.6, 0.4]
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([size, 0, 0]),
            pos=np.asarray(waypoints[i], dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array(rgba),
        )
        scn.ngeom += 1

    # 3. Error line (drone -> current waypoint, color-coded)
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            0.003,
            np.asarray(drone_pos, dtype=np.float64),
            np.asarray(waypoints[current_wp_idx], dtype=np.float64),
        )
        err_ratio = min(pos_err / 1.0, 1.0)
        scn.geoms[scn.ngeom].rgba[:] = [err_ratio, 1.0 - err_ratio, 0.0, 0.8]
        scn.ngeom += 1

    # 4. Flight trail (blue dots)
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

    # 5. Ground shadow
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


def evaluate(model_path: str, num_episodes: int = 5, render: bool = True,
             plot: bool = False):
    """Evaluate a trained policy.

    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to run
        render: Whether to render visualization
        plot: Whether to generate performance plots
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")

    # Resolve model directory for saving plots and description
    model_dir = os.path.dirname(os.path.abspath(model_path))

    # Auto-detect wrapper from config.json
    wrapper_cls = None
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        wrapper_cls = get_wrapper(config.get("wrapper", "none"))
        wrapper_name = config.get("wrapper", "none")
        print(f"Wrapper: {wrapper_name}")

    # Create environment
    env = HoverEnv()
    if wrapper_cls:
        env = wrapper_cls(env)
    base_env = env.unwrapped

    # Get max_rate from wrapper if available (for plot denormalization)
    max_rate = np.rad2deg(env.max_rate_rad) if hasattr(env, "max_rate_rad") else 360.0

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False
        trail = deque(maxlen=200)

        # Data collection for plots
        ep_data = {
            "times": [], "positions": [], "targets": [],
            "attitudes": [], "velocities": [], "angular_velocities": [],
            "motor_commands": [], "actions": [], "rewards": [],
        }

        # Setup viewer for this episode
        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data)
            viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            viewer.cam.distance = 7.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

        target = info["target"]
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"  Target position: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")

        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            # Collect data
            if plot:
                state = info["state"]
                ep_data["times"].append(step_count * base_env.dt * base_env.frame_skip)
                ep_data["positions"].append(state[:3].copy())
                ep_data["targets"].append(info["target"].copy())
                ep_data["attitudes"].append(state[3:6].copy())
                ep_data["velocities"].append(state[6:9].copy())
                ep_data["angular_velocities"].append(state[9:12].copy())
                ep_data["motor_commands"].append(info["motor_commands"].copy())
                ep_data["actions"].append(action.copy())
                ep_data["rewards"].append(reward)

            # Trail + visuals
            drone_pos = info["state"][:3]
            pos_err = float(np.linalg.norm(drone_pos - info["target"]))
            if step_count % 5 == 0:
                trail.append(drone_pos.copy())

            # Render
            if render and viewer is not None and viewer.is_running():
                _update_visuals(viewer, drone_pos, info["target"], trail, pos_err)
                viewer.sync()
                time.sleep(base_env.dt * base_env.frame_skip)
            elif render and viewer is not None and not viewer.is_running():
                break

            # Print status periodically
            if step_count % 100 == 0:
                state = info["state"]
                print(f"  Step {step_count}: pos=[{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}], "
                      f"reward={reward:.2f}")

        if viewer is not None:
            viewer.close()

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        status = "TERMINATED" if terminated else "TRUNCATED (max steps)"
        print(f"  {status} after {step_count} steps, total reward: {total_reward:.2f}")

        if plot and len(ep_data["times"]) > 0:
            plot_episode(ep_data, ep + 1, save_dir=os.path.join(model_dir, "plots"),
                        max_rate=max_rate)

    env.close()

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")

    # Prompt for description and save to model directory
    print("\nEnter a brief description of this policy (or press Enter to skip):")
    description = input("> ").strip()
    if description:
        desc_path = os.path.join(model_dir, "description.txt")
        with open(desc_path, "w") as f:
            f.write(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}\n")
            f.write(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}\n\n")
            f.write(f"{description}\n")
        print(f"Description saved to {desc_path}")


def evaluate_trajectory(model_path: str, trajectory: str = "eight",
                        spacing: float = 0.5, reach_radius: float = 0.25,
                        max_steps: int = 5000, render: bool = True):
    """Evaluate a trained policy on a single-lap waypoint trajectory.

    The drone starts at the first waypoint with zero velocities and follows a
    sequence of waypoints, advancing to the next when within reach_radius.
    Stops after completing one full lap. Plots are always generated.

    Args:
        model_path: Path to the trained model (.zip file).
        trajectory: Trajectory type key ("eight", "circle", "square").
        spacing: Distance between consecutive waypoints (meters).
        reach_radius: Distance threshold to switch to next waypoint (meters).
        max_steps: Maximum simulation steps (0 = run until viewer closed).
        render: Whether to render visualization.
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    model_dir = os.path.dirname(os.path.abspath(model_path))

    # Auto-detect wrapper from config.json
    wrapper_cls = None
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        wrapper_cls = get_wrapper(config.get("wrapper", "none"))
        print(f"Wrapper: {config.get('wrapper', 'none')}")

    # Create environment with extended episode length
    effective_max = max_steps if max_steps > 0 else 100_000
    env = HoverEnv(max_episode_steps=effective_max)
    if wrapper_cls:
        env = wrapper_cls(env)
    base_env = env.unwrapped

    max_rate = np.rad2deg(env.max_rate_rad) if hasattr(env, "max_rate_rad") else 360.0

    # Generate waypoints
    gen_fn = TRAJECTORY_GENERATORS[trajectory]
    waypoints = gen_fn(spacing=spacing)
    num_waypoints = len(waypoints)
    print(f"Trajectory: {trajectory}, {num_waypoints} waypoints, spacing={spacing}m")

    # Reset environment
    obs, info = env.reset()

    # Override initial state: start at first waypoint, identity quaternion, zero velocities
    start_pos = waypoints[0]
    qpos_init = np.array([start_pos[0], start_pos[1], start_pos[2], 1.0, 0.0, 0.0, 0.0])
    qvel_init = np.zeros(6)
    base_env.set_state(qpos_init, qvel_init)

    # Set second waypoint as target (drone is already at the first)
    wp_idx = 1 % num_waypoints
    base_env.target_state.state[0:3] = waypoints[wp_idx].astype(np.float32)

    # Re-obtain observation after state override
    obs = base_env._get_obs()
    if hasattr(env, 'observation') and callable(env.observation):
        obs = env.observation(obs)

    # Reset wrapper integral state if using rate control wrapper
    if hasattr(env, '_rate_int_torque'):
        env._rate_int_torque = np.zeros(3)

    # Setup viewer
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data)
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -35

    # Tracking variables
    trail = deque(maxlen=500)
    step_count = 0
    total_reward = 0.0
    waypoints_reached = 0
    laps_completed = 0
    done = False

    ep_data = {
        "times": [], "positions": [], "targets": [],
        "attitudes": [], "velocities": [], "angular_velocities": [],
        "motor_commands": [], "actions": [], "rewards": [],
    }

    print(f"\n--- Trajectory: {trajectory} ---")
    print(f"  Start (WP #0): [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]")
    print(f"  Target (WP #{wp_idx}): [{waypoints[wp_idx][0]:.2f}, {waypoints[wp_idx][1]:.2f}, {waypoints[wp_idx][2]:.2f}]")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        drone_pos = info["state"][:3]
        current_target = waypoints[wp_idx]
        dist_to_wp = float(np.linalg.norm(drone_pos - current_target))

        # Waypoint switching
        if dist_to_wp < reach_radius:
            waypoints_reached += 1
            wp_idx = (wp_idx + 1) % num_waypoints
            if wp_idx == 0:
                laps_completed += 1
                print(f"  Lap completed at step {step_count}!")
                done = True
            if not done:
                base_env.target_state.state[0:3] = waypoints[wp_idx].astype(np.float32)
                print(f"  WP {waypoints_reached}: reached! Next #{wp_idx} "
                      f"[{waypoints[wp_idx][0]:.2f}, {waypoints[wp_idx][1]:.2f}, {waypoints[wp_idx][2]:.2f}]")

        # Data collection (always, for plotting)
        state = info["state"]
        ep_data["times"].append(step_count * base_env.dt * base_env.frame_skip)
        ep_data["positions"].append(state[:3].copy())
        ep_data["targets"].append(info["target"].copy())
        ep_data["attitudes"].append(state[3:6].copy())
        ep_data["velocities"].append(state[6:9].copy())
        ep_data["angular_velocities"].append(state[9:12].copy())
        ep_data["motor_commands"].append(info["motor_commands"].copy())
        ep_data["actions"].append(action.copy())
        ep_data["rewards"].append(reward)

        # Trail
        if step_count % 3 == 0:
            trail.append(drone_pos.copy())

        # Render
        if render and viewer is not None and viewer.is_running():
            _update_trajectory_visuals(viewer, drone_pos, waypoints, wp_idx,
                                       trail, dist_to_wp)
            viewer.sync()
            time.sleep(base_env.dt * base_env.frame_skip)
        elif render and viewer is not None and not viewer.is_running():
            break

        # Termination
        if terminated:
            print(f"  TERMINATED at step {step_count} (out of bounds)")
            done = True
        elif truncated:
            print(f"  Max steps ({effective_max}) reached")
            done = True

        # Status
        if step_count % 200 == 0:
            print(f"  Step {step_count}: wp={wp_idx}/{num_waypoints}, "
                  f"dist={dist_to_wp:.2f}m, laps={laps_completed}")

    if viewer is not None:
        viewer.close()
    env.close()

    # Summary
    print(f"\n=== Trajectory Evaluation Summary ===")
    print(f"Trajectory:        {trajectory}")
    print(f"Steps:             {step_count}")
    print(f"Waypoints reached: {waypoints_reached}")
    print(f"Laps completed:    {laps_completed}")
    print(f"Total reward:      {total_reward:.2f}")

    if len(ep_data["times"]) > 0:
        plot_episode(ep_data, episode_num=0,
                     save_dir=os.path.join(model_dir, "plots_trajectory"),
                     max_rate=max_rate)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained hover policy")
    parser.add_argument(
        "--model",
        type=str,
        default="./models_trained/20260213_081613/best_model.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate performance plots (saved to model directory)"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        choices=list(TRAJECTORY_GENERATORS.keys()),
        help="Trajectory type for waypoint tracking (enables trajectory mode)"
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.5,
        help="Distance between waypoints in meters (default: 0.5)"
    )
    parser.add_argument(
        "--reach-radius",
        type=float,
        default=0.25,
        help="Distance threshold to switch waypoints in meters (default: 0.25)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Max simulation steps for trajectory mode (default: 5000, 0=unlimited)"
    )
    args = parser.parse_args()

    if args.trajectory:
        evaluate_trajectory(
            model_path=args.model,
            trajectory=args.trajectory,
            spacing=args.spacing,
            reach_radius=args.reach_radius,
            max_steps=args.max_steps,
            render=not args.no_render,
        )
    else:
        evaluate(args.model, args.episodes, render=not args.no_render,
                 plot=args.plot)


if __name__ == "__main__":
    main()
