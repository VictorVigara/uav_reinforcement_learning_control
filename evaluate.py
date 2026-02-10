"""Evaluation and visualization script for trained hover policy."""

import argparse
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs import HoverEnv


def plot_episode(data: dict, episode_num: int, save_dir: str = "./plots"):
    """Generate performance plots for a single evaluation episode.

    Args:
        data: Dict with keys: times, positions, targets, attitudes,
              velocities, angular_velocities, motor_commands, rewards.
        episode_num: Episode number (for title and filename).
        save_dir: Directory to save plot images.
    """
    os.makedirs(save_dir, exist_ok=True)

    t = np.array(data["times"])
    pos = np.array(data["positions"])
    tgt = np.array(data["targets"])
    att = np.rad2deg(np.array(data["attitudes"]))
    vel = np.array(data["velocities"])
    ang_vel = np.rad2deg(np.array(data["angular_velocities"]))
    motors = np.array(data["motor_commands"])
    rewards = np.array(data["rewards"])
    pos_err = np.linalg.norm(pos - tgt, axis=1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
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
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular vel (deg/s)")
    ax.set_title("Angular Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Motor commands
    ax = axes[2, 1]
    for i in range(4):
        ax.plot(t, motors[:, i], label=f"M{i+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust (N)")
    ax.set_title("Motor Commands")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(save_dir, f"episode_{episode_num}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {filepath}")


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

    # Create environment
    env = HoverEnv()

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        # Data collection for plots
        ep_data = {
            "times": [], "positions": [], "targets": [],
            "attitudes": [], "velocities": [], "angular_velocities": [],
            "motor_commands": [], "rewards": [],
        }

        # Setup viewer for this episode
        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

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
                ep_data["times"].append(step_count * env.dt * env.frame_skip)
                ep_data["positions"].append(state[:3].copy())
                ep_data["targets"].append(info["target"].copy())
                ep_data["attitudes"].append(state[3:6].copy())
                ep_data["velocities"].append(state[6:9].copy())
                ep_data["angular_velocities"].append(state[9:12].copy())
                ep_data["motor_commands"].append(info["motor_commands"].copy())
                ep_data["rewards"].append(reward)

            # Render
            if render and viewer is not None and viewer.is_running():
                viewer.sync()
                # Sync with real time
                time.sleep(env.dt * env.frame_skip)
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
            plot_episode(ep_data, ep + 1)

    env.close()

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")


def interactive_control(render: bool = True):
    """Run environment with hover thrust (no learned policy).

    Useful for testing that the simulation works correctly.
    """
    env = HoverEnv()
    obs, info = env.reset()

    # Calculate hover thrust
    # Total mass = 0.2 (core) + 4*0.025 (arms) + 4*0.025 (thrusters) = 0.4 kg
    # Hover thrust per motor = (mass * g) / (4 * gear_ratio) = (0.4 * 9.81) / (4 * 2) = 0.49
    # In [-1, 1] space: 0.49 -> 2*0.49 - 1 = -0.02
    hover_thrust_normalized = 2 * 0.49 - 1  # Convert [0,1] to [-1,1]

    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    print("Running with constant hover thrust...")
    print(f"Hover thrust (normalized): {hover_thrust_normalized:.3f}")
    print("Press Ctrl+C to stop")

    try:
        step = 0
        while True:
            # Constant hover action: [thrust, roll_rate, pitch_rate, yaw_rate]
            # All in [-1, 1] space
            action = np.array([hover_thrust_normalized, 0.0, 0.0, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if render and viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(env.dt * env.frame_skip)

            step += 1
            if step % 100 == 0:
                state = info["state"]
                print(f"Step {step}: z={state[2]:.3f}, reward={reward:.2f}")

            if terminated or truncated:
                print(f"Episode ended after {step} steps")
                obs, info = env.reset()
                step = 0

    except KeyboardInterrupt:
        print("\nStopped by user")

    if viewer is not None:
        viewer.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate or test hover policy")
    parser.add_argument(
        "--model",
        type=str,
        default="./models_trained/20260209_182641/best_model.zip",
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
        help="Generate performance plots (saved to ./plots/)"
    )
    parser.add_argument(
        "--test-hover",
        action="store_true",
        help="Test with constant hover thrust (no learned policy)"
    )

    args = parser.parse_args()

    if args.test_hover:
        interactive_control(render=not args.no_render)
    else:
        evaluate(args.model, args.episodes, render=not args.no_render,
                 plot=args.plot)


if __name__ == "__main__":
    main()
