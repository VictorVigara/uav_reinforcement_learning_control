"""PPO training script for quadrotor hover control."""

import argparse
import os
import json
import inspect
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from drones import DEFAULT_DRONE, available_drones, load_drone
from envs import DEFAULT_WRAPPER, TASKS, make_env as make_drone_env
from datetime import datetime


class CurriculumCallback(BaseCallback):
    """Widens the randomization (env.set_difficulty) from `start` to 1 over the first
    `until` share of training, then keeps it at 1."""

    def __init__(self, total_timesteps: int, start: float, until: float):
        super().__init__()
        self.total, self.start, self.until = total_timesteps, start, until
        self.level = None

    def _set(self, level: float):
        self.training_env.env_method("set_difficulty", level)
        self.level = level
        self.logger.record("curriculum/difficulty", level)

    def _on_training_start(self):
        self._set(self.start)

    def _on_step(self) -> bool:
        frac = min(self.num_timesteps / (self.until * self.total), 1.0) if self.until > 0 else 1.0
        level = round(self.start + (1.0 - self.start) * frac, 2)
        if level != self.level:
            self._set(level)
        return True


def main():
    parser = argparse.ArgumentParser(description="Train a PPO hover policy")
    parser.add_argument("--drone", default=DEFAULT_DRONE, choices=available_drones(),
                        help="Drone from the registry in drones/ (default: %(default)s)")
    parser.add_argument("--task", default="hover", choices=["hover", *TASKS],
                        help="hover: fly to a target (default). althold: the pilot flies roll/pitch/yaw, "
                             "the policy holds altitude with throttle")
    parser.add_argument("--curriculum", type=float, nargs=2, default=[0.2, 0.4], metavar=("START", "UNTIL"),
                        help="althold: randomization starts at START x its ranges and reaches the full ranges "
                             "after UNTIL x the timesteps (default: %(default)s; 1 0 turns it off)")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--device", default="cpu",
                        help="Torch device for the policy: cpu, cuda or auto (default: %(default)s)")
    args = parser.parse_args()

    # Configuration
    spec = load_drone(args.drone)
    total_timesteps = args.timesteps
    n_envs = args.n_envs
    checkpoint_freq = 50_000
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_timestamp if spec.name == DEFAULT_DRONE else f"{spec.name}_{run_timestamp}"
    if args.task != "hover":
        run_name = f"{spec.name}_{args.task}_{run_timestamp}"
    log_dir = f"./logs/{run_name}"
    model_dir = f"./models_trained/{run_name}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Wrapper on top of the drone's environment: see DEFAULT_WRAPPER in envs/__init__.py
    wrapper_cls = DEFAULT_WRAPPER.get(spec.name) if args.task == "hover" else None
    wrapper_name = wrapper_cls.__name__ if wrapper_cls else "none"

    def make_env():
        return make_drone_env(spec, wrapper=wrapper_cls, task=args.task)

    # Validate environment
    print("Validating environment...")
    check_env(make_env())
    print("Environment validation passed!")

    # Create vectorized environments for parallel training
    print(f"Creating {n_envs} parallel environments...")
    # Environments that simulate a flight controller are slow enough per step
    # to be worth one process each.
    vec_env_cls = SubprocVecEnv if spec.frame_skip > 1 and n_envs > 1 else None
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=vec_env_cls)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=0.0001547818138087132,
        n_steps=1024,
        batch_size=128,
        n_epochs=20,
        gamma=0.9906345854291289,
        gae_lambda=0.9079441765099094,
        clip_range=0.19153175856282983,
        ent_coef=9.106557393423481e-05,
        policy_kwargs={
            "net_arch": [128, 128],
            "activation_fn": nn.ReLU,
        },
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=model_dir,
        name_prefix="hover_policy",
    )

    # Create evaluation environment
    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Save run config: reward function, observation space, and hyperparameters
    env_tmp = make_env()
    base_env = env_tmp.unwrapped
    config = {
        "timestamp": run_timestamp,
        "drone": spec.name,
        "task": args.task,
        "observer": base_env.observer.config() if hasattr(base_env, "observer") else None,
        "curriculum": args.curriculum if hasattr(base_env, "set_difficulty") else None,
        "drone_params": spec.params,
        "command_limits": base_env.limits.as_dict() if hasattr(base_env, "limits") else None,
        "action_bounds": {
            "low": base_env._action_bounds.low.tolist(),
            "high": base_env._action_bounds.high.tolist(),
        },
        "control_dt": float(base_env.dt * base_env.frame_skip),
        "observation_names": list(getattr(base_env, "OBS_NAMES", [])) or None,
        "action_names": list(getattr(base_env, "ACTION_NAMES", [])) or None,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "wrapper": wrapper_name,
        "wrapper_source": inspect.getsource(wrapper_cls) if wrapper_cls else None,
        "reward_function": inspect.getsource(base_env._get_reward),
        "observation_function": inspect.getsource(base_env._get_obs),
        "observation_bounds": {
            "low": base_env._obs_bounds.low.tolist(),
            "high": base_env._obs_bounds.high.tolist(),
        },
        "state_bounds": {
            "low": base_env._state_bounds.low.tolist(),
            "high": base_env._state_bounds.high.tolist(),
        },
        "target_pos_bounds": {
            "low": base_env._target_pos_bounds.low.tolist(),
            "high": base_env._target_pos_bounds.high.tolist(),
        },
        "ppo": {
            "learning_rate": model.learning_rate,
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
            "clip_range": float(model.clip_range(1)),
            "ent_coef": model.ent_coef,
            "net_arch": [128, 128],
            "activation_fn": "ReLU",
        },
    }
    env_tmp.close()
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved to {config_path}")

    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    print("Monitor progress with: tensorboard --logdir ./logs")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
                 + ([CurriculumCallback(total_timesteps, *args.curriculum)] if hasattr(base_env, "set_difficulty") else []),
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(model_dir, "hover_policy_final")
    model.save(final_path)
    print(f"Training complete! Model saved to {final_path}.zip")

    # Cleanup
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
