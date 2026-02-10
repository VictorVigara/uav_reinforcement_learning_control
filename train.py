"""PPO training script for quadrotor hover control."""

import os
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs import HoverEnv
from datetime import datetime

def main():
    # Configuration
    total_timesteps = 10000000  # 10 million timesteps
    n_envs = 16
    checkpoint_freq = 50_000
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{run_timestamp}"
    model_dir = f"./models_trained/{run_timestamp}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Validate environment
    print("Validating environment...")
    env = HoverEnv()
    check_env(env)
    print("Environment validation passed!")
    env.close()

    # Create vectorized environments for parallel training
    print(f"Creating {n_envs} parallel environments...")
    vec_env = make_vec_env(HoverEnv, n_envs=n_envs)

    # PPO configuration
    policy_kwargs = {
        "net_arch": [256, 256],
        "activation_fn": nn.Tanh,
    }

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=800,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu",  # MLP policy is faster on CPU anyway
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=model_dir,
        name_prefix="hover_policy",
    )

    # Create evaluation environment
    eval_env = HoverEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    print("Monitor progress with: tensorboard --logdir ./logs")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
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
