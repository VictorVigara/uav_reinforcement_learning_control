"""Optuna hyperparameter optimization for PPO quadrotor hover control."""

import argparse
import os
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import optuna
import torch as th
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from drones import DEFAULT_DRONE, available_drones
from envs import DEFAULT_WRAPPER, make_env, RelPosActWrapper
from envs.rate_wrapper import RateControlWrapper

# ---------------------------------------------------------------------------
# Wrapper selection — change this to switch action/observation interface
# ---------------------------------------------------------------------------
# wrapper_cls = None                # bare HoverEnv (12D obs, direct torques)
# wrapper_cls = RelPosActWrapper    # 7D obs, direct torques
wrapper_cls = RateControlWrapper    # 12D obs, body-rate actions (CTBR)
drone_name = DEFAULT_DRONE          # set by --drone


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample PPO hyperparameters for one Optuna trial."""

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 800, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10, 20])
    gamma = 1.0 - trial.suggest_float("gamma_inv", 0.001, 0.05, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.1, log=True)
    net_arch_name = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    activation_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Store readable values for the report
    trial.set_user_attr("gamma", gamma)
    trial.set_user_attr("net_arch_name", net_arch_name)

    net_arch = {
        "small": [128, 128],
        "medium": [256, 256],
        "large": [512, 256],
    }[net_arch_name]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_name]

    # batch_size must divide n_steps * n_envs; enforce here
    # (n_envs is set globally; we just clamp batch_size)
    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }


# ---------------------------------------------------------------------------
# Callback that reports to Optuna and handles pruning
# ---------------------------------------------------------------------------

class TrialEvalCallback(EvalCallback):
    """EvalCallback that reports mean reward to Optuna and prunes bad trials."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _make_env():
    """Create the drone's environment, optionally wrapped by wrapper_cls."""
    return make_env(drone_name, wrapper=wrapper_cls)


def make_objective(n_timesteps: int, n_envs: int, n_eval_envs: int,
                   n_eval_episodes: int, eval_freq: int):
    """Create an objective closure with the given configuration."""

    def objective(trial: optuna.Trial) -> float:
        sampled_params = sample_ppo_params(trial)

        # Ensure batch_size divides rollout buffer size (n_steps * n_envs)
        rollout_size = sampled_params["n_steps"] * n_envs
        if rollout_size % sampled_params["batch_size"] != 0:
            # Pick the largest valid batch_size <= sampled one
            for bs in sorted([64, 128, 256, 512], reverse=True):
                if bs <= sampled_params["batch_size"] and rollout_size % bs == 0:
                    sampled_params["batch_size"] = bs
                    break
            else:
                sampled_params["batch_size"] = rollout_size  # single batch

        vec_env = make_vec_env(_make_env, n_envs=n_envs)
        eval_env = make_vec_env(_make_env, n_envs=n_eval_envs)

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=0,
            device="cpu",
            **sampled_params,
        )

        eval_callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq // n_envs,
            deterministic=True,
            verbose=0,
        )

        nan_encountered = False
        try:
            model.learn(n_timesteps, callback=eval_callback)
        except AssertionError:
            nan_encountered = True
        finally:
            vec_env.close()
            eval_env.close()

        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_best_params(study: optuna.Study) -> None:
    """Print the best trial in a copy-paste friendly format."""
    trial = study.best_trial
    print("\n" + "=" * 60)
    print("Best trial")
    print("=" * 60)
    print(f"  Mean reward : {trial.value:.2f}")
    print(f"  Trial number: {trial.number}")
    print("\n  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
    print("\n  User attrs:")
    for k, v in trial.user_attrs.items():
        print(f"    {k}: {v}")

    # Build a ready-to-paste config
    p = trial.params
    gamma = trial.user_attrs["gamma"]
    net_arch_name = trial.user_attrs["net_arch_name"]
    net_arch = {"small": [128, 128], "medium": [256, 256], "large": [512, 256]}[net_arch_name]
    act = "nn.Tanh" if p["activation_fn"] == "tanh" else "nn.ReLU"

    print("\n  Ready-to-use config for train.py:")
    print("  " + "-" * 40)
    print(f"""
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate={p['learning_rate']},
        n_steps={p['n_steps']},
        batch_size={p['batch_size']},
        n_epochs={p['n_epochs']},
        gamma={gamma},
        gae_lambda={p['gae_lambda']},
        clip_range={p['clip_range']},
        ent_coef={p['ent_coef']},
        policy_kwargs={{
            "net_arch": {net_arch},
            "activation_fn": {act},
        }},
        verbose=1,
        device="cpu",
    )
""")


def main():
    global drone_name, wrapper_cls
    parser = argparse.ArgumentParser(description="Optuna HPO for PPO hover control")
    parser.add_argument("--n-trials", type=int, default=50, help="Max number of trials")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel Optuna jobs")
    parser.add_argument("--timeout", type=int, default=None, help="Total timeout in seconds (default: no limit)")
    wrapper_tag = wrapper_cls.__name__ if wrapper_cls else "base"
    default_study = f"ppo_hover_{wrapper_tag}"
    parser.add_argument("--study-name", type=str, default=default_study, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///optuna.db)")
    parser.add_argument("--n-timesteps", type=int, default=500_000, help="Training steps per trial")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel training envs per trial")
    parser.add_argument("--drone", default=DEFAULT_DRONE, choices=available_drones(),
                        help="Drone from the registry in drones/ (default: %(default)s)")
    args = parser.parse_args()

    # Other drones train with their own default wrapper and get their own study.
    if args.drone != DEFAULT_DRONE:
        drone_name = args.drone
        wrapper_cls = DEFAULT_WRAPPER.get(args.drone)
        if args.study_name == default_study:
            args.study_name = f"ppo_hover_{args.drone}"

    # Faster single-threaded PyTorch
    th.set_num_threads(1)

    n_evaluations = 10
    eval_freq = args.n_timesteps // n_evaluations

    sampler = TPESampler(n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=n_evaluations // 3)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    objective = make_objective(
        n_timesteps=args.n_timesteps,
        n_envs=args.n_envs,
        n_eval_envs=5,
        n_eval_episodes=10,
        eval_freq=eval_freq,
    )

    print(f"Starting Optuna study '{args.study_name}'")
    print(f"  Wrapper  : {wrapper_cls.__name__ if wrapper_cls else 'None (bare HoverEnv)'}")
    print(f"  Trials    : {args.n_trials}")
    print(f"  Timeout   : {f'{args.timeout}s' if args.timeout else 'none'}")
    print(f"  Steps/trial: {args.n_timesteps}")
    print(f"  Envs/trial : {args.n_envs}")
    print()

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted.")

    print(f"\nFinished trials: {len(study.trials)}")

    if study.best_trial is not None:
        print_best_params(study)

    # Save results
    csv_path = f"study_results_{args.study_name}.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
