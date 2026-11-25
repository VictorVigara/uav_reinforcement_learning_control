#!/usr/bin/env python3
"""Score an altitude-hold policy against a hand-tuned PI, and plot an episode.

    python evaluate_althold.py models_trained/air65_althold_<stamp>
    python evaluate_althold.py models_trained/air65_althold_<stamp> --episodes 50 --plot 3

Conditions:
    nominal     the identified drone, sensor noise on, calm-to-busy pilot
    randomized  full training ranges
    hard        full ranges, the pilot at full aggressiveness every episode
Metrics are on the true altitude: survival (no floor/ceiling/tilt), RMS error,
95th percentile of |error|, and the mean |change| of throttle per tick (chatter).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drones import spec_from_params  # noqa: E402
from envs import AltHoldEnv  # noqa: E402


def find_model(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("best_model.zip", "hover_policy_final.zip"):
        if (path / name).exists():
            return path / name
    raise FileNotFoundError(f"no best_model.zip or hover_policy_final.zip in {path}")


def build_env(run_dir: Path, **kwargs) -> AltHoldEnv:
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    drone = config.get("drone", "air65")
    if config.get("drone_params"):
        drone = spec_from_params(drone, config["drone_params"])
    return AltHoldEnv(drone, **kwargs)


def pi_policy(env: AltHoldEnv):
    """PD on the measured error and climb, plus the integral and a tilt feedforward from the sticks."""
    o = env.observer
    hover = env.fc.p.hover_throttle

    def act(obs):
        err, vz, integ, tilt = obs[0] * o.err_scale, obs[1] * o.vz_scale, obs[2] * o.int_limit, obs[7] * o.tilt_full
        return np.array([o.action_for(hover * (1 + 0.5 * tilt) + 0.15 * err - 0.08 * vz + 0.25 * integ)])
    return act


def run(env: AltHoldEnv, act, episodes: int, seed: int, hard: bool = False, keep: int = 0):
    out, kept = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if hard:
            env.pilot = type(env.pilot)(env.np_random, env._pilot_cfg, angle_full=env.angle_full,
                                        dt=env.control_dt, aggressiveness=1.0)
        log = {k: [] for k in ("t", "z", "target", "z_meas", "age", "throttle", "roll", "pitch", "yaw", "true_roll", "true_pitch")}
        terminated = False
        for k in range(env.max_episode_steps):
            obs, _, terminated, truncated, info = env.step(act(obs))
            s, cmd, m = info["state"], info["command"], info["measured"]
            for key, v in zip(log, (k * env.control_dt, s[2], info["target_true_z"], m[0], m[2], cmd[0],
                                    cmd[1], cmd[2], cmd[3], s[3], s[4])):
                log[key].append(float(v))
            if terminated or truncated:
                break
        err = np.array(log["z"]) - np.array(log["target"])
        out.append((not terminated, math.sqrt(float(np.mean(err ** 2))), float(np.percentile(np.abs(err), 95)),
                    float(np.mean(np.abs(np.diff(log["throttle"])))) if len(err) > 1 else 0.0))
        if ep < keep:
            kept.append({k: np.array(v) for k, v in log.items()})
    a = np.array(out)
    return {"survival": a[:, 0].mean(), "rms": float(np.median(a[:, 1])), "p95": float(np.median(a[:, 2])),
            "chatter": float(np.median(a[:, 3]))}, kept


def plot(log: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = log["t"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, log["z"], label="true z")
    ax[0].plot(t, log["z_meas"], lw=0.8, alpha=0.7, label="camera z")
    ax[0].plot(t, log["target"], "k--", label="target")
    ax[0].set_ylabel("altitude, m")
    ax[0].legend(loc="upper right")
    ax[1].plot(t, log["throttle"], label="throttle (policy)")
    ax[1].set_ylabel("throttle")
    ax1b = ax[1].twinx()
    ax1b.plot(t, log["age"], color="tab:red", lw=0.6, alpha=0.6)
    ax1b.set_ylabel("camera age, s", color="tab:red")
    ax[2].plot(t, np.degrees(log["roll"]), label="roll stick")
    ax[2].plot(t, np.degrees(log["pitch"]), label="pitch stick")
    ax[2].plot(t, np.degrees(log["true_roll"]), ":", label="true roll")
    ax[2].plot(t, np.degrees(log["true_pitch"]), ":", label="true pitch")
    ax[2].set_ylabel("deg")
    ax[2].set_xlabel("s")
    ax[2].legend(loc="upper right", ncol=4)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="run directory (models_trained/...) or a .zip")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--plot", type=int, default=2, help="episodes to plot per condition")
    ap.add_argument("--no-baseline", action="store_true", help="skip the PI comparison")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model_path = find_model(args.run)
    run_dir = model_path.parent
    model = PPO.load(model_path, device="cpu")
    policy = lambda obs: model.predict(obs, deterministic=True)[0]
    out_dir = Path("plots") / f"althold_{run_dir.name}"
    os.makedirs(out_dir, exist_ok=True)

    conditions = {"nominal": dict(randomize=False), "randomized": dict(randomize=True),
                  "hard": dict(randomize=True)}
    print(f"{'condition':11} {'controller':7} {'survive':>8} {'rms m':>7} {'p95 m':>7} {'chatter':>8}")
    for name, kwargs in conditions.items():
        env = build_env(run_dir, **kwargs)
        controllers = {"policy": policy} if args.no_baseline else {"policy": policy, "PI": pi_policy(env)}
        for label, act in controllers.items():
            stats, logs = run(env, act, args.episodes, args.seed, hard=name == "hard",
                              keep=args.plot if label == "policy" else 0)
            print(f"{name:11} {label:7} {stats['survival']:8.2f} {stats['rms']:7.3f} {stats['p95']:7.3f} "
                  f"{stats['chatter']:8.4f}")
            for i, log in enumerate(logs):
                plot(log, out_dir / f"{name}_{i}.png", f"{run_dir.name}: {name}, episode {i}")
    print(f"plots in {out_dir}/")


if __name__ == "__main__":
    main()
