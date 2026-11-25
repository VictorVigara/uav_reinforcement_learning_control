#!/usr/bin/env python3
"""Export an altitude-hold policy for skybridge: numpy weights, no torch needed.

    python export_policy.py models_trained/air65_althold_<stamp>
    python export_policy.py models_trained/air65_althold_<stamp> -o ~/skybridge/models/althold.npz

The bundle (.npz) holds:
    layer{i}_w, layer{i}_b   the actor MLP, input to output; hidden layers use `activation`
    meta                     JSON: observer settings, throttle range, the run it came from
    parity_*                 a few simulated episodes: what the observer was given, the
                             observation it built and the action the policy chose. skybridge
                             replays these when it loads the bundle and refuses it if its own
                             observer or MLP gives anything else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_althold import build_env, find_model  # noqa: E402

FORMAT = "althold-policy/1"


def actor_layers(model) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """The deterministic actor as (weight, bias) pairs and the hidden activation."""
    import torch.nn as nn
    policy = model.policy
    if not isinstance(policy.features_extractor, nn.Flatten) and type(policy.features_extractor).__name__ != "FlattenExtractor":
        raise ValueError("only the default flatten feature extractor is supported")
    layers, acts = [], set()
    for module in policy.mlp_extractor.policy_net:
        if isinstance(module, nn.Linear):
            layers.append(module)
        elif isinstance(module, (nn.ReLU, nn.Tanh)):
            acts.add(type(module).__name__.lower())
        else:
            raise ValueError(f"unsupported layer {module}")
    if len(acts) > 1:
        raise ValueError(f"mixed activations {acts}")
    layers.append(policy.action_net)
    pairs = [(m.weight.detach().cpu().numpy().astype(np.float32), m.bias.detach().cpu().numpy().astype(np.float32))
             for m in layers]
    return pairs, (acts.pop() if acts else "relu")


def mlp(pairs, activation: str, obs: np.ndarray) -> np.ndarray:
    """The same forward pass skybridge runs."""
    x = np.asarray(obs, dtype=np.float32)
    act = (lambda v: np.maximum(v, 0.0)) if activation == "relu" else np.tanh
    for i, (w, b) in enumerate(pairs):
        x = w @ x + b
        if i < len(pairs) - 1:
            x = act(x)
    return np.clip(x, -1.0, 1.0)


def record_parity(env, predict, episodes: int, steps: int, seed: int):
    """Run the env and log every observer call: inputs, observation, action."""
    obs_dim = env.observation_space.shape[0]
    inputs, observations, actions = [], [], []
    observer = env.observer
    engage, observe = observer.engage, observer.observe

    def rec_engage(z, throttle, sticks):
        inputs.append([0.0, z, throttle, 0.0, *np.asarray(sticks, dtype=float)])
        observations.append(np.full(obs_dim, np.nan))
        actions.append(np.nan)
        engage(z, throttle, sticks)

    def rec_observe(z, vz, age, sticks):
        out = observe(z, vz, age, sticks)
        inputs.append([1.0, z, vz, age, *np.asarray(sticks, dtype=float)])
        observations.append(out.astype(np.float64))
        actions.append(float(predict(out)))
        return out

    observer.engage, observer.observe = rec_engage, rec_observe
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            for _ in range(steps):
                obs, _, terminated, truncated, _ = env.step(np.array([actions[-1]], dtype=np.float32))
                if terminated or truncated:
                    break
    finally:
        observer.engage, observer.observe = engage, observe
    return np.array(inputs), np.array(observations), np.array(actions)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="run directory or .zip")
    ap.add_argument("-o", "--out", type=Path, help="bundle path (default: <run dir>/althold_policy.npz)")
    ap.add_argument("--episodes", type=int, default=3, help="parity episodes")
    ap.add_argument("--steps", type=int, default=300, help="parity steps per episode")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model_path = find_model(args.run)
    run_dir = model_path.parent
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("task") != "althold":
        sys.exit(f"{run_dir} is not an althold run (task: {config.get('task')!r})")
    model = PPO.load(model_path, device="cpu")
    pairs, activation = actor_layers(model)

    env = build_env(run_dir, randomize=True)
    observer_cfg = env.observer.config()
    if observer_cfg != config.get("observer"):
        sys.exit("the observer this code builds differs from the one the run was trained with:\n"
                 f"  now: {observer_cfg}\n  run: {config.get('observer')}")

    # The numpy forward pass must agree with SB3 before anything is exported.
    sb3 = lambda o: float(model.predict(o, deterministic=True)[0][0])
    inputs, observations, actions = record_parity(env, sb3, args.episodes, args.steps, seed=12345)
    rows = inputs[:, 0] == 1
    ours = np.array([mlp(pairs, activation, o)[0] for o in observations[rows]])
    worst = float(np.max(np.abs(ours - actions[rows])))
    if worst > 1e-4:
        sys.exit(f"numpy actor disagrees with SB3 by {worst:.2e}")

    meta = {
        "format": FORMAT,
        "run": run_dir.name,
        "model": model_path.name,
        "drone": config.get("drone"),
        "activation": activation,
        "observer": observer_cfg,
        "observation_names": config.get("observation_names"),
        "command_limits": config.get("command_limits"),
        "parity_tolerance": {"obs": 1e-5, "action": 1e-4},
    }
    out = args.out or run_dir / "althold_policy.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"layer{i}_{k}": v for i, (w, b) in enumerate(pairs) for k, v in (("w", w), ("b", b))}
    np.savez(out, meta=np.array(json.dumps(meta)), parity_inputs=inputs, parity_obs=observations,
             parity_actions=actions, **arrays)
    sizes = " -> ".join(str(w.shape[1]) for w, _ in pairs) + f" -> {pairs[-1][0].shape[0]}"
    print(f"wrote {out}")
    print(f"  actor {sizes} ({activation}), numpy vs SB3 max diff {worst:.1e}")
    print(f"  parity: {int(rows.sum())} observer steps over {args.episodes} episodes")
    print(f"  sticks: {observer_cfg['angle_full_deg']:g} deg / {observer_cfg['yaw_full_dps']:g} deg/s at full deflection")


if __name__ == "__main__":
    main()
