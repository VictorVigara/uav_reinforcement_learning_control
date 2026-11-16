#!/usr/bin/env python3
"""Closed-loop check: skybridge's altitude-hold PID, flown in the simulator.

    python sysid/validate_althold.py

The flight logs hold two altitude-hold segments. For each, the simulated drone
starts at the recorded height and climb rate, the PID from
skybridge/control/altitude.py (same gains as skybridge's configs/drone.yaml)
drives the throttle from the simulated camera, and the pilot's recorded roll
and pitch are replayed. The real hover throttle during the segment is taken
from the recorded throttle once it has settled.

A match here means the loop the laptop closes (delay, thrust slope, drag,
sensor lag) behaves like the real one, which open-loop replay cannot show.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from drones.constraints import skybridge_config_path           # noqa: E402
from envs import make_env                                       # noqa: E402
from fit_air65 import ANGLE_LIMIT, load_flights                # noqa: E402

RC_DT = 0.02
MAX_AHEAD = 0.2


class AltitudeHold:
    """skybridge.control.altitude.AltitudeHold, without the engage/disengage guards."""

    def __init__(self, kp, ki, kd, authority, i_limit, **_):
        self.kp, self.ki, self.kd, self.authority, self.i_limit = kp, ki, kd, authority, i_limit

    def engage(self, z: float, throttle: float) -> None:
        self.setpoint, self.trim, self.integral = z, throttle, 0.0

    def update(self, z: float, vz: float, dt: float) -> float:
        clamp = lambda v, lim: max(-lim, min(lim, v))
        error = self.setpoint - z
        delta = self.kp * error + self.integral - self.kd * vz
        if not (abs(delta) >= self.authority and error * delta > 0):
            self.integral = clamp(self.integral + self.ki * error * dt, self.i_limit)
        return self.trim + clamp(self.kp * error + self.integral - self.kd * vz, self.authority)


def segments(d, min_len: float = 10.0):
    t = d["ah_t"]
    for s in np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > 0.1) + 1):
        if t[s[-1]] - t[s[0]] >= min_len:
            yield s


def simulate(env, d, s, gains: dict, duration: float, seed: int = 0) -> dict:
    col = {n: d["ah"][s, i] for i, n in enumerate(d["ah_names"])}
    t = d["ah_t"][s]
    n = min(int(duration / RC_DT), len(s))
    settled = (t - t[0] > min(8.0, 0.5 * (t[-1] - t[0]))) & (t - t[0] < duration)
    hover = float(np.mean(col["throttle"][settled]))        # what held the real drone up
    z0, vz0 = col["z"][0], col["vz"][0]
    state = np.array([0, 0, 1.0, 0, 0, 0, 0, 0, vz0, 0, 0, 0], dtype=float)
    rc = lambda key, tt: np.deg2rad(ANGLE_LIMIT) * np.interp(tt, d["rc_t"], d[key])
    env.reset(seed=seed, options={"state": state, "target": [0, 0, 1.0],
                                  "command": [col["throttle"][0], rc("rc_roll", t[0]), rc("rc_pitch", t[0]), 0.0]})
    env.fc.thrust_scale = 1.0
    env.fc.thrust_scale = env.fc.hover_motor_thrust / env.fc.motor_thrust_for(hover) * (env._mass / env.spec.mass)
    env.fc.reset(throttle=col["throttle"][0])

    pid = AltitudeHold(**gains)
    pid.engage(1.0, float(col["throttle"][0]))      # sim height 1.0 stands for the recorded setpoint
    out = {"t": [], "z": [], "throttle": []}
    throttle = float(col["throttle"][0])
    for k in range(n):
        tt = t[0] + k * RC_DT
        _, m = env.step_command([throttle, rc("rc_roll", tt), rc("rc_pitch", tt), 0.0])
        z = m["pos"][2] + m["vel"][2] * min(m["age"], MAX_AHEAD)    # DroneEstimate.at(now)
        throttle = pid.update(z, m["vel"][2], RC_DT)
        out["t"].append((k + 1) * RC_DT); out["z"].append(z - 1.0 + col["setpoint"][0]); out["throttle"].append(throttle)
    out = {k: np.array(v) for k, v in out.items()}
    out.update(rec_t=t[:n] - t[0], rec_z=col["z"][:n], rec_throttle=col["throttle"][:n],
               setpoint=float(col["setpoint"][0]), hover=hover)
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path, _ = skybridge_config_path("air65")
    gains = yaml.safe_load(path.read_text())["control"]["controllers"]["altitude_hold"]["params"]
    env = make_env("air65", randomize=False, noise=True)
    runs = [(name, simulate(env, d, s, gains, duration=30.0))
            for name, d in load_flights().items() if "ah_t" in d for s in segments(d)]
    fig, axes = plt.subplots(2, len(runs), figsize=(6.5 * len(runs), 6), squeeze=False, sharex="col")
    print(f"{'segment':30} {'hover thr':>9} {'height error std (mm): real / sim':>36} {'throttle std: real / sim':>28} {'z rmse (mm)':>12}")
    for j, (name, r) in enumerate(runs):
        sim_z = np.interp(r["rec_t"], r["t"], r["z"])
        sim_thr = np.interp(r["rec_t"], r["t"], r["throttle"])
        print(f"{name:30} {r['hover']:>9.3f} {1e3 * np.std(r['rec_z'] - r['setpoint']):>22.0f} / {1e3 * np.std(sim_z - r['setpoint']):<11.0f}"
              f"{np.std(r['rec_throttle']):>18.4f} / {np.std(sim_thr):<8.4f} {1e3 * np.sqrt(np.mean((sim_z - r['rec_z']) ** 2)):>10.0f}")
        ax = axes[0, j]
        ax.axhline(r["setpoint"], color="0.7", lw=1)
        ax.plot(r["rec_t"], r["rec_z"], label="recorded"); ax.plot(r["t"], r["z"], "--", label="sim")
        ax.set_ylabel("height (m)"); ax.set_title(name, fontsize=9); ax.legend()
        ax = axes[1, j]
        ax.axhline(r["hover"], color="0.7", lw=1)
        ax.plot(r["rec_t"], r["rec_throttle"], label="recorded"); ax.plot(r["t"], r["throttle"], "--", label="sim")
        ax.set_ylabel("throttle"); ax.set_xlabel("s after engaging")
    fig.suptitle("Altitude hold: skybridge's PID on the real drone vs in the simulator")
    fig.tight_layout()
    (HERE / "plots").mkdir(exist_ok=True)
    fig.savefig(HERE / "plots" / "althold.png", dpi=110)
    print(f"wrote {HERE / 'plots' / 'althold.png'}")


if __name__ == "__main__":
    main()
