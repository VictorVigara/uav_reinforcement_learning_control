#!/usr/bin/env python3
"""Replay recorded flights through the simulator and compare.

    python sysid/validate_replay.py                 # table + plots in sysid/plots/
    python sysid/validate_replay.py --tune          # also search the link/controller
                                                    # parameters that best match the logs

Each flight is cut into short windows. A window starts the simulated drone in
the recorded state, sends it the recorded RC commands, and compares what the
laptop would have seen with what it did see:

    attitude   simulated telemetry (delayed) vs recorded telemetry, by arrival time
    velocity   simulated tracker velocity vs recorded, by capture time

The windows are open loop, and the real drone is never trimmed exactly level:
at zero commanded tilt it still accelerates a little. That shows up as a
velocity error growing linearly in time. Its slope is reported as the "trim"
(as acceleration and as the equivalent tilt), and removed before scoring.

Frames: the camera world is x right, y forward, z up; the simulator's is
x forward, y left, z up, so sim = (y_cam, -x_cam, z_cam). Heading is
psi0 - (FC yaw - FC yaw at the flight's first window), psi0 from fit_air65.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from envs import make_env                                       # noqa: E402
from fit_air65 import ANGLE_LIMIT, load_flights, windows       # noqa: E402

RC_DT = 0.02
YAW_FULL_STICK = np.deg2rad(670.0)      # rc_rate 67, linear
TUNABLE = {     # name -> (where it lives, candidate values)
    "action_delay": ("link", [0.03, 0.045, 0.06, 0.075, 0.09, 0.11]),
    "attitude_latency": ("link", [0.03, 0.05, 0.07, 0.09, 0.11, 0.13]),
    "angle_ff": ("fc", [0.0, 0.4, 0.7, 0.85, 1.0]),
    "ff_tau": ("fc", [0.01, 0.03, 0.06, 0.1]),
    "level_kp": ("fc", [3.5, 5.0, 7.0, 10.0]),
    "velocity_tau": ("link", [0.01, 0.03, 0.05, 0.08, 0.12]),
    "drag_xy": ("aero", [0.2, 0.35, 0.5, 0.7, 1.0]),
    "drag_z": ("aero", [0.1, 0.25, 0.4, 0.6]),
}


def cam_to_sim(v: np.ndarray) -> np.ndarray:
    return np.stack([v[..., 1], -v[..., 0], v[..., 2]], axis=-1)


def set_param(env, name: str, value: float) -> None:
    where = TUNABLE[name][0]
    if where == "link":
        env._link[name] = value
    elif where == "aero":
        env._aero[name] = value
    else:
        setattr(env.fc.p, name, value)


def get_param(env, name: str):
    where = TUNABLE[name][0]
    value = {"link": env._link, "aero": env._aero}[where][name] if where != "fc" else getattr(env.fc.p, name)
    return float(np.mean(value))


def replay_window(env, d, w, psi0: float, yaw_ref: float, hover_u: float) -> dict:
    """Simulate one window. Returns simulated and recorded series on the RC grid."""
    et, rt = d["est_t"], d["rc_t"]
    t0, t1 = et[w[0]], et[w[-1]]
    grid = np.arange(t0, t1, RC_DT)
    fc_yaw = np.unwrap(np.deg2rad(d["att_yaw"]))
    heading = lambda t: psi0 - (np.interp(t, d["att_t"], fc_yaw) - yaw_ref)
    rc = lambda key, t: np.interp(t, rt, d[key])
    command = lambda t: np.array([rc("rc_throttle", t), np.deg2rad(ANGLE_LIMIT) * rc("rc_roll", t),
                                  np.deg2rad(ANGLE_LIMIT) * rc("rc_pitch", t), YAW_FULL_STICK * rc("rc_yaw", t)])

    pos0 = cam_to_sim(np.array([np.interp(t0, et, d["pos"][:, k]) for k in range(3)]))
    vel0 = cam_to_sim(np.array([np.interp(t0, et, d["vel"][:, k]) for k in range(3)]))
    lat = env._link["attitude_latency"]
    lat = float(np.mean(lat))
    roll0 = np.deg2rad(np.interp(t0 + lat, d["att_t"], d["att_roll"]))
    pitch0 = -np.deg2rad(np.interp(t0 + lat, d["att_t"], d["att_pitch"]))
    state = np.concatenate([[0.0, 0.0, 1.0], [roll0, pitch0, heading(t0)], vel0, np.zeros(3)])
    env.reset(seed=0, options={"state": state, "target": [0, 0, 1.0], "command": command(t0)})
    # Battery state for this flight: make the sim hover at the flight's hover throttle.
    env.fc.thrust_scale = 1.0
    env.fc.thrust_scale = env.fc.hover_motor_thrust / env.fc.motor_thrust_for(hover_u) * (env._mass / env.spec.mass)
    env.fc.reset(throttle=command(t0)[0], roll=command(t0)[1], pitch=command(t0)[2])

    sim = {"roll": [], "pitch": [], "vel": [], "true_roll": [], "true_pitch": []}
    for t in grid:
        truth, m = env.step_command(command(t))
        sim["roll"].append(m["roll"]); sim["pitch"].append(m["pitch"])
        sim["vel"].append(env._vel_filtered.copy())
        sim["true_roll"].append(truth[3]); sim["true_pitch"].append(truth[4])
        if not np.isfinite(truth).all() or abs(truth[3]) > 1.2 or abs(truth[4]) > 1.2:
            break
    n = len(sim["roll"])
    tt = grid[:n] + RC_DT           # the time each step ends
    out = {k: np.array(v) for k, v in sim.items()}
    out["t"] = tt - t0
    out["rec_roll"] = np.deg2rad(np.interp(tt, d["att_t"], d["att_roll"]))
    out["rec_pitch"] = -np.deg2rad(np.interp(tt, d["att_t"], d["att_pitch"]))
    out["rec_vel"] = cam_to_sim(np.stack([np.interp(tt, et, d["vel"][:, k]) for k in range(3)], axis=-1))
    out["cmd_roll"] = np.deg2rad(ANGLE_LIMIT) * rc("rc_roll", tt)
    out["cmd_pitch"] = np.deg2rad(ANGLE_LIMIT) * rc("rc_pitch", tt)
    out["complete"] = n == len(grid)
    return out


def _detrend(t, e):
    return e - np.polyval(np.polyfit(t, e, 1), t) if len(t) > 2 else e


def trim_accel(r: dict) -> np.ndarray:
    """Constant acceleration (forward, left, up) the sim lacks in this window, m/s^2."""
    return np.array([np.polyfit(r["t"], r["rec_vel"][:, k] - r["vel"][:, k], 1)[0] for k in range(3)])


def score(runs: list[dict]) -> dict:
    att_err, att_ref, v_err, v_ref, vz_err, vz_ref = [], [], [], [], [], []
    trim = np.array([np.hypot(*trim_accel(r)[:2]) for r in runs])
    for r in runs:
        for k in ("roll", "pitch"):
            att_err.append(r[k] - r["rec_" + k]); att_ref.append(r["rec_" + k] - r["rec_" + k].mean())
        for k in range(2):
            v_err.append(_detrend(r["t"], r["vel"][:, k] - r["rec_vel"][:, k]))
            v_ref.append(r["rec_vel"][:, k] - r["rec_vel"][:, k].mean())
        vz_err.append(_detrend(r["t"], r["vel"][:, 2] - r["rec_vel"][:, 2]))
        vz_ref.append(r["rec_vel"][:, 2] - r["rec_vel"][:, 2].mean())
    rms = lambda parts: float(np.sqrt(np.mean(np.concatenate(parts) ** 2)))
    return {"att_rmse_deg": np.rad2deg(rms(att_err)), "att_std_deg": np.rad2deg(rms(att_ref)),
            "vxy_rmse": rms(v_err), "vxy_std": rms(v_ref), "vz_rmse": rms(vz_err), "vz_std": rms(vz_ref),
            "trim_median": float(np.median(trim)), "trim_p90": float(np.percentile(trim, 90)),
            "windows": len(runs), "incomplete": sum(not r["complete"] for r in runs)}


def cost(s: dict) -> float:
    return s["att_rmse_deg"] / s["att_std_deg"] + s["vxy_rmse"] / s["vxy_std"] + s["vz_rmse"] / s["vz_std"]


def run_all(env, flights, fit, length: float, stride: int = 1) -> dict[str, list[dict]]:
    out = {}
    for name, d in flights.items():
        f = fit["flights"][name]
        if "error" in f["horizontal"]:
            continue
        wins = windows(d, length)[::stride]
        if not wins:
            continue
        fc_yaw = np.unwrap(np.deg2rad(d["att_yaw"]))
        yaw_ref = np.interp(d["est_t"][windows(d, 3.0)[0][0]], d["att_t"], fc_yaw)
        out[name] = [replay_window(env, d, w, np.deg2rad(f["horizontal"]["psi0_deg"]), yaw_ref,
                                   f["vertical"]["hover_u"]) for w in wins]
    return out


def tune(env, flights, fit, length: float, names: list[str], rounds: int = 2) -> dict[str, float]:
    total = lambda: cost(score([r for runs in run_all(env, flights, fit, length, stride=2).values() for r in runs]))
    best = total()
    print(f"tuning on {list(flights)}; start cost {best:.3f}")
    for _ in range(rounds):
        for name in names:
            keep = get_param(env, name)
            for value in TUNABLE[name][1]:
                set_param(env, name, value)
                c = total()
                if c < best - 1e-4:
                    best, keep = c, value
            set_param(env, name, keep)
            print(f"  {name:17} -> {keep:<6g} cost {best:.3f}")
    return {name: get_param(env, name) for name in names}


def plot(runs: dict[str, list[dict]], out_dir: Path, per_flight: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rs in runs.items():
        # the windows with the most stick activity say the most
        rs = sorted(rs, key=lambda r: -(np.std(r["cmd_roll"]) + np.std(r["cmd_pitch"])))[:per_flight]
        fig, axes = plt.subplots(5, len(rs), figsize=(5 * len(rs), 11), squeeze=False, sharex="col")
        for j, r in enumerate(rs):
            t = r["t"]
            for i, k in enumerate(("roll", "pitch")):
                ax = axes[i, j]
                ax.plot(t, np.rad2deg(r["cmd_" + k]), color="0.6", lw=1, label="command")
                ax.plot(t, np.rad2deg(r["rec_" + k]), color="C0", label="recorded telemetry")
                ax.plot(t, np.rad2deg(r[k]), color="C1", ls="--", label="sim telemetry")
                ax.set_ylabel(f"{k} (deg)")
            for i, k in enumerate(("forward", "left", "up")):
                ax = axes[2 + i, j]
                drift = np.polyval(np.polyfit(t, r["rec_vel"][:, i] - r["vel"][:, i], 1), t)
                ax.plot(t, r["rec_vel"][:, i], color="C0", label="recorded")
                ax.plot(t, r["vel"][:, i] + drift, color="C1", ls="--", label="sim, trim removed")
                ax.plot(t, r["vel"][:, i], color="C1", ls=":", lw=0.8, label="sim, raw")
                ax.set_ylabel(f"v {k} (m/s)")
            axes[0, j].legend(fontsize=7); axes[2, j].legend(fontsize=7)
            axes[-1, j].set_xlabel("s")
        fig.suptitle(f"{name}: recorded vs simulated (open-loop replay)")
        fig.tight_layout()
        fig.savefig(out_dir / f"replay_{name}.png", dpi=110)
        plt.close(fig)
        print(f"wrote {out_dir / f'replay_{name}.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drone", default="air65")
    ap.add_argument("--length", type=float, default=3.0, help="window length, seconds")
    ap.add_argument("--tune", action="store_true", help="search link/controller parameters")
    ap.add_argument("--params", nargs="*", default=list(TUNABLE), help="parameters to tune")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    fit = json.loads((HERE / "air65_fit.json").read_text())
    flights = load_flights()
    env = make_env(args.drone, randomize=False, noise=False)
    for name in TUNABLE:                 # ranges -> their midpoints, so they can be tuned
        set_param(env, name, get_param(env, name))

    if args.tune:
        # Flights where the pilot moved the sticks; a flight spent in altitude hold says little.
        active = {n: d for n, d in flights.items()
                  if "error" not in fit["flights"][n]["horizontal"] and fit["flights"][n]["horizontal"]["v_std"] > 0.1}
        tuned = tune(env, active, fit, args.length, args.params)
        print("tuned:", json.dumps(tuned, indent=2))
        (HERE / "air65_tuned.json").write_text(json.dumps(tuned, indent=2))

    runs = run_all(env, flights, fit, args.length)
    print(f"\n{'flight':28} {'windows':>7} {'attitude rmse / spread (deg)':>30} {'v_xy rmse / spread':>22} "
          f"{'v_z rmse / spread':>22} {'trim median / p90 (m/s2, deg)':>32}")
    for name, rs in runs.items():
        s = score(rs)
        print(f"{name:28} {s['windows']:>7} {s['att_rmse_deg']:>14.2f} / {s['att_std_deg']:<13.2f} "
              f"{s['vxy_rmse']:>10.3f} / {s['vxy_std']:<9.3f} {s['vz_rmse']:>10.3f} / {s['vz_std']:<9.3f}"
              f"{s['trim_median']:>10.2f} / {s['trim_p90']:.2f}  ({np.rad2deg(np.arctan(s['trim_median'] / 9.81)):.1f} / "
              f"{np.rad2deg(np.arctan(s['trim_p90'] / 9.81)):.1f} deg)"
              + (f"  ({s['incomplete']} diverged)" if s["incomplete"] else ""))
    if not args.no_plots:
        plot(runs, HERE / "plots")


if __name__ == "__main__":
    main()
