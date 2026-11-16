#!/usr/bin/env python3
"""Identify the Air65 model from the decoded logs in sysid/data.

    python sysid/fit_air65.py            # prints the fits, writes sysid/air65_fit.json

Blackbox logs (bbl_*.npz) give the motor: command -> rpm map, battery
dependence, motor time constant, and bounds on the inertia.
Flight logs (flight_*.npz) give what the laptop sees end to end: throttle ->
climb, commanded angle -> horizontal motion, drag and the delays.

Throttle convention: `u` is skybridge's fraction of 1000..2000 us. Betaflight
starts its throttle at min_check, so its own throttle is (u - 0.05) / 0.95,
and the motor output above idle equals that (verified in the blackbox).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import binary_dilation
from scipy.signal import butter, filtfilt, lfilter

HERE = Path(__file__).parent
G = 9.81
MIN_CHECK = 0.05            # (min_check - 1000) / 1000
ANGLE_LIMIT = 20.0          # deg at full stick when the logs were flown
MASS = 0.028                # kg, assumed all-up mass (only used to put a scale on the inertia)
ARM = 0.02298               # m, motor offset along x and y
HOVER_U = 0.305             # hover throttle in the flight logs


def lag(x, tau, dt, x0=None):
    """First-order lag y' = (x - y) / tau, explicit Euler, y[0] = x0 (default x[0])."""
    a = min(dt / max(tau, dt), 1.0)
    y0 = x[0] if x0 is None else x0
    y, _ = lfilter([0.0, a], [1.0, a - 1.0], x, zi=[(1.0 - a) * y0])
    return y


def integrate(acc, drag, dt, v0):
    """v' = acc - drag * v, explicit Euler, v[0] = v0."""
    k = 1.0 - drag * dt
    v, _ = lfilter([0.0, dt], [1.0, -k], acc, zi=[k * v0])
    return v


def bf_throttle(u):
    return np.clip((np.asarray(u, dtype=float) - MIN_CHECK) / (1.0 - MIN_CHECK), 0.0, 1.0)


# ── Blackbox: motor ──────────────────────────────────────────────────────────
def load_bbl():
    return [dict(np.load(f)) for f in sorted(glob.glob(str(HERE / "data/bbl_*.npz")))]


def rpm_model(p, u, v):
    """Steady rpm for motor output u (0..1 above idle) at battery voltage v."""
    k, u0, expo, vexp = p
    return k * (v / 3.7) ** vexp * (u + u0) ** expo


def fit_motor_map(logs):
    b, a = butter(2, 3 / 500)
    U, R, V = [], [], []
    for d in logs:
        if len(d["t"]) < 200:
            continue
        v = filtfilt(b, a, d["vbat"])
        for i in range(4):
            u = filtfilt(b, a, d["motor"][:, i])
            r = filtfilt(b, a, d["rpm"][:, i])
            m = (np.abs(np.gradient(u)) * 1000 < 0.15) & (r > 3000) & (u > 0)
            U.append(u[m][::25]); R.append(r[m][::25]); V.append(v[m][::25])
    U, R, V = map(np.concatenate, (U, R, V))
    res = least_squares(lambda p: (rpm_model(p, U, V) - R) / 1000, [60000, 0.05, 0.8, 0.8],
                        bounds=([1e4, 0.0, 0.3, 0.0], [3e5, 0.3, 2.0, 2.0]), loss="soft_l1")
    rms = float(np.sqrt(np.mean((rpm_model(res.x, U, V) - R) ** 2)))
    return res.x, rms, float(U.max())


def clean_mask(d):
    """Samples in steady flight. The logs include crashes and stalled motors;
    anything within 0.3 s of one is dropped."""
    bad = (np.abs(d["gyro"]).max(1) > 250) | (d["rpm"].min(1) < 3000) | (d["motor"].max(1) > 0.9)
    return ~binary_dilation(bad, iterations=300) & (d["motor"].mean(1) > 0.12)


def fit_motor_tau(logs, p):
    """Lag from the steady rpm map to the measured rpm, on the 2-60 Hz band
    where the flight controller's corrections live."""
    b, a = butter(2, [2 / 500, 60 / 500], "band")
    taus = (0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.07)
    errors = []
    for tau in taus:
        num = den = 0.0
        for d in logs:
            m = clean_mask(d)
            if m.sum() < 1000:
                continue
            for i in range(4):
                target = rpm_model(p, np.clip(d["motor"][:, i], 0, 1), d["vbat"])
                measured = filtfilt(b, a, d["rpm"][:, i])
                e = filtfilt(b, a, lag(target, tau, 0.001, d["rpm"][0, i])) - measured
                num += np.sum(e[m] ** 2); den += np.sum(measured[m] ** 2)
        errors.append(float(np.sqrt(num / den)))
    k = int(np.argmin(errors))
    return taus[k], errors[k], dict(zip(taus, errors))


# ── Blackbox: inertia ────────────────────────────────────────────────────────
def fit_inertia(logs):
    """Angular acceleration vs differential rpm^2, on the 3-35 Hz band.

    alpha = (kf * L / I) * sum(s_i * rpm_i^2), with thrust = kf * rpm^2.
    Betaflight motor order: 1 rear right, 2 front right, 3 rear left, 4 front left.
    Both signals are noisy, so the regression slope underestimates the gain;
    slope / r^2 is the upper bound. The two together bound the inertia.
    """
    b, a = butter(2, [3 / 500, 35 / 500], "band")
    signs = {"roll": [-1, -1, 1, 1], "pitch": [1, -1, 1, -1]}
    out = {}
    for ax, (name, s) in enumerate(signs.items()):
        X, Y = [], []
        for d in logs:
            m = clean_mask(d)
            if m.sum() < 1000:
                continue
            alpha = np.gradient(filtfilt(b, a, np.deg2rad(d["gyro"][:, ax])), d["t"])
            X.append(filtfilt(b, a, (d["rpm"] ** 2) @ np.array(s, float))[m]); Y.append(alpha[m])
        X, Y = np.concatenate(X), np.concatenate(Y)
        gain = float(np.dot(X, Y) / np.dot(X, X))
        r2 = float(1 - np.sum((Y - gain * X) ** 2) / np.sum(Y ** 2))
        out[name] = {"gain_low": abs(gain), "gain_high": abs(gain) / r2, "r2": r2}
    return out


# ── Flights ──────────────────────────────────────────────────────────────────
def load_flights():
    return {Path(f).stem: dict(np.load(f)) for f in sorted(glob.glob(str(HERE / "data/flight_*.npz")))}


def windows(d, length, min_height=0.25):
    """Index arrays into est_t for airborne, tracked windows of `length` seconds."""
    et, z = d["est_t"], d["pos"][:, 2]
    air = d["rc_armed"] & (d["rc_throttle"] > 0.2)
    ok = (np.interp(et, d["rc_t"], air.astype(float)) > 0.99) & (d["since_update"] == 0) \
        & (z > z.min() + min_height)
    idx = np.flatnonzero(ok)
    out = []
    for s in np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1):
        if len(s) < 2:
            continue
        a = et[s[0]]
        while a + length <= et[s[-1]]:
            w = s[(et[s] >= a) & (et[s] < a + length)]
            if len(w) > 20 * length:
                out.append(w)
            a += length
    return out


def _detrend(t, e):
    return e - np.polyval(np.polyfit(t - t[0], e, 1), t - t[0])


def fit_vertical(d, u0):
    """vz' = g * (thrust(u(t - delay)) / thrust(u_hover) * cos(tilt) - 1) - cd * vz, lagged by tau."""
    et, rt = d["est_t"], d["rc_t"]
    vz = d["vel"][:, 2]
    cos_t = np.cos(np.deg2rad(np.interp(et, d["att_t"], d["att_roll"]))) * \
        np.cos(np.deg2rad(np.interp(et, d["att_t"], d["att_pitch"])))
    wins = windows(d, 4.0)
    thr = bf_throttle(d["rc_throttle"])

    def resid(p):
        uh, expo, delay, tau, cd = p
        out = []
        for w in wins:
            g = np.arange(et[w[0]], et[w[-1]], 0.005)
            u = np.interp(g - delay, rt, thr); c = np.interp(g, et, cos_t)
            acc = lag(G * (((u + u0) / (uh + u0)) ** expo * c - 1), tau, 0.005)
            v = integrate(acc, cd, 0.005, vz[w[0]])
            out.append(_detrend(et[w], vz[w] - np.interp(et[w], g, v)))
        return np.concatenate(out)

    best = None
    for e0 in (1.0, 1.7, 2.5):
        r = least_squares(resid, [0.27, e0, 0.08, 0.05, 0.5],
                          bounds=([0.15, 0.3, 0, 0.005, 0], [0.45, 5, 0.4, 0.5, 5]))
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    spread = float(np.std(np.concatenate([vz[w] - vz[w].mean() for w in wins])))
    return {"hover_bf_throttle": p[0], "hover_u": p[0] * (1 - MIN_CHECK) + MIN_CHECK,
            "expo": p[1], "delay": p[2], "tau": p[3], "drag_z": p[4],
            "rmse": float(np.sqrt(np.mean(best.fun ** 2))), "vz_std": spread, "windows": len(wins)}


def fit_horizontal(d):
    """Commanded angle -> horizontal velocity, in the camera world frame.

    angle = first order (tau) of the stick angle delayed by `delay`;
    accel_body = gain * g * tan(angle) - cd * v; rotated by heading = psi0 - (FC yaw - FC yaw at start).
    """
    et, rt = d["est_t"], d["rc_t"]
    vel = d["vel"][:, :2]
    wins = windows(d, 3.0)
    yaw = np.unwrap(np.deg2rad(d["att_yaw"]))
    yaw0 = np.interp(et[wins[0][0]], d["att_t"], yaw)

    def resid(p):
        delay, tau, cd, gain, psi0 = p
        out = []
        for w in wins:
            g = np.arange(et[w[0]] - 0.5, et[w[-1]], 0.005)     # 0.5 s lead-in settles the angle lag
            roll = lag(np.deg2rad(ANGLE_LIMIT) * np.interp(g - delay, rt, d["rc_roll"]), tau, 0.005)
            pitch = lag(np.deg2rad(ANGLE_LIMIT) * np.interp(g - delay, rt, d["rc_pitch"]), tau, 0.005)
            head = psi0 - (np.interp(g, d["att_t"], yaw) - yaw0)
            right, fwd = gain * G * np.tan(roll), gain * G * np.tan(pitch)
            c, s_ = np.cos(head), np.sin(head)
            acc = np.stack([c * right - s_ * fwd, s_ * right + c * fwd], axis=1)
            i0 = np.searchsorted(g, et[w[0]])
            for k in range(2):
                v = integrate(acc[i0:, k], cd, 0.005, vel[w[0], k])
                out.append(_detrend(et[w], vel[w, k] - np.interp(et[w], g[i0:], v)))
        return np.concatenate(out)

    best = None
    for psi in (0.0, 0.5, -0.5):
        r = least_squares(resid, [0.12, 0.06, 0.5, 1.0, psi],
                          bounds=([0, 0.005, 0, 0.3, -np.pi], [0.4, 0.5, 5, 2.0, np.pi]))
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    spread = float(np.std(np.concatenate([vel[w] - vel[w].mean(0) for w in wins])))
    return {"delay": p[0], "tau": p[1], "drag_xy": p[2], "gain": p[3], "psi0_deg": float(np.rad2deg(p[4])),
            "rmse": float(np.sqrt(np.mean(best.fun ** 2))), "v_std": spread, "windows": len(wins)}


def fit_attitude_telemetry(d):
    """Commanded angle -> telemetry angle: first order + delay. delay here minus
    the motion delay is the telemetry latency."""
    rt = d["rc_t"]; air = d["rc_armed"] & (d["rc_throttle"] > 0.2)
    g = np.arange(rt[air][0], rt[air][-1], 0.005)
    out = {}
    for name, sign in (("roll", 1.0), ("pitch", -1.0)):
        y = np.interp(g, d["att_t"], d["att_" + name]) * sign
        best = None
        for delay in np.arange(0, 0.3, 0.01):
            u = ANGLE_LIMIT * np.interp(g - delay, rt, d["rc_" + name])
            for tau in (0.02, 0.04, 0.06, 0.08, 0.1, 0.15):
                ys = lag(u, tau, 0.005, y[0])
                A = np.c_[ys, np.ones_like(ys)]; c = np.linalg.lstsq(A, y, rcond=None)[0]
                e = float(np.sqrt(np.mean((A @ c - y) ** 2)))
                if best is None or e < best[0]:
                    best = (e, delay, tau, c[0], c[1])
        out[name] = dict(zip(("rmse_deg", "delay", "tau", "gain", "offset_deg"), map(float, best)))
    dt = np.diff(d["att_t"])
    out["period"] = {"median": float(np.median(dt)), "p95": float(np.percentile(dt, 95))}
    return out


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    fit: dict = {}
    logs = load_bbl()
    p, rms, umax = fit_motor_map(logs)
    fit["rpm_map"] = {"k": p[0], "u0": p[1], "expo": p[2], "vbat_expo": p[3], "rmse_rpm": rms,
                      "valid_up_to_u": umax}
    print(f"rpm = {p[0]:.0f} * (V/3.7)^{p[3]:.2f} * (u + {p[1]:.3f})^{p[2]:.3f}   rmse {rms:.0f} rpm, "
          f"data up to u = {umax:.2f}")
    print(f"  thrust ~ rpm^2 ~ (u + {p[1]:.3f})^{2 * p[2]:.2f} * V^{2 * p[3]:.2f}")
    tau, err, curve = fit_motor_tau(logs, p)
    fit["motor_tau"] = {"tau": tau, "relative_error": err, "curve": {str(k): v for k, v in curve.items()}}
    print(f"motor lag: {tau * 1000:.0f} ms (relative error {err:.2f}; "
          + ", ".join(f"{k * 1000:.0f} ms: {v:.2f}" for k, v in curve.items()) + ")")
    # Thrust per rpm^2 from hover: the flights hover at hover_u with a mid battery.
    hover_rpm = rpm_model(p, bf_throttle(HOVER_U), 3.6)
    kf = MASS * G / 4 / hover_rpm ** 2
    fit["hover_rpm"] = float(hover_rpm)
    fit["inertia"] = {}
    for name, v in fit_inertia(logs).items():
        low, high = kf * ARM / v["gain_high"], kf * ARM / v["gain_low"]
        fit["inertia"][name] = {"low": low, "high": high, "r2": v["r2"]}
        print(f"inertia {name}: {low:.1e} .. {high:.1e} kg m^2 (r2 {v['r2']:.2f}; assumes mass {MASS * 1000:.0f} g, "
              f"hover at {hover_rpm:.0f} rpm)")
    print("rate loop: not identifiable, the blackbox logs were flown with the sticks centred")

    fit["flights"] = {}
    for name, d in load_flights().items():
        f = {"vertical": fit_vertical(d, p[1]), "attitude": fit_attitude_telemetry(d)}
        try:
            f["horizontal"] = fit_horizontal(d)
        except Exception as exc:                      # too little motion in this flight
            f["horizontal"] = {"error": str(exc)}
        fit["flights"][name] = f
        v, h, a = f["vertical"], f["horizontal"], f["attitude"]
        print(f"\n{name}")
        print(f"  vertical:   hover u {v['hover_u']:.3f}, thrust expo {v['expo']:.2f}, delay {v['delay'] * 1000:.0f} ms, "
              f"lag {v['tau'] * 1000:.0f} ms, drag {v['drag_z']:.2f} 1/s, rmse {v['rmse']:.3f} of {v['vz_std']:.3f} m/s ({v['windows']} windows)")
        if "error" not in h:
            print(f"  horizontal: delay {h['delay'] * 1000:.0f} ms, angle tau {h['tau'] * 1000:.0f} ms, drag {h['drag_xy']:.2f} 1/s, "
                  f"gain {h['gain']:.2f}, heading at start {h['psi0_deg']:.0f} deg, rmse {h['rmse']:.3f} of {h['v_std']:.3f} m/s ({h['windows']} windows)")
        for k in ("roll", "pitch"):
            print(f"  telemetry {k}: delay {a[k]['delay'] * 1000:.0f} ms, tau {a[k]['tau'] * 1000:.0f} ms, gain {a[k]['gain']:.2f}, rmse {a[k]['rmse_deg']:.2f} deg")
        print(f"  telemetry period: {a['period']['median'] * 1000:.0f} ms median, {a['period']['p95'] * 1000:.0f} ms p95")

    (HERE / "air65_fit.json").write_text(json.dumps(fit, indent=2, default=float))
    print(f"\nwrote {HERE / 'air65_fit.json'}")


if __name__ == "__main__":
    main()
