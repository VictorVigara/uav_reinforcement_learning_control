#!/usr/bin/env python3
"""Export skybridge flight recordings (.rrd) to small .npz files.

Run with skybridge's interpreter, which has `rerun`:

    ~/skybridge/.venv/bin/python sysid/extract_flight_logs.py \
        [--skybridge ~/skybridge] [--out sysid/data]

Each `data/live/*_flight.rrd` becomes `flight_<stamp>.npz` holding, on the
recording's monotonic clock (seconds):

    rc_t, rc_throttle (0..1), rc_roll/rc_pitch/rc_yaw (deflection -1..1), rc_armed
    est_t, pos (Nx3 m), vel (Nx3 m/s), pos_std, since_update     camera tracker
    att_t, att_roll, att_pitch, att_yaw (deg)                    FC telemetry
    ah_t, ah (Nx6), ah_names, ah_engaged_t, ah_engaged           altitude hold, if flown
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def scalar_series(path: Path) -> tuple[dict, dict]:
    import rerun.chunk as rrc

    names, times, values = {}, {}, {}
    for chunk in rrc.RrdReader(str(path)).stream():
        entity = chunk.entity_path
        if not entity.startswith("/plots/"):
            continue
        batch = chunk.to_record_batch()
        if chunk.is_static:
            if "SeriesLines:names" in batch.schema.names:
                names[entity] = batch.column("SeriesLines:names")[0].as_py()
            continue
        if "Scalars:scalars" not in batch.schema.names:
            continue
        times.setdefault(entity, []).append(
            batch.column("monotonic").cast("int64").to_numpy() / 1e9)
        values.setdefault(entity, []).extend(batch.column("Scalars:scalars").to_pylist())
    out = {}
    for entity in times:
        t = np.concatenate(times[entity])
        v = np.array(values[entity], dtype=float)
        order = np.argsort(t, kind="stable")
        out[entity] = (t[order], v[order])
    return out, names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skybridge", type=Path, default=Path.home() / "skybridge")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = ap.parse_args()

    sys.path.insert(0, str(args.skybridge / "src"))
    from skybridge.units import throttle_fraction, tick_to_deflection, us_to_tick

    args.out.mkdir(parents=True, exist_ok=True)
    for rrd in sorted((args.skybridge / "data" / "live").glob("*_flight.rrd")):
        series, names = scalar_series(rrd)
        if "/plots/rc" not in series or "/plots/position" not in series:
            print(f"skip {rrd.name}: no rc or position")
            continue
        rc_t, rc = series["/plots/rc"]
        col = {n: rc[:, i] for i, n in enumerate(names["/plots/rc"])}
        defl = lambda us: np.array([tick_to_deflection(us_to_tick(u)) for u in us])
        data = {
            "rc_t": rc_t,
            "rc_throttle": np.array([throttle_fraction(us_to_tick(u)) for u in col["throttle"]]),
            "rc_roll": defl(col["roll"]), "rc_pitch": defl(col["pitch"]), "rc_yaw": defl(col["yaw"]),
            "rc_armed": col["arm"] > 1700,
        }
        data["est_t"], data["pos"] = series["/plots/position"]
        _, data["vel"] = series["/plots/velocity"]
        _, data["pos_std"] = series["/plots/position_std"]
        _, since = series["/plots/since_update"]
        data["since_update"] = since[:, 0]
        if "/plots/telemetry/attitude" in series:
            t, a = series["/plots/telemetry/attitude"]
            order = names.get("/plots/telemetry/attitude", ["pitch_deg", "roll_deg", "yaw_deg"])
            data["att_t"] = t
            for i, n in enumerate(order):
                data["att_" + n.removesuffix("_deg")] = a[:, i]
        if "/plots/control/altitude_hold" in series:
            data["ah_t"], data["ah"] = series["/plots/control/altitude_hold"]
            data["ah_names"] = np.array(names["/plots/control/altitude_hold"])
            data["ah_engaged_t"], eng = series["/plots/control_engaged/altitude_hold"]
            data["ah_engaged"] = eng[:, 0]
        stamp = rrd.name.removeprefix("flight_").removesuffix("_flight.rrd")
        dst = args.out / f"flight_{stamp}.npz"
        np.savez_compressed(dst, **data)
        print(f"{dst.name}: rc {rc_t.size}, est {data['est_t'].size}, "
              f"att {data.get('att_t', np.empty(0)).size}, alt-hold {data.get('ah_t', np.empty(0)).size}")


if __name__ == "__main__":
    main()
