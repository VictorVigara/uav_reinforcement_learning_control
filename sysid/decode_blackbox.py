#!/usr/bin/env python3
"""Decode Betaflight blackbox logs (.bbl) to .npz, using `orangebox`.

    python sysid/decode_blackbox.py [--src ~/skybridge/data/blackbox] [--out sysid/data]

`pip install orangebox==0.4.0` (0.5.0 has a broken entry point).

Each log becomes `bbl_<name>.npz`:

    t            seconds
    gyro         Nx3 deg/s (filtered)      setpoint   Nx3 deg/s (roll, pitch, yaw)
    throttle_sp  0..1 (setpoint[3]/1000)   rc         Nx4 rcCommand
    motor        Nx4 0..1, 0 = idle output, 1 = full
    erpm         Nx4 electrical rpm        rpm        Nx4 mechanical rpm
    vbat         volts                     acc        Nx3 in g
    pid_p/i/d/f  Nx3 PID terms
    header       the raw headers, as a JSON string
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def decode(path: Path) -> dict:
    from orangebox import Parser

    logging.getLogger("orangebox").setLevel(logging.ERROR)
    parser = Parser.load(str(path))
    names = parser.field_names
    ix = {k: i for i, k in enumerate(names)}
    width = ix["eRPM[3]"] + 1
    rows = np.array([[_num(v) for v in fr.data[:width]]
                     for fr in parser.frames() if len(fr.data) >= width])
    h = parser.headers
    cols = lambda fmt, n: rows[:, [ix[fmt.format(i)] for i in range(n)]]
    lo, hi = (int(v) for v in str(h["motorOutput"]).split(",")) if isinstance(
        h["motorOutput"], str) else h["motorOutput"]
    erpm = cols("eRPM[{}]", 4) * 100.0
    d = np.c_[cols("axisD[{}]", 2), np.zeros(len(rows))]
    return {
        "t": rows[:, ix["time"]] / 1e6,
        "gyro": cols("gyroADC[{}]", 3),
        "setpoint": cols("setpoint[{}]", 3),
        "throttle_sp": rows[:, ix["setpoint[3]"]] / 1000.0,
        "rc": cols("rcCommand[{}]", 4),
        "motor": (cols("motor[{}]", 4) - lo) / (hi - lo),
        "erpm": erpm,
        "rpm": erpm / (int(h["motor_poles"]) / 2),
        "vbat": rows[:, ix["vbatLatest"]] / 100.0,
        "acc": cols("accSmooth[{}]", 3) / float(h["acc_1G"]),
        "pid_p": cols("axisP[{}]", 3), "pid_i": cols("axisI[{}]", 3),
        "pid_d": d, "pid_f": cols("axisF[{}]", 3),
        "header": json.dumps({k: str(v) for k, v in h.items()}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=Path.home() / "skybridge/data/blackbox")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for bbl in sorted(args.src.glob("btfl_0*.bbl")):
        data = decode(bbl)
        dst = args.out / f"bbl_{bbl.stem.removeprefix('btfl_')}.npz"
        np.savez_compressed(dst, **data)
        t = data["t"]
        print(f"{dst.name}: {t.size} rows, {t[-1] - t[0]:.1f} s, "
              f"motor max {np.nanmax(data['motor']):.2f}, rpm max {np.nanmax(data['rpm']):.0f}, "
              f"vbat {np.nanmedian(data['vbat']):.2f} V")


if __name__ == "__main__":
    main()
