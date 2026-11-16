"""Command limits of the real drone, read from skybridge's config.

skybridge's `configs/drone.yaml` mirrors the flight controller and holds the
safety clamps the laptop applies. It is the one place those limits are set, so
the simulated policy takes its action range from the same file:

    $SKYBRIDGE_CONFIG, else ~/skybridge/configs/drone.yaml,
    else the snapshot drones/<name>/skybridge_drone.yaml (with a warning).

Raise `angle_limit` in Betaflight -> edit skybridge's YAML -> retrain. Each
training run stores the limits it used in its config.json.

    python -m drones.constraints            # show the limits in use
    python -m drones.constraints --sync air65   # refresh the snapshot
"""
from __future__ import annotations

import argparse
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

DRONES_DIR = Path(__file__).resolve().parent
DEFAULT_SKYBRIDGE_CONFIG = Path.home() / "skybridge" / "configs" / "drone.yaml"
SNAPSHOT_HEADER = (
    "# SNAPSHOT of skybridge configs/drone.yaml, used only when skybridge is not on\n"
    "# this machine (e.g. the training cluster). Refresh with:\n"
    "#     python -m drones.constraints --sync {name}\n\n"
)


@dataclass(frozen=True)
class CommandLimits:
    """What the laptop is allowed to command, in skybridge's units."""

    max_angle_deg: float        # roll and pitch: min(Betaflight angle_limit, safety.max_angle)
    angle_limit_deg: float      # Betaflight's own angle_limit (full stick)
    max_yaw_rate_dps: float     # min(safety.max_rate, the yaw rate curve's maximum)
    max_throttle: float         # 0..1 of the 1000..2000 us range
    min_check: float            # throttle fraction below which Betaflight's throttle is 0
    rc_hz: float                # command rate
    source: str                 # the file these came from

    def as_dict(self) -> dict:
        return asdict(self)


def skybridge_config_path(drone: str = "air65") -> tuple[Path, bool]:
    """The config to read and whether it is the live skybridge file."""
    env = os.environ.get("SKYBRIDGE_CONFIG")
    if env:
        path = Path(env).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SKYBRIDGE_CONFIG points to a missing file: {path}")
        return path, True
    if DEFAULT_SKYBRIDGE_CONFIG.exists():
        return DEFAULT_SKYBRIDGE_CONFIG, True
    return DRONES_DIR / drone / "skybridge_drone.yaml", False


def load_limits(drone: str = "air65") -> CommandLimits:
    path, live = skybridge_config_path(drone)
    if not live:
        warnings.warn(f"skybridge config not found; using the snapshot {path}. "
                      "Check that it still matches the real drone.", stacklevel=2)
    raw = yaml.safe_load(path.read_text())
    bf = raw.get("betaflight", {})
    safety = raw.get("safety", {})
    rates = bf.get("rates", {})
    if rates.get("type", "actual") != "actual":
        raise ValueError(f"{path}: only Betaflight ACTUAL rates are supported")

    def per_axis(key: str, default: float) -> float:
        value = rates.get(key, default)
        return float(value["yaw"] if isinstance(value, dict) else value)

    # ACTUAL rates: full stick commands srate * 10 deg/s (rc_rate * 10 when linear).
    yaw_curve_max = 10.0 * max(per_axis("srate", 67), per_axis("rc_rate", 67))
    angle_limit = float(bf.get("angle_limit", 60))
    return CommandLimits(
        max_angle_deg=min(angle_limit, float(safety.get("max_angle", 30))),
        angle_limit_deg=angle_limit,
        max_yaw_rate_dps=min(float(safety.get("max_rate", 200)), yaw_curve_max),
        max_throttle=float(raw.get("rc", {}).get("max_throttle", 1.0)),
        min_check=(float(bf.get("min_check", 1050)) - 1000.0) / 1000.0,
        rc_hz=float(raw.get("serial", {}).get("rc_hz", 50)),
        source=str(path),
    )


def sync_snapshot(drone: str) -> Path:
    """Copy the live skybridge config over the drone's snapshot."""
    src, live = skybridge_config_path(drone)
    if not live:
        raise FileNotFoundError("no live skybridge config to copy from")
    dst = DRONES_DIR / drone / "skybridge_drone.yaml"
    dst.write_text(SNAPSHOT_HEADER.format(name=drone) + src.read_text())
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Show or snapshot the real drone's command limits")
    ap.add_argument("--sync", metavar="DRONE", help="refresh drones/DRONE/skybridge_drone.yaml")
    args = ap.parse_args()
    if args.sync:
        print(f"wrote {sync_snapshot(args.sync)}")
    for key, value in load_limits(args.sync or "air65").as_dict().items():
        print(f"{key:18} {value}")
