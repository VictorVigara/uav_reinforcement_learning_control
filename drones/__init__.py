"""Drone registry: one folder per drone, each with a `params.yaml`.

    from drones import load_drone
    spec = load_drone("air65")

To add a drone: create `drones/<name>/params.yaml` (copy an existing one), put
its MuJoCo model under `model/<name>/`, and pick the `interface` that says how
it is commanded. `envs.make_env(<name>)` then builds the right environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
DRONES_DIR = Path(__file__).resolve().parent
DEFAULT_DRONE = "small"

# interface name -> what the policy commands
INTERFACES = ("thrust_torque", "betaflight_angle")


@dataclass(frozen=True)
class DroneSpec:
    """Physical parameters of one drone. SI units."""

    name: str
    interface: str                  # one of INTERFACES
    model_path: Path                # MuJoCo XML
    mass: float                     # kg
    inertia: tuple[float, float, float]     # Ixx, Iyy, Izz, kg m^2
    arm_length: float               # motor offset from the centre along body x and y, m
    max_motor_thrust: float         # N per motor
    yaw_torque_coeff: float         # motor reaction torque / thrust, m
    dt: float                       # physics timestep, s
    frame_skip: int = 1             # physics steps per policy step
    g: float = 9.81
    params: dict[str, Any] = field(default_factory=dict, compare=False)  # the whole YAML

    @property
    def control_dt(self) -> float:
        return self.dt * self.frame_skip

    @property
    def max_total_thrust(self) -> float:
        return 4 * self.max_motor_thrust

    @property
    def max_torque(self) -> float:
        return 2 * self.max_motor_thrust * self.arm_length

    @property
    def hover_thrust_per_motor(self) -> float:
        return self.mass * self.g / 4

    def section(self, key: str) -> dict[str, Any]:
        """A sub-dictionary of the YAML, empty if absent."""
        return dict(self.params.get(key) or {})


def available_drones() -> list[str]:
    return sorted(p.parent.name for p in DRONES_DIR.glob("*/params.yaml"))


@lru_cache(maxsize=None)
def load_drone(name: str = DEFAULT_DRONE) -> DroneSpec:
    path = DRONES_DIR / name / "params.yaml"
    if not path.exists():
        raise ValueError(f"unknown drone {name!r}; available: {available_drones()}")
    raw = yaml.safe_load(path.read_text())
    if raw["interface"] not in INTERFACES:
        raise ValueError(f"{path}: interface must be one of {INTERFACES}")
    return DroneSpec(
        name=name,
        interface=raw["interface"],
        model_path=REPO / raw["model"],
        mass=float(raw["mass"]),
        inertia=tuple(float(x) for x in raw["inertia"]),
        arm_length=float(raw["arm_length"]),
        max_motor_thrust=float(raw["max_motor_thrust"]),
        yaw_torque_coeff=float(raw["yaw_torque_coeff"]),
        dt=float(raw["dt"]),
        frame_skip=int(raw.get("frame_skip", 1)),
        g=float(raw.get("g", 9.81)),
        params=raw,
    )


def resolve(drone: str | DroneSpec | None) -> DroneSpec:
    if isinstance(drone, DroneSpec):
        return drone
    return load_drone(drone or DEFAULT_DRONE)
