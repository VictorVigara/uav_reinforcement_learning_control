"""Physical parameters of the default drone, as module constants.

The values live in `drones/<name>/params.yaml` (see `drones/__init__.py`);
this module exposes the default drone's for code that predates the registry.
New code should take a `DroneSpec` from `drones.load_drone(name)` instead.
"""

from drones import DEFAULT_DRONE, load_drone

_SPEC = load_drone(DEFAULT_DRONE)

# ── Base parameters ──
MAX_MOTOR_THRUST = _SPEC.max_motor_thrust       # N per motor
ARM_LENGTH = _SPEC.arm_length                   # m
YAW_TORQUE_COEFF = _SPEC.yaw_torque_coeff       # motor reaction-torque / thrust ratio
MASS = _SPEC.mass                               # kg
G = _SPEC.g                                     # m/s²
DT = _SPEC.dt                                   # s (MuJoCo timestep)
IXX, IYY, IZZ = _SPEC.inertia                   # kg·m²

# ── Derived parameters ──
MAX_TOTAL_THRUST = _SPEC.max_total_thrust               # N
MAX_TORQUE = _SPEC.max_torque                           # N·m
HOVER_THRUST_PER_MOTOR = _SPEC.hover_thrust_per_motor   # N
