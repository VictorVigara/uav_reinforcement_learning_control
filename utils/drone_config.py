"""Central drone physical parameters — single source of truth.

All Python modules should import from here instead of hardcoding values.
The MuJoCo XML (model/drone/drone.xml) must be kept in sync manually;
update ctrlrange and keyframe hover ctrl when changing MAX_MOTOR_THRUST.
"""

# ── Base parameters (change these to update everywhere) ──
MAX_MOTOR_THRUST = 1.47         # N per motor  (150 g full battery)
ARM_LENGTH = 0.039799           # m
YAW_TORQUE_COEFF = 0.0201       # motor reaction-torque / thrust ratio
MASS = 0.150                    # kg (real measured mass, including Raspberry Pi)
G = 9.81                        # m/s²
DT = 0.01                       # s (MuJoCo timestep)
IXX = 1.44e-4                   # kg·m² (roll inertia)
IYY = 1.46e-4                   # kg·m² (pitch inertia)
IZZ = 2.11e-4                   # kg·m² (yaw inertia)

# ── Derived parameters ──
MAX_TOTAL_THRUST = 4 * MAX_MOTOR_THRUST                # N  (5.88)
MAX_TORQUE = 2 * MAX_MOTOR_THRUST * ARM_LENGTH          # N·m (~0.117)
HOVER_THRUST_PER_MOTOR = MASS * G / 4                   # N  (~0.368)
