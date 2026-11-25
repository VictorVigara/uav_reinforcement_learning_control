"""A Betaflight angle-mode flight controller, as far as the laptop can tell.

The real drone takes four RC channels at 50 Hz: throttle, roll angle, pitch
angle and yaw rate. Everything between those and the propellers runs on the
flight controller. This module stands in for it:

    RC command -> RC smoothing -> angle loop (P + feedforward) -> rate loop
               -> mixer with airmode -> motor lag -> thrust per motor

The structure follows Betaflight; the numbers are fitted to flight and
blackbox logs (see sysid/) instead of being converted from Betaflight's PID
units. Angles and rates are in the simulator's body frame (x forward, y left,
z up): positive roll tilts right, positive pitch tilts the nose down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass
class BetaflightParams:
    # ── throttle -> thrust ──
    hover_throttle: float = 0.305   # throttle (0..1 of 1000..2000 us) that holds the nominal mass
    min_check: float = 0.05         # Betaflight's throttle is 0 below this
    thrust_u0: float = 0.047        # thrust ~ (bf_throttle + u0) ** expo
    thrust_expo: float = 1.67
    max_thrust_ratio: float = 3.4   # per-motor thrust ceiling / hover thrust
    thrust_soft_cap: float = 0.0    # 0: the curve is cut off at the ceiling. n > 0: it bends smoothly
                                    # into it, ratio / (1 + (ratio / ceiling)^n)^(1/n) (scaled to 1 at
                                    # hover), so full thrust comes near full stick, as Betaflight
                                    # spreads motor output over the whole stick
    motor_tau_up: float = 0.03      # s, in rpm (thrust = rpm^2)
    motor_tau_down: float = 0.05
    # ── attitude ──
    rc_tau: float = 0.013           # s, RC smoothing
    level_kp: float = 5.0           # 1/s: rate setpoint per unit of angle error
    angle_ff: float = 0.8           # share of the setpoint's own rate fed forward
    ff_tau: float = 0.03            # s, smoothing of that feedforward
    rate_tau: tuple[float, float, float] = (0.02, 0.02, 0.04)   # s, closed rate loop

    @classmethod
    def from_dict(cls, raw: dict) -> BetaflightParams:
        known = {f.name for f in fields(cls)}
        if unknown := set(raw) - known:
            raise ValueError(f"betaflight: unknown params {sorted(unknown)}")
        params = cls(**raw)
        params.rate_tau = tuple(float(x) for x in params.rate_tau)
        return params


class BetaflightAngleController:
    """Turns a (delayed) RC command into four motor thrusts, once per physics step.

    Written with plain floats: it runs several times per policy step, and small
    numpy arrays would dominate the simulation time.
    """

    # Motor order: 1 front right, 2 rear right, 3 rear left, 4 front left.
    ROLL_SIGN = (-1.0, -1.0, 1.0, 1.0)      # left motors up -> roll right
    PITCH_SIGN = (-1.0, 1.0, 1.0, -1.0)     # rear motors up -> nose down
    YAW_SIGN = (1.0, -1.0, 1.0, -1.0)

    def __init__(self, params: BetaflightParams, *, mass: float, inertia, arm_length: float,
                 yaw_torque_coeff: float, g: float = 9.81):
        self.p = params
        self.inertia = tuple(float(x) for x in inertia)
        self.hover_motor_thrust = mass * g / 4.0    # N, at the nominal mass and battery
        self._arm = float(arm_length)
        self._yaw_coeff = float(yaw_torque_coeff)
        self.motor_tau_scale = 1.0
        self.angle_trim = (0.0, 0.0)    # rad: the FC's level differs from true level by this
        self.thrust_scale = 1.0         # battery state: scales every motor's thrust
        self.reset()

    # ── throttle curve ──
    @property
    def thrust_scale(self) -> float:
        return self._thrust_scale

    @thrust_scale.setter
    def thrust_scale(self, value: float) -> None:
        self._thrust_scale = float(value)
        p = self.p
        self._hover_base = self._bf_throttle(p.hover_throttle) + p.thrust_u0
        self._unit = self.hover_motor_thrust * self._thrust_scale   # thrust per motor at hover throttle
        self._high = self._unit * p.max_thrust_ratio
        self._low = self.motor_thrust_for(0.0)

    def _bf_throttle(self, u: float) -> float:
        return min(max((u - self.p.min_check) / (1.0 - self.p.min_check), 0.0), 1.0)

    def _cap(self, ratio: float) -> float:
        p = self.p
        if p.thrust_soft_cap > 0.0:     # normalized so hover thrust stays at the measured hover throttle
            n, top = p.thrust_soft_cap, p.max_thrust_ratio
            bend = lambda r: r / (1.0 + (r / top) ** n) ** (1.0 / n)
            return min(bend(ratio) / bend(1.0), top)
        return min(ratio, p.max_thrust_ratio)

    def motor_thrust_for(self, throttle: float) -> float:
        """Steady thrust per motor (N) at this throttle, all motors equal."""
        p = self.p
        ratio = ((self._bf_throttle(throttle) + p.thrust_u0) / self._hover_base) ** p.thrust_expo
        return self._unit * self._cap(ratio)

    def throttle_for_hover(self, mass: float, g: float = 9.81) -> float:
        """Throttle that holds `mass` level at the current battery state."""
        p = self.p
        ratio = (mass * g / 4.0) / self._unit
        if p.thrust_soft_cap > 0.0:     # invert the soft cap (it is monotonic) by bisection
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if self.motor_thrust_for(mid) < ratio * self._unit else (lo, mid)
            return 0.5 * (lo + hi)
        bf = ratio ** (1.0 / p.thrust_expo) * self._hover_base - p.thrust_u0
        return min(max(bf * (1.0 - p.min_check) + p.min_check, 0.0), 1.0)

    @property
    def thrust_limits(self) -> tuple[float, float]:
        """Idle and maximum thrust per motor, N."""
        return self._low, self._high

    # ── state ──
    def reset(self, throttle: float | None = None, roll: float = 0.0, pitch: float = 0.0) -> None:
        """Start with the sticks at `throttle` and the given angles, motors settled."""
        throttle = self.p.hover_throttle if throttle is None else throttle
        self._rc = [throttle, roll, pitch, 0.0]             # smoothed command
        self._ff = [0.0, 0.0]                               # smoothed setpoint rate
        self._rpm = [math.sqrt(self.motor_thrust_for(throttle))] * 4   # sqrt(thrust)

    def step(self, command, roll: float, pitch: float, gyro, dt: float) -> list[float]:
        """Advance by `dt`.

        command: [throttle 0..1, roll rad, pitch rad, yaw rate rad/s].
        roll, pitch: true attitude, rad. gyro: body rates, rad/s.
        Returns the four motor thrusts in newtons.
        """
        p, rc, ff = self.p, self._rc, self._ff

        # RC smoothing
        a = min(dt / p.rc_tau, 1.0)
        prev_roll, prev_pitch = rc[1], rc[2]
        for i in range(4):
            rc[i] += a * (command[i] - rc[i])

        # Angle loop: P on the error, plus the setpoint's own rate fed forward
        a = min(dt / p.ff_tau, 1.0)
        ff[0] += a * ((rc[1] - prev_roll) / dt - ff[0])
        ff[1] += a * ((rc[2] - prev_pitch) / dt - ff[1])
        rate_sp = (p.level_kp * (rc[1] - roll - self.angle_trim[0]) + p.angle_ff * ff[0],
                   p.level_kp * (rc[2] - pitch - self.angle_trim[1]) + p.angle_ff * ff[1],
                   rc[3])

        # Rate loop: first-order closed loop with time constant rate_tau
        inertia, tau = self.inertia, p.rate_tau
        tx = inertia[0] * (rate_sp[0] - gyro[0]) / tau[0] / (4.0 * self._arm)
        ty = inertia[1] * (rate_sp[1] - gyro[1]) / tau[1] / (4.0 * self._arm)
        tz = inertia[2] * (rate_sp[2] - gyro[2]) / tau[2] / (4.0 * self._yaw_coeff)

        # Mixer, with airmode: keep the torque and shift the collective when a motor would clip
        low, high = self._low, self._high
        base = self.motor_thrust_for(rc[0])
        thrust = [base + self.ROLL_SIGN[i] * tx + self.PITCH_SIGN[i] * ty + self.YAW_SIGN[i] * tz
                  for i in range(4)]
        lo, hi = min(thrust), max(thrust)
        if hi - lo > high - low:
            k = (high - low) / (hi - lo)
            thrust = [low + (f - lo) * k for f in thrust]
        elif lo < low:
            thrust = [f + low - lo for f in thrust]
        elif hi > high:
            thrust = [f - (hi - high) for f in thrust]

        # Motor lag, in rpm (thrust = rpm^2)
        rpm = self._rpm
        up, down = p.motor_tau_up * self.motor_tau_scale, p.motor_tau_down * self.motor_tau_scale
        for i in range(4):
            target = math.sqrt(thrust[i])
            rpm[i] += min(dt / (up if target > rpm[i] else down), 1.0) * (target - rpm[i])
        return [r * r for r in rpm]
