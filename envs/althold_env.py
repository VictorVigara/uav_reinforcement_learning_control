"""Altitude hold: the pilot flies roll, pitch and yaw; the policy only sets throttle.

    pilot sticks (roll, pitch, yaw) ──────────────┐
                                                  ├─> Betaflight angle mode -> motors
    depth camera z, vz -> policy -> throttle ─────┘

The policy sees no attitude telemetry (it is 100-200 ms old and noisy). It is
told the pilot's sticks instead: in angle mode a stick deflection asks for an
angle, so the sticks say how far the drone is about to tilt, and so how much
of the thrust is about to go sideways. The drone gets there late, overshoots,
and its level is not quite level: all of that is randomized here, so the policy
learns to use the sticks as a hint and not as the truth.

On engagement the policy holds the altitude the camera measured at that moment.

    action       throttle, normalized to [-1, 1] over `althold.throttle_range`
    observation  see AltHoldObserver: altitude error, climb rate, error integral,
                 age of the camera data, the last few stick commands, the last
                 few throttle commands

Everything the real deployment needs (AltHoldObserver, AltHoldAssist) is plain
Python with no simulator in it, so the laptop builds the exact observation the
policy was trained on. The ranges live in `drones/<name>/params.yaml`, section
`althold`.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import Callable

import numpy as np
import mujoco
from gymnasium.spaces import Box

from drones import DroneSpec, resolve
from drones.constraints import CommandLimits
from utils import QuadState

from .angle_env import AngleModeEnv, _euler
from .hover_env import HoverEnv
from .sim2real import draw

G = 9.81


# ── observation: shared by training, rc_sim and the real drone ──────────────
class AltHoldObserver:
    """Builds the policy's observation from what the laptop knows.

    Call `engage` once when the policy takes over, then every RC tick
    `observe` (before the policy runs) and `push_throttle` (with its action).
    """

    STICK_FEATURES = ("roll", "pitch", "yaw", "tilt")

    def __init__(self, cfg: dict, *, angle_full: float, yaw_full: float,
                 throttle_range: tuple[float, float], control_dt: float):
        self.stick_history = int(cfg.get("stick_history", 5))
        self.throttle_history = int(cfg.get("throttle_history", 10))
        scales = cfg.get("obs", {}) or {}
        self.err_scale = float(scales.get("err", 1.0))          # m
        self.vz_scale = float(scales.get("vz", 2.0))            # m/s
        self.int_limit = float(scales.get("int", 0.5))          # m s, where the integral saturates
        self.age_scale = float(scales.get("age", 0.5))          # s
        self.int_zone = float(cfg.get("int_zone", 0.3))         # m: integrate only while the error is below this
        self.angle_full = float(angle_full)                     # rad at full stick
        self.yaw_full = float(yaw_full)                         # rad/s at full stick
        self.tilt_full = 1.0 / math.cos(self.angle_full) ** 2 - 1.0     # tilt term, both sticks at full
        self.thr_lo, self.thr_hi = float(throttle_range[0]), float(throttle_range[1])
        self.dt = float(control_dt)
        self.names = (("alt_err", "climb", "alt_int", "cam_age")
                      + tuple(f"stick{k}_{n}" for k in range(self.stick_history) for n in self.STICK_FEATURES)
                      + tuple(f"thr{k}" for k in range(self.throttle_history)))
        self.size = len(self.names)
        self.target = 0.0
        self.integral = 0.0
        self._started = False
        self._sticks: deque = deque(maxlen=self.stick_history)
        self._throttles: deque = deque(maxlen=self.throttle_history)

    # throttle <-> action
    def throttle(self, action) -> float:
        a = float(np.clip(np.asarray(action, dtype=np.float64).ravel()[0], -1.0, 1.0))
        return self.thr_lo + (a + 1.0) / 2.0 * (self.thr_hi - self.thr_lo)

    def action_for(self, throttle: float) -> float:
        return float(np.clip(2.0 * (throttle - self.thr_lo) / (self.thr_hi - self.thr_lo) - 1.0, -1.0, 1.0))

    def _stick_features(self, sticks) -> tuple[float, ...]:
        """sticks: [roll rad, pitch rad, yaw rate rad/s] as commanded (skybridge signs)."""
        roll, pitch, yaw = (float(x) for x in sticks)
        tilt = 1.0 / max(math.cos(roll) * math.cos(pitch), 0.1) - 1.0     # extra thrust the tilt costs
        return (roll / self.angle_full, pitch / self.angle_full, yaw / self.yaw_full, tilt / self.tilt_full)

    def engage(self, z: float, throttle: float, sticks) -> None:
        """Take over at measured altitude `z`, with the pilot holding `throttle` and `sticks`."""
        self.target = float(z)
        self.integral = 0.0
        self._started = False
        self._sticks = deque([self._stick_features(sticks)] * self.stick_history, maxlen=self.stick_history)
        self._throttles = deque([self.action_for(throttle)] * self.throttle_history, maxlen=self.throttle_history)

    def observe(self, z: float, vz: float, age: float, sticks) -> np.ndarray:
        """z, vz: the latest camera measurement (m, m/s, up+). age: how old it is, s.
        sticks: the stick command about to be sent with the policy's throttle."""
        err = self.target - float(z)
        if self._started and abs(err) < self.int_zone:
            self.integral = float(np.clip(self.integral + err * self.dt, -self.int_limit, self.int_limit))
        self._started = True
        self._sticks.appendleft(self._stick_features(sticks))
        head = [err / self.err_scale, float(vz) / self.vz_scale, self.integral / self.int_limit,
                min(max(float(age), 0.0) / self.age_scale, 1.0)]
        obs = np.concatenate([head, np.ravel(list(self._sticks)), list(self._throttles)])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def push_throttle(self, action) -> None:
        self._throttles.appendleft(float(np.clip(np.asarray(action, dtype=np.float64).ravel()[0], -1.0, 1.0)))

    def config(self) -> dict:
        return {"stick_history": self.stick_history, "throttle_history": self.throttle_history,
                "angle_full_deg": math.degrees(self.angle_full), "yaw_full_dps": math.degrees(self.yaw_full),
                "throttle_range": [self.thr_lo, self.thr_hi], "obs": {"err": self.err_scale, "vz": self.vz_scale,
                "int": self.int_limit, "age": self.age_scale}, "int_zone": self.int_zone, "control_dt": self.dt}


class AltHoldAssist:
    """A trained policy holding altitude on live data, one call per RC tick.

    If the camera data gets older than `timeout` seconds the throttle goes to
    zero and stays there until the next `engage` (it does not jump back when
    the camera returns).
    """

    def __init__(self, observer: AltHoldObserver, policy: Callable[[np.ndarray], np.ndarray],
                 timeout: float = 1.0):
        self.observer = observer
        self.policy = policy
        self.timeout = float(timeout)
        self.engaged = False
        self.timed_out = False

    def engage(self, z: float, throttle: float, sticks) -> None:
        self.observer.engage(z, throttle, sticks)
        self.engaged, self.timed_out = True, False

    def disengage(self) -> None:
        self.engaged = False

    def step(self, z: float, vz: float, age: float, sticks) -> float:
        """Throttle (0..1) to send with `sticks`."""
        if age > self.timeout:
            self.timed_out = True
        if self.timed_out:
            return 0.0
        obs = self.observer.observe(z, vz, age, sticks)
        action = float(np.asarray(self.policy(obs)).ravel()[0])
        self.observer.push_throttle(action)
        return self.observer.throttle(action)


# ── a simulated pilot ────────────────────────────────────────────────────────
class SimPilot:
    """Roll, pitch and yaw sticks of a pilot keeping the drone in the camera's view.

    A slow position-return (with a reaction delay) keeps it near the centre;
    `aggressiveness` (0..1) adds wandering, sharp punches and yaw spins on top.
    Output is [roll rad, pitch rad, yaw rate rad/s], skybridge signs.
    """

    def __init__(self, rng: np.random.Generator, cfg: dict, *, angle_full: float, dt: float,
                 aggressiveness: float):
        u = lambda key, default: float(rng.uniform(*cfg.get(key, default)))
        self.rng, self.dt, self.a = rng, dt, float(aggressiveness)
        self.angle_full = angle_full
        self.kp = u("return_kp", [0.5, 1.5])
        self.kd = u("return_kd", [0.8, 1.6])
        self._seen: deque = deque(maxlen=int(round(u("reaction_s", [0.15, 0.35]) / dt)) + 1)
        self.wander_sigma = self.a * float(cfg.get("wander", 0.6)) * angle_full
        self.wander_tau = u("wander_tau", [0.3, 2.0])
        self.punch_rate = self.a * float(cfg.get("punch_rate", 0.4))
        self.punch_s = cfg.get("punch_s", [0.2, 1.0])
        self.yaw_sigma = self.a * math.radians(float(cfg.get("yaw_wander_dps", 60.0)))
        self.spin_rate = self.a * float(cfg.get("spin_rate", 0.1))
        self.spin_dps = cfg.get("spin_dps", [90.0, 200.0])
        self.noise = float(cfg.get("stick_noise", 0.003)) * angle_full
        self._wander = np.zeros(2)
        self._yaw = 0.0
        self._punch: list | None = None     # [steps left, roll, pitch]
        self._spin: list | None = None      # [steps left, rate]

    def sticks(self, pos, vel, yaw: float) -> np.ndarray:
        rng, dt = self.rng, self.dt
        self._seen.append((np.array(pos[0:2], dtype=float), np.array(vel[0:2], dtype=float), float(yaw)))
        p, v, yaw_seen = self._seen[0]

        # Back towards the centre of the view
        acc = -self.kp * p - self.kd * v
        cap = G * math.tan(0.7 * self.angle_full)
        n = float(np.linalg.norm(acc))
        if n > cap:
            acc *= cap / n
        c, s = math.cos(yaw_seen), math.sin(yaw_seen)
        forward, left = c * acc[0] + s * acc[1], -s * acc[0] + c * acc[1]
        roll, pitch = -math.atan(left / G), math.atan(forward / G)

        # Wandering
        k = dt / self.wander_tau
        self._wander += -k * self._wander + self.wander_sigma * math.sqrt(2 * k) * rng.normal(0, 1, 2)
        roll += self._wander[0]
        pitch += self._wander[1]

        # Punches: a sharp tilt held for a moment
        if self._punch is None and rng.random() < self.punch_rate * dt:
            ang = rng.uniform(-math.pi, math.pi)
            mag = rng.uniform(0.5, 1.0) * self.angle_full
            self._punch = [int(rng.uniform(*self.punch_s) / dt), mag * math.cos(ang), mag * math.sin(ang)]
        if self._punch is not None:
            roll, pitch = self._punch[1], self._punch[2]
            self._punch[0] -= 1
            if self._punch[0] <= 0:
                self._punch = None

        # Yaw: wandering plus the odd spin
        self._yaw += -dt * self._yaw + self.yaw_sigma * math.sqrt(2 * dt) * rng.normal()
        yaw_rate = self._yaw
        if self._spin is None and rng.random() < self.spin_rate * dt:
            self._spin = [int(rng.uniform(0.5, 2.0) / dt), math.radians(rng.uniform(*self.spin_dps)) * rng.choice([-1, 1])]
        if self._spin is not None:
            yaw_rate = self._spin[1]
            self._spin[0] -= 1
            if self._spin[0] <= 0:
                self._spin = None

        roll += rng.normal(0, self.noise)
        pitch += rng.normal(0, self.noise)
        lim = self.angle_full
        return np.array([min(max(roll, -lim), lim), min(max(pitch, -lim), lim), yaw_rate])


# ── the environment ──────────────────────────────────────────────────────────
class AltHoldEnv(AngleModeEnv):
    """Hold the altitude at engagement with throttle alone, while a simulated pilot flies.

    Args:
        drone: a registry name or DroneSpec with an `althold` section.
        randomize: per-episode variation of drone, flight controller, air, link and sensors.
        noise: sensor noise, dropped frames, camera outages, gusts and loop jitter.
        difficulty: 0..1, scales the randomization around nominal and the pilot's
            aggressiveness (a curriculum sets it during training).
    """

    ACTION_NAMES = ("throttle",)

    def __init__(self, drone: str | DroneSpec = "air65", render_mode: str | None = None,
                 max_episode_steps: int | None = None, limits: CommandLimits | None = None,
                 randomize: bool = True, noise: bool = True, difficulty: float = 1.0):
        spec = resolve(drone)
        alt = spec.section("althold")
        if not alt:
            raise ValueError(f"drone {spec.name!r} has no 'althold' section in its params.yaml")
        super().__init__(drone=spec, render_mode=render_mode, limits=limits, randomize=randomize, noise=noise,
                         max_episode_steps=max_episode_steps or int(alt.get("episode_steps", 1000)))
        self._alt = alt
        # The althold section widens and extends the drone's own ranges.
        self._ranges = {**self._ranges, **(alt.get("randomize") or {})}
        self._link = {**self._link, **(alt.get("link") or {})}
        self._noise = {**self._noise, **(alt.get("noise") or {})}
        self._aero = {**self._aero, **(alt.get("aero") or {})}
        self._reward = dict(alt.get("reward") or {})
        self._pilot_cfg = dict(alt.get("pilot") or {})
        self.difficulty = float(difficulty)
        self._fc_nominal = replace(self.fc.p)

        thr_lo, thr_hi = alt.get("throttle_range", [0.05, 0.56])
        thr_hi = min(float(thr_hi), self.limits.max_throttle)
        self.angle_full = math.radians(self.limits.angle_limit_deg)
        self.yaw_full = math.radians(self.limits.yaw_curve_dps or self.limits.max_yaw_rate_dps)
        self.observer = AltHoldObserver(alt, angle_full=self.angle_full, yaw_full=self.yaw_full,
                                        throttle_range=(float(thr_lo), thr_hi), control_dt=self.control_dt)
        self.OBS_NAMES = self.observer.names
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.observer.size,), dtype=np.float32)
        self._action_bounds = Box(low=np.array([thr_lo], dtype=np.float32), high=np.array([thr_hi], dtype=np.float32))
        self._obs_bounds = Box(low=-np.ones(self.observer.size, dtype=np.float32),
                               high=np.ones(self.observer.size, dtype=np.float32))   # scales: observer.config()

        z_lo, z_hi = alt.get("z_bounds", [0.05, 2.2])
        self._max_tilt = math.radians(float(alt.get("max_tilt_deg", 75.0)))
        inf = np.inf
        tilt = np.array([self._max_tilt, self._max_tilt, np.pi])
        self._state_bounds = Box(low=np.concatenate([[-inf, -inf, z_lo], -tilt, [-inf] * 6]).astype(np.float32),
                                 high=np.concatenate([[inf, inf, z_hi], tilt, [inf] * 6]).astype(np.float32))
        self._z_bounds = (float(z_lo), float(z_hi))

        self._sticks = np.zeros(3)
        self._target_true = 0.0
        self._prev_a = 0.0
        self._action_change = 0.0
        self._outage_end = -1.0

    def set_difficulty(self, difficulty: float) -> None:
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # ── per-episode variation ──
    def _draw(self, key: str, nominal: float) -> float:
        """A draw from `randomize[key]` ([low, high], or a scalar r for nominal +- r),
        pulled towards nominal by the difficulty."""
        r = self._ranges.get(key)
        if not self.randomize or r is None:
            return nominal
        lo, hi = (nominal - float(r), nominal + float(r)) if np.isscalar(r) else (float(r[0]), float(r[1]))
        return nominal + self.difficulty * (float(self.np_random.uniform(lo, hi)) - nominal)

    def _randomize_episode(self) -> None:
        super()._randomize_episode()        # mass, battery, drag, trim, link delays, gusts
        rng, on, nom = self.np_random, self.randomize, self._fc_nominal
        sc = lambda key: self._draw(key, 1.0)

        # Flight controller, motors and throttle curve
        rate_scale = sc("rate_tau")
        self.fc.p = replace(
            nom,
            thrust_expo=self._draw("thrust_expo", nom.thrust_expo),
            thrust_u0=self._draw("thrust_u0", nom.thrust_u0),
            max_thrust_ratio=self._draw("max_thrust_ratio", nom.max_thrust_ratio),
            motor_tau_up=nom.motor_tau_up * sc("motor_tau_up"),
            motor_tau_down=nom.motor_tau_down * sc("motor_tau_down"),
            level_kp=nom.level_kp * sc("level_kp"),
            angle_ff=self._draw("angle_ff", nom.angle_ff),
            rc_tau=nom.rc_tau * sc("rc_tau"),
            rate_tau=tuple(t * rate_scale for t in nom.rate_tau))
        self.fc.motor_tau_scale = 1.0                  # up and down are drawn separately above
        self.fc.thrust_scale = self.fc.thrust_scale    # rebuild the curve with the new parameters
        self._thrust0 = self.fc.thrust_scale
        self._sag = self._draw("thrust_sag", 0.0)      # share of thrust lost over an episode
        # The real inertia changes; the FC's tuning (fc.inertia) does not.
        self.model.body_inertia[self._body] *= sc("inertia")

        # Response to the sticks, air
        self._angle_scale = sc("angle_scale")          # real angle per commanded angle
        self._lift = self._draw("translational_lift", 0.0)
        self._descent_loss = self._draw("descent_loss", 0.0)
        self._gust_std *= self.difficulty

        # Camera and link
        link, noise = self._link, self._noise
        self._depth_scale = 1.0 + self._draw("depth_scale", 0.0)
        self._camera_drop = draw(rng, link.get("camera_drop", 0.0), on) if self.noise_on else 0.0
        self._outage_rate = draw(rng, link.get("outage_rate", 0.0), on) * self.difficulty if self.noise_on else 0.0
        self._jitter = float(link.get("loop_jitter", 0.0)) if self.noise_on else 0.0
        self._missed = float(link.get("missed_step", 0.0)) if self.noise_on else 0.0
        self._pos_noise = draw(rng, noise["position"], on)
        self._vel_noise = draw(rng, noise["velocity"], on)

        # The pilot: some episodes are hands-off apart from staying in view
        p = self._pilot_cfg
        lo, hi = p.get("aggressiveness", [0.0, 1.0])
        agg = rng.uniform(lo, lo + (hi - lo) * self.difficulty) if on else 0.5 * (lo + hi)
        if on and rng.random() < float(p.get("calm_share", 0.2)):
            agg = 0.0
        self.pilot = SimPilot(rng, p, angle_full=self.angle_full, dt=self.control_dt, aggressiveness=agg)

    # ── physics ──
    def push_command(self, command) -> None:
        """Send [throttle, roll rad, pitch rad, yaw rate rad/s] (skybridge signs) over the radio link."""
        rng = self.np_random
        if self._missed and rng.random() < self._missed:
            return                                      # the laptop missed this tick
        delay = self._action_delay + (rng.uniform(-self._jitter, self._jitter) if self._jitter else 0.0)
        s = self._angle_scale
        self._command_line.push(self._t + delay, (float(command[0]), float(command[1]) * s,
                                                  float(command[2]) * s, -float(command[3])))

    def _update_disturbances(self) -> None:
        """Once per RC tick: the wandering air, and the battery sagging."""
        if self._gust_std:
            a = self.control_dt / self._aero.get("gust_tau", 1.0)
            self._gust += -a * self._gust + self._gust_std * math.sqrt(2.0 * a) * self.np_random.normal(0, 1, 3)
        if self._sag:
            self.fc.thrust_scale = self._thrust0 * (1.0 - self._sag * min(self._t / self._episode_time, 1.0))

    def _thrust_factor(self) -> float:
        """Translational lift with horizontal speed, thrust loss when sinking into the own wake."""
        v = self.data.qvel
        k = 1.0 + self._lift * min(math.hypot(v[0], v[1]), 1.5)
        sink = -v[2] - 0.5
        if sink > 0.0:
            k *= 1.0 - self._descent_loss * min(sink, 1.0)
        return k

    def _substep(self, armed: bool = True, push: bool = True) -> None:
        """One physics step: flight controller, motors, air, camera."""
        data = self.data
        if armed:
            roll, pitch, _ = _euler(data.qpos[3:7])
            thrust = self.fc.step(self._command_line.get(self._t), roll, pitch, data.qvel[3:6].tolist(), self.dt)
            k = self._thrust_factor()
            data.ctrl[:] = [f * k for f in thrust]
        else:
            data.ctrl[:] = 0.0
        force = -self._mass * self._drag * data.qvel[0:3]
        if push:
            force = force + self._mass * self._gust
        data.xfrc_applied[self._body, 0:3] = force
        mujoco.mj_step(self.model, data)
        self._t += self.dt
        self._sense_camera(self.dt)

    def _advance(self, command: np.ndarray) -> np.ndarray:
        self.push_command(command)
        self._update_disturbances()
        for _ in range(self.frame_skip):
            self._substep()
        self._state.set_from_mujoco(self.data.qpos[:7], self.data.qvel[:6])
        return self.data.ctrl.copy()

    # ── camera ──
    def _sense_camera(self, dt: float) -> None:
        rng = self.np_random
        self._vel_filtered += min(dt / self._velocity_tau, 1.0) * (self.data.qvel[0:3] - self._vel_filtered)
        if self._outage_rate and self._t >= self._outage_end and rng.random() < self._outage_rate * dt:
            self._outage_end = self._t + draw(rng, self._link.get("outage_duration", [0.1, 0.5]))
        if self._t < self._next_camera:
            return
        self._next_camera += self._link["camera_period"]
        if self._t < self._outage_end or (self._camera_drop and rng.random() < self._camera_drop):
            return
        z = float(self.data.qpos[2]) * self._depth_scale
        vz = float(self._vel_filtered[2]) * self._depth_scale
        if self.noise_on:
            z += rng.normal(0, self._pos_noise)
            vz += rng.normal(0, self._vel_noise)
        self._camera_line.push(self._t + self._camera_latency, (z, vz, self._t))

    def altitude_measurement(self) -> tuple[float, float, float]:
        """(z, vz, age): the newest camera altitude the laptop has, and how old it is."""
        z, vz, captured = self._camera_line.get(self._t)
        return z, vz, self._t - captured

    # ── RL interface ──
    def _get_obs(self) -> np.ndarray:
        """AltHoldObserver.observe on the latest camera data and the sticks about to be sent."""
        z, vz, age = self.altitude_measurement()
        return self.observer.observe(z, vz, age, self._sticks)

    def _get_reward(self) -> float:
        """Gaussians of the true altitude error (broad to come back, narrow to stay),
        minus costs for climbing/sinking and for jerky throttle."""
        r = self._reward
        err = float(self._state.position[2]) - self._target_true
        vz = float(self._state.velocity[2])
        reward = r.get("alive", 0.2) + math.exp(-(err / r.get("alt_scale", 0.3)) ** 2)
        reward += r.get("alt_fine", 1.0) * math.exp(-(err / r.get("alt_fine_scale", 0.05)) ** 2)
        reward -= r.get("climb", 0.1) * vz * vz
        reward -= r.get("action_rate", 1.0) * self._action_change ** 2
        return float(reward)

    def _is_terminated(self) -> bool:
        state = self._state.vec()
        if not np.isfinite(state).all():
            return True
        if not self._z_bounds[0] <= state[2] <= self._z_bounds[1]:
            return True
        return bool(max(abs(state[3]), abs(state[4])) > self._max_tilt)

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float64).ravel()[0], -1.0, 1.0))
        self._action_change = a - self._prev_a
        self._prev_a = a
        throttle = self.observer.throttle(a)
        command = np.array([throttle, *self._sticks])
        self.observer.push_throttle(a)
        motors = self._advance(command)
        self._step_count += 1

        s = self._state
        self._sticks = self.pilot.sticks(s.position, s.velocity, _euler(self.data.qpos[3:7])[2])
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_episode_steps
        z, vz, age = self.altitude_measurement()
        info = {"state": s.vec().copy(), "motor_commands": motors, "target": self.target_state.position.copy(),
                "command": command, "measured": np.array([z, vz, age]), "target_true_z": self._target_true}
        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Start an episode in flight, engaged at the altitude the camera reads.

        options: `state` (12D true state), `command` ([throttle, roll, pitch, yaw rate] held
        at engagement; default: about hover, pilot's sticks), `target_z` (measured units).
        """
        super(HoverEnv, self).reset(seed=seed)
        options = options or {}
        rng, cfg = self.np_random, self._alt
        self._step_count = 0
        self._t = 0.0
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_episode()
        self._episode_time = self.max_episode_steps * self.control_dt
        self._outage_end = -1.0

        if "state" in options:
            state = np.asarray(options["state"], dtype=np.float64)
        else:
            state = np.zeros(12)
            xy = float(cfg.get("start_xy", 0.6))
            tilt = math.radians(float(cfg.get("start_tilt_deg", 10.0)))
            state[0:2] = rng.uniform(-xy, xy, 2)
            state[2] = rng.uniform(*cfg.get("start_z", [0.4, 1.4]))
            state[3:5] = rng.uniform(-tilt, tilt, 2)
            state[5] = rng.uniform(-math.pi, math.pi)
            state[6:8] = rng.uniform(-0.5, 0.5, 2)
            state[8] = rng.uniform(*cfg.get("start_vz", [-0.5, 0.5]))
            state[9:12] = rng.uniform(-0.5, 0.5, 3)
        initial = QuadState()
        initial.state[:] = state
        self.set_state(*initial.get_mujoco_state())
        self._state.set_from_mujoco(self.data.qpos[:7], self.data.qvel[:6])

        pos, vel = self.data.qpos[0:3], self.data.qvel[0:3]
        self._vel_filtered = vel.copy()
        self._camera_line.reset((float(pos[2]) * self._depth_scale, float(vel[2]) * self._depth_scale, 0.0))
        self._next_camera = rng.uniform(0, self._link["camera_period"])

        if "command" in options:
            command = np.asarray(options["command"], dtype=np.float64)
            throttle, self._sticks = float(command[0]), command[1:4].copy()
        else:
            self._sticks = self.pilot.sticks(pos, vel, _euler(self.data.qpos[3:7])[2])
            throttle = self.fc.throttle_for_hover(self._mass, self.spec.g)
            if self.randomize:      # the pilot's throttle when handing over is only about right
                err = float(cfg.get("start_throttle_error", 0.1)) * self.difficulty
                throttle *= 1.0 + rng.uniform(-err, err)
        s = self._angle_scale
        self.fc.reset(throttle=throttle, roll=self._sticks[0] * s, pitch=self._sticks[1] * s)
        self._command_line.reset((throttle, self._sticks[0] * s, self._sticks[1] * s, -self._sticks[2]))

        if "target_z" in options:
            target = float(options["target_z"])
        else:
            target = self.altitude_measurement()[0]
            if self.randomize and rng.random() < float(cfg.get("target_offset_share", 0.5)):
                target += self.difficulty * rng.uniform(*cfg.get("target_offset", [-0.3, 0.3]))
        self._target_true = target / self._depth_scale
        self.target_state = QuadState()
        self.target_state.state[0:3] = [pos[0], pos[1], self._target_true]

        self.observer.engage(target, throttle, self._sticks)
        self._prev_a = self.observer.action_for(throttle)
        self._action_change = 0.0
        obs = self._get_obs()
        info = {"state": self._state.vec().copy(), "target": self.target_state.position.copy(),
                "target_true_z": self._target_true}
        return obs, info
