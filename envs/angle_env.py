"""Hover environment for a drone flown through Betaflight angle mode.

This is the laptop's view of a drone like the BETAFPV Air65 behind skybridge:

    action       [throttle, roll angle, pitch angle, yaw rate], at the RC rate (50 Hz)
    observation  what the laptop can measure: position and velocity from the
                 depth camera, roll/pitch/heading from telemetry, the running
                 integral of the position error, and its own last few commands.
                 No body rates.

The integral is there because the real drone is never trimmed level and its
hover throttle moves with the battery: a policy that only sees the current
error would settle beside the target. It is kept in the world frame (it must not
turn with the drone while it yaws) and only grows near the target, so it does
not wind up on the way there.

The command history is there because of the delays: a command takes 30-70 ms to
reach the drone and the attitude the policy sees is 100-200 ms old. A policy
that only knows its last command cannot tell which of its earlier ones are still
on their way, reacts to stale data and oscillates. `env.action_history` sets how
many commands it sees (1 gives the original 16D observation).

Sign conventions, chosen to match skybridge's `set_angle()`:
    roll  +  tilts right          pitch +  tilts the nose down (flies forward)
    yaw rate + turns right (clockwise from above)
    heading  + is to the left (counter-clockwise), as in skybridge.heading

The flight controller, the radio link delay, the sensor rates and their latency
are simulated (see betaflight.py, sim2real.py), with per-episode randomization.
"""
from __future__ import annotations

import math

import numpy as np
import mujoco
from gymnasium.spaces import Box

from drones import DroneSpec, resolve
from drones.constraints import CommandLimits, load_limits
from utils import QuadState

from .betaflight import BetaflightAngleController, BetaflightParams
from .hover_env import HoverEnv
from .sim2real import DelayLine, draw, scale, symmetric


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _euler(quat) -> tuple[float, float, float]:
    """Roll, pitch, yaw from a MuJoCo quaternion (w, x, y, z)."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(min(max(2 * (w * y - z * x), -1.0), 1.0))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class AngleModeEnv(HoverEnv):
    """Hover at a target, commanding throttle, roll, pitch and yaw rate.

    Args:
        drone: a registry name or DroneSpec with `interface: betaflight_angle`.
        limits: command limits; default is read from skybridge's config.
        randomize: per-episode variation of the model, delays and sensor errors.
            Off gives the nominal drone, for validation against flight logs.
        noise: sensor noise on or off.
    """

    SENSED_NAMES = ("err_forward", "err_left", "err_up", "vel_forward", "vel_left", "vel_up",
                    "roll", "pitch", "heading_err", "int_forward", "int_left", "int_up")
    ACTION_NAMES = ("throttle", "roll", "pitch", "yaw_rate")

    def __init__(self, drone: str | DroneSpec = "air65", render_mode: str | None = None,
                 max_episode_steps: int | None = None, limits: CommandLimits | None = None,
                 randomize: bool = True, noise: bool = True):
        spec = resolve(drone)
        if spec.interface != "betaflight_angle":
            raise ValueError(f"drone {spec.name!r} is not flown through Betaflight angle mode")
        cfg = spec.section("env")
        super().__init__(render_mode=render_mode, drone=spec,
                         max_episode_steps=max_episode_steps or int(cfg.get("episode_steps", 500)))
        self.limits = limits or load_limits(spec.name)
        self.randomize = randomize
        self.noise_on = noise
        self.frame_skip = spec.frame_skip
        self.control_dt = spec.control_dt
        self._aero = spec.section("aero")
        self._link = spec.section("link")
        self._noise = spec.section("noise")
        self._ranges = spec.section("randomize")
        self._reward = spec.section("reward")

        # ── Action: normalized [-1, 1] -> skybridge units ──
        thr_lo, thr_hi = cfg.get("throttle_range", [0.05, 0.56])
        thr_hi = min(thr_hi, self.limits.max_throttle)
        angle = np.deg2rad(self.limits.max_angle_deg)
        yaw_rate = np.deg2rad(min(self.limits.max_yaw_rate_dps, cfg.get("max_yaw_rate_dps", 200.0)))
        self._action_bounds = Box(
            low=np.array([thr_lo, -angle, -angle, -yaw_rate], dtype=np.float32),
            high=np.array([thr_hi, angle, angle, yaw_rate], dtype=np.float32))
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # ── Observation: what is sensed, then the last `history` commands, newest first ──
        self._history = int(cfg.get("action_history", 1))
        if self._history < 1:
            raise ValueError("env.action_history must be at least 1")
        self.OBS_NAMES = self.SENSED_NAMES + tuple(
            f"prev{'' if k == 0 else k + 1}_{name}" for k in range(self._history) for name in self.ACTION_NAMES)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(len(self.OBS_NAMES),), dtype=np.float32)
        pos, vel, tilt = cfg.get("obs_pos", 2.0), cfg.get("obs_vel", 3.0), np.deg2rad(cfg.get("obs_tilt_deg", 60.0))
        self._int_limit = float(cfg.get("obs_int", 1.0))    # m s: the integral saturates here
        zone = cfg.get("int_zone")                          # m: integrate an axis only while its error is below this
        self._int_zone = float(zone) if zone is not None else np.inf
        self._obs_scale = np.array([pos] * 3 + [vel] * 3 + [tilt, tilt, np.pi] + [self._int_limit] * 3,
                                   dtype=np.float32)
        self._err_int = np.zeros(3)

        # ── Episode ranges (true state: pos, att, vel, body rates) ──
        box = lambda key: Box(low=np.array(cfg[key]["low"], dtype=np.float32),
                              high=np.array(cfg[key]["high"], dtype=np.float32))
        self._initial_state_bounds = box("initial")
        self._target_pos_bounds = box("target")
        self._pos_bounds = box("bounds")
        self._max_tilt = np.deg2rad(cfg.get("max_tilt_deg", 60.0))
        self._state = QuadState()
        # Recorded in a run's config.json: the scale of each observation (see OBS_NAMES)
        # and the true-state box that ends an episode.
        full_scale = np.concatenate([self._obs_scale, np.ones(4 * self._history, dtype=np.float32)])
        self._obs_bounds = Box(low=-full_scale, high=full_scale)
        free = np.full(6, np.inf, dtype=np.float32)
        tilt = np.array([self._max_tilt, self._max_tilt, np.pi], dtype=np.float32)
        self._state_bounds = Box(low=np.concatenate([self._pos_bounds.low, -tilt, -free]),
                                 high=np.concatenate([self._pos_bounds.high, tilt, free]))

        # ── Model: params.yaml is the source of truth for mass, inertia and thrust ──
        self._body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.model.actuator_ctrlrange[:, 1] = 4.0 * spec.max_motor_thrust
        self.model.actuator_gear[:, 5] = spec.yaw_torque_coeff * np.array([1, -1, 1, -1])
        self.fc = BetaflightAngleController(
            BetaflightParams.from_dict(spec.section("betaflight")), mass=spec.mass,
            inertia=spec.inertia, arm_length=spec.arm_length,
            yaw_torque_coeff=spec.yaw_torque_coeff, g=spec.g)
        self._command_line, self._camera_line, self._attitude_line = DelayLine(), DelayLine(), DelayLine()
        self._apply_model(1.0)
        self._t = 0.0
        self._mass = spec.mass
        self._drag = np.zeros(3)

    # ── model ──
    def _apply_model(self, mass_scale: float) -> None:
        self._mass = self.spec.mass * mass_scale
        self.model.body_mass[self._body] = self._mass
        self.model.body_inertia[self._body] = np.asarray(self.spec.inertia) * mass_scale

    def _randomize_episode(self) -> None:
        rng, r, on = self.np_random, self._ranges, self.randomize
        self._apply_model(scale(rng, r, "mass", on))
        self.fc.thrust_scale = scale(rng, r, "thrust", on)
        self.fc.motor_tau_scale = scale(rng, r, "motor_tau", on)
        self.fc.angle_trim = tuple(np.deg2rad(symmetric(rng, r, "angle_trim_deg", on, size=2)))
        drag = scale(rng, r, "drag", on)
        self._drag = drag * np.array([self._aero.get("drag_xy", 0.0)] * 2 + [self._aero.get("drag_z", 0.0)])
        link = self._link
        self._action_delay = draw(rng, link["action_delay"], on)
        self._camera_latency = draw(rng, link["camera_latency"], on)
        self._attitude_latency = draw(rng, link["attitude_latency"], on)
        self._velocity_tau = draw(rng, link["velocity_tau"], on)
        self._heading_bias = np.deg2rad(symmetric(rng, r, "heading_error_deg", on))
        # Slow air disturbance: an acceleration that wanders (Ornstein-Uhlenbeck), per axis.
        self._gust_std = draw(rng, self._aero.get("gust_accel", 0.0), on) if self.noise_on else 0.0
        self._gust = np.zeros(3)

    # ── commands ──
    def denormalize_action(self, action) -> np.ndarray:
        """Normalized action -> [throttle 0..1, roll rad, pitch rad, yaw rate rad/s], skybridge signs."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        low, high = self._action_bounds.low, self._action_bounds.high
        return (a + 1.0) / 2.0 * (high - low) + low

    def normalize_command(self, command) -> np.ndarray:
        low, high = self._action_bounds.low, self._action_bounds.high
        return (2.0 * (np.asarray(command, dtype=np.float64) - low) / (high - low) - 1.0).astype(np.float32)

    # ── sensors ──
    def _sense(self, dt: float, roll: float, pitch: float, yaw: float) -> None:
        """Advance the camera and telemetry by one physics step."""
        vel = self.data.qvel[0:3]
        self._vel_filtered += min(dt / self._velocity_tau, 1.0) * (vel - self._vel_filtered)
        if self._t >= self._next_camera:
            rng, n = self.np_random, self._noise
            self._next_camera += self._link["camera_period"]
            if not self.noise_on:
                self._camera_line.push(self._t + self._camera_latency,
                                       (self.data.qpos[0:3].copy(), self._vel_filtered.copy(), self._t))
            elif rng.random() >= self._link.get("camera_drop", 0.0):
                self._camera_line.push(self._t + self._camera_latency,
                                       (self.data.qpos[0:3] + rng.normal(0, n["position"], 3),
                                        self._vel_filtered + rng.normal(0, n["velocity"], 3), self._t))
        if self._t >= self._next_attitude:
            rng = self.np_random
            self._next_attitude += draw(rng, self._link["attitude_period"], self.noise_on)
            sample = np.array([roll, pitch, yaw + self._heading_bias])
            if self.noise_on:
                sample += rng.normal(0, math.radians(self._noise["attitude_deg"]), 3)
            self._attitude_line.push(self._t + self._attitude_latency, sample)

    def measurements(self) -> dict[str, np.ndarray]:
        """What the laptop knows right now (delayed, sampled, noisy)."""
        pos, vel, captured = self._camera_line.get(self._t)
        att = self._attitude_line.get(self._t)
        return {"pos": pos, "vel": vel, "age": self._t - captured,
                "roll": att[0], "pitch": att[1], "heading": att[2]}

    def _get_obs(self) -> np.ndarray:
        """Target error and velocity in the heading frame, roll, pitch, heading error,
        integral of the error (heading frame, clipped), then the last commands, newest first."""
        m = self.measurements()
        c, s = np.cos(m["heading"]), np.sin(m["heading"])
        to_heading = lambda v: np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])
        err_world = self.target_state.position - m["pos"]
        err = to_heading(err_world)
        vel = to_heading(m["vel"])
        self._heading_err = _wrap(self.target_heading - m["heading"])
        if self._integrate:     # once per policy step, not at reset
            near = np.abs(err_world) < self._int_zone
            self._err_int = np.clip(self._err_int + near * err_world * self.control_dt,
                                    -self._int_limit, self._int_limit)
        sensed = np.concatenate([err, vel, [m["roll"], m["pitch"], self._heading_err],
                                 to_heading(self._err_int)]) / self._obs_scale
        return np.concatenate([np.clip(sensed, -1.0, 1.0), self._action_hist.ravel()]).astype(np.float32)

    # ── reward and termination (true state) ──
    def _get_reward(self) -> float:
        """Gaussians of the true position error (a broad one to find the target, a narrow one
        to stay on it), minus costs for jerky commands, velocity and heading error."""
        r = self._reward
        pos_error = float(np.linalg.norm(self._state.position - self.target_state.position))
        reward = r.get("alive", 0.0) + np.exp(-(pos_error / r.get("pos_scale", 1.0)) ** 2)
        if r.get("pos_fine", 0.0):
            reward += r["pos_fine"] * np.exp(-(pos_error / r["pos_fine_scale"]) ** 2)
        reward -= r.get("action_rate", 0.0) * float(np.mean(self._action_change ** 2))
        reward -= r.get("velocity", 0.0) * float(np.dot(self._state.velocity, self._state.velocity))
        reward -= r.get("heading", 0.0) * (_wrap(self.target_heading - float(self._state.attitude[2])) / np.pi) ** 2
        return float(reward)

    def _is_terminated(self) -> bool:
        state = self._state.vec()
        if not np.isfinite(state).all():
            return True
        if not self._pos_bounds.contains(state[0:3].astype(np.float32)):
            return True
        return bool(max(abs(state[3]), abs(state[4])) > self._max_tilt)

    # ── stepping ──
    def _advance(self, command: np.ndarray) -> np.ndarray:
        """Send one RC command (skybridge signs) and run the physics up to the next one."""
        sim_command = (float(command[0]), float(command[1]), float(command[2]), -float(command[3]))   # yaw: right+ -> left+
        self._command_line.push(self._t + self._action_delay, sim_command)
        dt, data, body = self.dt, self.data, self._body
        drag = -self._mass * self._drag
        if self._gust_std:
            a = self.control_dt / self._aero.get("gust_tau", 1.0)
            self._gust += -a * self._gust + self._gust_std * math.sqrt(2.0 * a) * self.np_random.normal(0, 1, 3)
        push = self._mass * self._gust
        roll, pitch, yaw = _euler(data.qpos[3:7])
        for _ in range(self.frame_skip):
            thrust = self.fc.step(self._command_line.get(self._t), roll, pitch, data.qvel[3:6].tolist(), dt)
            data.ctrl[:] = thrust
            data.xfrc_applied[body, 0:3] = drag * data.qvel[0:3] + push
            mujoco.mj_step(self.model, data)
            self._t += dt
            roll, pitch, yaw = _euler(data.qpos[3:7])
            self._sense(dt, roll, pitch, yaw)
        thrust = np.array(thrust)
        self._state.set_from_mujoco(self.data.qpos[:7], self.data.qvel[:6])
        return thrust

    def step_command(self, command) -> tuple[np.ndarray, dict]:
        """Step with a physical command [throttle, roll rad, pitch rad, yaw rate rad/s]. For replay."""
        thrust = self._advance(np.asarray(command, dtype=np.float64))
        return self._state.vec().copy(), {"motor_commands": thrust.copy(), **self.measurements()}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._action_change = action - self._prev_action
        self._action_hist = np.roll(self._action_hist, 1, axis=0)
        self._action_hist[0] = action
        self._prev_action = action.copy()
        command = self.denormalize_action(action)
        thrust = self._advance(command)
        self._step_count += 1

        self._integrate = True
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_episode_steps
        info = {"state": self._state.vec().copy(), "motor_commands": thrust.copy(),
                "target": self.target_state.position.copy(), "command": command}
        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Start an episode in flight: random state, target and model variation.

        options: `state` (12D true state) and `target` (3D) override the random draw;
        `command` is the RC command being held when the episode starts (default: hover, level).
        """
        super(HoverEnv, self).reset(seed=seed)
        options = options or {}
        self._step_count = 0
        self._t = 0.0
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_episode()

        initial = QuadState()
        if "state" in options:
            initial.state[:] = np.asarray(options["state"], dtype=np.float32)
        else:
            initial.random_reset(self.np_random, self._initial_state_bounds)
        self.set_state(*initial.get_mujoco_state())
        self._state.set_from_mujoco(self.data.qpos[:7], self.data.qvel[:6])

        self.target_state = QuadState()
        target = options["target"] if "target" in options else self.np_random.uniform(
            self._target_pos_bounds.low, self._target_pos_bounds.high)
        self.target_state.state[0:3] = np.asarray(target, dtype=np.float32)
        self.target_heading = float(options.get("target_heading", 0.0))

        # The pilot (or a previous controller) was holding hover when the policy takes over.
        hover = self.fc.throttle_for_hover(self._mass, self.spec.g)
        start = np.asarray(options.get("command", [hover, 0.0, 0.0, 0.0]), dtype=np.float64)
        self.fc.reset(throttle=start[0], roll=start[1], pitch=start[2])
        self._command_line.reset((start[0], start[1], start[2], -start[3]))
        self._prev_action = np.clip(self.normalize_command(start), -1.0, 1.0)
        self._action_hist = np.tile(self._prev_action, (self._history, 1))
        self._action_change = np.zeros(4, dtype=np.float32)

        pos, vel = self.data.qpos[0:3], self.data.qvel[0:3]
        roll, pitch, yaw = _euler(self.data.qpos[3:7])
        self._vel_filtered = vel.copy()
        self._camera_line.reset((pos.copy(), vel.copy(), 0.0))
        self._attitude_line.reset(np.array([roll, pitch, yaw + self._heading_bias]))
        self._next_camera = self.np_random.uniform(0, self._link["camera_period"])
        self._next_attitude = self.np_random.uniform(0, 0.06)

        self._err_int = np.zeros(3)
        self._integrate = False
        obs = self._get_obs()
        info = {"state": self._state.vec().copy(), "target": self.target_state.position.copy()}
        return obs, info
