#!/usr/bin/env python3
"""Fly the simulated Air65 with the real radio, to judge how close the sim feels.

    Boxer (EdgeTX "USB Joystick") --USB--> /dev/input/js0 --> this script --> MuJoCo

The sticks are read exactly as skybridge's rc_fly.py reads them and turned into
what Betaflight would make of them in angle mode:

    roll / pitch   stick deflection x angle_limit       (skybridge configs/drone.yaml)
    yaw            stick deflection x srate*10 deg/s     (ACTUAL rates, linear)
    throttle       0..1 of 1000..2000 us, min_check applies

That command then goes through the same model the policy trains on
(envs/angle_env.py): radio link delay, Betaflight angle/rate loops, airmode
mixer, motor lag, the fitted throttle curve, drag and the slow air disturbance.
What is NOT simulated: ground effect, prop wash, crash physics (touching the
ground just sets the drone down level).

Switches, as on the real drone:
    CH5 (AUX1) arm      high = armed; arming needs the switch seen OFF and throttle low
    CH6 (AUX2) mode     only angle mode is simulated; acro shows a warning

Keys in the viewer window:
    R   reset to the ground at the origin (disarm first to re-arm)
    C   toggle between the fixed pilot view (default) and a camera following the drone

Thrust tuning, live (the letters are all taken by MuJoCo's viewer):
    Up / Down          maximum thrust, as a multiple of hover thrust (T/W at full stick)
    PgUp / PgDn        thrust expo: how steeply thrust grows with throttle around hover
    Right / Left       vertical drag, 1/s: how quickly a climb or a sink dies out
Hover throttle stays where it was measured whatever you change. On exit the
values are printed as params.yaml lines; only then do they affect training.

The pilot view is fixed, as if you stood in a room behind the takeoff pad: the
drone starts with its nose pointing away from you, so the sticks act as in line
of sight from behind. Move the pilot with --eye / --look, the room with --room.

Altitude hold with a trained policy (--policy): you fly roll, pitch and yaw, the
policy flies throttle. Engage in the air with Enter (or a radio switch, see
--engage-axis): it holds the altitude the simulated camera reads at that moment,
and your throttle stick is ignored until you disengage. If the camera data gets
older than --camera-timeout the throttle goes to zero and stays there until you
disengage and engage again, as it will on the real drone.

Usage:
    python rc_sim.py                        # nominal drone, nominal delays
    python rc_sim.py --policy models_trained/air65_althold_<stamp>   # altitude hold, Enter toggles
    python rc_sim.py --randomize            # a random drone/battery/delay draw, as in training
    python rc_sim.py --hover-throttle 0.32  # e.g. a sagging battery
    python rc_sim.py --record flight.npz    # save sticks and true state for comparison
    python rc_sim.py --max-thrust 2.5 --drag-z 0.8   # start from other thrust settings
"""
from __future__ import annotations

import argparse
import array
import fcntl
import math
import os
import select
import struct
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataclasses import replace  # noqa: E402
import json  # noqa: E402

from drones import spec_from_params  # noqa: E402
from drones.constraints import load_limits, skybridge_config_path  # noqa: E402
from envs.althold_env import AltHoldAssist, AltHoldEnv  # noqa: E402
from envs.angle_env import AngleModeEnv, _euler  # noqa: E402

GROUND_Z = 0.013        # m: body origin height when the battery rests on the floor
CRASH_SPEED = 3.0       # m/s into the ground
CRASH_TILT = math.radians(80)
ARENA = 8.0             # m: beyond this from the origin (or above it) counts as flown away


# ── radio ────────────────────────────────────────────────────────────────────
class Joystick:
    """Background reader of a Linux joydev device (same protocol as skybridge.rc.joystick)."""

    EVENT = struct.Struct("IhBB")   # time_ms, value, type, number
    AXIS, INIT = 0x02, 0x80

    def __init__(self, device: str):
        try:
            self.fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        except FileNotFoundError:
            sys.exit(f"{device} not found: is the radio plugged in, with 'USB Joystick' selected on its screen?")
        name = array.array("B", [0] * 128)
        fcntl.ioctl(self.fd, 0x80006A13 + (128 << 16), name)
        self.name = name.tobytes().split(b"\0", 1)[0].decode(errors="replace")
        n = array.array("B", [0])
        fcntl.ioctl(self.fd, 0x80016A11, n)
        self.axes = [0] * n[0]
        self._seen: set[int] = set()
        self.alive = True
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        buf = bytearray()
        while True:
            try:
                if not select.select([self.fd], [], [], 0.1)[0]:
                    continue
                buf += os.read(self.fd, self.EVENT.size * 64)
            except BlockingIOError:
                continue
            except OSError:
                self.alive = False
                return
            n = len(buf) // self.EVENT.size * self.EVENT.size
            with self._lock:
                for _, value, kind, number in self.EVENT.iter_unpack(bytes(buf[:n])):
                    if kind & self.AXIS and number < len(self.axes):
                        self.axes[number] = value
                        self._seen.add(number)
            del buf[:n]

    def read(self) -> list[int] | None:
        """Axis values (-32767..32767), or None until every axis has reported."""
        with self._lock:
            return list(self.axes) if len(self._seen) == len(self.axes) else None


def axis_us(value: int) -> float:
    """Axis -> pulse width as Betaflight decodes it: -100..+100 % = 988..2012 us."""
    return 1500.0 + value / 32767.0 * 512.0


class Sticks:
    """Radio axes -> Betaflight angle-mode command [throttle, roll rad, pitch rad, yaw rate rad/s]."""

    def __init__(self, sky: dict):
        rc_map = sky.get("rc", {}).get("mapping", {})
        axis = lambda ch, default: int(rc_map.get(ch, {}).get("axis", default))
        invert = lambda ch: bool(rc_map.get(ch, {}).get("invert", False))
        self.channels = {name: (axis(ch, i), invert(ch)) for name, ch, i in
                         [("roll", "ch1", 0), ("pitch", "ch2", 1), ("throttle", "ch3", 2), ("yaw", "ch4", 3),
                          ("arm", "ch5", 4), ("mode", "ch6", 5)]}
        bf = sky.get("betaflight", {})
        rates = bf.get("rates", {})
        per_axis = lambda key, ax: float(rates.get(key, {}).get(ax, 67) if isinstance(rates.get(key), dict)
                                         else rates.get(key, 67))
        self.angle_limit = math.radians(float(bf.get("angle_limit", 20)))
        self.yaw_max = math.radians(10.0 * per_axis("srate", "yaw"))
        if per_axis("rc_rate", "yaw") != per_axis("srate", "yaw"):
            print("warning: yaw rc_rate != srate, the yaw curve is treated as linear anyway")
        modes = sky.get("modes", {})
        self.arm_range = modes.get("arm", {}).get("range", [1700, 2100])
        self.angle_range = modes.get("angle", {}).get("range", [900, 1300])

    def us(self, axes, name) -> float:
        idx, inv = self.channels[name]
        value = axes[idx] if idx < len(axes) else 0
        return axis_us(-value if inv else value)

    def command(self, axes) -> np.ndarray:
        deflect = lambda name: float(np.clip((self.us(axes, name) - 1500.0) / 500.0, -1.0, 1.0))
        throttle = float(np.clip((self.us(axes, "throttle") - 1000.0) / 1000.0, 0.0, 1.0))
        return np.array([throttle, deflect("roll") * self.angle_limit, deflect("pitch") * self.angle_limit,
                         deflect("yaw") * self.yaw_max])

    def arm_switch(self, axes) -> bool:
        lo, hi = self.arm_range
        return lo <= self.us(axes, "arm") < hi

    def angle_mode(self, axes) -> bool:
        lo, hi = self.angle_range
        return lo <= self.us(axes, "mode") < hi


# ── thrust tuning ────────────────────────────────────────────────────────────
class ThrustTuning:
    """The throttle -> climb knobs a pilot can feel, applied on top of params.yaml."""

    STEPS = {"max_thrust_ratio": 0.2, "thrust_expo": 0.1, "drag_z": 0.1}
    LIMITS = {"max_thrust_ratio": (1.2, 6.0), "thrust_expo": (0.5, 3.0), "drag_z": (0.0, 5.0)}

    def __init__(self, env: AngleModeEnv, max_thrust=None, expo=None, drag_z=None, soft_cap=None):
        p = env.fc.p
        self.yaml = {"max_thrust_ratio": p.max_thrust_ratio, "thrust_expo": p.thrust_expo,
                     "thrust_soft_cap": p.thrust_soft_cap, "drag_z": env._aero.get("drag_z", 0.0)}
        self.values = dict(self.yaml)
        for key, value in (("max_thrust_ratio", max_thrust), ("thrust_expo", expo),
                           ("drag_z", drag_z), ("thrust_soft_cap", soft_cap)):
            if value is not None:
                self.values[key] = float(value)

    def nudge(self, key: str, sign: int):
        lo, hi = self.LIMITS[key]
        self.values[key] = round(min(max(self.values[key] + sign * self.STEPS[key], lo), hi), 3)

    def apply(self, env: AngleModeEnv):
        fc, v = env.fc, self.values
        fc.p.max_thrust_ratio = v["max_thrust_ratio"]
        fc.p.thrust_expo = v["thrust_expo"]
        fc.p.thrust_soft_cap = v["thrust_soft_cap"]
        fc.thrust_scale = fc.thrust_scale           # rebuilds the throttle curve around hover
        nominal = env._aero.get("drag_z", 0.0)
        scale = env._drag[2] / nominal if nominal else 1.0     # keep a randomized episode's draw
        env._drag[2] = v["drag_z"] * scale

    def summary(self, env: AngleModeEnv) -> str:
        fc, weight = env.fc, env._mass * env.spec.g
        tw = lambda u: 4 * fc.motor_thrust_for(u) / weight
        climb = lambda u: (tw(u) - 1) * env.spec.g / env._drag[2] if env._drag[2] else float("inf")
        v = self.values
        return (f"max T/W {v['max_thrust_ratio']:.1f}  expo {v['thrust_expo']:.2f}  drag_z {v['drag_z']:.2f}"
                f"  soft cap {v['thrust_soft_cap']:g}  |  T/W at stick 40/50/70/100%: "
                + " ".join(f"{tw(u):.2f}" for u in (0.4, 0.5, 0.7, 1.0))
                + f"  |  top climb {climb(1.0):.1f} m/s")

    def changes(self) -> list[str]:
        """params.yaml lines for the values that differ from it."""
        section = {"max_thrust_ratio": "betaflight", "thrust_expo": "betaflight",
                   "thrust_soft_cap": "betaflight", "drag_z": "aero"}
        return [f"{section[k]}.{k}: {v:g}    (params.yaml: {self.yaml[k]:g})"
                for k, v in self.values.items() if abs(v - self.yaml[k]) > 1e-9]


# ── simulation ───────────────────────────────────────────────────────────────
class RcSim:
    """The angle-mode env, stepped by hand: arming, a floor, and no episode ends."""

    def __init__(self, env: AngleModeEnv, min_check: float, delay: float | None = None,
                 tuning: ThrustTuning | None = None):
        self.env = env
        self.min_check = min_check
        self.delay = delay        # s: fixed radio link delay instead of the env's draw
        self.tuning = tuning
        self.reset()

    def reset(self):
        env = self.env
        start = np.zeros(12)
        start[2] = GROUND_Z
        env.reset(options={"state": start, "target": [0.0, 0.0, 0.5], "command": [0.0, 0.0, 0.0, 0.0]})
        if self.delay is not None:
            env._action_delay = self.delay
        if self.tuning is not None:
            self.tuning.apply(env)
        self.armed = False
        self.arm_allowed = False      # the arm switch has to be seen OFF first
        self.crashed = ""
        self.on_ground = True
        self.max_speed = 0.0

    def update_arming(self, switch: bool, throttle: float):
        if not switch:
            self.arm_allowed = True
            if self.armed:
                self.armed = False
            return
        if not self.armed and self.arm_allowed and not self.crashed:
            if throttle < self.min_check:
                self.armed = True
                env = self.env
                env.fc.reset(throttle=0.0)
                env._command_line.reset((0.0, 0.0, 0.0, 0.0))
            self.arm_allowed = throttle < self.min_check     # throttle up -> cycle the switch

    def step(self, command: np.ndarray):
        """One RC tick (1/50 s) of physics, like AngleModeEnv._advance plus a floor."""
        env, data, body = self.env, self.env.data, self.env._body
        sim_command = (float(command[0]), float(command[1]), float(command[2]), -float(command[3]))
        env._command_line.push(env._t + env._action_delay, sim_command)
        drag = -env._mass * env._drag
        if env._gust_std:
            a = env.control_dt / env._aero.get("gust_tau", 1.0)
            env._gust += -a * env._gust + env._gust_std * math.sqrt(2 * a) * env.np_random.normal(0, 1, 3)
        if isinstance(env, AltHoldEnv):      # the same physics as in althold training
            env.push_command(command)
            env._update_disturbances()
            for _ in range(env.frame_skip):
                env._substep(armed=self.armed, push=not self.on_ground)
                self._floor()
            self._after_step()
            return
        push = env._mass * env._gust * (not self.on_ground)
        for _ in range(env.frame_skip):
            roll, pitch, _ = _euler(data.qpos[3:7])
            if self.armed:
                data.ctrl[:] = env.fc.step(env._command_line.get(env._t), roll, pitch, data.qvel[3:6].tolist(), env.dt)
            else:
                data.ctrl[:] = 0.0
            data.xfrc_applied[body, 0:3] = drag * data.qvel[0:3] + push
            mujoco.mj_step(env.model, data)
            env._t += env.dt
            self._floor()
        self._after_step()

    def _after_step(self):
        data = self.env.data
        self.max_speed = max(self.max_speed, float(np.linalg.norm(data.qvel[0:3])))
        if np.linalg.norm(data.qpos[0:2]) > ARENA or data.qpos[2] > ARENA:
            self._crash("flew out of the arena")

    def _floor(self):
        data = self.env.data
        if data.qpos[2] > GROUND_Z:
            self.on_ground = False
            return
        roll, pitch, yaw = _euler(data.qpos[3:7])
        if not self.on_ground and (-data.qvel[2] > CRASH_SPEED or max(abs(roll), abs(pitch)) > CRASH_TILT):
            self._crash(f"hit the ground at {-data.qvel[2]:.1f} m/s, tilt {math.degrees(max(abs(roll), abs(pitch))):.0f} deg")
        self.on_ground = True
        data.qpos[2] = GROUND_Z
        data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]   # sits level
        data.qvel[0:2] = 0.0
        data.qvel[2] = max(data.qvel[2], 0.0)
        data.qvel[3:6] = 0.0

    def _crash(self, why: str):
        if not self.crashed:
            self.crashed = why
            self.armed = False
            self.arm_allowed = False


# ── viewer decorations ──────────────────────────────────────────────────────
class Room:
    """A room drawn around the flight space: floor grid (1 m), side walls, far wall, takeoff pad.
    The model has no geometry besides the drone, so this is purely visual (nothing collides)."""

    def __init__(self, x: tuple[float, float], half_width: float, height: float):
        self.x0, self.x1 = x
        self.w, self.h = half_width, height

    def contains(self, pos, margin: float = 0.04) -> bool:
        return (self.x0 + margin < pos[0] < self.x1 - margin and abs(pos[1]) < self.w - margin
                and pos[2] < self.h - margin)

    def draw(self, viewer, drone_pos):
        scn = viewer.user_scn
        scn.ngeom = 0

        def add(type_, size, p, rgba):
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type_, np.asarray(size, float), np.asarray(p, float),
                                    np.eye(3).ravel(), np.asarray(rgba, np.float32))
                scn.ngeom += 1

        box = mujoco.mjtGeom.mjGEOM_BOX
        x0, x1, w, h = self.x0, self.x1, self.w, self.h
        cx, lx = (x0 + x1) / 2, (x1 - x0) / 2
        line = [0.55, 0.55, 0.6, 1]
        wall = [0.78, 0.76, 0.72, 1]
        add(box, [lx, w, 0.001], [cx, 0, -0.001], [0.3, 0.3, 0.33, 1])            # floor
        for k in range(math.ceil(x0), math.floor(x1) + 1):                         # grid across
            add(box, [0.004, w, 0.0005], [k, 0, 0.0005], line)
        for k in range(-math.floor(w), math.floor(w) + 1):                         # grid along
            add(box, [lx, 0.004, 0.0005], [cx, k, 0.0005], line)
        add(box, [0.01, w, h / 2], [x1 + 0.01, 0, h / 2], wall)                    # far wall
        add(box, [lx, 0.01, h / 2], [cx, w + 0.01, h / 2], wall)                   # left wall
        add(box, [lx, 0.01, h / 2], [cx, -w - 0.01, h / 2], wall)                  # right wall
        for z in np.arange(0.5, h, 0.5):                                           # height marks every 0.5 m
            mark = [0.45, 0.45, 0.5, 1] if z % 1 else [0.25, 0.25, 0.3, 1]
            add(box, [0.012, w, 0.004], [x1, 0, z], mark)
            add(box, [lx, 0.012, 0.004], [cx, w, z], mark)
            add(box, [lx, 0.012, 0.004], [cx, -w, z], mark)
        for k in range(math.ceil(x0), math.floor(x1) + 1):                         # 1 m posts on the side walls
            add(box, [0.006, 0.012, h / 2], [k, w, h / 2], line)
            add(box, [0.006, 0.012, h / 2], [k, -w, h / 2], line)
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.12, 0.001, 0], [0, 0, 0.001], [0.9, 0.5, 0.1, 1])   # takeoff pad
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.035, 0.0005, 0], [drone_pos[0], drone_pos[1], 0.003],
            [0, 0, 0, 0.55])                                                       # shadow


def setup_visuals(model, fov: float):
    """Room-scale rendering. MuJoCo sizes the clip planes and lights from the model's extent,
    which is the 6 cm drone: without this, everything beyond ~2.5 m is clipped away."""
    model.stat.extent = 10.0                    # m: znear = 3 mm, zfar = 500 m
    model.vis.map.znear = 0.0003
    model.vis.map.zfar = 50.0
    model.vis.global_.fovy = fov
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    model.vis.headlight.specular[:] = 0.1


def look_from(cam, eye, target):
    """Point a free camera from `eye` at `target` (world coordinates)."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    dist = float(np.linalg.norm(d))
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = target
    cam.distance = dist
    cam.azimuth = math.degrees(math.atan2(d[1], d[0]))
    cam.elevation = math.degrees(math.asin(d[2] / dist))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None, help="joystick device (default: skybridge rc.device or /dev/input/js0)")
    ap.add_argument("--randomize", action="store_true", help="draw a random drone/battery/delay, as in training")
    ap.add_argument("--no-gusts", action="store_true", help="turn off the slow air disturbance")
    ap.add_argument("--hover-throttle", type=float, help="override betaflight.hover_throttle (battery state)")
    ap.add_argument("--delay", type=float, help="radio link delay in seconds (default: params.yaml / nominal)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--record", type=Path, help="save sticks, command and true state to this .npz")
    ap.add_argument("--max-thrust", type=float, help="max thrust / hover thrust (params.yaml: max_thrust_ratio)")
    ap.add_argument("--thrust-expo", type=float, help="thrust ~ throttle^expo (params.yaml: thrust_expo)")
    ap.add_argument("--drag-z", type=float, help="vertical drag, 1/s (params.yaml: aero.drag_z)")
    ap.add_argument("--soft-cap", type=float, default=None,
                    help="bend the thrust curve into its maximum so it comes at full stick (0 = hard cut-off, "
                         "as params.yaml has it now); default 4, or params.yaml's with --policy")
    ap.add_argument("--policy", type=Path, help="altitude-hold run directory (or .zip): the policy flies throttle")
    ap.add_argument("--engage-axis", type=int, help="radio axis index of a switch that engages altitude hold "
                    "(high = on); default: the Enter key toggles it")
    ap.add_argument("--camera-timeout", type=float, default=1.0,
                    help="s: camera data older than this -> throttle 0 until re-engaged (default %(default)s)")
    ap.add_argument("--eye", type=float, nargs=3, default=[-2.0, 0.0, 1.6], metavar=("X", "Y", "Z"),
                    help="pilot's eye position, m (default: 2 m behind the pad, eye height 1.6 m)")
    ap.add_argument("--look", type=float, nargs=3, default=[1.5, 0.0, 0.45], metavar=("X", "Y", "Z"),
                    help="point the pilot looks at, m")
    ap.add_argument("--fov", type=float, default=60.0, help="vertical field of view, deg")
    ap.add_argument("--room", type=float, nargs=3, default=[6.0, 5.0, 3.0], metavar=("LENGTH", "WIDTH", "HEIGHT"),
                    help="room size, m; it starts 0.5 m behind the pilot")
    args = ap.parse_args()

    sky_path, live = skybridge_config_path("air65")
    sky = yaml.safe_load(sky_path.read_text())
    sticks = Sticks(sky)
    limits = load_limits("air65")
    joy = Joystick(args.device or sky.get("rc", {}).get("device", "/dev/input/js0"))

    assist = None
    if args.policy:
        from stable_baselines3 import PPO
        model_path = args.policy
        if model_path.is_dir():
            model_path = next((model_path / n for n in ("best_model.zip", "hover_policy_final.zip")
                               if (model_path / n).exists()), None)
            if model_path is None:
                sys.exit(f"no best_model.zip or hover_policy_final.zip in {args.policy}")
        config_path = model_path.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        drone = spec_from_params("air65", config["drone_params"]) if config.get("drone_params") else "air65"
        env = AltHoldEnv(drone, randomize=args.randomize, noise=not args.no_gusts)
        if args.hover_throttle is not None:
            env._fc_nominal = replace(env._fc_nominal, hover_throttle=args.hover_throttle)
        model = PPO.load(model_path, device="cpu")
        assist = AltHoldAssist(env.observer, lambda obs: model.predict(obs, deterministic=True)[0],
                               timeout=args.camera_timeout)
        print(f"policy     {model_path}  (altitude hold: {'switch on axis %d' % args.engage_axis if args.engage_axis is not None else 'Enter'} toggles)")
    else:
        env = AngleModeEnv("air65", randomize=args.randomize, noise=not args.no_gusts)
        if args.hover_throttle is not None:
            env.fc.p.hover_throttle = args.hover_throttle   # the throttle curve is rebuilt at reset
    env.reset(seed=args.seed)
    soft_cap = args.soft_cap if args.soft_cap is not None else (None if assist else 4.0)
    tuning = ThrustTuning(env, args.max_thrust, args.thrust_expo, args.drag_z, soft_cap)
    sim = RcSim(env, limits.min_check, args.delay, tuning)

    print(f"radio      {joy.name}  ({len(joy.axes)} axes)")
    print(f"config     {sky_path}{'' if live else '  (snapshot)'}")
    print(f"sticks     angle limit {math.degrees(sticks.angle_limit):.0f} deg, yaw {math.degrees(sticks.yaw_max):.0f} deg/s")
    fc = env.fc
    print(f"drone      mass {env._mass * 1000:.1f} g, hover throttle {fc.throttle_for_hover(env._mass, env.spec.g):.3f}, "
          f"thrust x{fc.thrust_scale:.2f}, link delay {env._action_delay * 1000:.0f} ms, "
          f"drag {env._drag.round(2)}, trim {np.degrees(fc.angle_trim).round(1)} deg")
    print(f"view       pilot at {args.eye}, looking at {args.look}; the drone's nose points away from you (+x)")
    print(f"thrust     {tuning.summary(env)}")
    for line in tuning.changes():
        print(f"           sim only: {line}")
    print("keys       R reset, C pilot view / follow camera.  Arm: switch OFF, throttle low, switch ON.")
    print("           Up/Down max thrust, PgUp/PgDn thrust expo, Right/Left vertical drag.\n")

    keys = {"reset": False, "cam": False, "tune": [], "hold": False}
    tune_keys = {265: ("max_thrust_ratio", 1), 264: ("max_thrust_ratio", -1),     # GLFW Up / Down
                 266: ("thrust_expo", 1), 267: ("thrust_expo", -1),               # PgUp / PgDn
                 262: ("drag_z", 1), 263: ("drag_z", -1)}                         # Right / Left

    def on_key(key):
        if key == ord("R"):
            keys["reset"] = True
        elif key == ord("C"):
            keys["cam"] = True
        elif key == 257:                        # GLFW Enter: altitude hold on/off
            keys["hold"] = not keys["hold"]
        elif key in tune_keys:
            keys["tune"].append(tune_keys[key])

    log = {k: [] for k in ("t", "axes", "command", "armed", "state", "motors", "hold", "z_meas")} if args.record else None
    setup_visuals(env.model, args.fov)
    length, width, height = args.room
    back = args.eye[0] - 0.5
    room = Room((back, back + length), width / 2, height)
    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key,
                                          show_left_ui=False, show_right_ui=False)
    look_from(viewer.cam, args.eye, args.look)
    follow = False

    period = env.control_dt
    next_tick = time.perf_counter()
    last_print = 0.0
    try:
        while viewer.is_running():
            if not joy.alive:
                print("\nradio unplugged")
                break
            axes = joy.read()
            if axes is None:
                time.sleep(0.05)
                continue
            if keys["reset"]:
                keys["reset"] = False
                sim.reset()
            if keys["tune"]:
                while keys["tune"]:
                    tuning.nudge(*keys["tune"].pop(0))
                tuning.apply(env)
                print(f"\rthrust     {tuning.summary(env)}" + " " * 10)
            if keys["cam"]:
                keys["cam"] = False
                follow = not follow
                if follow:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = env._body
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.6, 0, -20
            if not follow:
                look_from(viewer.cam, args.eye, args.look)     # every tick: the mouse cannot move it

            command = sticks.command(axes)
            sim.update_arming(sticks.arm_switch(axes), command[0])
            if assist is not None:
                want = (axis_us(axes[args.engage_axis]) > 1700 if args.engage_axis is not None
                        and args.engage_axis < len(axes) else keys["hold"])
                can = sim.armed and not sim.crashed and not sim.on_ground
                z_meas, vz_meas, age = env.altitude_measurement()
                if want and can and not assist.engaged:
                    assist.engage(z_meas, command[0], command[1:4])
                elif assist.engaged and not (want and can):
                    assist.disengage()
                    keys["hold"] = False
                if assist.engaged:
                    command[0] = assist.step(z_meas, vz_meas, age, command[1:4])
            with viewer.lock():
                sim.step(command)
                if not room.contains(env.data.qpos[0:3]):
                    sim._crash("hit a wall or the ceiling")
                room.draw(viewer, env.data.qpos[0:3])
            viewer.sync()

            pos, vel = env.data.qpos[0:3], env.data.qvel[0:3]
            roll, pitch, yaw = _euler(env.data.qpos[3:7])
            if sim.crashed:
                state = f"CRASHED: {sim.crashed} (R to reset)"
            elif sim.armed:
                state = "ARMED"
            elif not sim.arm_allowed:
                state = "disarmed (cycle arm switch with throttle low)"
            else:
                state = "disarmed"
            mode = "" if sticks.angle_mode(axes) else "  ACRO SWITCH ON: sim flies angle mode only"
            if assist is not None:
                z_meas, _, age = env.altitude_measurement()
                if assist.engaged and assist.timed_out:
                    mode += f"  CAMERA TIMEOUT: throttle 0 (disengage to recover)"
                elif assist.engaged:
                    mode += f"  ALT HOLD {assist.observer.target:.2f} m (cam {z_meas:.2f}, {age * 1000:.0f} ms)"
                else:
                    mode += "  alt hold off"
            hud = (f"{state}{mode}\n"
                   f"thr {command[0]:.2f}  roll {math.degrees(command[1]):+5.1f}  pitch {math.degrees(command[2]):+5.1f}  "
                   f"yaw {math.degrees(command[3]):+5.0f}/s\n"
                   f"alt {pos[2]:.2f} m  speed {np.linalg.norm(vel):.2f} m/s  climb {vel[2]:+.2f}\n"
                   f"tilt {math.degrees(roll):+5.1f} {math.degrees(pitch):+5.1f}  hdg {math.degrees(yaw):+4.0f}  "
                   f"pos {pos[0]:+.2f} {pos[1]:+.2f}\n"
                   f"max T/W {tuning.values['max_thrust_ratio']:.1f} (Up/Dn)  expo {tuning.values['thrust_expo']:.2f} (PgUp/Dn)  "
                   f"drag_z {tuning.values['drag_z']:.2f} (Rt/Lt)")
            try:
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, hud, ""))
            except Exception:
                pass
            now = time.perf_counter()
            if now - last_print > 0.2:
                last_print = now
                print("\r" + hud.rsplit("\n", 1)[0].replace("\n", " | ") + " " * 4, end="", flush=True)

            if log is not None:
                log["t"].append(env._t)
                log["axes"].append(axes)
                log["command"].append(command)
                log["armed"].append(sim.armed)
                log["state"].append(np.concatenate([pos, [roll, pitch, yaw], vel, env.data.qvel[3:6]]))
                log["motors"].append(env.data.ctrl.copy())
                log["hold"].append(bool(assist and assist.engaged))
                log["z_meas"].append(env.altitude_measurement()[0] if assist else float("nan"))

            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:        # fell behind (window dragged, ...): don't try to catch up
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        print()
        viewer.close()
        if changes := tuning.changes():
            print("thrust settings flown that differ from drones/air65/params.yaml:")
            for line in changes:
                print(f"    {line}")
        if log is not None and log["t"]:
            np.savez(args.record, **{k: np.asarray(v) for k, v in log.items()},
                     state_names=np.array(["x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "wx", "wy", "wz"]))
            print(f"saved {len(log['t'])} ticks to {args.record}")


if __name__ == "__main__":
    main()
