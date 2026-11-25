from drones import DroneSpec, resolve

from .hover_env import HoverEnv
from .angle_env import AngleModeEnv
from .althold_env import AltHoldEnv, AltHoldObserver, AltHoldAssist
from .wrappers import RelPosActWrapper, get_wrapper
from .rate_wrapper import RateControlWrapper

# How a drone is commanded -> the environment that simulates it.
ENV_FOR_INTERFACE = {
    "thrust_torque": HoverEnv,
    "betaflight_angle": AngleModeEnv,
}

# Wrapper each drone trains with by default (None = the env's own spaces).
DEFAULT_WRAPPER = {
    "small": RateControlWrapper,    # 12D obs, body-rate actions (CTBR). Or RelPosActWrapper.
    "air65": None,                  # AngleModeEnv already speaks skybridge's angle mode
}


# Tasks other than flying to a target, by name.
TASKS = {
    "althold": AltHoldEnv,      # pilot flies roll/pitch/yaw, the policy holds altitude with throttle
}


def make_env(drone: str | DroneSpec | None = None, wrapper=None, task: str = "hover", **kwargs):
    """Build the environment for a drone from the registry.

    `wrapper` is a wrapper class, a name from the wrapper registry, or None.
    `task` is "hover" (the drone's own environment) or a name from TASKS.
    """
    spec = resolve(drone)
    env_cls = ENV_FOR_INTERFACE[spec.interface] if task == "hover" else TASKS[task]
    env = env_cls(drone=spec, **kwargs)
    if isinstance(wrapper, str):
        wrapper = get_wrapper(wrapper)
    return wrapper(env) if wrapper else env


__all__ = ["HoverEnv", "AngleModeEnv", "AltHoldEnv", "AltHoldObserver", "AltHoldAssist", "TASKS", "RelPosActWrapper", "RateControlWrapper", "get_wrapper",
           "make_env", "ENV_FOR_INTERFACE", "DEFAULT_WRAPPER"]
