"""Observation and action wrappers for HoverEnv."""

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from .rate_wrapper import RateControlWrapper


WRAPPER_REGISTRY = {}


class RelPosActWrapper(gym.ObservationWrapper):
    """7D observation: [normalized_rel_pos(3), prev_action(4)].

    Selects the normalized relative target position (first 3 elements of
    the base env observation) and appends the previous action (already in [-1, 1]).
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs[0:3], self.unwrapped._prev_action]).astype(np.float32)


WRAPPER_REGISTRY["RelPosActWrapper"] = RelPosActWrapper
WRAPPER_REGISTRY["RateControlWrapper"] = RateControlWrapper


def get_wrapper(name):
    """Look up wrapper class by name. Returns None if name is 'none' or None."""
    if name is None or name == "none":
        return None
    return WRAPPER_REGISTRY[name]
