"""What separates the laptop from the drone: latency, sample rates, noise.

The policy runs on a laptop. Its commands reach the motors late, and it sees
the drone through a 30 Hz depth camera and a slow telemetry link. These helpers
model that, and draw the per-episode variation that makes a policy robust to
what the identification could not pin down.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class DelayLine:
    """Values that become visible some time after they were produced."""

    def __init__(self) -> None:
        self._queue: deque[tuple[float, Any]] = deque()
        self._current: Any = None

    def reset(self, value: Any) -> None:
        self._queue.clear()
        self._current = value

    def push(self, available_at: float, value: Any) -> None:
        self._queue.append((available_at, value))

    def get(self, now: float) -> Any:
        """The newest value available by `now`."""
        while self._queue and self._queue[0][0] <= now + 1e-9:
            self._current = self._queue.popleft()[1]
        return self._current


def draw(rng: np.random.Generator, value: Any, randomize: bool = True) -> float:
    """A number from a YAML entry: a scalar, or a `[low, high]` range.

    With `randomize` off, a range gives its midpoint.
    """
    if isinstance(value, (list, tuple)):
        low, high = float(value[0]), float(value[1])
        return float(rng.uniform(low, high)) if randomize else 0.5 * (low + high)
    return float(value)


def scale(rng: np.random.Generator, ranges: dict, key: str, randomize: bool) -> float:
    """A multiplier from `ranges[key] = [low, high]`; 1 when absent or not randomizing."""
    if not randomize or key not in ranges:
        return 1.0
    return float(rng.uniform(*ranges[key]))


def symmetric(rng: np.random.Generator, ranges: dict, key: str, randomize: bool, size=None):
    """Uniform in +-ranges[key]; 0 when absent or not randomizing."""
    bound = float(ranges.get(key, 0.0)) if randomize else 0.0
    return rng.uniform(-bound, bound, size=size) if bound else (np.zeros(size) if size else 0.0)
