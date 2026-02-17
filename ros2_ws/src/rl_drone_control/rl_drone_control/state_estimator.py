"""Velocity estimation from mocap position via filtered differentiation."""

import numpy as np


class VelocityEstimator:
    """First-order low-pass filtered differentiation of position.

    Suitable for high-rate, low-noise sources like motion capture (100-200 Hz).

    v_raw = (pos - pos_prev) / dt
    v_filtered = alpha * v_prev + (1 - alpha) * v_raw

    Args:
        alpha: Low-pass filter coefficient in [0, 1). Higher = smoother but more lag.
        max_dt: Maximum dt to accept (s). If exceeded, velocity resets to zero
                (likely a dropout or large gap in data).
    """

    def __init__(self, alpha: float = 0.8, max_dt: float = 0.1):
        self.alpha = alpha
        self.max_dt = max_dt
        self._prev_pos: np.ndarray | None = None
        self._prev_time: float | None = None
        self._velocity = np.zeros(3)

    def update(self, position: np.ndarray, timestamp: float) -> np.ndarray:
        """Update with a new position measurement and return filtered velocity.

        Args:
            position: 3D position [x, y, z] in meters.
            timestamp: Time in seconds (from message header stamp).

        Returns:
            Filtered velocity estimate [vx, vy, vz] in m/s.
        """
        if self._prev_pos is None:
            self._prev_pos = position.copy()
            self._prev_time = timestamp
            self._velocity = np.zeros(3)
            return self._velocity.copy()

        dt = timestamp - self._prev_time
        if dt <= 0 or dt > self.max_dt:
            # Invalid dt — reset filter state
            self._prev_pos = position.copy()
            self._prev_time = timestamp
            self._velocity = np.zeros(3)
            return self._velocity.copy()

        v_raw = (position - self._prev_pos) / dt
        self._velocity = self.alpha * self._velocity + (1.0 - self.alpha) * v_raw

        self._prev_pos = position.copy()
        self._prev_time = timestamp

        return self._velocity.copy()

    def reset(self):
        """Reset estimator state."""
        self._prev_pos = None
        self._prev_time = None
        self._velocity = np.zeros(3)

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate."""
        return self._velocity.copy()
