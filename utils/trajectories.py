"""Trajectory generators for waypoint-based evaluation."""

import numpy as np


def generate_figure_eight(spacing: float = 0.5, radius: float = 1.0,
                          center: np.ndarray = None) -> list[np.ndarray]:
    """Generate figure-8 waypoints in the XY plane at fixed altitude.

    Uses parametric lemniscate: x=radius*cos(t), y=(radius/2)*sin(2t).
    Waypoints are arc-length sampled so consecutive points are ~spacing apart.
    """
    if center is None:
        center = np.array([0.0, 0.0, 1.0])
    # Compute arc length numerically
    num_samples = 1000
    t_dense = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    dx = -radius * np.sin(t_dense)
    dy = radius * np.cos(2.0 * t_dense)
    dt_param = 2.0 * np.pi / num_samples
    ds = np.sqrt(dx**2 + dy**2) * dt_param
    cumulative = np.cumsum(ds)
    total_length = cumulative[-1]

    n = max(int(np.ceil(total_length / spacing)), 8)
    target_distances = np.linspace(0, total_length, n, endpoint=False)
    t_values = np.interp(target_distances, cumulative, t_dense)

    waypoints = []
    for t in t_values:
        x = center[0] + radius * np.cos(t)
        y = center[1] + (radius / 2.0) * np.sin(2.0 * t)
        waypoints.append(np.array([x, y, center[2]]))
    return waypoints


def generate_circle(spacing: float = 0.5, radius: float = 1.0,
                    center: np.ndarray = None) -> list[np.ndarray]:
    """Generate circular waypoints in the XY plane at fixed altitude."""
    if center is None:
        center = np.array([0.0, 0.0, 1.0])
    circumference = 2.0 * np.pi * radius
    n = max(int(np.ceil(circumference / spacing)), 4)
    waypoints = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        waypoints.append(np.array([x, y, center[2]]))
    return waypoints


def generate_square(spacing: float = 0.5, side_length: float = 1.5,
                    center: np.ndarray = None) -> list[np.ndarray]:
    """Generate square waypoints in the XY plane at fixed altitude."""
    if center is None:
        center = np.array([0.0, 0.0, 1.0])
    half = side_length / 2.0
    corners = [
        np.array([center[0] + half, center[1] + half, center[2]]),
        np.array([center[0] - half, center[1] + half, center[2]]),
        np.array([center[0] - half, center[1] - half, center[2]]),
        np.array([center[0] + half, center[1] - half, center[2]]),
    ]
    waypoints = []
    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]
        edge_length = np.linalg.norm(end - start)
        n_seg = max(int(np.ceil(edge_length / spacing)), 1)
        for j in range(n_seg):
            frac = j / n_seg
            waypoints.append(start + frac * (end - start))
    return waypoints


TRAJECTORY_GENERATORS = {
    "eight": generate_figure_eight,
    "circle": generate_circle,
    "square": generate_square,
}
