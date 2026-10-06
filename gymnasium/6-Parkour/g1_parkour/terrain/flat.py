"""Flat terrain: a plain ground plane with a straight route."""

from __future__ import annotations

import numpy as np

from .core import TerrainSpec


def make_flat(
    rng: np.random.Generator,
    difficulty: float = 0.0,
    course_length: float = 20.0,
    waypoint_spacing: float = 2.5,
    lateral_jitter: float = 0.0,
) -> TerrainSpec:
    """Flat plane with waypoints every ``waypoint_spacing`` metres along +x."""
    num = max(2, int(course_length / waypoint_spacing))
    xs = np.linspace(waypoint_spacing, course_length, num)
    ys = rng.uniform(-lateral_jitter, lateral_jitter, size=num) if lateral_jitter else np.zeros(num)
    waypoints = np.stack([xs, ys, np.zeros(num)], axis=1)
    return TerrainSpec(
        name="flat",
        waypoints=waypoints,
        spawn_pos=(0.0, 0.0, 0.0),
        add_ground_plane=True,
        difficulty=difficulty,
        modules=["flat"],
        course_length=course_length,
    )
