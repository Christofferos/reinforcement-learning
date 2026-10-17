"""Observation terms and the observation config used for MDP ablations."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

# Only terrain geoms live in group 1, so height-scan rays never hit the robot.
TERRAIN_RAY_MASK = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)


@dataclass
class ObservationCfg:
    """Which terms enter the policy observation (the main MDP ablation surface)."""

    base_lin_vel: bool = True
    base_ang_vel: bool = True
    projected_gravity: bool = True
    base_height: bool = True
    joint_pos: bool = True
    joint_vel: bool = True
    last_action: bool = True
    height_scan: bool = True
    overhead_scan: bool = False
    overhead_scan_observation: bool = True
    waypoint_command: bool = True
    speed_command: bool = False
    exteroceptive_delay: int = 0
    """Number of control steps the height scan lags behind (sim-to-real style latency)."""

    # Height-scan geometry (body frame, metres).
    scan_forward: tuple[float, float] = (-0.4, 1.6)
    scan_lateral: tuple[float, float] = (-0.5, 0.5)
    scan_shape: tuple[int, int] = (9, 5)
    scan_max_depth: float = 3.0
    scan_clip: tuple[float, float] = (-1.0, 1.5)
    overhead_offsets: tuple[float, ...] = (0.0, 0.2, 0.4)
    """Forward offsets of the upward rays. Spaced closer than the 0.24 m tunnel bar so it is
    never between rays; 0.5 m spacing missed it while its centre was 0.13-0.38 m ahead."""
    overhead_origin_height: float = 0.5
    overhead_max_depth: float = 2.0

    # Gaussian observation noise (0 disables).
    noise_lin_vel: float = 0.0
    noise_ang_vel: float = 0.0
    noise_gravity: float = 0.0
    noise_joint_pos: float = 0.0
    noise_joint_vel: float = 0.0
    noise_height_scan: float = 0.0

    history_length: int = 1
    """Number of stacked control steps (1 = no stacking)."""

    scan_points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nx, ny = self.scan_shape
        xs = np.linspace(self.scan_forward[0], self.scan_forward[1], nx)
        ys = np.linspace(self.scan_lateral[0], self.scan_lateral[1], ny)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.scan_points = np.stack([gx.ravel(), gy.ravel()], axis=1)

    @property
    def num_scan_points(self) -> int:
        return int(self.scan_shape[0] * self.scan_shape[1])


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.ascontiguousarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def yaw_from_quat(quat: np.ndarray) -> float:
    mat = quat_to_mat(quat)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """World -z axis expressed in the base frame (upright => [0, 0, -1])."""
    return quat_to_mat(quat).T @ np.array([0.0, 0.0, -1.0])


def ground_ray(model, data, start, ceilings: frozenset[int] = frozenset(),
               walls: frozenset[int] = frozenset()) -> float:
    """Distance straight down from ``start`` to the terrain, or -1 on a miss.

    A ray that meets one of the ``ceilings`` geom ids (from above, or from inside, where
    ``mj_ray`` reports its underside) continues from just below it, so a tunnel bar ahead of a
    standing robot reads as the floor beneath it rather than a wall. Ceilings must be boxes.
    A ray that starts inside one of the ``walls`` (boxes taller than a ray's start, as in a
    tilt passage) reads it at distance 0: ``mj_ray`` skips a geom that contains its start, so
    the wall read as the floor beside it. ``height_scan`` checks its whole grid at once instead.
    """
    origin = np.array(start, dtype=np.float64)
    if any(_inside_box(model, data, wall, origin) for wall in walls):
        return 0.0
    direction = np.array([0.0, 0.0, -1.0])
    geom_id = np.zeros(1, dtype=np.int32)
    travelled = 0.0
    while True:
        dist = mujoco.mj_ray(model, data, origin, direction, TERRAIN_RAY_MASK, 1, -1, geom_id)
        geom = int(geom_id[0])
        if dist < 0 or geom not in ceilings:
            return travelled + dist if dist >= 0 else -1.0
        half_height = float(np.abs(data.geom_xmat[geom].reshape(3, 3)[2]) @ model.geom_size[geom])
        step = origin[2] - (data.geom_xpos[geom, 2] - half_height) + 1e-6
        origin[2] -= step
        travelled += step


def height_scan(
    model,
    data,
    origin: np.ndarray,
    yaw: float,
    cfg: ObservationCfg,
    ceilings: frozenset[int] = frozenset(),
    walls: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast a yaw-aligned grid downwards.

    Returns ``(relative_heights, hit_points)`` where ``relative_heights`` is the base
    height above the terrain sample and ``hit_points`` are world-space hits (for debug
    markers).  Misses are clamped to ``scan_max_depth``. Rays pass through ``ceilings``.
    """
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    points = cfg.scan_points
    world_x = origin[0] + cos_y * points[:, 0] - sin_y * points[:, 1]
    world_y = origin[1] + sin_y * points[:, 0] + cos_y * points[:, 1]

    heights = np.empty(len(points))
    hits = np.empty((len(points), 3))
    starts = np.column_stack([world_x, world_y, np.full(len(points), origin[2])])
    in_wall = np.zeros(len(points), dtype=bool)
    for wall in walls:  # one vectorised check per wall instead of one call per ray
        in_wall |= _inside_box_points(model, data, wall, starts)
    for i in range(len(points)):
        dist = 0.0 if in_wall[i] else ground_ray(model, data, starts[i], ceilings)
        if dist < 0 or dist > cfg.scan_max_depth:
            dist = cfg.scan_max_depth
        heights[i] = dist
        hits[i] = (world_x[i], world_y[i], origin[2] - dist)
    return heights, hits


def overhead_scan(
    model,
    data,
    origin: np.ndarray,
    yaw: float,
    cfg: ObservationCfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast upwards from a low, yaw-aligned line; cap misses at the maximum range.

    A ray that starts inside a terrain box or cylinder (a hurdle, climb, stair or sand bank
    taller than ``overhead_origin_height`` ahead) counts as a miss: ``mj_ray`` reports that box's top face
    from the inside, which is something to step onto, not duck under. Heightfields are not
    checked.
    """
    offsets = np.asarray(cfg.overhead_offsets)
    direction = np.array([0.0, 0.0, 1.0])
    geom_id = np.zeros(1, dtype=np.int32)
    depths = np.empty(len(offsets))
    hits = np.empty((len(offsets), 3))
    start = origin.copy()
    for index, offset in enumerate(offsets):
        start[:2] = origin[:2] + offset * np.array([np.cos(yaw), np.sin(yaw)])
        distance = mujoco.mj_ray(
            model, data, start, direction, TERRAIN_RAY_MASK, 1, -1, geom_id,
        )
        if distance >= 0 and _inside_box(model, data, int(geom_id[0]), start):
            distance = -1.0
        if distance < 0 or distance > cfg.overhead_max_depth:
            distance = cfg.overhead_max_depth
        depths[index] = distance
        hits[index] = start + direction * distance
    return depths, hits


def _inside_box_points(model, data, geom: int, points: np.ndarray) -> np.ndarray:
    """Which of ``points`` (N x 3) lie inside box ``geom``, rotated or not."""
    local = (points - data.geom_xpos[geom]) @ data.geom_xmat[geom].reshape(3, 3)
    return np.all(np.abs(local) <= model.geom_size[geom], axis=1)


def _inside_box(model, data, geom: int, point: np.ndarray) -> bool:
    """Whether ``point`` lies inside ``geom`` when it is a box or a cylinder (a water tank's sand
    bank), rotated ones included."""
    kind = model.geom_type[geom]
    if kind not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
        return False
    local = data.geom_xmat[geom].reshape(3, 3).T @ (point - data.geom_xpos[geom])
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return bool(np.hypot(local[0], local[1]) <= size[0] and abs(local[2]) <= size[1])
    return bool(np.all(np.abs(local) <= size))
