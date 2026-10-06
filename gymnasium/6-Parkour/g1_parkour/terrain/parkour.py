"""Procedural parkour terrain: a randomly sequenced chain of skill modules.

Each module appends geometry along +x and returns the route waypoints that cross
it, so a course is a random *skill sequence* (gap -> stairs -> beam -> vault ...)
rather than one hand-authored track.  ``difficulty`` in [0, 1] scales gap widths,
step heights, beam narrowness and overhead clearance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import Box, TerrainSpec

PLATFORM_THICKNESS = 0.4
DEFAULT_WIDTH = 2.4


@dataclass
class Cursor:
    """End of the course so far: x reach, lateral centre and surface height."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    index: int = 0


@dataclass
class ModuleOutput:
    boxes: list[Box] = field(default_factory=list)
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)


def _lerp(low: float, high: float, t: float) -> float:
    return low + (high - low) * t


def _platform(name: str, x0: float, x1: float, y: float, z: float, width: float, **kw) -> Box:
    kw.setdefault("material", "Concrete")
    return Box(
        name=name,
        pos=((x0 + x1) / 2.0, y, z - PLATFORM_THICKNESS / 2.0),
        size=((x1 - x0) / 2.0, width / 2.0, PLATFORM_THICKNESS / 2.0),
        **kw,
    )


# --------------------------------------------------------------------------------------
# Modules.  Signature: (rng, difficulty, cursor) -> ModuleOutput, mutating the cursor.
# --------------------------------------------------------------------------------------

def mod_run(rng, d, c: Cursor) -> ModuleOutput:
    length = _lerp(3.0, 2.0, d) + rng.uniform(0.0, 1.0)
    box = _platform(f"run_{c.index}", c.x, c.x + length, c.y, c.z, DEFAULT_WIDTH)
    out = ModuleOutput([box], [(c.x + length * 0.5, c.y, c.z), (c.x + length, c.y, c.z)])
    c.x += length
    return out


def mod_gap(rng, d, c: Cursor) -> ModuleOutput:
    gap = _lerp(0.45, 1.5, d) * rng.uniform(0.85, 1.15)
    landing = rng.uniform(1.6, 2.6)
    drop = rng.uniform(-0.15, 0.15) * d
    x0 = c.x + gap
    box = _platform(
        f"gap_land_{c.index}", x0, x0 + landing, c.y, c.z + drop, DEFAULT_WIDTH,
        rgba=(0.7, 0.1, 0.2, 1.0), material=None,
    )
    c.x = x0 + landing
    c.z += drop
    return ModuleOutput([box], [(x0 + 0.4, c.y, c.z), (c.x, c.y, c.z)])


def mod_stepping_stones(rng, d, c: Cursor) -> ModuleOutput:
    count = int(_lerp(3, 6, d))
    spacing = _lerp(0.85, 1.35, d)
    half = _lerp(0.45, 0.28, d)
    boxes, waypoints = [], []
    for i in range(count):
        x = c.x + spacing * (i + 1)
        y = c.y + rng.uniform(-0.5, 0.5) * d
        z = c.z + rng.uniform(-0.1, 0.1) * d
        boxes.append(
            Box(
                f"stone_{c.index}_{i}",
                pos=(x, y, z - PLATFORM_THICKNESS / 2.0),
                size=(half, half, PLATFORM_THICKNESS / 2.0),
                rgba=(1.0, 0.647, 0.0, 1.0),
                material=None,
            )
        )
        waypoints.append((x, y, z))
    c.x += spacing * (count + 1)
    landing = _platform(f"stone_land_{c.index}", c.x - 0.6, c.x + 1.6, c.y, c.z, DEFAULT_WIDTH)
    boxes.append(landing)
    c.x += 1.6
    waypoints.append((c.x, c.y, c.z))
    return ModuleOutput(boxes, waypoints)


def mod_hurdle(rng, d, c: Cursor) -> ModuleOutput:
    length = 4.0
    height = _lerp(0.2, 0.65, d)
    boxes = [_platform(f"hurdle_run_{c.index}", c.x, c.x + length, c.y, c.z, DEFAULT_WIDTH)]
    wall_x = c.x + length / 2.0
    boxes.append(
        Box(
            f"hurdle_{c.index}",
            pos=(wall_x, c.y, c.z + height / 2.0),
            size=(0.12, DEFAULT_WIDTH / 2.0, height / 2.0),
            rgba=(0.2, 0.8, 0.2, 1.0),
            material=None,
        )
    )
    c.x += length
    return ModuleOutput(boxes, [(wall_x + 0.8, c.y, c.z), (c.x, c.y, c.z)])


def mod_stairs(rng, d, c: Cursor, direction: int = 1) -> ModuleOutput:
    steps = int(_lerp(3, 6, d))
    rise = _lerp(0.10, 0.22, d) * direction
    run = _lerp(0.45, 0.32, d)
    boxes, waypoints = [], []
    for i in range(steps):
        z = c.z + rise * (i + 1)
        x0 = c.x + run * i
        length = run * (steps - i) + 0.4 if direction > 0 else run
        if direction < 0 and i == steps - 1:
            length += 0.4
        boxes.append(
            _platform(
                f"stair_{c.index}_{i}", x0, x0 + length, c.y, z, DEFAULT_WIDTH,
                rgba=(0.7, 0.1, 0.2, 1.0), material=None,
            )
        )
        waypoints.append((x0 + run * 0.5, c.y, z))
    c.x += run * steps + 0.4
    c.z += rise * steps
    boxes.append(_platform(f"stair_top_{c.index}", c.x, c.x + 1.5, c.y, c.z, DEFAULT_WIDTH))
    c.x += 1.5
    waypoints.append((c.x, c.y, c.z))
    return ModuleOutput(boxes, waypoints)


def mod_stairs_down(rng, d, c: Cursor) -> ModuleOutput:
    return mod_stairs(rng, d, c, direction=-1)


def mod_ramp(rng, d, c: Cursor) -> ModuleOutput:
    angle = _lerp(8.0, 22.0, d) * (1 if rng.random() < 0.6 else -1)
    length = rng.uniform(2.5, 3.5)
    rad = np.deg2rad(angle)
    rise = length * np.tan(rad)
    cx = c.x + length / 2.0
    cz = c.z + rise / 2.0
    boxes = [
        Box(
            f"ramp_{c.index}",
            pos=(cx, c.y, cz - PLATFORM_THICKNESS / 2.0 * np.cos(rad)),
            size=(length / 2.0 / np.cos(rad), DEFAULT_WIDTH / 2.0, PLATFORM_THICKNESS / 2.0),
            euler=(0.0, -angle, 0.0),
            material="Metal" if rng.random() < 0.3 else "Concrete",
            friction=(0.6 if rng.random() < 0.3 else 1.0, 0.1, 0.1),
        )
    ]
    c.x += length
    c.z += rise
    boxes.append(_platform(f"ramp_top_{c.index}", c.x, c.x + 1.5, c.y, c.z, DEFAULT_WIDTH))
    c.x += 1.5
    return ModuleOutput(boxes, [(cx, c.y, cz), (c.x, c.y, c.z)])


def mod_beam(rng, d, c: Cursor) -> ModuleOutput:
    length = _lerp(2.5, 4.0, d)
    width = _lerp(0.6, 0.28, d)
    boxes = [
        _platform(f"beam_{c.index}", c.x, c.x + length, c.y, c.z, width, material="Wood"),
    ]
    mid = c.x + length / 2.0
    c.x += length
    boxes.append(_platform(f"beam_land_{c.index}", c.x, c.x + 1.6, c.y, c.z, DEFAULT_WIDTH))
    c.x += 1.6
    return ModuleOutput(boxes, [(mid, c.y, c.z), (c.x, c.y, c.z)])


def mod_tunnel(rng, d, c: Cursor) -> ModuleOutput:
    length = 3.5
    clearance = _lerp(1.45, 1.05, d)
    boxes = [_platform(f"tunnel_run_{c.index}", c.x, c.x + length, c.y, c.z, DEFAULT_WIDTH)]
    bar_x = c.x + length / 2.0
    boxes.append(
        Box(
            f"tunnel_bar_{c.index}",
            pos=(bar_x, c.y, c.z + clearance + 0.1),
            size=(0.12, DEFAULT_WIDTH / 2.0, 0.1),
            material="Wood",
        )
    )
    for side, sign in (("l", 1), ("r", -1)):
        boxes.append(
            Box(
                f"tunnel_post_{c.index}_{side}",
                pos=(bar_x, c.y + sign * DEFAULT_WIDTH / 2.0, c.z + (clearance + 0.1) / 2.0),
                size=(0.12, 0.12, (clearance + 0.1) / 2.0),
                material="Wood",
            )
        )
    c.x += length
    return ModuleOutput(boxes, [(bar_x, c.y, c.z), (c.x, c.y, c.z)])


def mod_climb(rng, d, c: Cursor) -> ModuleOutput:
    rise = _lerp(0.3, 0.75, d)
    gap = _lerp(0.2, 0.6, d)
    x0 = c.x + gap
    length = rng.uniform(2.0, 3.0)
    boxes = [
        _platform(
            f"climb_{c.index}", x0, x0 + length, c.y, c.z + rise, DEFAULT_WIDTH,
            rgba=(0.4, 0.7, 0.9, 1.0), material=None,
        )
    ]
    c.x = x0 + length
    c.z += rise
    return ModuleOutput(boxes, [(x0 + 0.5, c.y, c.z), (c.x, c.y, c.z)])


def mod_lateral_shift(rng, d, c: Cursor) -> ModuleOutput:
    shift = _lerp(0.8, 2.0, d) * (1 if rng.random() < 0.5 else -1)
    length = rng.uniform(2.0, 3.0)
    boxes = [
        _platform(f"shift_pad_{c.index}", c.x, c.x + 1.6, c.y, c.z, DEFAULT_WIDTH + abs(shift)),
    ]
    mid_y = c.y + shift / 2.0
    c.x += 1.6
    c.y += shift
    boxes.append(_platform(f"shift_run_{c.index}", c.x, c.x + length, c.y, c.z, DEFAULT_WIDTH))
    waypoints = [(c.x - 0.8, mid_y, c.z), (c.x + length, c.y, c.z)]
    c.x += length
    return ModuleOutput(boxes, waypoints)


def mod_rubble(rng, d, c: Cursor) -> ModuleOutput:
    length = 4.0
    boxes = [_platform(f"rubble_run_{c.index}", c.x, c.x + length, c.y, c.z, DEFAULT_WIDTH)]
    for i in range(int(_lerp(4, 12, d))):
        h = rng.uniform(0.04, 0.04 + 0.16 * d)
        boxes.append(
            Box(
                f"rubble_{c.index}_{i}",
                pos=(rng.uniform(c.x + 0.5, c.x + length - 0.5), c.y + rng.uniform(-1.0, 1.0), c.z + h / 2),
                size=(rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3), h / 2),
                euler=(0.0, 0.0, rng.uniform(0, 90)),
                rgba=(0.45, 0.42, 0.4, 1.0),
                material=None,
            )
        )
    c.x += length
    return ModuleOutput(boxes, [(c.x - length / 2, c.y, c.z), (c.x, c.y, c.z)])


MODULES: dict[str, Callable[..., ModuleOutput]] = {
    "run": mod_run,
    "gap": mod_gap,
    "stepping_stones": mod_stepping_stones,
    "hurdle": mod_hurdle,
    "stairs_up": mod_stairs,
    "stairs_down": mod_stairs_down,
    "ramp": mod_ramp,
    "beam": mod_beam,
    "tunnel": mod_tunnel,
    "climb": mod_climb,
    "lateral_shift": mod_lateral_shift,
    "rubble": mod_rubble,
}

DEFAULT_SKILLS = tuple(MODULES)


def make_parkour(
    rng: np.random.Generator,
    difficulty: float = 0.5,
    num_modules: int = 8,
    skills: tuple[str, ...] = DEFAULT_SKILLS,
    skill_weights: dict[str, float] | None = None,
    start_pad_length: float = 3.0,
    pit_depth: float = 4.0,
    ramp_difficulty: bool = True,
) -> TerrainSpec:
    """Sample a parkour course by chaining ``num_modules`` random skill modules.

    ``ramp_difficulty`` linearly grows the per-module difficulty from ``0.35 *
    difficulty`` at the start of the course to ``difficulty`` at the end.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    unknown = set(skills) - set(MODULES)
    if unknown:
        raise KeyError(f"unknown parkour skills {sorted(unknown)}; available: {sorted(MODULES)}")

    probs = None
    if skill_weights:
        weights = np.array([max(skill_weights.get(s, 0.0), 0.0) for s in skills], dtype=float)
        if weights.sum() > 0:
            probs = weights / weights.sum()

    cursor = Cursor()
    boxes = [
        _platform("start_pad", -1.5, start_pad_length, 0.0, 0.0, 3.0, rgba=(0.4, 0.5, 0.6, 1.0)),
    ]
    cursor.x = start_pad_length
    waypoints: list[tuple[float, float, float]] = [(start_pad_length, 0.0, 0.0)]
    module_names: list[str] = ["start_pad"]

    for i in range(num_modules):
        cursor.index = i
        name = str(rng.choice(skills, p=probs))
        t = (i + 1) / num_modules if ramp_difficulty else 1.0
        local_d = difficulty * (0.35 + 0.65 * t)
        out = MODULES[name](rng, local_d, cursor)
        boxes.extend(out.boxes)
        waypoints.extend(out.waypoints)
        module_names.append(name)

    boxes.append(
        _platform("finish_pad", cursor.x, cursor.x + 2.5, cursor.y, cursor.z, 3.0, material="grid")
    )
    waypoints.append((cursor.x + 2.0, cursor.y, cursor.z))

    return TerrainSpec(
        name="parkour",
        boxes=boxes,
        waypoints=np.asarray(waypoints, dtype=np.float64),
        spawn_pos=(0.0, 0.0, 0.0),
        add_ground_plane=True,
        ground_plane_z=-pit_depth,
        resolve_waypoint_z=False,
        difficulty=difficulty,
        modules=module_names,
        course_length=cursor.x + 2.5,
    )
