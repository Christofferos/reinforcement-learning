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

from ..mdp.events import ICE_MATERIAL, ICE_RGBA
from .actors import CrumbleActor, ElevatorActor, RunClockActor, TrampolineActor, WindActor
from .core import Box, TerrainBody, TerrainSpec, WaterVolume, segment

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
    water: list[WaterVolume] = field(default_factory=list)
    bodies: list[TerrainBody] = field(default_factory=list)
    actors: list = field(default_factory=list)
    speed_boost: float = 0.0
    """m/s added to the speed command of every waypoint of the module, as a sprint."""


def _lerp(low: float, high: float, t: float) -> float:
    return low + (high - low) * t


EXIT_RUNOUT = 1.8
"""Flat run-out after a module's last hazard, with the module's last waypoint at its end. The
0.7 m waypoint radius then counts the module passed 1.1 m past the hazard, so a fall while
recovering from it (standing up after a tunnel, landing a gap) is charged to that module, not
the next one, by the per-skill curriculum."""


def _platform(name: str, x0: float, x1: float, y: float, z: float, width: float,
              thickness: float = PLATFORM_THICKNESS, **kw) -> Box:
    kw.setdefault("material", "Concrete")
    return Box(
        name=name,
        pos=((x0 + x1) / 2.0, y, z - thickness / 2.0),
        size=((x1 - x0) / 2.0, width / 2.0, thickness / 2.0),
        **kw,
    )


def _slope(name: str, x0: float, z0: float, length: float, angle: float, y: float, width: float,
           **kw) -> Box:
    """A slab whose top face runs from (x0, z0) over ``length`` m of x, rising ``angle`` degrees."""
    rad = np.deg2rad(angle)
    cx = x0 + length / 2.0
    cz = z0 + length * np.tan(rad) / 2.0
    return Box(
        name,
        # Centre the slab half its thickness below its top face along the face normal,
        # (-sin, cos) in x-z. Offsetting z alone slid the face 0.2 * sin(angle) m along the
        # slope, leaving a 14 cm deep notch where a 22-degree ramp met the platform before it.
        pos=(cx + PLATFORM_THICKNESS / 2.0 * np.sin(rad), y, cz - PLATFORM_THICKNESS / 2.0 * np.cos(rad)),
        size=(length / 2.0 / np.cos(rad), width / 2.0, PLATFORM_THICKNESS / 2.0),
        euler=(0.0, -angle, 0.0),
        **kw,
    )


# --------------------------------------------------------------------------------------
# Modules.  Signature: (rng, difficulty, cursor) -> ModuleOutput, mutating the cursor.
# --------------------------------------------------------------------------------------

RUN_CLOCK_SHARE: tuple[float, float] | None = None
"""Off for now. As (0.5, 0.65): the share of the run's commanded speed the robot must average
across it at ``d`` = 0 and 1, or the episode ends ``"too_slow"`` (``RunClockActor``).
procedural_0.0.6 averaged 64%, 59% and 55% of it at d = 0, 0.5 and 1 (1.20-1.32 m/s against
1.90-2.41 m/s asked), whatever the boost. To come back once most skills are at high levels."""


def mod_run(rng, d, c: Cursor) -> ModuleOutput:
    """A flat sprint: its waypoints ask for the course's sampled speed plus 0.5 m/s at d = 0 to
    1.0 m/s at d = 1. With ``RUN_CLOCK_SHARE`` set it also runs against a clock."""
    length = _lerp(3.0, 2.0, d) + rng.uniform(0.0, 1.0)
    box = _platform(f"run_{c.index}", c.x, c.x + length, c.y, c.z, DEFAULT_WIDTH)
    actors = []
    if RUN_CLOCK_SHARE is not None:
        actors.append(RunClockActor((c.x, c.x + length), c.y, c.z, _lerp(*RUN_CLOCK_SHARE, d)))
    out = ModuleOutput([box], [(c.x + length * 0.5, c.y, c.z), (c.x + length, c.y, c.z)],
                       speed_boost=_lerp(0.5, 1.0, d), actors=actors)
    c.x += length
    return out


def mod_gap(rng, d, c: Cursor) -> ModuleOutput:
    # Up to about the longest stride the humanoid has: legs ~0.8 m, spread ~35 degrees each way
    # from the hip, is ~0.9 m foot to foot, a 0.8 m gap with each foot 5 cm off the edges
    # (``feet_edge``). It reached 1.73 m, a running long jump for a walker stepping 0.3-0.45 m.
    gap = _lerp(0.3, 0.8, d) * rng.uniform(0.9, 1.1)
    landing = rng.uniform(1.6, 2.6)
    # Landing height against the take-off. Up to d = 0.5 it may sit up to 0.15 d higher (tapering
    # to level at 0.5) or 0.15 d lower. Beyond, the longer the gap the lower the landing, 0.20 to
    # 0.35 m down at d = 1: stepping off with no upward speed, 0.3 m of drop is 0.25 s more in the
    # air, about 0.3 m more reach at the robot's 1.3 m/s take-off. Never deeper than 0.35 m, where
    # the ground scan (clipped at -0.5 m, 0.4 m below the floor it stands on) would read the
    # landing the same as falling off the course. It was ±0.15 d at every level: at d = 1 an
    # 0.8 m gap could land 15 cm up, which a robot that jumps 7 cm cannot reach.
    if d <= 0.5:
        lowest, highest = -0.15 * d, 0.15 * d * (1.0 - 2.0 * d)
    else:
        t = (d - 0.5) / 0.5
        lowest, highest = -_lerp(0.075, 0.35, t), -_lerp(0.0, 0.2, t)
    offset = rng.uniform(lowest, highest)
    x0 = c.x + gap
    box = _platform(
        f"gap_land_{c.index}", x0, x0 + landing, c.y, c.z + offset, DEFAULT_WIDTH,
        rgba=(0.7, 0.1, 0.2, 1.0), material=None,
    )
    c.x = x0 + landing
    c.z += offset
    return ModuleOutput([box], [(x0 + 0.4, c.y, c.z), (c.x, c.y, c.z)])


def mod_stepping_stones(rng, d, c: Cursor) -> ModuleOutput:
    """Stones a step apart: the same edge-to-edge gap before the first stone, between stones
    and after the last, from a 0.1 m crack to 0.3 m. With 5 cm kept off each edge
    (``feet_edge``) the longest step needed is 0.4 m, a long step rather than a jump. Stones
    used to sit at even centre spacing from the previous section with the landing reaching
    0.6 m back, so at d=1 the first gap was 1.07 m and at d=0 the last stone overlapped it.
    """
    count = int(_lerp(3, 6, d))
    half = _lerp(0.4, 0.25, d)
    gap = _lerp(0.1, 0.3, d)
    # Neighbours are at most 0.4 m apart sideways, so 0.5 m stones still overlap by 0.1 m.
    lateral = 0.2 * d
    boxes, waypoints = [], []
    near_edge = c.x + gap
    for i in range(count):
        x = near_edge + half
        y = c.y + rng.uniform(-lateral, lateral)
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
        near_edge = x + half + gap
    boxes.append(_platform(f"stone_land_{c.index}", near_edge, near_edge + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH))
    c.x = near_edge + EXIT_RUNOUT
    waypoints.append((c.x, c.y, c.z))
    return ModuleOutput(boxes, waypoints)


def mod_hurdle(rng, d, c: Cursor) -> ModuleOutput:
    length = 4.0
    # Up to 0.5 m, 63% of the humanoid's 0.79 m hip height; 0.65 m was 82%. Grows with d², so
    # the low curriculum levels stay near what the walker can step over: procedural_0.0.2
    # passed 15/15 at 0.20 m and 3/15 at 0.24 m, and with linear growth every hurdle along a
    # d=0.38 course was already 0.24-0.31 m.
    height = 0.2 + 0.3 * d**2
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
    rise = length * np.tan(np.deg2rad(angle))
    cx = c.x + length / 2.0
    cz = c.z + rise / 2.0
    boxes = [
        _slope(
            f"ramp_{c.index}", c.x, c.z, length, angle, c.y, DEFAULT_WIDTH,
            material="Metal" if rng.random() < 0.3 else "Concrete",
            friction=(0.6 if rng.random() < 0.3 else 1.0, 0.1, 0.1),
        )
    ]
    c.x += length
    c.z += rise
    boxes.append(_platform(f"ramp_top_{c.index}", c.x, c.x + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH))
    c.x += EXIT_RUNOUT
    return ModuleOutput(boxes, [(cx, c.y, cz), (c.x, c.y, c.z)])


def mod_beam(rng, d, c: Cursor) -> ModuleOutput:
    length = _lerp(2.5, 4.0, d)
    # 1.0 m at d=0, so Procedural's ±0.4 m ground scan stays on the beam and it reads like a
    # walkway; the outer rays first hang over the pit near d=0.3 (0.8 m). It started at 0.6 m,
    # where 18 of 45 rays were over the pit from the start and procedural_0.0.1 failed 17/17.
    width = _lerp(1.0, 0.28, d)
    boxes = [
        _platform(f"beam_{c.index}", c.x, c.x + length, c.y, c.z, width, material="Wood"),
    ]
    mid = c.x + length / 2.0
    c.x += length
    boxes.append(_platform(f"beam_land_{c.index}", c.x, c.x + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH))
    c.x += EXIT_RUNOUT
    return ModuleOutput(boxes, [(mid, c.y, c.z), (c.x, c.y, c.z)])


TUNNEL_POST_LENGTH = 0.175
TUNNEL_POST_WIDTH = 0.12
"""The roof's pillars, along and across the course: half the 0.35 x 0.24 m they were."""


def mod_tunnel(rng, d, c: Cursor) -> ModuleOutput:
    # 1.25 m at d=1, 0.28 m below the standing humanoid's head top (1.53 m). It went to 1.05 m,
    # then 1.2 m, where procedural_0.0.3 cleared 0 of 16 d = 0.9 bars (stuck or fell crouching).
    clearance = _lerp(1.45, 1.25, d)
    # The covered stretch grows with d, from a 0.24 m bar to a 1 m roof along the course, which
    # has to be crouch-walked (the height termination relaxes under it).
    half_depth = _lerp(0.12, 0.5, d)
    # 1.75 m to the roof's middle, then EXIT_RUNOUT plus 0.4 m past its end: procedural_0.0.5 fell
    # 0.3-1.5 m past d = 0.9 roofs standing up out of the crouch, which then counted against the
    # next module (the run-out was 1.25 m at d = 1, the module passed 0.55 m past the roof).
    length = 1.75 + half_depth + EXIT_RUNOUT + 0.4
    boxes = [_platform(f"tunnel_run_{c.index}", c.x, c.x + length, c.y, c.z, DEFAULT_WIDTH)]
    bar_x = c.x + 1.75
    boxes.append(
        Box(
            f"tunnel_bar_{c.index}",
            pos=(bar_x, c.y, c.z + clearance + 0.1),
            size=(half_depth, DEFAULT_WIDTH / 2.0, 0.1),
            material="Wood",
            ceiling=True,
        )
    )
    # Pillars at most TUNNEL_POST_LENGTH along the course under the middle of the roof, not side
    # walls as long as a 1 m roof.
    post_half = min(half_depth, TUNNEL_POST_LENGTH / 2.0)
    for side, sign in (("l", 1), ("r", -1)):
        boxes.append(
            Box(
                f"tunnel_post_{c.index}_{side}",
                # Standing on the floor's edge, outer face flush with it, 1.08 m out from the route.
                pos=(bar_x, c.y + sign * (DEFAULT_WIDTH - TUNNEL_POST_WIDTH) / 2.0,
                     c.z + (clearance + 0.1) / 2.0),
                size=(post_half, TUNNEL_POST_WIDTH / 2.0, (clearance + 0.1) / 2.0),
                material="Wood",
            )
        )
    c.x += length
    return ModuleOutput(boxes, [(bar_x, c.y, c.z), (c.x, c.y, c.z)])


def mod_climb(rng, d, c: Cursor) -> ModuleOutput:
    # The humanoid's knee is 0.43 m and its hip 0.79 m above the floor, and it has no hands:
    # the old 0.75 m top was 95% of hip height. Now from Rough's tallest rubble step at d=0
    # to about knee height. The riser starts where the floor ends, with no gap, so the robot
    # can stand against it before stepping up; the block reaches down to the floor's
    # underside, so above a 0.4 m rise no slot opens under the riser for a foot to slip into.
    # Grows with d², like the hurdle: procedural_0.0.2 climbed 0.20 m every time, 0.24 m 10/15
    # and 0.30 m 1/15.
    rise = 0.15 + 0.3 * d**2
    x0 = c.x
    length = rng.uniform(2.0, 3.0)
    boxes = [
        _platform(
            f"climb_{c.index}", x0, x0 + length, c.y, c.z + rise, DEFAULT_WIDTH,
            thickness=rise + PLATFORM_THICKNESS, rgba=(0.4, 0.7, 0.9, 1.0), material=None,
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
    field = 4.0  # blocks centred 0.5-3.5 m in, each at most 0.43 m from its centre
    length = field + EXIT_RUNOUT
    boxes = [_platform(f"rubble_run_{c.index}", c.x, c.x + length, c.y, c.z, DEFAULT_WIDTH)]
    # 8 blocks at d = 0 to 24 at 1 over the 3 m x 2 m field (it was 4 to 12, which
    # procedural_0.0.4 crossed 16/16 at d = 0.6): about 4 per square metre at d = 1, so a
    # straight line through the field meets several.
    for i in range(int(_lerp(8, 24, d))):
        h = rng.uniform(0.04, 0.04 + 0.16 * d)
        boxes.append(
            Box(
                f"rubble_{c.index}_{i}",
                pos=(rng.uniform(c.x + 0.5, c.x + field - 0.5), c.y + rng.uniform(-1.0, 1.0), c.z + h / 2),
                size=(rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3), h / 2),
                euler=(0.0, 0.0, rng.uniform(0, 90)),
                rgba=(0.45, 0.42, 0.4, 1.0),
                material=None,
            )
        )
    mid = c.x + field / 2.0
    c.x += length
    return ModuleOutput(boxes, [(mid, c.y, c.z), (c.x, c.y, c.z)])


SAND_SOLREF = (0.05, 1.5)
SAND_SOLIMP = (0.3, 0.95, 0.05)
"""Soft, heavily damped contact: a standing foot sinks about 3.5 cm and nothing bounces."""
SAND = dict(material="MatSand", friction=(0.9, 0.1, 0.1), solref=SAND_SOLREF, solimp=SAND_SOLIMP)
WATER_SURFACE_BELOW_RIM = 0.1
"""The water stands this far below the rim the robot drops in from."""
SAND_BANK_MAX_SLOPE = 35.0
"""Steepest flank of a sand bank, in degrees, where it meets the floor."""


def _sand_banks(rng, d, c: Cursor, x0: float, length: float, floor_z: float, height: float) -> list[Box]:
    """Sand banks across the tank floor: wide cylinders lying across the course, buried so only
    a ``height``-high crest shows, slanted up to 15 degrees off square to the course.

    A cylinder rather than a heightfield: one contact per foot instead of a dozen, and the
    upward rays can tell when they start inside one. Its radius keeps the flanks at
    ``SAND_BANK_MAX_SLOPE``, gentle enough that ``feet_edge`` does not count them as a drop.
    """
    radius = height / (1.0 - np.cos(np.deg2rad(SAND_BANK_MAX_SLOPE)))
    crest_half_width = np.sqrt(2.0 * radius * height - height**2)
    banks = []
    for i in range(int(_lerp(2, 4, d))):
        slant = rng.uniform(-15.0, 15.0)
        tan = np.tan(np.deg2rad(abs(slant)))
        # Ends 5 cm into the 10 cm side walls, so no end face lies flush with a wall's face.
        half_length = DEFAULT_WIDTH / 2.0 / np.cos(np.deg2rad(slant)) - crest_half_width * tan + 0.05
        banks.append(Box(
            f"water_tank_bank_{c.index}_{i}",
            pos=(rng.uniform(x0 + 0.8, x0 + length - 0.6), c.y, floor_z + height - radius),
            # A cylinder's axis is its local z: 90 degrees about x lays it along y, then the
            # second angle turns it about the world z axis.
            size=(radius, half_length, 0.0),
            euler=(90.0, slant, 0.0),
            shape="cylinder",
            rgba=(0.82, 0.72, 0.5, 1.0),
            **SAND,
        ))
    return banks


WATER_TANK_EASE = 0.3 / 0.9
"""Scale on the water tank's difficulty: d = 0.9 builds what d = 0.3 did before 2026-10-17."""


def mod_water_tank(rng, d, c: Cursor) -> ModuleOutput:
    """A deck, a drop into a water tank with sand banks on the floor, and a sand ramp out.

    The deck is the tank's own: one block from the rim down to the tank's base, so its end is
    the riser and no wall overlaps the previous module. The robot steps or jumps 0.3-0.45 m down into water 0.2-0.35 m deep, which drags on the
    shins and feet and buoys them up, wades across sand banks 0.05-0.25 m high, and climbs a
    20-degree sand ramp back to the rim height. Floor, banks and ramp are soft contacts; the
    water has no geom, so the ground scan reads the floor under it.
    """
    # Eased 2026-10-17: d = 0.9 builds what d = 0.3 did, every value a third as far along (so
    # d = 1 is the old 0.33). procedural_0.0.7-0.0.8 reached only levels 0.22-0.3 here and
    # passed 5 of 16 at d = 0.3, failing on height in the water.
    d *= WATER_TANK_EASE
    # At most 0.45 m: Procedural's ground scan clips at -0.5 m, where a deeper floor would read
    # the same as falling off the course.
    drop = _lerp(0.3, 0.45, d)
    bank = _lerp(0.05, 0.25, d)
    length = _lerp(3.0, 5.0, d) + rng.uniform(0.0, 1.0)
    ramp_angle = 20.0
    ramp_length = drop / np.tan(np.deg2rad(ramp_angle))
    floor_z = c.z - drop
    deck = 0.8
    x0 = c.x + deck
    x1 = x0 + length
    span = length + ramp_length
    wall = 0.1
    depth = drop + PLATFORM_THICKNESS

    boxes = [
        _platform(f"water_tank_deck_{c.index}", c.x, x0, c.y, c.z, DEFAULT_WIDTH, thickness=depth),
        _platform(f"water_tank_floor_{c.index}", x0, x1, c.y, floor_z, DEFAULT_WIDTH, rgba=None, **SAND),
        _slope(f"water_tank_ramp_{c.index}", x1, floor_z, ramp_length, ramp_angle, c.y, DEFAULT_WIDTH, **SAND),
        *_sand_banks(rng, d, c, x0, length, floor_z, bank),
    ]
    # Side walls up to the rim, between the deck and the exit platform.
    for side, sign in (("l", 1), ("r", -1)):
        boxes.append(Box(
            f"water_tank_wall_{side}_{c.index}",
            pos=(x0 + span / 2.0, c.y + sign * (DEFAULT_WIDTH + wall) / 2.0, c.z - depth / 2.0),
            size=(span / 2.0, wall / 2.0, depth / 2.0),
        ))
    water = [
        WaterVolume(
            name=f"water_tank_{c.index}_water",
            center=(x0 + span / 2.0, c.y),
            half_size=(span / 2.0, DEFAULT_WIDTH / 2.0),
            surface_z=c.z - WATER_SURFACE_BELOW_RIM,
            floor_z=floor_z,
        )
    ]
    c.x = x1 + ramp_length
    boxes.append(_platform(f"water_tank_exit_{c.index}", c.x, c.x + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH))
    waypoints = [(x0 + 0.8, c.y, floor_z), ((x0 + x1) / 2.0, c.y, floor_z), (c.x, c.y, c.z)]
    c.x += EXIT_RUNOUT
    waypoints.append((c.x, c.y, c.z))
    return ModuleOutput(boxes, waypoints, water)


# --------------------------------------------------------------------------------------
# Experimental obstacle modules (not in Procedural): moving parts, special contacts, wind.
# --------------------------------------------------------------------------------------

WOOD_RGBA = (0.55, 0.38, 0.2, 1.0)
ROPE_RGBA = (0.72, 0.6, 0.4, 1.0)


def mod_spinning_barrel(rng, d, c: Cursor) -> ModuleOutput:
    """One to three barrels lying across a pit, free to spin on fixed axles.

    A foot pushing on a barrel's side turns it, so the robot has to step onto the crest and
    off again before it rolls away underfoot. Narrower, lighter-damped barrels at higher ``d``.
    """
    radius = 0.35
    count = int(np.floor(_lerp(1, 3, d) + 0.5))  # halves up: round() would make 2.5 barrels 2
    width = _lerp(2.0, 1.2, d)
    damping = _lerp(25.0, 4.0, d)
    gap = 0.04
    deck = 1.5
    boxes = [_platform(f"barrel_deck_{c.index}", c.x, c.x + deck, c.y, c.z, DEFAULT_WIDTH)]
    bodies, waypoints = [], []
    x = c.x + deck + gap + radius
    for i in range(count):
        geoms = [Box(
            f"barrel_{c.index}_{i}", (0.0, 0.0, 0.0), (radius, width / 2.0, 0.0),
            euler=(90.0, 0.0, 0.0), shape="cylinder", material="Wood", rgba=WOOD_RGBA, density=80.0,
        )]
        # Pale staves around the barrel, so its spin shows when rendering.
        for k in range(6):
            angle = 60.0 * k
            geoms.append(Box(
                f"barrel_{c.index}_{i}_stave_{k}",
                (radius * np.cos(np.deg2rad(angle)), 0.0, radius * np.sin(np.deg2rad(angle))),
                (0.025, width / 2.0 + 0.002, 0.004), euler=(0.0, 90.0 - angle, 0.0),
                rgba=(0.9, 0.8, 0.55, 1.0), material=None, visual=True,
            ))
        bodies.append(TerrainBody(
            f"barrel_body_{c.index}_{i}", (x, c.y, c.z - radius), geoms,
            joint="hinge", axis=(0.0, 1.0, 0.0), damping=damping,
        ))
        waypoints.append((x, c.y, c.z))
        x += 2.0 * radius + gap
    landing_x = x - radius
    boxes.append(_platform(f"barrel_land_{c.index}", landing_x, landing_x + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH))
    c.x = landing_x + EXIT_RUNOUT
    waypoints.append((c.x, c.y, c.z))
    return ModuleOutput(boxes, waypoints, bodies=bodies)


SNOW_RGBA = (0.97, 0.97, 1.0, 1.0)
ICE_SLIDE_FRICTION = (0.30, 0.22)
"""Slide friction at d = 0 and d = 1, never below 0.22."""
ICE_SLIDE_REFLECTANCE = (0.1, 0.2, 0.3, 0.4, 0.5)
"""Reflectance of the slide's ice materials ``MatIceSlide0``..``4`` (scene.py makes them), from
the grippiest ice to the most slippery: the less friction, the glassier it looks."""


def ice_slide_material(friction: float) -> str:
    """Name of the slide ice material for ``friction``: more reflective the more slippery."""
    grip, slick = ICE_SLIDE_FRICTION
    t = float(np.clip((grip - friction) / (grip - slick), 0.0, 1.0))
    return f"MatIceSlide{int(np.floor(t * (len(ICE_SLIDE_REFLECTANCE) - 1) + 0.5))}"


def _snowman(name: str, base: tuple[float, float, float], facing: float) -> list[Box]:
    """A 0.6 m snowman standing on ``base``, its carrot nose pointing ``facing`` (radians from
    +x): three snowballs, coal eyes and buttons, twig arms and a top hat. Decoration only."""
    x, y, z = base
    ahead = np.array([np.cos(facing), np.sin(facing), 0.0])
    side = np.array([-ahead[1], ahead[0], 0.0])
    deco = dict(material=None, visual=True)
    parts = []
    centre = np.array([x, y, z])
    for k, (radius, height) in enumerate(((0.16, 0.15), (0.115, 0.37), (0.085, 0.53))):
        parts.append(Box(f"{name}_snow_{k}", tuple(centre + (0, 0, height)), (radius, 0.0, 0.0),
                         shape="sphere", rgba=SNOW_RGBA, **deco))
    head = centre + np.array([0.0, 0.0, 0.53])
    for k, sign in enumerate((-1, 1)):
        eye = head + 0.075 * ahead + 0.03 * sign * side + np.array([0.0, 0.0, 0.025])
        parts.append(Box(f"{name}_eye_{k}", tuple(eye), (0.012, 0.0, 0.0), shape="sphere",
                         rgba=(0.05, 0.05, 0.05, 1.0), **deco))
    nose = head + 0.08 * ahead
    parts.append(segment(f"{name}_carrot", nose, nose + 0.1 * ahead + np.array([0.0, 0.0, -0.01]), 0.016,
                         shape="ellipsoid", rgba=(1.0, 0.5, 0.08, 1.0), **deco))
    for k, height in enumerate((0.31, 0.37, 0.43)):
        button = centre + np.array([0.0, 0.0, height]) + 0.11 * ahead
        parts.append(Box(f"{name}_button_{k}", tuple(button), (0.012, 0.0, 0.0), shape="sphere",
                         rgba=(0.05, 0.05, 0.05, 1.0), **deco))
    for k, sign in enumerate((-1, 1)):
        shoulder = centre + np.array([0.0, 0.0, 0.4]) + 0.1 * sign * side
        parts.append(segment(f"{name}_arm_{k}", shoulder, shoulder + 0.2 * sign * side + np.array([0.0, 0.0, 0.1]),
                             0.008, rgba=(0.35, 0.22, 0.1, 1.0), **deco))
    parts.append(Box(f"{name}_hat_brim", tuple(head + (0, 0, 0.07)), (0.08, 0.005, 0.0), shape="cylinder",
                     rgba=(0.08, 0.08, 0.1, 1.0), **deco))
    parts.append(Box(f"{name}_hat", tuple(head + (0, 0, 0.12)), (0.055, 0.05, 0.0), shape="cylinder",
                     rgba=(0.08, 0.08, 0.1, 1.0), **deco))
    return parts


def mod_icy_slide(rng, d, c: Cursor) -> ModuleOutput:
    """An icy slope down onto a run-out, with a snowman in one of the run-out's corners.

    Friction is the main lever: it limits every step's grip, not just standing still, and
    at 0.12 the robot could not push off or brake even on the flat. The slope stays gentle,
    6° to 9° (it went to 15°, then 12°), and friction never drops below 0.22 (it went to 0.12,
    then 0.18): at d = 1 the decline's tangent, 0.16, is well under the friction, so the ice is
    slippery underfoot but holds still. The ice gets glassier as friction drops.
    """
    angle = _lerp(6.0, 9.0, d)
    mu = _lerp(*ICE_SLIDE_FRICTION, d)
    length = _lerp(2.5, 3.5, d)
    drop = length * np.tan(np.deg2rad(angle))
    # 1 cm narrower each side than the platforms: the slab's upper end tucks under the previous
    # platform, and side faces lying in one plane there flickered between ice and concrete.
    boxes = [_slope(
        f"icy_slide_{c.index}", c.x, c.z, length, -angle, c.y, DEFAULT_WIDTH - 0.02,
        material=ice_slide_material(mu), rgba=tuple(ICE_RGBA), friction=(mu, 0.005, 0.0005),
    )]
    mid = (c.x + length / 2.0, c.y, c.z - drop / 2.0)
    c.x += length
    c.z -= drop
    # The slab's lower end is square to the slope and leans back from the run-out, leaving a
    # wedge-shaped crack under the junction; starting the run-out under the slab fills it.
    tuck = PLATFORM_THICKNESS * np.sin(np.deg2rad(angle))
    runout = 2.5
    boxes.append(_platform(f"icy_slide_runout_{c.index}", c.x - tuck, c.x + runout, c.y, c.z, DEFAULT_WIDTH))
    # The snowman stands in a random corner of the run-out, off the robot's line, facing the slide.
    corner_x = c.x + (0.35 if rng.random() < 0.5 else runout - 0.35)
    corner_y = c.y + (DEFAULT_WIDTH / 2.0 - 0.3) * (1 if rng.random() < 0.5 else -1)
    boxes += _snowman(f"icy_slide_snowman_{c.index}", (corner_x, corner_y, c.z), np.arctan2(c.y - corner_y, -1.0))
    c.x += runout
    return ModuleOutput(boxes, [mid, (c.x, c.y, c.z)])


def mod_seesaw(rng, d, c: Cursor) -> ModuleOutput:
    """A plank on a damped hinge, resting with its near end down at the deck. Its pivot sits
    ``offset`` past the plank's middle, so it stays down until the robot walks past the pivot
    (0.5 m at d = 0, 0.15 m at d = 1), then tips (in about 0.9 s down to 0.5 s) and sets its far
    end down at the exit."""
    length = _lerp(3.0, 4.0, d)
    width = _lerp(1.2, 0.6, d)
    tilt = _lerp(8.0, 14.0, d)
    # A lighter plank, a smaller pivot offset and less damping at higher d: with a 42 kg load it
    # tips from 0.52 m past the pivot in 0.94 s at d = 0, and from 0.16 m in 0.49 s at d = 1.
    damping = _lerp(800.0, 190.0, d)
    density = _lerp(400.0, 250.0, d)
    offset = _lerp(0.12, 0.07, d)
    thickness, gap, deck = 0.1, 0.08, 1.0
    rad = np.deg2rad(tilt)
    boxes = [_platform(f"seesaw_deck_{c.index}", c.x, c.x + deck, c.y, c.z, DEFAULT_WIDTH)]
    near_end = c.x + deck + gap
    # In the body frame the plank's top runs from -(L/2 + offset) to (L/2 - offset) along x.
    pivot_x = near_end + (length / 2.0 + offset) * np.cos(rad)
    pivot_z = c.z + (length / 2.0 + offset) * np.sin(rad)
    plank = Box(
        f"seesaw_plank_{c.index}", (-offset, 0.0, -thickness / 2.0), (length / 2.0, width / 2.0, thickness / 2.0),
        material="Wood", rgba=WOOD_RGBA, density=density,
    )
    hub = Box(
        f"seesaw_hub_{c.index}", (0.0, 0.0, -thickness), (0.08, width / 2.0 + 0.05, 0.0),
        euler=(90.0, 0.0, 0.0), shape="cylinder", material=None, rgba=(0.3, 0.3, 0.35, 1.0), visual=True,
    )
    bodies = [TerrainBody(
        f"seesaw_body_{c.index}", (pivot_x, c.y, pivot_z), [plank, hub],
        joint="hinge", axis=(0.0, 1.0, 0.0), joint_range=(0.0, 2.0 * tilt), damping=damping,
        euler=(0.0, -tilt, 0.0),
    )]
    # The A-frame the axle rests on, down to the pit floor.
    for side in (-1, 1):
        for lean in (-1, 1):
            boxes.append(segment(
                f"seesaw_leg_{c.index}_{side}_{lean}",
                (pivot_x, c.y + side * (width / 2.0 + 0.1), pivot_z - thickness),
                (pivot_x + lean * 0.8, c.y + side * (width / 2.0 + 0.1), pivot_z - 3.0),
                0.05, material=None, rgba=(0.35, 0.35, 0.4, 1.0), visual=True,
            ))
    far_end = pivot_x + (length / 2.0 - offset) * np.cos(rad)
    exit_z = pivot_z - (length / 2.0 - offset) * np.sin(rad)
    c.x = far_end + gap
    c.z = exit_z
    boxes.append(_platform(f"seesaw_exit_{c.index}", c.x, c.x + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH))
    waypoints = [(pivot_x - 0.5, c.y, pivot_z - 0.5 * np.sin(rad)), (pivot_x + 0.8, c.y, pivot_z),
                 (c.x + EXIT_RUNOUT, c.y, c.z)]
    c.x += EXIT_RUNOUT
    return ModuleOutput(boxes, waypoints, bodies=bodies)


TRAMPOLINE_STIFFNESS = 5000.0
"""Spring under the bed in N/m: the standing robot sinks it about 10 cm, and a 0.4 m drop
compresses it about 0.27 m."""
TRAMPOLINE_BED_MASS = 10.0
TRAMPOLINE_LAUNCH = 3.5
"""Speed in m/s, along the bed's normal, the bed's rebound gives a robot that dropped onto it
(``TrampolineActor``)."""
TRAMPOLINE_TILT = 5.0
"""Degrees the bed slopes down towards the slot."""
TRAMPOLINE_RAMP_ANGLE = 10.0
"""Degrees the landing ramp beyond the slot rises, over 0.8 m (0.14 m)."""
TRAMPOLINE_SLOT = (0.2, 0.25)
"""Width in m of the open slot between the bed and the landing, at ``d`` = 0 and 1: at least
one 0.2 m scan row, so the scan always sees it, and narrower than the 0.27 m shortest gap."""


def mod_trampoline(rng, d, c: Cursor) -> ModuleOutput:
    """Drop onto a tilted trampoline bed below the deck and bounce over a slot onto a landing.

    The bed is a light body on a spring (a slide joint with stiffness and little damping),
    not a soft contact: soft contacts either barely gave or, underdamped, became unstable. A
    landing launches the robot at up to ``TRAMPOLINE_LAUNCH`` (``TrampolineActor``).

    The shape is the trampoline's signature in the height scan, found in no other module: the
    bed slopes down ``TRAMPOLINE_TILT`` degrees to a full-width slot that reads as a pit, and
    the landing beyond it ramps back up at ``TRAMPOLINE_RAMP_ANGLE``. Mid-bed a flat trampoline looked like a climb. The
    bed's spring and launch run along its normal, so the bounce also throws the robot forward
    over the slot. The landing's near edge is 0.1 m above the bed's low end at ``d`` = 0 and
    0.4 m at 1, then rises another 0.14 m over 0.8 m.
    """
    drop = _lerp(0.25, 0.35, d)
    slot = _lerp(*TRAMPOLINE_SLOT, d)
    tilt = np.deg2rad(TRAMPOLINE_TILT)
    bed_length, bed_width, pad, bed_thickness = 1.0, 2.0, 0.2, 0.1
    deck = 1.2
    back_gap = 0.04  # the squashed bed slides back along its normal (up to 0.04 m) clear of the deck
    bed_x, bed_z = c.x + deck + back_gap, c.z - drop

    def on_bed(u: float, w: float) -> tuple[float, float]:
        """World (x, z) of a point ``u`` m along the bed's top from its high end, ``w`` m above it."""
        return bed_x + u * np.cos(tilt) + w * np.sin(tilt), bed_z - u * np.sin(tilt) + w * np.cos(tilt)

    front_x, front_z = on_bed(bed_length, 0.0)
    normal = (float(np.sin(tilt)), 0.0, float(np.cos(tilt)))
    boxes = [
        _platform(f"trampoline_deck_{c.index}", c.x, bed_x, c.y, c.z, DEFAULT_WIDTH, thickness=drop + PLATFORM_THICKNESS),
    ]
    bed = [Box(
        f"trampoline_bed_{c.index}", (bed_length / 2.0, 0.0, -bed_thickness / 2.0),
        (bed_length / 2.0, bed_width / 2.0, bed_thickness / 2.0), material=None, rgba=(0.08, 0.08, 0.1, 1.0),
        density=TRAMPOLINE_BED_MASS / (bed_length * bed_width * bed_thickness),
    )]
    # The net: a grid of thin strands on the bed, moving with it.
    for k, x in enumerate(np.arange(0.05, bed_length, 0.1)):
        bed.append(Box(f"trampoline_net_x_{c.index}_{k}", (x, 0.0, 0.002), (0.004, bed_width / 2.0, 0.002),
                       material=None, rgba=(0.85, 0.85, 0.85, 1.0), visual=True))
    for k, y in enumerate(np.arange(-bed_width / 2.0 + 0.05, bed_width / 2.0, 0.1)):
        bed.append(Box(f"trampoline_net_y_{c.index}_{k}", (bed_length / 2.0, y, 0.002), (bed_length / 2.0, 0.004, 0.002),
                       material=None, rgba=(0.85, 0.85, 0.85, 1.0), visual=True))
    sag = TRAMPOLINE_BED_MASS * 9.81 * np.cos(tilt) / TRAMPOLINE_STIFFNESS
    bodies = [TerrainBody(
        f"trampoline_body_{c.index}", (bed_x, c.y, bed_z), bed,
        joint="slide", axis=(0.0, 0.0, 1.0), joint_range=(-0.45, 0.05),  # the bed's own normal
        stiffness=TRAMPOLINE_STIFFNESS, springref=sag, damping=30.0, euler=(0.0, TRAMPOLINE_TILT, 0.0),
    )]
    for side in (-1, 1):  # padded frame rails along both sides of the bed, back to the slot
        rail_x, rail_z = on_bed((bed_length - pad) / 2.0, -0.02)
        boxes.append(Box(
            f"trampoline_pad_{c.index}_{side}", (rail_x, c.y + side * (bed_width + pad) / 2.0, rail_z),
            ((bed_length + pad) / 2.0, pad / 2.0, 0.07), material=None, rgba=(0.1, 0.35, 0.85, 1.0),
            euler=(0.0, TRAMPOLINE_TILT, 0.0),
        ))
    landing_x, landing_z = front_x + slot, front_z + _lerp(0.1, 0.4, d)
    ramp, ramp_slope = 0.8, np.tan(np.deg2rad(TRAMPOLINE_RAMP_ANGLE))
    boxes.append(_slope(f"trampoline_ramp_{c.index}", landing_x, landing_z, ramp, TRAMPOLINE_RAMP_ANGLE, c.y, DEFAULT_WIDTH))
    # Fills in under the sloped slab down to the landing's depth; its top stays inside the slab.
    boxes.append(_platform(
        f"trampoline_ramp_base_{c.index}", landing_x + 0.01, landing_x + ramp, c.y, landing_z - 0.01, DEFAULT_WIDTH,
        thickness=landing_z - 0.01 - bed_z + PLATFORM_THICKNESS,
    ))
    c.z = landing_z + ramp * ramp_slope
    boxes.append(_platform(
        f"trampoline_land_{c.index}", landing_x + ramp, landing_x + ramp + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH,
        thickness=c.z - bed_z + PLATFORM_THICKNESS,
    ))
    mid_x, mid_z = on_bed(bed_length / 2.0, 0.0)
    waypoints = [
        (mid_x, c.y, mid_z), (landing_x + 0.6, c.y, landing_z + 0.6 * ramp_slope), (landing_x + ramp + EXIT_RUNOUT, c.y, c.z),
    ]
    c.x = landing_x + ramp + EXIT_RUNOUT
    actor = TrampolineActor(
        (bed_x, front_x), bodies[0].joint_name, f"trampoline_bed_{c.index}", TRAMPOLINE_LAUNCH, normal=normal,
    )
    return ModuleOutput(boxes, waypoints, bodies=bodies, actors=[actor])


def mod_wind_turbine(rng, d, c: Cursor) -> ModuleOutput:
    """A walkway past a wind cannon on its left or right that, while the robot is on the
    walkway, fires balls that kick its base sideways (the random pushes, from one side and
    only here) and, at higher ``d``, blows a steady crosswind. Painted arrows show the wind's
    direction."""
    length = _lerp(4.0, 6.0, d)
    side = 1.0 if rng.random() < 0.5 else -1.0
    x0, x1 = c.x, c.x + length
    mid = (x0 + x1) / 2.0
    boxes = [_platform(f"wind_run_{c.index}", x0, x1, c.y, c.z, DEFAULT_WIDTH)]
    housing_y = c.y + side * (DEFAULT_WIDTH / 2.0 + 0.7)
    mouth_y = housing_y - side * 0.45
    mouth = (mid, mouth_y, c.z + 1.0)
    boxes += [
        Box(f"wind_tower_{c.index}", (mid, housing_y, c.z - 1.5), (0.25, 0.25, 2.1),
            material="Metal", rgba=(0.75, 0.75, 0.8, 1.0)),
        Box(f"wind_housing_{c.index}", (mid, housing_y, c.z + 1.0), (0.42, 0.45, 0.0),
            euler=(90.0, 0.0, 0.0), shape="cylinder", material="Metal", rgba=(0.9, 0.9, 0.95, 1.0)),
        Box(f"wind_hole_{c.index}", (mid, mouth_y - side * 0.004, c.z + 1.0), (0.34, 0.006, 0.0),
            euler=(90.0, 0.0, 0.0), shape="cylinder", material=None, rgba=(0.05, 0.05, 0.07, 1.0), visual=True),
    ]
    # Arrows painted across the walkway, pointing away from the cannon.
    def paint(name, start, end):
        (xa, ya), (xb, yb) = start, end
        boxes.append(Box(
            name, ((xa + xb) / 2.0, (ya + yb) / 2.0, c.z + 0.003), (np.hypot(xb - xa, yb - ya) / 2.0, 0.04, 0.002),
            euler=(0.0, 0.0, float(np.degrees(np.arctan2(yb - ya, xb - xa)))),
            material=None, rgba=(0.95, 0.85, 0.1, 1.0), visual=True,
        ))

    head = np.deg2rad(35.0)
    for k, x in enumerate(np.linspace(x0 + 0.8, x1 - 0.8, 3)):
        tip = (x, c.y - side * 0.7)
        paint(f"wind_arrow_{c.index}_{k}", (x, c.y + side * 0.7), tip)
        for wing, sign in (("a", 1), ("b", -1)):
            paint(f"wind_arrow_{c.index}_{k}_{wing}", tip,
                  (tip[0] + sign * 0.35 * np.sin(head), tip[1] + side * 0.35 * np.cos(head)))
    # Shots every 1.6 s down to 0.8 s, kicks up to 0.3 m/s up to 0.8 m/s (Rough's strongest
    # random push is 0.75), and a steady crosswind of up to 10 N (2.4% of the humanoid's weight)
    # to lean into between shots. Sideways velocity it is dealt per second at d = 1: about 1.0
    # m/s, against 0.66 with no crosswind and 0.7 m/s kicks; 2.0 (0.6 s, 1.0 m/s, 30 N) was too much.
    actor = WindActor(
        (x0, x1), mouth, side, interval=_lerp(1.6, 0.8, d), strength=_lerp(0.3, 0.8, d),
        steady=_lerp(0.0, 10.0, d),
    )
    boxes.append(_platform(f"wind_exit_{c.index}", x1, x1 + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH))
    c.x = x1 + EXIT_RUNOUT
    return ModuleOutput(boxes, [(mid, c.y, c.z), (c.x, c.y, c.z)], actors=[actor])


SLIDING_ELEVATOR_EASE = 0.5 / 0.9
"""Scale on the sliding elevator's difficulty: d = 0.9 builds what d = 0.5 did before 2026-10-17."""


def mod_sliding_elevator(rng, d, c: Cursor) -> ModuleOutput:
    """A platform shuttling across a pit between the boarding deck and the exit. The robot
    has to wait at the edge for it, step on, ride across and step off; it moves only while
    the robot is near (``ElevatorActor``)."""
    # Capped at what used to be d = 0.5, which is reached only at d = 1 now: above 0.6 the old
    # range (up to 1.25 m of travel at 0.5 m/s with 1.2 s stops) was far too hard. Every value
    # changes half as fast along the curriculum. Eased again 2026-10-17: d = 0.9 builds what
    # d = 0.5 did (d = 1 the old 0.56); procedural_0.0.8 had reached only 0.34 here.
    d *= SLIDING_ELEVATOR_EASE
    travel = _lerp(0.5, 0.875, d)
    platform = _lerp(1.4, 1.3, d)
    width = _lerp(1.6, 1.45, d)
    speed = _lerp(0.375, 0.4375, d)
    dwell = _lerp(1.5, 1.35, d)
    clear, deck, thickness, mass = 0.03, 1.5, 0.2, 2000.0
    deck_end = c.x + deck
    exit_x = deck_end + clear + platform + travel + clear
    boxes = [
        _platform(f"elevator_deck_{c.index}", c.x, deck_end, c.y, c.z, DEFAULT_WIDTH),
        _platform(f"elevator_exit_{c.index}", exit_x, exit_x + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH),
    ]
    for side in (-1, 1):  # rails under the platform's path
        boxes.append(segment(
            f"elevator_rail_{c.index}_{side}", (deck_end, c.y + side * width / 3.0, c.z - thickness - 0.05),
            (exit_x, c.y + side * width / 3.0, c.z - thickness - 0.05), 0.03,
            material=None, rgba=(0.4, 0.4, 0.45, 1.0), visual=True,
        ))
    volume = platform * width * thickness
    geoms = [Box(
        f"elevator_platform_{c.index}", (platform / 2.0, 0.0, -thickness / 2.0),
        (platform / 2.0, width / 2.0, thickness / 2.0), material=None,
        rgba=(0.95, 0.75, 0.1, 1.0), density=mass / volume,
    )]
    for k, x in enumerate((0.03, platform - 0.03)):  # hazard edges
        geoms.append(Box(
            f"elevator_edge_{c.index}_{k}", (x, 0.0, 0.002), (0.03, width / 2.0, 0.002),
            material=None, rgba=(0.1, 0.1, 0.1, 1.0), visual=True,
        ))
    joint = f"elevator_body_{c.index}_joint"
    bodies = [TerrainBody(
        f"elevator_body_{c.index}", (deck_end + clear, c.y, c.z), geoms,
        joint="slide", axis=(1.0, 0.0, 0.0), joint_range=(0.0, travel),
    )]
    actor = ElevatorActor(
        (c.x, exit_x + EXIT_RUNOUT), joint, travel, speed, dwell, boarding=(deck_end - 1.2, deck_end + 0.05),
    )
    waypoints = [(deck_end - 0.4, c.y, c.z), (exit_x + 0.5, c.y, c.z), (exit_x + EXIT_RUNOUT, c.y, c.z)]
    c.x = exit_x + EXIT_RUNOUT
    return ModuleOutput(boxes, waypoints, bodies=bodies, actors=[actor])


def mod_crumbling_bridge(rng, d, c: Cursor) -> ModuleOutput:
    """A sagging rope bridge of wooden planks over a pit; each plank drops 1 s (d = 0) to 0.5 s
    (d = 1) after the robot first touches it (``CrumbleActor``), so stopping on it is a fall."""
    span = _lerp(3.0, 5.0, d)
    plank_length = 0.3
    gap = _lerp(0.04, 0.15, d)
    width = _lerp(1.0, 0.7, d)
    sag = _lerp(0.05, 0.2, d)
    deck = 1.0
    x0 = c.x + deck
    boxes = [_platform(f"bridge_deck_{c.index}", c.x, x0, c.y, c.z, DEFAULT_WIDTH)]
    count = max(2, int((span + gap) / (plank_length + gap)))
    pitch = (span - plank_length) / (count - 1)

    def surface(x: float) -> float:
        s = (x - x0) / span
        return c.z - 4.0 * sag * s * (1.0 - s)

    rail_height = 0.95
    bodies, names = [], []
    for i in range(count):
        x = x0 + plank_length / 2.0 + i * pitch
        z = surface(x)
        slope = np.degrees(np.arctan((surface(x + 0.01) - surface(x - 0.01)) / 0.02))
        geoms = [Box(
            f"bridge_plank_{c.index}_{i}", (0.0, 0.0, -0.025), (plank_length / 2.0, width / 2.0, 0.025),
            material="Wood", rgba=WOOD_RGBA,
        )]
        # Hangers up to the hand ropes on every other plank, falling with it. Decoration is
        # drawn twice a frame (scene and shadow), so the bridge keeps it sparse: ~60 thin
        # capsules cost 4 ms of a 21 ms full-HD frame. Cylinders, as their ends never show.
        for side in (-1, 1) if i % 2 == 0 or i == count - 1 else ():
            geoms.append(segment(
                f"bridge_hanger_{c.index}_{i}_{side}", (0.0, side * width / 2.0, 0.0),
                (0.0, side * width / 2.0, rail_height), 0.008,
                shape="cylinder", material=None, rgba=ROPE_RGBA, visual=True,
            ))
        name = f"bridge_body_{c.index}_{i}"
        bodies.append(TerrainBody(name, (x, c.y, z), geoms, mocap=True, euler=(0.0, -slope, 0.0)))
        names.append(name)
    # Hand ropes and the ropes under the planks' ends, between the posts.
    for side in (-1, 1):
        for level, height in (("hand", rail_height), ("foot", -0.03)):
            points = [(x, c.y + side * width / 2.0, surface(x) + height) for x in np.linspace(x0, x0 + span, 5)]
            for k, (a, b) in enumerate(zip(points, points[1:])):
                boxes.append(segment(f"bridge_rope_{level}_{c.index}_{side}_{k}", a, b, 0.012,
                                     shape="cylinder", material=None, rgba=ROPE_RGBA, visual=True))
        for k, x in enumerate((x0, x0 + span)):
            boxes.append(segment(
                f"bridge_post_{c.index}_{side}_{k}", (x, c.y + side * (width / 2.0 + 0.05), c.z - 0.3),
                (x, c.y + side * (width / 2.0 + 0.05), c.z + rail_height + 0.15), 0.06,
                shape="cylinder", material=None, rgba=(0.4, 0.27, 0.13, 1.0), visual=True,
            ))
    c.x = x0 + span
    boxes.append(_platform(f"bridge_land_{c.index}", c.x, c.x + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH))
    waypoints = [(x0 + span / 2.0, c.y, surface(x0 + span / 2.0)), (c.x + EXIT_RUNOUT, c.y, c.z)]
    c.x += EXIT_RUNOUT
    actor = CrumbleActor((x0, x0 + span), names, delay=_lerp(1.0, 0.5, d))
    return ModuleOutput(boxes, waypoints, bodies=bodies, actors=[actor])


TILT_PASSAGE_WIDTH = (0.8, 0.28)
"""Gap between the walls at ``d`` = 0 and 1, metres. 0.8 m leaves three free scan columns and is
walked facing forward. Without touching the walls the humanoid needs 0.32-0.34 m facing forward
in any leg pose, 0.33 m sideways in its bent-knee stance and 0.24 m sideways with straight legs,
so 0.28 m makes it turn sideways and straighten up."""
TILT_PASSAGE_WALL_HEIGHT = 1.0
"""Below the shoulders (1.2-1.3 m, where the arms stick out to 0.7 m), so only the legs and hips
have to fit, and below a standing robot's ground-ray start (~1.4 m)."""


def mod_tilt_passage(rng, d, c: Cursor) -> ModuleOutput:
    """A corridor between two walls, narrowing with ``d`` until the robot has to turn sideways.

    The walls fill the floor out to its edges, so there is no way round. The width narrows
    smoothly: turned by an angle the body presents its width times cos plus its depth times sin,
    so every few degrees of turn help a little. The walls are ``Box.wall``: a ground ray that
    starts inside one (a crouching robot's) reads it as a wall, not the floor beside it. The
    corridor is flat and empty, since turned sideways the scan sees only 0.4 m along it.
    """
    width = _lerp(*TILT_PASSAGE_WIDTH, d)
    length = _lerp(1.5, 3.0, d)
    run_in = 1.2
    x0 = c.x + run_in
    x1 = x0 + length
    boxes = [_platform(f"tilt_run_{c.index}", c.x, x1 + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH)]
    half = (DEFAULT_WIDTH - width) / 4.0  # each wall's half-width, from the corridor to the edge
    for side, sign in (("l", 1), ("r", -1)):
        boxes.append(Box(
            f"tilt_wall_{c.index}_{side}",
            pos=((x0 + x1) / 2.0, c.y + sign * (width / 2.0 + half), c.z + TILT_PASSAGE_WALL_HEIGHT / 2.0),
            size=(length / 2.0, half, TILT_PASSAGE_WALL_HEIGHT / 2.0),
            material=None, rgba=(0.62, 0.36, 0.3, 1.0), wall=True,
        ))
    waypoints = [(x0 - 0.4, c.y, c.z), ((x0 + x1) / 2.0, c.y, c.z), (x1 + EXIT_RUNOUT, c.y, c.z)]
    c.x = x1 + EXIT_RUNOUT
    return ModuleOutput(boxes, waypoints)


BALL_RGBA = ((0.85, 0.2, 0.2, 1.0), (0.95, 0.75, 0.15, 1.0), (0.2, 0.45, 0.85, 1.0))


def mod_ball_balance(rng, d, c: Cursor) -> ModuleOutput:
    """A row of fixed balls over a pit, their tops level with the deck: step from crest to crest.

    Only a ball's top is level, so each step is a balance on a curved patch. At ``d`` = 0, two
    big balls (0.675 m radius) 0.6 m apart, crest to crest, overlap into a bumpy walkway with
    0.07 m dips; at ``d`` = 1, four smaller ones (0.4 m) 0.8 m apart just touch, with a sideways
    jitter up to 0.12 m. The crests stay a stride apart however big the balls: touching 0.675 m
    balls would be 1.35 m apart. The 0.2 m scan puts two to four rays on every ball. (Until
    2026-10-17: 2-6 balls of 0.6-0.35 m.)
    """
    radius = _lerp(0.675, 0.4, d)
    count = int(np.floor(_lerp(2, 4, d) + 0.5))
    spacing = _lerp(0.6, 0.8, d)  # crest to crest, also from the deck's edge and to the landing's
    lateral = _lerp(0.0, 0.12, d)
    deck = 1.2
    boxes = [_platform(f"ball_deck_{c.index}", c.x, c.x + deck, c.y, c.z, DEFAULT_WIDTH)]
    waypoints = []
    x = c.x + deck + spacing / 2.0
    for i in range(count):
        y = c.y + rng.uniform(-lateral, lateral)
        boxes.append(Box(
            f"ball_{c.index}_{i}", (x, y, c.z - radius), (radius, radius, radius), shape="sphere",
            material=None, rgba=BALL_RGBA[i % len(BALL_RGBA)],
        ))
        waypoints.append((x, y, c.z))
        x += spacing
    landing_x = x - spacing / 2.0
    boxes.append(_platform(f"ball_land_{c.index}", landing_x, landing_x + EXIT_RUNOUT, c.y, c.z, DEFAULT_WIDTH))
    c.x = landing_x + EXIT_RUNOUT
    waypoints.append((c.x, c.y, c.z))
    return ModuleOutput(boxes, waypoints)


def _ceiling_slab(name: str, x0: float, z0: float, x1: float, z1: float, y: float, width: float,
                  thickness: float, **kw) -> Box:
    """A slab whose underside runs from (x0, z0) to (x1, z1): by default a roof (``ceiling``), like
    a vent's funnel or sloped duct roof, or with ``ceiling=False`` a wall standing on a ramp."""
    rad = np.arctan2(z1 - z0, x1 - x0)
    mid_x, mid_z = (x0 + x1) / 2.0, (z0 + z1) / 2.0
    kw.setdefault("ceiling", True)
    return Box(
        name,
        pos=(mid_x - thickness / 2.0 * np.sin(rad), y, mid_z + thickness / 2.0 * np.cos(rad)),
        size=(np.hypot(x1 - x0, z1 - z0) / 2.0, width / 2.0, thickness / 2.0),
        euler=(0.0, -np.degrees(rad), 0.0), **kw,
    )


AIR_VENT_CLEARANCE = (1.2, 0.95)
"""Height of the vent's roof above its floor at ``d`` = 0 and 1, the same all along the sloped
duct: a deep duck-walk (the standing head top is 1.53 m), down to hands and knees (it was 1.15
to 0.85 m, and 1.0 to 0.7 m before). Both are within ``TerminationCfg.crawl_ceiling_clearance``,
so the crawl's lower torso and horizontal body are allowed while the upward rays see the roof."""
AIR_VENT_EXIT_RISE = (0.15, 0.25)
"""How far the vent's floor ramps up from its inlet to its exit, at ``d`` = 0 and 1."""


def mod_air_vent(rng, d, c: Cursor) -> ModuleOutput:
    """A low duct to crawl up through on hands and knees, then stand up and walk on.

    A sloped roof funnels down from 1.45 m (a tunnel's height) to the vent's clearance over 1 m.
    From there the floor ramps up ``AIR_VENT_EXIT_RISE`` over the duct's ``length`` to the exit,
    the roof following it at the same height above the floor, and the duct opens where the ramp
    ends. The run-out after it is 0.6 m longer than ``EXIT_RUNOUT``, for standing up from a
    crawl (procedural_0.0.5 fell 0.3-1.5 m past tunnel roofs standing up from a crouch). Walls
    at the floor's edges, standing on the ramp, close the duct's sides.
    """
    clearance = _lerp(*AIR_VENT_CLEARANCE, d)
    rise = _lerp(*AIR_VENT_EXIT_RISE, d)
    length = _lerp(1.5, 3.0, d)
    run_in, funnel_in, roof = 1.2, 1.0, 0.15
    xa = c.x + run_in  # the inlet funnel starts
    xb = xa + funnel_in  # the inlet ends, the ramp and its roof start
    xc = xb + length  # the ramp ends, the duct opens
    end = xc + EXIT_RUNOUT + 0.6
    wall_t = 0.12
    metal = dict(material="Metal")
    top = c.z + rise  # the exit's floor
    angle = np.degrees(np.arctan2(rise, length))
    boxes = [
        _platform(f"vent_run_{c.index}", c.x, xb, c.y, c.z, DEFAULT_WIDTH),
        _slope(f"vent_ramp_{c.index}", xb, c.z, length, angle, c.y, DEFAULT_WIDTH),
        _platform(f"vent_exit_{c.index}", xc, end, c.y, top, DEFAULT_WIDTH, thickness=rise + PLATFORM_THICKNESS),
        _ceiling_slab(f"vent_in_{c.index}", xa, c.z + 1.45, xb, c.z + clearance, c.y, DEFAULT_WIDTH, roof, **metal),
        _ceiling_slab(f"vent_roof_{c.index}", xb, c.z + clearance, xc, top + clearance, c.y, DEFAULT_WIDTH, roof, **metal),
    ]
    for side, sign in (("l", 1), ("r", -1)):
        y = c.y + sign * (DEFAULT_WIDTH - wall_t) / 2.0  # on the floor's edges, like tunnel pillars
        boxes.append(_ceiling_slab(
            f"vent_wall_{c.index}_{side}", xb, c.z, xc, top, y, wall_t, clearance, ceiling=False, **metal,
        ))
        boxes.append(Box(  # posts under the inlet funnel's open end
            f"vent_post_{c.index}_in_{side}", (xa, y, c.z + 1.45 / 2.0),
            (wall_t / 2.0, wall_t / 2.0, 1.45 / 2.0), **metal,
        ))
    waypoints = [(xa - 0.2, c.y, c.z), ((xb + xc) / 2.0, c.y, c.z + rise / 2.0), (xc, c.y, top), (end, c.y, top)]
    c.x, c.z = end, top
    return ModuleOutput(boxes, waypoints)


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
    "water_tank": mod_water_tank,
    "spinning_barrel": mod_spinning_barrel,
    "icy_slide": mod_icy_slide,
    "seesaw": mod_seesaw,
    "trampoline": mod_trampoline,
    "wind_turbine": mod_wind_turbine,
    "sliding_elevator": mod_sliding_elevator,
    "crumbling_bridge": mod_crumbling_bridge,
    "tilt_passage": mod_tilt_passage,
    "ball_balance": mod_ball_balance,
    "air_vent": mod_air_vent,
}

EXPERIMENTAL_SKILLS = ("air_vent", "trampoline")
INTRODUCED_SKILLS = (
    "water_tank", "icy_slide", "spinning_barrel", "wind_turbine", "crumbling_bridge",
    "sliding_elevator", "seesaw", "tilt_passage", "ball_balance",
)
"""Experimental until 2026-10-17, now in the default mix (Procedural trains them at weight 0.3)."""
"""Modules courses only get when asked for by name in ``skills``, not by default."""
DEFAULT_SKILLS = tuple(name for name in MODULES if name not in EXPERIMENTAL_SKILLS)


def make_parkour(
    rng: np.random.Generator,
    difficulty: float = 0.5,
    num_modules: int = 8,
    num_modules_growth: int = 0,
    skills: tuple[str, ...] = DEFAULT_SKILLS,
    skill_weights: dict[str, float] | None = None,
    start_pad_length: float = 3.0,
    pit_depth: float = 4.0,
    skill_difficulty: dict[str, float] | None = None,
    review_share: float = 0.0,
    preview_share: float = 0.0,
    preview_range: float = 0.1,
) -> TerrainSpec:
    """Sample a parkour course by chaining random skill modules: ``num_modules`` at difficulty 0,
    ``num_modules_growth`` more at difficulty 1, rounded to the nearest whole module.

    Every module is built at its skill's level in ``skill_difficulty`` (the per-skill
    curriculum), or at ``difficulty`` for a skill it leaves out, the same along the whole
    course. Courses used to ramp from 0.35 x ``difficulty`` at the start to ``difficulty`` at
    the end, so only the last module was at the stated level and hurdles above the curriculum's
    top height (0.35 m at d = 0.7) almost never appeared.

    For practice around the levels, ``review_share`` of the modules are built at a random easier
    level (uniform below the skill's) and ``preview_share`` up to ``preview_range`` above it;
    ``TerrainSpec.module_kinds`` records which, since only modules at the level count toward
    the skill's promotion.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    skill_difficulty = skill_difficulty or {}
    num_modules += int(np.floor(num_modules_growth * difficulty + 0.5))
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
    boosts: list[float] = [0.0]
    owners: list[int] = [0]  # the start pad's
    kinds: list[str] = ["pad"]
    water: list[WaterVolume] = []
    bodies: list[TerrainBody] = []
    actors: list = []
    module_names: list[str] = ["start_pad"]

    for i in range(num_modules):
        cursor.index = i
        name = str(rng.choice(skills, p=probs))
        local_d = float(np.clip(skill_difficulty.get(name, difficulty), 0.0, 1.0))
        kind = "level"
        if review_share or preview_share:  # no draw otherwise, so such courses stay as they were
            draw = rng.random()
            if draw < review_share and local_d > 0.0:
                local_d, kind = float(rng.uniform(0.0, local_d)), "review"
            elif review_share <= draw < review_share + preview_share and local_d < 1.0:
                local_d, kind = float(min(1.0, local_d + rng.uniform(0.0, preview_range))), "preview"
        kinds.append(kind)
        out = MODULES[name](rng, local_d, cursor)
        boxes.extend(out.boxes)
        waypoints.extend(out.waypoints)
        boosts.extend([out.speed_boost] * len(out.waypoints))
        owners.extend([len(module_names)] * len(out.waypoints))
        water.extend(out.water)
        bodies.extend(out.bodies)
        actors.extend(out.actors)
        module_names.append(name)

    boxes.append(
        _platform("finish_pad", cursor.x, cursor.x + 2.5, cursor.y, cursor.z, 3.0, material="grid")
    )
    waypoints.append((cursor.x + 2.0, cursor.y, cursor.z))
    boosts.append(0.0)
    owners.append(-1)  # the finish pad's

    return TerrainSpec(
        name="parkour",
        boxes=boxes,
        water=water,
        bodies=bodies,
        actors=actors,
        waypoints=np.asarray(waypoints, dtype=np.float64),
        waypoint_speed_boost=np.asarray(boosts, dtype=np.float64),
        waypoint_modules=np.asarray(owners, dtype=np.int64),
        spawn_pos=(0.0, 0.0, 0.0),
        add_ground_plane=True,
        ground_plane_z=-pit_depth,
        resolve_waypoint_z=False,
        difficulty=difficulty,
        modules=module_names,
        module_kinds=kinds,
        course_length=cursor.x + 2.5,
    )
