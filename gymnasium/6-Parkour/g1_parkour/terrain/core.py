"""Terrain primitives shared by the flat / rough / procedural-parkour generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


TERRAIN_GEOM_ATTRS = 'group="1" priority="1"'
"""Attributes of every terrain geom. Group 1 is what the scans ray-cast against. Priority 1 above
the robot's 0 makes MuJoCo take a contact's friction from the terrain alone; with equal
priorities it takes the larger of the two, and the feet's 1.0 hid every slippery ground draw,
ice section and metal ramp below it."""


def contact_attrs(solref: tuple[float, ...] | None, solimp: tuple[float, ...] | None) -> str:
    """``solref``/``solimp`` attributes for a soft terrain geom, or "" for MuJoCo's stiff default.

    Terrain priority 1 makes a contact take these from the terrain alone, as with friction.
    """
    attrs = []
    if solref is not None:
        attrs.append('solref="' + " ".join(f"{v:g}" for v in solref) + '"')
    if solimp is not None:
        attrs.append('solimp="' + " ".join(f"{v:g}" for v in solimp) + '"')
    return " ".join(attrs)


@dataclass
class Box:
    """An axis-aligned (optionally rotated) collision box.

    ``size`` uses MuJoCo half-extents, ``full_size`` is available for readability. With
    ``shape="cylinder"`` or ``"capsule"`` it is that instead, ``size`` holding its radius and
    half-length along its local z axis (``top_z`` then assumes that axis is vertical).
    """

    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: str | None = "Concrete"
    rgba: tuple[float, float, float, float] | None = None
    friction: tuple[float, float, float] = (1.0, 0.1, 0.1)
    contype: int = 1
    ceiling: bool = False
    """Something to pass under, like a tunnel bar: downward ground rays continue below it, while
    the upward overhead scan still sees it."""
    wall: bool = False
    """Tall enough to contain the start of a downward ground ray (over ~1 m, like a tilt
    passage's walls): a ray starting inside it reads it as a wall, not the floor beside it."""
    solref: tuple[float, float] | None = None
    solimp: tuple[float, float, float] | None = None
    """Soft ground such as sand or foam; None keeps MuJoCo's stiff contact."""
    shape: str = "box"
    visual: bool = False
    """Decoration only, such as a rope or a painted arrow: no contacts, no mass, outside the
    scans' geom group, and left out of the scene unless it renders."""
    density: float | None = None
    """Mass per cubic metre for a geom of a moving ``TerrainBody``; MuJoCo's 1000 when None."""
    quat: tuple[float, float, float, float] | None = None
    """Orientation as a (w, x, y, z) quaternion, used instead of ``euler`` when set."""

    @classmethod
    def from_full_size(cls, name: str, pos, full_size, **kwargs) -> "Box":
        return cls(name, tuple(pos), tuple(s / 2.0 for s in full_size), **kwargs)

    def to_xml(self) -> str:
        if self.visual:
            attrs = [
                f'name="{self.name}"',
                f'type="{self.shape}"',
                'contype="0" conaffinity="0" group="2" density="0"',
                f'pos="{self.pos[0]:.4f} {self.pos[1]:.4f} {self.pos[2]:.4f}"',
                f'size="{self.size[0]:.4f} {self.size[1]:.4f} {self.size[2]:.4f}"',
            ]
        else:
            attrs = [
                f'name="{self.name}"',
                f'type="{self.shape}"',
                TERRAIN_GEOM_ATTRS,
                'condim="3"',
                f'pos="{self.pos[0]:.4f} {self.pos[1]:.4f} {self.pos[2]:.4f}"',
                f'size="{self.size[0]:.4f} {self.size[1]:.4f} {self.size[2]:.4f}"',
                f'friction="{self.friction[0]} {self.friction[1]} {self.friction[2]}"',
                f'contype="{self.contype}"',
            ]
            if self.density is not None:
                attrs.append(f'density="{self.density:g}"')
        if self.quat is not None:
            attrs.append('quat="' + " ".join(f"{q:.6f}" for q in self.quat) + '"')
        elif any(self.euler):
            attrs.append(f'euler="{self.euler[0]} {self.euler[1]} {self.euler[2]}"')
        if self.material:
            attrs.append(f'material="{self.material}"')
        if self.rgba:
            attrs.append('rgba="' + " ".join(f"{c}" for c in self.rgba) + '"')
        soft = contact_attrs(self.solref, self.solimp)
        if soft:
            attrs.append(soft)
        return "    <geom " + " ".join(attrs) + "/>"

    @property
    def top_z(self) -> float:
        return self.pos[2] + self.size[2]


def segment(name: str, start, end, radius: float, **kwargs) -> Box:
    """A capsule (or another ``shape``) from ``start`` to ``end``: a rope, a rail, a strut or a
    carrot nose."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    direction = axis / length if length > 0 else np.array([0.0, 0.0, 1.0])
    # Shortest rotation taking the geom's local z axis onto the segment's direction.
    w = 1.0 + direction[2]
    if w < 1e-9:
        quat = (0.0, 1.0, 0.0, 0.0)
    else:
        q = np.array([w, -direction[1], direction[0], 0.0])
        quat = tuple(q / np.linalg.norm(q))
    kwargs.setdefault("shape", "capsule")
    if kwargs["shape"] in ("box", "ellipsoid"):
        size = (radius, radius, length / 2.0)
    else:
        size = (radius, length / 2.0, 0.0)
    return Box(name, tuple((start + end) / 2.0), size, quat=quat, **kwargs)


@dataclass
class TerrainBody:
    """A terrain part that moves: on a hinge or slide ``joint`` that physics drives (a barrel,
    a seesaw, an elevator a module's actor pushes along), or a ``mocap`` body that an actor
    places every control step (a crumbling bridge plank). ``geoms`` are positioned relative to
    ``pos``. The scene writes bodies after the robot, so the robot's free joint stays first.
    """

    name: str
    pos: tuple[float, float, float]
    geoms: list[Box]
    joint: str | None = None
    """``"hinge"`` or ``"slide"``, named ``<name>_joint``; None for a mocap or fixed body."""
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    joint_range: tuple[float, float] | None = None
    """Degrees for a hinge, metres for a slide."""
    damping: float = 0.0
    stiffness: float = 0.0
    """Spring pulling the joint back to ``springref``, as under a trampoline bed."""
    springref: float = 0.0
    mocap: bool = False
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Orientation at a joint position of zero, in degrees."""

    @property
    def joint_name(self) -> str:
        return f"{self.name}_joint"

    def to_xml(self, visuals: bool) -> str:
        head = f'name="{self.name}" pos="{self.pos[0]:.4f} {self.pos[1]:.4f} {self.pos[2]:.4f}"'
        if any(self.euler):
            head += f' euler="{self.euler[0]:g} {self.euler[1]:g} {self.euler[2]:g}"'
        if self.mocap:
            head += ' mocap="true"'
        lines = [f"    <body {head}>"]
        if self.joint:
            joint = (
                f'name="{self.joint_name}" type="{self.joint}" '
                f'axis="{self.axis[0]:g} {self.axis[1]:g} {self.axis[2]:g}" damping="{self.damping:g}"'
            )
            # Explicit, so a robot's <default> joint settings (limits, armature) do not apply.
            joint += ' armature="0"'
            if self.stiffness:
                joint += f' stiffness="{self.stiffness:g}" springref="{self.springref:g}"'
            if self.joint_range is not None:
                joint += f' limited="true" range="{self.joint_range[0]:g} {self.joint_range[1]:g}"'
            else:
                joint += ' limited="false"'
            lines.append(f"      <joint {joint}/>")
        lines.extend("  " + geom.to_xml() for geom in self.geoms if visuals or not geom.visual)
        lines.append("    </body>")
        return "\n".join(lines)


@dataclass
class HeightField:
    """A MuJoCo ``hfield`` asset plus the geom that instantiates it."""

    name: str
    data: np.ndarray  # (nrow, ncol) in [0, 1]
    radius_x: float
    radius_y: float
    elevation: float
    base: float = 0.25
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def asset_xml(self) -> str:
        image = (np.clip(self.data, 0.0, 1.0) * 255.0).astype(np.uint8)
        elevations = " ".join(str(value) for value in np.flipud(image).ravel())
        source = f'nrow="{image.shape[0]}" ncol="{image.shape[1]}" elevation="{elevations}"'
        return (
            f'    <hfield name="{self.name}" {source} '
            f'size="{self.radius_x} {self.radius_y} {self.elevation} {self.base}"/>'
        )

    def geom_xml(self) -> str:
        return (
            f'    <geom name="{self.name}_geom" type="hfield" hfield="{self.name}" '
            f'pos="{self.pos[0]} {self.pos[1]} {self.pos[2]}" {TERRAIN_GEOM_ATTRS} condim="3" '
            f'friction="1 0.1 0.1" material="Concrete"/>'
        )


@dataclass
class WaterVolume:
    """A box of still water: buoyancy and drag on whatever is below ``surface_z`` inside it.

    It has no collision geom, so the ground scan and contacts see the floor under it; the
    environment applies its forces, and draws it only when rendering.
    """

    name: str
    center: tuple[float, float]
    half_size: tuple[float, float]
    surface_z: float
    floor_z: float
    """Lowest point of the tank, for drawing the water body."""
    density: float = 1000.0
    drag_coefficient: float = 1.0


@dataclass
class TerrainSpec:
    """A fully described course: geometry, route waypoints and spawn pose."""

    name: str
    boxes: list[Box] = field(default_factory=list)
    heightfields: list[HeightField] = field(default_factory=list)
    water: list[WaterVolume] = field(default_factory=list)
    bodies: list[TerrainBody] = field(default_factory=list)
    actors: list[Any] = field(default_factory=list)
    """Runtime behaviour of moving parts (``terrain.actors.Actor``), stepped by the environment."""
    waypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    waypoint_speed_boost: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """m/s added to each waypoint's sampled speed command (a module's sprint); empty for none."""
    waypoint_modules: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    """Index into ``modules`` of the module each waypoint belongs to (0 the start pad, -1 the
    finish), so the per-skill curriculum can credit a pass or a failure; empty for courses
    without modules."""
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Terrain-surface spawn point; the robot's spawn height is added on top."""
    add_ground_plane: bool = True
    ground_plane_z: float = 0.0
    resolve_waypoint_z: bool = True
    """Ray-cast the terrain surface under each waypoint instead of trusting its z."""
    difficulty: float = 0.0
    modules: list[str] = field(default_factory=list)
    module_kinds: list[str] = field(default_factory=list)
    """Per module: ``"level"`` (at its skill's level), ``"review"`` (easier) or ``"preview"``
    (harder), ``"pad"`` for the start pad; empty for courses without modules."""
    course_length: float = 10.0

    def extend(self, other: "TerrainSpec") -> None:
        self.boxes.extend(other.boxes)
        self.heightfields.extend(other.heightfields)
        self.water.extend(other.water)
        self.bodies.extend(other.bodies)
        self.actors.extend(other.actors)
        self.modules.extend(other.modules)
        self.module_kinds.extend(other.module_kinds)
