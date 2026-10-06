"""Terrain primitives shared by the flat / rough / procedural-parkour generators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Box:
    """An axis-aligned (optionally rotated) collision box.

    ``size`` uses MuJoCo half-extents, ``full_size`` is available for readability.
    """

    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: str | None = "Concrete"
    rgba: tuple[float, float, float, float] | None = None
    friction: tuple[float, float, float] = (1.0, 0.1, 0.1)
    contype: int = 1

    @classmethod
    def from_full_size(cls, name: str, pos, full_size, **kwargs) -> "Box":
        return cls(name, tuple(pos), tuple(s / 2.0 for s in full_size), **kwargs)

    def to_xml(self) -> str:
        attrs = [
            f'name="{self.name}"',
            'type="box"',
            'group="1"',
            'condim="3"',
            f'pos="{self.pos[0]:.4f} {self.pos[1]:.4f} {self.pos[2]:.4f}"',
            f'size="{self.size[0]:.4f} {self.size[1]:.4f} {self.size[2]:.4f}"',
            f'friction="{self.friction[0]} {self.friction[1]} {self.friction[2]}"',
            f'contype="{self.contype}"',
        ]
        if any(self.euler):
            attrs.append(f'euler="{self.euler[0]} {self.euler[1]} {self.euler[2]}"')
        if self.material:
            attrs.append(f'material="{self.material}"')
        if self.rgba:
            attrs.append('rgba="' + " ".join(f"{c}" for c in self.rgba) + '"')
        return "    <geom " + " ".join(attrs) + "/>"

    @property
    def top_z(self) -> float:
        return self.pos[2] + self.size[2]


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
            f'pos="{self.pos[0]} {self.pos[1]} {self.pos[2]}" group="1" condim="3" '
            f'friction="1 0.1 0.1" material="Concrete"/>'
        )


@dataclass
class TerrainSpec:
    """A fully described course: geometry, route waypoints and spawn pose."""

    name: str
    boxes: list[Box] = field(default_factory=list)
    heightfields: list[HeightField] = field(default_factory=list)
    waypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Terrain-surface spawn point; the robot's spawn height is added on top."""
    add_ground_plane: bool = True
    ground_plane_z: float = 0.0
    resolve_waypoint_z: bool = True
    """Ray-cast the terrain surface under each waypoint instead of trusting its z."""
    difficulty: float = 0.0
    modules: list[str] = field(default_factory=list)
    course_length: float = 10.0

    def extend(self, other: "TerrainSpec") -> None:
        self.boxes.extend(other.boxes)
        self.heightfields.extend(other.heightfields)
        self.modules.extend(other.modules)
