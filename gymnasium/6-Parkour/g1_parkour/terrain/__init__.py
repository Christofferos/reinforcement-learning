"""Terrain generators for the parkour task suite."""

from __future__ import annotations

import numpy as np

from .core import Box, HeightField, TerrainSpec
from .flat import make_flat
from .parkour import DEFAULT_SKILLS, MODULES, make_parkour
from .rough import fractal_field, make_rough

GENERATORS = {"flat": make_flat, "rough": make_rough, "parkour": make_parkour}


def build_terrain(kind: str, rng: np.random.Generator, **kwargs) -> TerrainSpec:
    if kind not in GENERATORS:
        raise KeyError(f"Unknown terrain '{kind}'. Available: {sorted(GENERATORS)}")
    return GENERATORS[kind](rng, **kwargs)


__all__ = [
    "Box",
    "HeightField",
    "TerrainSpec",
    "GENERATORS",
    "MODULES",
    "DEFAULT_SKILLS",
    "build_terrain",
    "make_flat",
    "make_rough",
    "make_parkour",
    "fractal_field",
]
