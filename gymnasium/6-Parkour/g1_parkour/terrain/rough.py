"""Rough terrain: fractal-noise heightfields with optional discrete obstacles."""

from __future__ import annotations

import numpy as np

from .core import Box, HeightField, TerrainSpec


def _value_noise(rng: np.random.Generator, shape: tuple[int, int], cells: tuple[int, int]) -> np.ndarray:
    """Bilinearly upsampled value noise in [0, 1] on a ``cells`` grid."""
    cr, cc = max(2, cells[0]), max(2, cells[1])
    coarse = rng.random((cr + 1, cc + 1))
    rows = np.linspace(0, cr, shape[0])
    cols = np.linspace(0, cc, shape[1])
    r0 = np.clip(np.floor(rows).astype(int), 0, cr - 1)
    c0 = np.clip(np.floor(cols).astype(int), 0, cc - 1)
    tr = (rows - r0)[:, None]
    tc = (cols - c0)[None, :]
    # Smoothstep for C1 continuity between cells.
    tr = tr * tr * (3 - 2 * tr)
    tc = tc * tc * (3 - 2 * tc)
    top = coarse[r0][:, c0] * (1 - tc) + coarse[r0][:, c0 + 1] * tc
    bottom = coarse[r0 + 1][:, c0] * (1 - tc) + coarse[r0 + 1][:, c0 + 1] * tc
    return top * (1 - tr) + bottom * tr


def fractal_field(
    rng: np.random.Generator,
    shape: tuple[int, int],
    octaves: int = 4,
    base_cells: tuple[int, int] = (4, 4),
    persistence: float = 0.65,
) -> np.ndarray:
    field = np.zeros(shape)
    amplitude, total = 1.0, 0.0
    for octave in range(octaves):
        cells = (base_cells[0] * 2**octave, base_cells[1] * 2**octave)
        field += amplitude * _value_noise(rng, shape, cells)
        total += amplitude
        amplitude *= persistence
    field /= total
    field -= field.min()
    return field / max(field.max(), 1e-6)


def make_rough(
    rng: np.random.Generator,
    difficulty: float = 0.5,
    course_length: float = 24.0,
    course_width: float = 12.0,
    resolution: float = 0.08,
    max_elevation: float = 0.35,
    base_wavelength: float = 3.0,
    octaves: int = 3,
    discrete_obstacles: int = 0,
    flat_start_radius: float = 1.5,
    course_length_growth: float = 0.0,
    base_wavelength_growth: float = -1.5,
    rubble_per_metre: float = 0.5,
    rubble_corridor: float = 1.0,
) -> TerrainSpec:
    """Noisy heightfield course with rubble blocks along the route.

    ``difficulty`` in [0, 1] raises the hills and deepens the pits (the noise is centred on
    the spawn height, so it goes both above and below it), shortens the noise features so
    bumps come more often, and scatters more and taller rubble blocks within
    ``rubble_corridor`` metres of the route, up to ``rubble_per_metre`` at difficulty 1.
    ``base_wavelength`` is the coarsest noise feature size in metres at difficulty 0 and
    changes by ``base_wavelength_growth`` at difficulty 1; each extra octave halves it.
    The course grows by ``course_length_growth`` metres at difficulty 1.
    ``discrete_obstacles`` fixes the block count instead.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    course_length += course_length_growth * difficulty
    wavelength = base_wavelength + base_wavelength_growth * difficulty
    radius_x, radius_y = course_length / 2.0 + 2.0, course_width / 2.0
    ncol = int(2 * radius_x / resolution)
    nrow = int(2 * radius_y / resolution)
    base_cells = (
        max(2, int(round(2 * radius_y / wavelength))),
        max(2, int(round(2 * radius_x / wavelength))),
    )
    data = fractal_field(rng, (nrow, ncol), octaves=octaves, base_cells=base_cells)

    elevation = max_elevation * (0.25 + 0.75 * difficulty)
    center_x = course_length / 2.0

    # Blend a disc around the spawn to the field's mid level, which sits at z = 0, so the
    # robot never starts inside a bump and the rest of the course has hills and pits.
    xs = np.linspace(center_x - radius_x, center_x + radius_x, ncol)
    ys = np.linspace(-radius_y, radius_y, nrow)
    gx, gy = np.meshgrid(xs, ys)
    dist = np.hypot(gx - 0.0, gy - 0.0)
    blend = np.clip((dist - flat_start_radius) / max(flat_start_radius, 1e-3), 0.0, 1.0)
    data = 0.5 + (data - 0.5) * blend
    ground = (data - 0.5) * elevation

    hfield = HeightField(
        name="rough_terrain",
        data=data,
        radius_x=radius_x,
        radius_y=radius_y,
        elevation=elevation,
        pos=(center_x, 0.0, -0.5 * elevation),
    )

    num_wp = max(2, int(course_length / 2.5))
    xs_wp = np.linspace(2.5, course_length, num_wp)
    ys_wp = rng.uniform(-1.0, 1.0, size=num_wp) * difficulty
    waypoints = np.stack([xs_wp, ys_wp, np.zeros(num_wp)], axis=1)
    route_x, route_y = np.concatenate([[0.0], xs_wp]), np.concatenate([[0.0], ys_wp])

    boxes: list[Box] = []
    count = discrete_obstacles or int(round(rubble_per_metre * difficulty * (course_length - 3.0)))
    for i in range(count):
        x = rng.uniform(3.0, course_length)
        y = np.interp(x, route_x, route_y) + rng.uniform(-rubble_corridor, rubble_corridor)
        half = rng.uniform(0.15, 0.4, size=2)
        height = rng.uniform(0.05, 0.05 + 0.25 * difficulty)
        # Rest the block on the ground under it and stand ``height`` above its highest point,
        # so a hill never buries it.
        under = ground[np.abs(ys - y) <= half[1]][:, np.abs(xs - x) <= half[0]]
        bottom, top = float(under.min()) - 0.05, float(under.max()) + height
        boxes.append(
            Box.from_full_size(
                f"rough_block_{i}",
                (x, y, (bottom + top) / 2.0),
                (2.0 * half[0], 2.0 * half[1], top - bottom),
                rgba=(0.45, 0.45, 0.5, 1.0),
                material=None,
            )
        )

    return TerrainSpec(
        name="rough",
        boxes=boxes,
        heightfields=[hfield],
        waypoints=waypoints,
        spawn_pos=(0.0, 0.0, 0.0),
        add_ground_plane=False,
        difficulty=difficulty,
        modules=["rough_noise"] + [f"block_{i}" for i in range(count)],
        course_length=course_length,
    )
