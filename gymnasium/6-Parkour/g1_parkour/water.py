"""Water volumes: simple buoyancy and drag for training, waves and splashes for watching.

``WaterPhysics`` is all that runs during training: once per control step, a few vectorised
numpy operations over the robot's collision geoms, skipped entirely while the robot is away
from every water volume. ``WaterEffects`` is render-only; it keeps a small particle state
and returns decor markers for the viewer, and is never created when nothing renders.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .terrain.core import WaterVolume

GRAVITY = 9.81
NEAR_WATER_MARGIN = 1.5
"""How far outside a volume's footprint the robot's root may be before forces are skipped."""


@dataclass
class _Geoms:
    """Static shape data of the geoms water acts on."""

    ids: np.ndarray
    body_ids: np.ndarray
    root_ids: np.ndarray
    capsule: np.ndarray
    radius: np.ndarray
    half_length: np.ndarray
    volume: np.ndarray
    area: np.ndarray
    """Broadside cross-section, used for drag whatever the flow direction."""

    @classmethod
    def of_robot(cls, model, root_body_id: int) -> "_Geoms":
        tree = model.body_rootid[root_body_id]
        ids = np.array([
            g for g in range(model.ngeom)
            if model.body_rootid[model.geom_bodyid[g]] == tree
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ], dtype=int)
        types = model.geom_type[ids]
        size = model.geom_size[ids]
        capsule = np.isin(types, (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER))
        sphere = types == mujoco.mjtGeom.mjGEOM_SPHERE
        # Spheres and capsules exactly; anything else (boxes, meshes) as its bounding sphere.
        radius = np.where(capsule | sphere, size[:, 0], model.geom_rbound[ids])
        half_length = np.where(capsule, size[:, 1], 0.0)
        volume = 4.0 / 3.0 * np.pi * radius**3 + np.pi * radius**2 * 2.0 * half_length
        area = np.pi * radius**2 + 2.0 * radius * 2.0 * half_length
        body_ids = model.geom_bodyid[ids]
        return cls(ids, body_ids, model.body_rootid[body_ids], capsule, radius, half_length, volume, area)

    def state(self, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World centre, vertical half-extent and linear velocity of each geom."""
        pos = data.geom_xpos[self.ids]
        # World z of each geom's local z axis: a capsule reaches half_length * |that| above its centre.
        axis_z = np.abs(data.geom_xmat[self.ids, 8])
        extent = self.radius + self.half_length * axis_z
        # cvel is [angular, linear] about the subtree centre of mass of the body's tree root.
        cvel = data.cvel[self.body_ids]
        velocity = cvel[:, 3:] + _cross(cvel[:, :3], pos - data.subtree_com[self.root_ids])
        return pos, extent, velocity


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross product; np.cross costs several times more on arrays this small."""
    return np.column_stack([
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
    ])


def submerged_fraction(volume: WaterVolume, pos: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Fraction of each geom's height below the surface, zero outside the volume's footprint."""
    inside = (
        (np.abs(pos[:, 0] - volume.center[0]) <= volume.half_size[0])
        & (np.abs(pos[:, 1] - volume.center[1]) <= volume.half_size[1])
    )
    depth = volume.surface_z - (pos[:, 2] - extent)
    return np.where(inside, np.clip(depth / (2.0 * extent), 0.0, 1.0), 0.0)


class WaterPhysics:
    """Buoyancy and quadratic drag on the robot inside water volumes, through ``xfrc_applied``.

    Buoyancy is the weight of the displaced water, from each geom's volume times the fraction
    of its height below the surface, pushing up at the geom's centre. Drag is
    ``0.5 * density * drag_coefficient * area * |v| * v`` against each geom's velocity, scaled
    by the same fraction, so a foot swung through the water is slowed and one lifted clear of
    it is not. The humanoid's geoms have water's density, so fully submerged limbs float.
    """

    def __init__(self, model, volumes: list[WaterVolume], root_body_id: int):
        self.volumes = list(volumes)
        self.geoms = _Geoms.of_robot(model, root_body_id)
        self.body_ids, slot = np.unique(self.geoms.body_ids, return_inverse=True)
        # Sums each geom's wrench into its body's row: one matrix product instead of np.add.at.
        self.geom_to_body = np.zeros((len(self.body_ids), len(self.geoms.ids)))
        self.geom_to_body[slot, np.arange(len(slot))] = 1.0
        self.root_body_id = root_body_id
        self.applying = False

    def near(self, data) -> list[WaterVolume]:
        x, y = data.xpos[self.root_body_id, :2]
        return [
            volume for volume in self.volumes
            if abs(x - volume.center[0]) <= volume.half_size[0] + NEAR_WATER_MARGIN
            and abs(y - volume.center[1]) <= volume.half_size[1] + NEAR_WATER_MARGIN
        ]

    def apply(self, data) -> None:
        """Set the water forces on the robot's bodies (zero when out of the water) until the next call."""
        volumes = self.near(data)
        if not volumes:
            if self.applying:
                data.xfrc_applied[self.body_ids] = 0.0
                self.applying = False
            return
        pos, extent, velocity = self.geoms.state(data)
        force = np.zeros((len(pos), 3))
        for volume in volumes:
            fraction = submerged_fraction(volume, pos, extent)
            if not fraction.any():
                continue
            speed = np.sqrt(np.einsum("ij,ij->i", velocity, velocity))
            force -= (
                0.5 * volume.density * volume.drag_coefficient * self.geoms.area * fraction * speed
            )[:, None] * velocity
            force[:, 2] += volume.density * GRAVITY * self.geoms.volume * fraction
        # xfrc_applied acts at each body's centre of mass: move the force there with its moment.
        wrench = np.hstack([force, _cross(pos - data.xipos[self.geoms.body_ids], force)])
        data.xfrc_applied[self.body_ids] = self.geom_to_body @ wrench
        self.applying = True


class WaterEffects:
    """Render-only water animation: rolling wave crests, ripples around wading legs and splashes.

    ``markers`` advances the effects to the simulation time and returns viewer decor markers.
    It reads the robot's geoms but never writes to the model or the simulation state.
    """

    WAVE_RATE = 0.12
    """New wave crests per metre of tank length per second; each lives 2.5-5 s, so a 5 m
    tank shows about two at a time."""
    WAVE_LIFETIME = (2.5, 5.0)
    WAVE_SPEED = (0.15, 0.45)
    WAVE_AMPLITUDE = (0.006, 0.02)
    RIPPLE_INTERVAL = 0.12
    RIPPLE_LIFETIME = 0.9
    RIPPLE_MIN_SPEED = 0.25
    SPLASH_MIN_SPEED = 0.6
    MAX_DROPLETS = 300
    DROPLET_LIFETIME = 2.0
    MARKER_GROUP = 2

    def __init__(self, model, volumes: list[WaterVolume], root_body_id: int, seed: int = 0):
        self.volumes = list(volumes)
        self.geoms = _Geoms.of_robot(model, root_body_id)
        self.rng = np.random.default_rng(seed)
        self.time: float | None = None
        self.was_under = np.zeros(len(self.geoms.ids), dtype=bool)
        self.last_ripple = np.full(len(self.geoms.ids), -np.inf)
        self.ripples: list[tuple[np.ndarray, float]] = []
        self.droplets = np.zeros((0, 8))
        """Rows of position (3), velocity (3), radius and age."""
        self.crests = np.zeros((0, 10))
        """Rows of volume index, x, y, speed, half-length, heading, amplitude, phase, age, lifetime."""

    def _surface_volume(self, pos: np.ndarray) -> list[WaterVolume | None]:
        found = []
        for p in pos:
            match = None
            for volume in self.volumes:
                if (abs(p[0] - volume.center[0]) <= volume.half_size[0]
                        and abs(p[1] - volume.center[1]) <= volume.half_size[1]):
                    match = volume
                    break
            found.append(match)
        return found

    def _advance(self, data, dt: float) -> None:
        pos, extent, velocity = self.geoms.state(data)
        volumes = self._surface_volume(pos)
        for i, volume in enumerate(volumes):
            if volume is None:
                self.was_under[i] = False
                continue
            surface = volume.surface_z
            under = pos[i, 2] - extent[i] < surface
            crossing = under and pos[i, 2] + extent[i] > surface
            if under and not self.was_under[i] and -velocity[i, 2] > self.SPLASH_MIN_SPEED:
                self._splash(np.array([pos[i, 0], pos[i, 1], surface]), -velocity[i, 2], self.geoms.radius[i])
            self.was_under[i] = under
            if crossing and np.linalg.norm(velocity[i, :2]) > self.RIPPLE_MIN_SPEED \
                    and data.time - self.last_ripple[i] >= self.RIPPLE_INTERVAL:
                self.ripples.append((np.array([pos[i, 0], pos[i, 1], surface]), data.time))
                self.last_ripple[i] = data.time
        self.ripples = [(p, t) for p, t in self.ripples if data.time - t < self.RIPPLE_LIFETIME]

        if len(self.droplets) and dt > 0:
            self.droplets[:, 5] -= GRAVITY * dt
            self.droplets[:, :3] += self.droplets[:, 3:6] * dt
            self.droplets[:, 7] += dt
            surfaces = np.array([
                volume.surface_z if volume is not None else -np.inf
                for volume in self._surface_volume(self.droplets[:, :3])
            ])
            landed = (self.droplets[:, 2] < surfaces) & (self.droplets[:, 5] < 0)
            for drop in self.droplets[landed & (self.rng.random(len(self.droplets)) < 0.15)]:
                self.ripples.append((np.array([drop[0], drop[1], drop[2]]), data.time))
            self.droplets = self.droplets[~landed & (self.droplets[:, 7] < self.DROPLET_LIFETIME)]

    def _splash(self, at: np.ndarray, speed: float, radius: float) -> None:
        count = int(np.clip(4 + 6 * speed * radius / 0.05, 4, 24))
        angle = self.rng.uniform(0.0, 2.0 * np.pi, count)
        outward = self.rng.uniform(0.2, 0.6, count) * min(speed, 2.5)
        upward = self.rng.uniform(0.4, 1.0, count) * min(speed, 2.5)
        offset = radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        new = np.column_stack([
            at[0] + offset[:, 0], at[1] + offset[:, 1], np.full(count, at[2] + 0.01),
            outward * np.cos(angle), outward * np.sin(angle), upward,
            self.rng.uniform(0.008, 0.018, count), np.zeros(count),
        ])
        self.droplets = np.vstack([self.droplets, new])[-self.MAX_DROPLETS:]
        self.ripples.append((at.copy(), self.time if self.time is not None else 0.0))

    def markers(self, data) -> list[dict]:
        if self.time is None or data.time < self.time:  # first frame, or a reset rewound time
            self.time = data.time
            self.was_under[:] = False
            self.last_ripple[:] = -np.inf
            self.ripples, self.droplets = [], np.zeros((0, 8))
            self.crests = self._spawn_crests(steady_state=True)
        dt = data.time - self.time
        self.time = data.time
        self._advance(data, dt)
        self._advance_crests(dt)

        markers = []
        for index, x, y, _, half_length, heading, amplitude, phase, age, lifetime in self.crests:
            volume = self.volumes[int(index)]
            fade = np.sin(np.pi * age / lifetime)  # rises out of the surface and sinks back
            bob = amplitude * fade * np.sin(phase + 4.0 * data.time)
            c, s = np.cos(heading), np.sin(heading)
            markers.append(self._marker(
                mujoco.mjtGeom.mjGEOM_ELLIPSOID, (x, y, volume.surface_z + bob),
                (0.05, half_length, 0.006), (0.8, 0.9, 1.0, 0.2 * fade),
                mat=(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0),
            ))
        for at, born in self.ripples:
            age = (data.time - born) / self.RIPPLE_LIFETIME
            markers.append(self._marker(
                mujoco.mjtGeom.mjGEOM_ELLIPSOID, at + np.array([0.0, 0.0, 0.004]),
                (0.06 + 0.4 * age, 0.06 + 0.4 * age, 0.003), (0.9, 0.96, 1.0, 0.45 * (1.0 - age)),
            ))
        for drop in self.droplets:
            markers.append(self._marker(
                mujoco.mjtGeom.mjGEOM_SPHERE, drop[:3], (drop[6],) * 3, (0.85, 0.93, 1.0, 0.8),
            ))
        return markers

    def _spawn_crests(self, dt: float = 0.0, steady_state: bool = False) -> np.ndarray:
        """New wave crests at random places, headings, lengths and speeds.

        ``steady_state`` fills the tanks with as many crests, at random ages, as they hold on
        average, so the first frame does not start from a flat surface.
        """
        rows = []
        mean_lifetime = float(np.mean(self.WAVE_LIFETIME))
        for index, volume in enumerate(self.volumes):
            span = 2.0 * volume.half_size[0]
            expected = self.WAVE_RATE * span * (mean_lifetime if steady_state else dt)
            for _ in range(self.rng.poisson(expected)):
                lifetime = self.rng.uniform(*self.WAVE_LIFETIME)
                half_length = self.rng.uniform(0.3, 0.9) * volume.half_size[1]
                rows.append((
                    index,
                    volume.center[0] + self.rng.uniform(-1.0, 1.0) * (volume.half_size[0] - 0.1),
                    volume.center[1] + self.rng.uniform(-1.0, 1.0) * (volume.half_size[1] - half_length),
                    self.rng.uniform(*self.WAVE_SPEED) * self.rng.choice((-1.0, 1.0), p=(0.3, 0.7)),
                    half_length,
                    np.deg2rad(self.rng.uniform(-25.0, 25.0)),
                    self.rng.uniform(*self.WAVE_AMPLITUDE),
                    self.rng.uniform(0.0, 2.0 * np.pi),
                    self.rng.uniform(0.0, lifetime) if steady_state else 0.0,
                    lifetime,
                ))
        return np.array(rows, dtype=float).reshape(-1, 10)

    def _advance_crests(self, dt: float) -> None:
        if dt <= 0:
            return
        crests = self.crests
        crests[:, 1] += crests[:, 3] * dt
        crests[:, 8] += dt
        inside = np.array([
            abs(x - self.volumes[int(i)].center[0]) <= self.volumes[int(i)].half_size[0] - 0.05
            for i, x in crests[:, :2]
        ], dtype=bool)
        self.crests = np.vstack([crests[inside & (crests[:, 8] < crests[:, 9])], self._spawn_crests(dt)])

    def _marker(self, kind, pos, size, rgba, mat=None) -> dict:
        marker = dict(
            type=kind, pos=np.asarray(pos, dtype=np.float64), size=np.asarray(size, dtype=np.float64),
            rgba=rgba, emission=0.3, label="", group=self.MARKER_GROUP,
        )
        if mat is not None:
            marker["mat"] = np.asarray(mat, dtype=np.float64)
        return marker
