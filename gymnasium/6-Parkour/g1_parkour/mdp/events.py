"""Events: domain randomisation and disturbances applied on reset / during rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class EventCfg:
    reset_noise_scale: float = 0.02
    reset_yaw_range: float = 0.0
    """Uniform yaw perturbation at spawn, radians."""
    reset_along_course: bool = False
    """Spawn at a random waypoint instead of the start (terrain curriculum trick)."""

    push_robot: bool = False
    push_interval_steps: int = 200
    """Mean control steps between pushes; each gap is drawn uniformly from 0.5-1.5x this."""
    push_velocity: float = 0.6
    """Maximum planar velocity kick in m/s; magnitude is drawn from 0.5-1x, direction uniform."""
    push_velocity_difficulty_shift: float = 0.0
    """Added to ``push_velocity`` at maximum terrain difficulty."""

    randomize_friction: bool = False
    friction_range: tuple[float, float] = (0.5, 1.5)
    randomize_mass: bool = False
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    randomize_gear: bool = False
    gear_scale_range: tuple[float, float] = (0.9, 1.1)

    action_delay_steps: int = 0


# Friction-to-appearance mapping for rubble terrain sections: low friction renders icy
# (pale blue-white and glossy), high friction muddy (brown and matte), so the current
# ground friction is visible while playing. Only rubble-named geoms are re-tinted.
ICE_MATERIAL = "MatIce"
MUD_MATERIAL = "MatMud"
ICE_RGBA = np.array([0.82, 0.90, 0.98, 1.0])
MUD_RGBA = np.array([0.30, 0.21, 0.12, 1.0])


def rubble_section_key(name: str) -> str | None:
    """Section grouping key for a rubble geom name, or None for non-rubble geoms.

    Parkour rubble modules (``rubble_run_<i>`` floors and ``rubble_<i>_<j>`` blocks) share
    one key per module instance; the rough heightfield is a single section and each
    ``rough_block_<i>`` its own.
    """
    if name == "rough_terrain_geom":
        return "rough_terrain"
    if name.startswith("rough_block_"):
        return name
    if name.startswith("rubble_run_"):
        return f"rubble_{name.removeprefix('rubble_run_')}"
    if name.startswith("rubble_") and len(name.split("_")) == 3:
        return "_".join(name.split("_")[:2])
    return None


def randomize_rubble_sections(
    model, rng: np.random.Generator, cfg: EventCfg, nominal: dict
) -> None:
    """Give each rubble section its own friction draw and matching ice/mud appearance."""
    sections: dict[str, list[int]] = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        key = rubble_section_key(name or "")
        if key is not None:
            sections.setdefault(key, []).append(geom_id)
    if not sections:
        return
    ice_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, ICE_MATERIAL)
    mud_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, MUD_MATERIAL)
    low, high = cfg.friction_range
    for geom_ids in sections.values():
        scale = rng.uniform(low, high)
        ids = np.asarray(geom_ids)
        model.geom_friction[ids, 0] = nominal["geom_friction"][ids, 0] * scale
        t = float(np.clip((scale - low) / max(high - low, 1e-9), 0.0, 1.0))
        model.geom_rgba[ids] = t * MUD_RGBA + (1.0 - t) * ICE_RGBA
        material_id = mud_id if t >= 0.5 else ice_id
        if material_id >= 0:
            model.geom_matid[ids] = material_id


def randomize_model(model, rng: np.random.Generator, cfg: EventCfg, nominal: dict) -> None:
    """Re-apply domain randomisation to a compiled model from its nominal values."""
    if cfg.randomize_friction:
        scale = rng.uniform(*cfg.friction_range)
        model.geom_friction[:, 0] = nominal["geom_friction"][:, 0] * scale
        randomize_rubble_sections(model, rng, cfg, nominal)
    if cfg.randomize_mass:
        scale = rng.uniform(*cfg.mass_scale_range, size=model.nbody)
        model.body_mass[:] = nominal["body_mass"] * scale
        model.body_inertia[:] = nominal["body_inertia"] * scale[:, None]
    if cfg.randomize_gear:
        scale = rng.uniform(*cfg.gear_scale_range, size=model.nu)
        model.actuator_gear[:, 0] = nominal["actuator_gear"][:, 0] * scale


def schedule_push(
    rng: np.random.Generator, cfg: EventCfg, current_step: int, difficulty: float = 0.0,
) -> tuple[int, np.ndarray]:
    """Draw the control step and planar velocity kick of the next push."""
    interval = max(1, cfg.push_interval_steps)
    delay = int(rng.integers(max(1, interval // 2), interval + interval // 2 + 1))
    angle = rng.uniform(0.0, 2.0 * np.pi)
    maximum = cfg.push_velocity + cfg.push_velocity_difficulty_shift * float(np.clip(difficulty, 0.0, 1.0))
    magnitude = rng.uniform(0.5, 1.0) * maximum
    return current_step + delay, magnitude * np.array([np.cos(angle), np.sin(angle)])


def push(data, kick: np.ndarray) -> None:
    """Apply an instantaneous planar velocity kick to the floating base."""
    data.qvel[0:2] += kick
