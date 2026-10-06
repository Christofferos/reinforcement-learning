"""Events: domain randomisation and disturbances applied on reset / during rollouts."""

from __future__ import annotations

from dataclasses import dataclass

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
    push_velocity: float = 0.6

    randomize_friction: bool = False
    friction_range: tuple[float, float] = (0.7, 1.3)
    randomize_mass: bool = False
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    randomize_gear: bool = False
    gear_scale_range: tuple[float, float] = (0.9, 1.1)

    action_delay_steps: int = 0


def randomize_model(model, rng: np.random.Generator, cfg: EventCfg, nominal: dict) -> None:
    """Re-apply domain randomisation to a compiled model from its nominal values."""
    if cfg.randomize_friction:
        scale = rng.uniform(*cfg.friction_range)
        model.geom_friction[:, 0] = nominal["geom_friction"][:, 0] * scale
    if cfg.randomize_mass:
        scale = rng.uniform(*cfg.mass_scale_range, size=model.nbody)
        model.body_mass[:] = nominal["body_mass"] * scale
        model.body_inertia[:] = nominal["body_inertia"] * scale[:, None]
    if cfg.randomize_gear:
        scale = rng.uniform(*cfg.gear_scale_range, size=model.nu)
        model.actuator_gear[:, 0] = nominal["actuator_gear"][:, 0] * scale


def push(data, rng: np.random.Generator, cfg: EventCfg) -> None:
    """Apply an instantaneous planar velocity kick to the floating base."""
    data.qvel[0:2] += rng.uniform(-cfg.push_velocity, cfg.push_velocity, size=2)
