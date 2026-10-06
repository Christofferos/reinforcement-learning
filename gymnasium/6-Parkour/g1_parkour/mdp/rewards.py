"""Reward terms and weights.  Set a weight to 0.0 to ablate the term."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardCfg:
    progress: float = 2.0
    """Potential-based shaping: distance closed toward the active waypoint, per second."""
    progress_per_second: bool = True
    """False rewards metres travelled instead of progress rate on every control step."""
    preferred_speed: float = 1.2
    overspeed: float = 0.0
    velocity_tracking: float = 0.0
    velocity_tracking_sigma: float = 0.35
    waypoint_bonus: float = 10.0
    goal_bonus: float = 50.0
    heading: float = 0.3
    """Alignment of the base forward axis with the direction to the waypoint."""
    alive: float = 0.5
    upright: float = 0.5
    base_height: float = 0.0
    lateral_velocity: float = -0.1
    angular_velocity: float = -0.02
    action_rate: float = -0.05
    ctrl_cost: float = -0.1
    energy: float = 0.0
    joint_limit: float = -0.1
    fall_penalty: float = -10.0
    clip_progress: float = 3.0
    """Upper bound on the raw progress rate (m/s) before weighting."""


def terms(
    cfg: RewardCfg,
    *,
    dt: float,
    progress: float,
    heading_alignment: float,
    upright: float,
    height_error: float,
    lateral_speed: float,
    ang_vel: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray,
    torque_power: float,
    joint_limit_violation: float,
    waypoints_reached: int,
    reached_goal: bool,
    fell: bool,
    forward_speed: float = 0.0,
    target_speed: float | None = None,
    velocity_error_squared: float = 0.0,
) -> dict[str, float]:
    progress_rate = float(np.clip(progress / dt, -cfg.clip_progress, cfg.clip_progress))
    speed_limit = cfg.preferred_speed if target_speed is None else target_speed
    out = {
        "progress": cfg.progress * progress_rate * (1.0 if cfg.progress_per_second else dt),
        "overspeed": cfg.overspeed * max(0.0, abs(forward_speed) - speed_limit) ** 2,
        "velocity_tracking": cfg.velocity_tracking * float(
            np.exp(-velocity_error_squared / cfg.velocity_tracking_sigma**2)
        ),
        "waypoint_bonus": cfg.waypoint_bonus * waypoints_reached,
        "goal_bonus": cfg.goal_bonus * float(reached_goal),
        "heading": cfg.heading * heading_alignment,
        "alive": cfg.alive,
        "upright": cfg.upright * upright,
        "base_height": cfg.base_height * -(height_error**2),
        "lateral_velocity": cfg.lateral_velocity * abs(lateral_speed),
        "angular_velocity": cfg.angular_velocity * float(np.sum(np.square(ang_vel[:2]))),
        "action_rate": cfg.action_rate * float(np.sum(np.square(action - prev_action))),
        "ctrl_cost": cfg.ctrl_cost * float(np.sum(np.square(action))),
        "energy": cfg.energy * torque_power,
        "joint_limit": cfg.joint_limit * joint_limit_violation,
        "fall_penalty": cfg.fall_penalty * float(fell),
    }
    return {k: v for k, v in out.items() if v != 0.0 or k in ("progress", "alive")}
