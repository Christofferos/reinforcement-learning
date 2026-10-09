"""Reward terms and weights.  Set a weight to 0.0 to ablate the term."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardCfg:
    progress: float = 20.0
    """Potential-based shaping: distance closed toward the active waypoint, per second."""
    progress_per_second: bool = True
    """False rewards metres travelled instead of progress rate on every control step."""
    velocity_tracking: float = 0.0
    """Maximum reward for matching the waypoint's commanded planar velocity."""
    velocity_tracking_sigma: float = 0.35
    """Velocity-error tolerance in m/s for the exponential tracking reward."""
    waypoint_bonus: float = 10.0
    goal_bonus: float = 50.0
    heading: float = 0.3
    """Alignment of the base forward axis with the direction to the waypoint."""
    alive: float = 0.5
    upright: float = 0.5
    base_height: float = 0.0
    base_height_clearance_margin: float = 0.05
    """Safety gap above the robot when lowering its posture target for overhead clearance."""
    lateral_velocity: float = -0.1
    angular_velocity: float = -0.02
    action_rate: float = -0.05
    ctrl_cost: float = -0.1
    energy: float = 0.0
    joint_limit: float = -0.1
    fall_penalty: float = -10.0
    clip_progress: float = 3.0
    """Upper bound on the raw progress rate (m/s) before weighting."""
    feet_air_time: float = 0.0
    """Biped single-stance reward: the shorter of the two feet's current swing/stance
    durations while exactly one foot is on the ground, capped at ``feet_air_time_threshold``.
    Only paid when a non-zero speed is commanded."""
    feet_air_time_threshold: float = 0.4
    feet_step_ahead: float = 0.0
    """Touchdown reward for alternating steps: paid when a foot lands and the other foot was
    the last to land, scaled by how far ahead it lands relative to the stance foot along the
    direction to the active waypoint, saturating at ``feet_step_ahead_margin`` and negative
    when it lands behind. A skip (same foot always leading) nets zero per stride whichever way
    the torso faces; a repeated hop on one foot earns nothing. Only paid when a non-zero speed
    is commanded."""
    feet_step_ahead_margin: float = 0.15
    feet_step_symmetry: float = 0.0
    """Penalty per metre of difference between consecutive left and right step lengths (along
    the direction to the waypoint), paid at each alternating touchdown. A gallop or limp, where
    one foot lands far ahead and the other only catches up, pays on every step; an even walk or
    run pays almost nothing. Unlike ``feet_step_ahead`` it never saturates, so it keeps a
    gradient toward even steps. Only paid when a non-zero speed is commanded."""
    feet_slide: float = 0.0
    """Penalty on the planar speed of feet that are in contact with the ground."""
    gate_on_support: bool = True
    """Pay ``progress`` and ``velocity_tracking`` only while the base is in a healthy pose
    (height and tilt inside the termination band), so a slow fall earns nothing."""


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
    target_speed: float | None = None,
    velocity_error_squared: float = 0.0,
    feet_air_time: float = 0.0,
    feet_step_ahead: float = 0.0,
    feet_step_asymmetry: float = 0.0,
    feet_slide_speed: float = 0.0,
    supported: bool = True,
) -> dict[str, float]:
    progress_limit = cfg.clip_progress if target_speed is None else min(cfg.clip_progress, target_speed)
    progress_rate = float(np.clip(progress / dt, -progress_limit, progress_limit))
    gate = 1.0 if (supported or not cfg.gate_on_support) else 0.0
    moving = target_speed is None or target_speed > 0.1
    out = {
        "progress": gate * cfg.progress * progress_rate * (1.0 if cfg.progress_per_second else dt),
        "velocity_tracking": gate * cfg.velocity_tracking * float(
            np.exp(-velocity_error_squared / cfg.velocity_tracking_sigma**2)
        ) if target_speed is not None else 0.0,
        "feet_air_time": cfg.feet_air_time * min(feet_air_time, cfg.feet_air_time_threshold)
        if moving else 0.0,
        "feet_step_ahead": cfg.feet_step_ahead * feet_step_ahead if moving else 0.0,
        "feet_step_symmetry": cfg.feet_step_symmetry * feet_step_asymmetry if moving else 0.0,
        "feet_slide": cfg.feet_slide * feet_slide_speed,
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
