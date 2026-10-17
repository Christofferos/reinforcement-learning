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
    base_height_low: float = 0.0
    """Penalty per metre per step that the torso is below its standing height (``nominal_base_height``
    over the ground holding it up, lowered under a ceiling, the ``base_height`` target) beyond
    ``base_height_low_deadband``. Linear and charged every step, so a crouched gait keeps costing:
    procedural_0.0.7 walked flat ground with its torso at 1.15 m and its knees at a median -83
    degrees (the action range allows -87), which the quadratic ``base_height`` charged 0.004 a
    step. Zero under a tunnel or vent roof that asks for the crouch."""
    base_height_low_deadband: float = 0.03
    vertical_progress: float = 0.0
    """Pay per m/s that the torso closes its height gap to standing height over the active
    waypoint (``progress``'s vertical twin, measured against the same target across the step).
    ``base_height`` measures against the ground under the torso, so with a foot on a climb's top
    and the torso still over the floor below it asked for nothing more; procedural_0.0.6-0.0.7
    failed climbs exactly there. The gap counts only beyond ``vertical_progress_deadband``, so
    descents, flat ground and gait bobbing pay nothing. It is the change of one height measure,
    rises and drops alike (a potential), so any stumble, crouch or jump and its recovery nets
    zero, and an ending repays the drops it charged and the robot did not climb back, so a fall
    costs nothing extra here (the gap a higher waypoint opens is never charged, so a failed climb
    is not paid as if it had climbed). Paying
    rises only with a foot down made it one-sided: replaying procedural_0.0.7 it then netted -9
    to -17 per module, charging every landing of a running stride; without the repayment a fall
    cost a second ~-50 on top of ``fall_penalty``, making an attempt dearer than a stall. Not
    clipped, which would break the balance."""
    vertical_progress_deadband: float = 0.15
    """Metres of gap ignored: procedural_0.0.7's gait rode 0.10 m below standing height on flat
    ground; a 0.35 m climb leaves 0.3 m, a 0.2 m stair step about 0.15 m."""
    ceiling_clearance: float = 0.0
    """Penalty per metre per step that the robot's top is above the lowest ceiling the upward
    rays see, less ``base_height_clearance_margin``. Zero without a ceiling. Linear, so it keeps
    pulling the robot down all the way, where the quadratic ``base_height`` target charged only
    0.072 per step for standing 0.38 m too tall under a d=1 tunnel bar."""
    ceiling_contact: float = 0.0
    """Penalty per step that any part of the robot touches a ceiling. ``ceiling_clearance``
    cannot charge for scraping under one: pressed against the roof, the robot's top is level
    with it. procedural_0.0.5 shoved under every d = 0.9 tunnel roof with its torso and arms,
    wedged at the front edge on 3 of 12 and pitched over past the end on 6."""
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
    feet_stumble: float = 0.0
    """Penalty per control step on which a foot or shin presses on a near-vertical terrain face,
    such as the side of a rubble block or a stair riser: the immediate cost of a trip, which
    otherwise only shows up later as a fall. Zero on flat ground and gentle slopes."""
    feet_edge: float = 0.0
    """Penalty per foot per control step standing within ``feet_edge_margin`` of a drop deeper
    than ``feet_edge_drop``: the edge of a block top, a stair nose, a gap or a stepping stone.
    A sphere foot there rolls off. Extreme Parkour's clearance term (Cheng et al. 2023), with
    the edges found by sampling the ground around the foot instead of a precomputed mask."""
    feet_edge_margin: float = 0.05
    feet_edge_drop: float = 0.05
    feet_clearance: float = 0.0
    """Swing-foot lift, paid only with an obstacle ahead: while the ground scan shows terrain at
    least ``feet_clearance_min_obstacle`` above the floor the robot stands on, within
    ``feet_clearance_window`` m ahead (and 0.2 m to either side), each foot in the air earns its
    height above that floor as a share of the obstacle plus ``feet_clearance_margin``, at most
    1. Nothing on flat ground, for a planted foot, or without forward motion, so it never pays
    for high-stepping in general or for lifting a foot in place in front of a hurdle.
    procedural_0.0.2-0.0.4 stalled at hurdles and climbs of 0.24, 0.30 and 0.33 m."""
    feet_clearance_min_obstacle: float = 0.1
    feet_clearance_margin: float = 0.05
    feet_clearance_window: tuple[float, float] = (-0.2, 0.6)
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
    stumbling: bool = False,
    feet_at_edge: int = 0,
    feet_clearance: float = 0.0,
    ceiling_overlap: float = 0.0,
    ceiling_touch: bool = False,
    supported: bool = True,
    height_deficit: float = 0.0,
    vertical_rate: float = 0.0,
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
        "feet_stumble": cfg.feet_stumble * float(stumbling),
        "feet_edge": cfg.feet_edge * feet_at_edge,
        "feet_clearance": cfg.feet_clearance * feet_clearance if moving else 0.0,
        "ceiling_clearance": cfg.ceiling_clearance * ceiling_overlap,
        "ceiling_contact": cfg.ceiling_contact * float(ceiling_touch),
        "waypoint_bonus": cfg.waypoint_bonus * waypoints_reached,
        "goal_bonus": cfg.goal_bonus * float(reached_goal),
        "heading": cfg.heading * heading_alignment,
        "alive": cfg.alive,
        "upright": cfg.upright * upright,
        "base_height": cfg.base_height * -(height_error**2),
        "base_height_low": cfg.base_height_low * -max(0.0, height_deficit - cfg.base_height_low_deadband),
        "vertical_progress": cfg.vertical_progress * vertical_rate,
        "lateral_velocity": cfg.lateral_velocity * abs(lateral_speed),
        "angular_velocity": cfg.angular_velocity * float(np.sum(np.square(ang_vel[:2]))),
        "action_rate": cfg.action_rate * float(np.sum(np.square(action - prev_action))),
        "ctrl_cost": cfg.ctrl_cost * float(np.sum(np.square(action))),
        "energy": cfg.energy * torque_power,
        "joint_limit": cfg.joint_limit * joint_limit_violation,
        "fall_penalty": cfg.fall_penalty * float(fell),
    }
    return {k: v for k, v in out.items() if v != 0.0 or k in ("progress", "alive")}
