"""Termination terms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TerminationCfg:
    bad_height: bool = True
    """Root height relative to the terrain below drops under the healthy lower bound.

    Only the lower bound is checked: a large height above terrain means the robot is
    airborne over a gap, which is legitimate, and actually falling into a pit is
    already caught by ``fell_off_course`` (which uses absolute z).
    """
    bad_height_scale: float = 1.0
    """Multiplier on the healthy lower bound; values below 1 tolerate deeper crouches
    and stumble-recoveries before terminating (used by the play script)."""
    bad_orientation: bool = True
    recovery_grace_steps: int = 0
    """Consecutive unhealthy steps tolerated before a height or tilt termination."""
    orientation_limit: float = 0.4
    """Terminate when the base z-axis tilts more than acos(limit) from vertical."""
    out_of_bounds: bool = True
    lateral_limit: float = 8.0
    backward_limit: float = 4.0
    fell_off_course: bool = True
    pit_margin: float = 1.0
    """Terminate once the root drops this far below the lowest waypoint."""
    stall: bool = True
    stall_steps: int = 250
    stall_distance: float = 0.5
    """Terminate if the robot has not closed this much distance in ``stall_steps``."""


def check(
    cfg: TerminationCfg,
    *,
    height_above_terrain: float,
    healthy_height_range: tuple[float, float],
    upright: float,
    position: tuple[float, float, float],
    min_course_z: float,
    stall_counter: int,
    bad_height_steps: int = 1,
    bad_orientation_steps: int = 1,
) -> tuple[bool, str]:
    if (
        cfg.bad_height
        and height_above_terrain < healthy_height_range[0] * cfg.bad_height_scale
        and bad_height_steps > cfg.recovery_grace_steps
    ):
        return True, "bad_height"
    if (cfg.bad_orientation and upright < cfg.orientation_limit
            and bad_orientation_steps > cfg.recovery_grace_steps):
        return True, "bad_orientation"
    if cfg.out_of_bounds and (
        abs(position[1]) > cfg.lateral_limit or position[0] < -cfg.backward_limit
    ):
        return True, "out_of_bounds"
    if cfg.fell_off_course and position[2] < min_course_z - cfg.pit_margin:
        return True, "fell_off_course"
    if cfg.stall and stall_counter >= cfg.stall_steps:
        return True, "stall"
    return False, ""
