"""Environment configuration dataclasses (terrain + MDP + runtime)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .mdp import EventCfg, ObservationCfg, RewardCfg, TerminationCfg


@dataclass
class TerrainCfg:
    kind: str = "parkour"
    """One of ``flat``, ``rough``, ``parkour``."""
    difficulty: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)
    """Extra kwargs forwarded to the terrain generator."""

    resample_every_n_resets: int = 1
    """0 keeps a single fixed course (fastest); 1 regenerates on every reset."""

    curriculum: bool = False
    difficulty_range: tuple[float, float] = (0.1, 1.0)
    promote_completion: float = 1.0
    """Completion required for a successful full-course curriculum episode."""
    curriculum_window: int = 15
    promote_success_rate: float = 0.8
    """Promote when at least this fraction of the window are full successes; demote when at
    least this fraction end with completion <= ``demote_completion``. Both decisions need a
    full window, which is cleared on every level change."""
    demote_completion: float = 0.25
    difficulty_step: float = 0.05

    per_skill_difficulty: bool = False
    """Parkour only: a level per skill instead of one for the course. Every module is built at
    its skill's level, which the curriculum moves on how often that skill's modules are passed
    (``skill_*`` below), so one weak skill no longer holds back the rest. A module is passed
    when the robot reaches its last waypoint and failed when the episode ends short of it.
    ``difficulty`` is then the starting level of every skill, and the course's own difficulty
    (its length) the skill-weighted mean level."""
    skill_difficulty: dict[str, float] | None = None
    """Starting levels for named skills, over ``difficulty`` (playback of a trained model)."""
    skill_window: int = 20
    """Module attempts per skill and worker before a promote or demote decision."""
    skill_maintain_share: float | None = None
    """Per-skill only: practice for a skill at the top level, as a share of what it gets at
    level 0. Each skill's ``skill_weights`` entry is scaled by ``1 - (1 - share) * level`` when a
    course is built, so a mastered skill is kept up with fewer modules and the ones still
    climbing get more. None samples by ``skill_weights`` alone."""
    skill_action_noise: dict[str, float] = field(default_factory=dict)
    """Per-skill only: scale on the policy's action noise in training while the robot is on that
    skill's module (its waypoints are the active target), 1 for skills it leaves out. The
    environment reports it as ``info["action_noise_scale"]``; ``SymmetricPPO`` samples with it
    and evaluates the update under it. Evaluation and playback use no noise at all."""
    skill_review_share: float = 0.0
    """Per-skill training only: the share of modules built at a random easier level (uniform
    below the skill's), so easier instances are not forgotten. They do not count toward the
    skill's promotion. Off while the curriculum is off (evaluation, playback)."""
    skill_preview_share: float = 0.0
    skill_preview_range: float = 0.1
    """Per-skill training only: the share of modules built up to ``skill_preview_range`` above
    the skill's level, a look at what promotion brings. Also not counted, also off without the
    curriculum."""
    skill_promote_rate: float = 0.85
    skill_demote_rate: float = 0.6
    """Promote a skill when at least ``skill_promote_rate`` of its window passed; demote it
    when fewer than ``skill_demote_rate`` did. The window clears on every level change."""


@dataclass
class ParkourEnvCfg:
    robot: str = "humanoid"
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    events: EventCfg = field(default_factory=EventCfg)

    frame_skip: int = 5
    episode_length_s: float = 25.0
    waypoint_radius: float = 0.7
    waypoint_lookahead: int = 2
    command_speed_range: tuple[float, float] | None = None
    """Sample a requested walking speed per waypoint; None keeps legacy rewards/observations."""
    command_speed_difficulty_shift: float = 0.0
    """Shift both speed bounds by this amount at maximum terrain difficulty."""
    zero_command_prob: float = 0.0
    """Fraction of episodes whose waypoint speeds are all zero (stand-and-balance practice)."""
    control_mode: str = "position"
    """``position``: actions are joint-position targets around the robot's standing pose,
    tracked by a per-substep PD loop and clipped to the motor torque limits.
    ``torque``: actions map directly to the actuator control range (legacy)."""
    position_action_scale: float = 1.0
    """Radians of joint-target offset per unit action in ``position`` mode."""
    nominal_base_height: float = 1.25
    debug_scan_markers: bool = False
    water_physics: bool = True
    """Buoyancy and drag inside the terrain's water volumes (``water_tank``). Courses without
    water never compute them; turning this off is for benchmarking."""

    def replace(self, **kwargs) -> "ParkourEnvCfg":
        return replace(self, **kwargs)
