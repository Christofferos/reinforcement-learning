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
    curriculum_window: int = 20
    promote_success_rate: float = 0.8
    demote_completion: float = 0.25
    difficulty_step: float = 0.05


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
    nominal_base_height: float = 1.25
    debug_scan_markers: bool = False

    def replace(self, **kwargs) -> "ParkourEnvCfg":
        return replace(self, **kwargs)
