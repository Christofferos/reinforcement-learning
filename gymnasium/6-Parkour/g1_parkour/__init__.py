"""Procedural MuJoCo parkour environments for legged/humanoid skill sequencing.

Quick start::

    import gymnasium as gym
    import g1_parkour.tasks  # registers the task ids

    env = gym.make("Parkour-Procedural-v0", render_mode="human")
"""

from .env import ParkourEnv
from .env_cfg import ParkourEnvCfg, TerrainCfg
from .mdp import EventCfg, ObservationCfg, RewardCfg, TerminationCfg
from .robots import HUMANOID, RobotSpec, get_robot
from .terrain import build_terrain

__all__ = [
    "EventCfg",
    "HUMANOID",
    "ObservationCfg",
    "ParkourEnv",
    "ParkourEnvCfg",
    "RewardCfg",
    "RobotSpec",
    "TerminationCfg",
    "TerrainCfg",
    "build_terrain",
    "get_robot",
]
