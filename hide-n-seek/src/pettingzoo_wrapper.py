"""
PettingZoo ParallelEnv wrapper for the Hide and Seek environment.

This wraps the custom MuJoCo environment so it can be used with
PettingZoo-compatible MARL frameworks (MAPPO, RLlib, etc.).
"""

from __future__ import annotations

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

from env import HideAndSeekEnv, AGENT_NAMES, HIDER_NAMES, SEEKER_NAMES


class HideAndSeekPZ(ParallelEnv):
    """PettingZoo Parallel API wrapper for Hide and Seek."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "hide_and_seek_v0"}

    def __init__(self, **kwargs):
        super().__init__()
        self._env = HideAndSeekEnv(**kwargs)

        self.possible_agents = list(AGENT_NAMES)
        self.agents = list(AGENT_NAMES)
        self.agent_name_mapping = {name: i for i, name in enumerate(AGENT_NAMES)}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._env.observation_space[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._env.action_space[agent]

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        obs, info = self._env.reset(seed=seed, options=options)
        self._global_state = info.get("global_state", None)

        infos = {agent: {"global_state": self._global_state} for agent in self.agents}
        return obs, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self._env.step(actions)
        self._global_state = info.get("global_state", None)

        # PettingZoo expects per-agent infos
        infos = {
            agent: {
                "global_state": self._global_state,
                "prep_phase": info.get("prep_phase", False),
            }
            for agent in self.agents
        }

        # Remove agents that are done
        self.agents = [
            a for a in self.agents
            if not (terminated.get(a, False) or truncated.get(a, False))
        ]

        return obs, rewards, terminated, truncated, infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def global_state_size(self):
        return self._env.global_state_size

    def get_global_state(self):
        return self._env.get_global_state()

    # Convenience
    @property
    def n_hiders(self):
        return self._env.n_hiders

    @property
    def n_seekers(self):
        return self._env.n_seekers


def raw_env(**kwargs):
    """Returns the AEC API version of the environment."""
    env = HideAndSeekPZ(**kwargs)
    env = parallel_to_aec(env)
    return env


def parallel_env(**kwargs) -> HideAndSeekPZ:
    """Returns the Parallel API version of the environment."""
    return HideAndSeekPZ(**kwargs)
