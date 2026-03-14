"""
Synchronous Vectorized Multi-Agent Environment
================================================
Wraps N independent HideAndSeekEnv instances and steps them in lockstep.

Each sub-env runs its own episode. When a sub-env finishes (truncated/terminated),
it is auto-reset and the terminal observation + info is preserved for correct
GAE bootstrapping.

This is intentionally synchronous (no multiprocessing). MuJoCo is fast enough
that the Python loop overhead dominates, and synchronous execution is far
simpler to debug.
"""

from __future__ import annotations

import numpy as np
from env import HideAndSeekEnv, AGENT_NAMES, HIDER_NAMES, SEEKER_NAMES


class SyncVectorMultiAgentEnv:
    """
    Vectorized wrapper for N copies of HideAndSeekEnv.

    All public methods return data indexed as [env_idx][agent_name].
    """

    def __init__(
        self,
        n_envs: int = 8,
        horizon: int = 240,
        prep_fraction: float = 0.4,
        reward_type: str = "joint_zero_sum",
        render_mode: str | None = None,
        procedural: bool = True,
    ):
        self.n_envs = n_envs
        self.envs = [
            HideAndSeekEnv(
                horizon=horizon,
                prep_fraction=prep_fraction,
                reward_type=reward_type,
                # Only render the first env (if requested)
                render_mode=render_mode if i == 0 else None,
                procedural=procedural,
            )
            for i in range(n_envs)
        ]

        # Expose these for convenience
        self.possible_agents = list(AGENT_NAMES)
        self.hider_names = list(HIDER_NAMES)
        self.seeker_names = list(SEEKER_NAMES)

    def reset(self, seed: int | None = None):
        """
        Reset all sub-environments.

        Returns:
            all_obs:  list[dict[str, ndarray]]  — length n_envs
            all_info: list[dict]                — length n_envs
        """
        all_obs = []
        all_info = []
        for i, env in enumerate(self.envs):
            env_seed = (seed + i) if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            all_obs.append(obs)
            all_info.append(info)
        return all_obs, all_info

    def step(self, all_actions: list[dict[str, np.ndarray]]):
        """
        Step all sub-environments.

        Args:
            all_actions: list[dict[str, ndarray]] — length n_envs

        Returns:
            all_obs, all_rewards, all_terminated, all_truncated, all_info
            Each is a list of length n_envs.

            When a sub-env finishes, it is auto-reset. The returned obs/info
            for that env is the NEW episode's initial obs/info. The terminal
            obs is stored in info[env_idx]["terminal_obs"] and the terminal
            info in info[env_idx]["terminal_info"].
        """
        all_obs = []
        all_rewards = []
        all_terminated = []
        all_truncated = []
        all_info = []

        for i, env in enumerate(self.envs):
            obs, rewards, terminated, truncated, info = env.step(all_actions[i])

            # Check if this sub-env's episode is done
            done = all(
                truncated.get(n, False) or terminated.get(n, False)
                for n in AGENT_NAMES
            )

            if done:
                # Save terminal data before auto-reset
                info["terminal_obs"] = obs
                info["terminal_info"] = {k: v for k, v in info.items()
                                         if k not in ("terminal_obs", "terminal_info")}
                # Auto-reset
                obs, reset_info = env.reset()
                # Merge reset info but keep terminal markers
                info["global_state"] = reset_info["global_state"]

            all_obs.append(obs)
            all_rewards.append(rewards)
            all_terminated.append(terminated)
            all_truncated.append(truncated)
            all_info.append(info)

        return all_obs, all_rewards, all_terminated, all_truncated, all_info

    def close(self):
        for env in self.envs:
            env.close()

    @property
    def single_observation_space(self):
        return self.envs[0].observation_space

    @property
    def single_action_space(self):
        return self.envs[0].action_space
