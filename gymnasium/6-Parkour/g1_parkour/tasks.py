"""Task registry.

Importing this module registers every task with Gymnasium, e.g.::

    import gymnasium as gym
    import g1_parkour.tasks  # noqa: F401
    env = gym.make("Parkour-Procedural-v0")
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Callable

from gymnasium.envs.registration import register, registry

from .env_cfg import ParkourEnvCfg, TerrainCfg
from .mdp import EventCfg, ObservationCfg, RewardCfg, TerminationCfg

ENTRY_POINT = "g1_parkour.env:ParkourEnv"


# --------------------------------------------------------------------------------------
# Base configs
# --------------------------------------------------------------------------------------

# Keeps the height scan (flat => constant)
def flat_cfg() -> ParkourEnvCfg:
    return ParkourEnvCfg(
        terrain=TerrainCfg(
            kind="flat",
            difficulty=0.0,
            params={"course_length": 12.0, "waypoint_spacing": 3.0, "lateral_jitter": 1.25},
            resample_every_n_resets=1,
        ),
        episode_length_s=40.0,
        command_speed_range=(1.25, 2.25),
        observation=ObservationCfg(speed_command=True, overhead_scan=True),
        reward=RewardCfg(
            progress=2.0, progress_per_second=False, clip_progress=1.5,
            velocity_tracking=1.5, waypoint_bonus=5.0, goal_bonus=20.0,
            alive=0.05, upright=0.3, heading=0.1, base_height=0.5,
            ctrl_cost=-0.01, action_rate=-0.02, angular_velocity=-0.05,
            lateral_velocity=-0.05, overspeed=-0.5, fall_penalty=-30.0,
        ),
        termination=TerminationCfg(recovery_grace_steps=10, stall_steps=500, stall_distance=0.15),
        events=EventCfg(reset_noise_scale=0.01, reset_yaw_range=0.25),
        debug_scan_markers=True,
    )


def rough_cfg() -> ParkourEnvCfg:
    cfg = ParkourEnvCfg(
        terrain=TerrainCfg(
            kind="rough",
            difficulty=0.5,
            params={"course_length": 28.0, "course_width": 12.0},
            resample_every_n_resets=5,
            curriculum=True,
        ),
        episode_length_s=70.0,
        command_speed_range=(0.75, 2.0),
        observation=ObservationCfg(speed_command=True, overhead_scan=True),
        reward=RewardCfg(
            progress_per_second=False, clip_progress=1.5,
            velocity_tracking=1.5, overspeed=-0.5,
        ),
        events=EventCfg(reset_noise_scale=0.02, randomize_friction=True, randomize_mass=True),
        debug_scan_markers=True,
    )
    return cfg


def parkour_cfg() -> ParkourEnvCfg:
    return ParkourEnvCfg(
        terrain=TerrainCfg(
            kind="parkour",
            difficulty=0.05,
            params={"num_modules": 8},
            resample_every_n_resets=1,
            curriculum=True,
        ),
        episode_length_s=60.0,
        command_speed_range=(1.0, 2.75),
        observation=ObservationCfg(speed_command=True, overhead_scan=True),
        reward=RewardCfg(
            progress=20.0, progress_per_second=False, clip_progress=1.2,
            waypoint_bonus=20.0, goal_bonus=200.0, alive=0.05, upright=0.1,
            heading=0.1, overspeed=-0.2, preferred_speed=1.2, fall_penalty=-30.0,
            ctrl_cost=-0.01, action_rate=-0.01, angular_velocity=-0.01,
            lateral_velocity=-0.05,
        ),
        termination=TerminationCfg(bad_height_scale=0.75, recovery_grace_steps=20),
        events=EventCfg(
            reset_noise_scale=0.02,
            reset_yaw_range=0.1,
            randomize_friction=True,
            randomize_mass=True,
            push_robot=True,
        ),
        debug_scan_markers=True,
    )


# --------------------------------------------------------------------------------------
# MDP ablations, applied on top of the procedural parkour task
# --------------------------------------------------------------------------------------

def _blind(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.observation = replace(cfg.observation, height_scan=False)
    return cfg


def _no_waypoints(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.observation = replace(cfg.observation, waypoint_command=False)
    return cfg


def _no_privileged_velocity(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.observation = replace(cfg.observation, base_lin_vel=False, base_height=False)
    return cfg


def _noisy_observations(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.observation = replace(
        cfg.observation,
        noise_lin_vel=0.1,
        noise_ang_vel=0.2,
        noise_gravity=0.05,
        noise_joint_pos=0.01,
        noise_joint_vel=1.5,
        noise_height_scan=0.05,
        exteroceptive_delay=2,
    )
    cfg.events = replace(cfg.events, action_delay_steps=1)
    return cfg


def _history(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.observation = replace(cfg.observation, history_length=5)
    return cfg


def _sparse_reward(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.reward = RewardCfg(
        progress=0.0, waypoint_bonus=10.0, goal_bonus=100.0, heading=0.0,
        alive=0.0, upright=0.0, lateral_velocity=0.0, angular_velocity=0.0,
        action_rate=0.0, ctrl_cost=0.0, joint_limit=0.0, fall_penalty=-10.0,
    )
    return cfg


def _no_style_costs(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.reward = replace(
        cfg.reward, action_rate=0.0, ctrl_cost=0.0, joint_limit=0.0,
        lateral_velocity=0.0, angular_velocity=0.0,
    )
    return cfg


def _energy_penalty(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.reward = replace(cfg.reward, energy=-2e-4)
    return cfg


def _no_early_termination(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.termination = TerminationCfg(
        bad_height=False, bad_orientation=False, out_of_bounds=True,
        fell_off_course=True, stall=False,
    )
    return cfg


def _no_domain_randomization(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.events = EventCfg(reset_noise_scale=0.02)
    return cfg


def _reset_along_course(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.events = replace(cfg.events, reset_along_course=True)
    return cfg


def _fixed_course(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.terrain = replace(cfg.terrain, resample_every_n_resets=0, curriculum=False)
    return cfg


def _no_curriculum(cfg: ParkourEnvCfg) -> ParkourEnvCfg:
    cfg.terrain = replace(cfg.terrain, curriculum=False, difficulty=1.0)
    return cfg


ABLATIONS: dict[str, Callable[[ParkourEnvCfg], ParkourEnvCfg]] = {
    "blind": _blind,
    "no-waypoints": _no_waypoints,
    "no-privileged-velocity": _no_privileged_velocity,
    "noisy-obs": _noisy_observations,
    "obs-history": _history,
    "sparse-reward": _sparse_reward,
    "no-style-costs": _no_style_costs,
    "energy-penalty": _energy_penalty,
    "no-early-termination": _no_early_termination,
    "no-domain-rand": _no_domain_randomization,
    "reset-along-course": _reset_along_course,
    "fixed-course": _fixed_course,
    "no-curriculum": _no_curriculum,
}


def make_ablation_cfg(name: str, base: ParkourEnvCfg | None = None) -> ParkourEnvCfg:
    if name not in ABLATIONS:
        raise KeyError(f"Unknown ablation '{name}'. Available: {sorted(ABLATIONS)}")
    return ABLATIONS[name](copy.deepcopy(base) if base else parkour_cfg())


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------

TASKS: dict[str, Callable[[], ParkourEnvCfg]] = {
    "Flat": flat_cfg,
    "Rough": rough_cfg,
    "Procedural": parkour_cfg,
}

for _ablation in ABLATIONS:
    _slug = "".join(part.capitalize() for part in _ablation.split("-"))
    TASKS[f"Parkour-Ablation-{_slug}-v0"] = (
        lambda name=_ablation: make_ablation_cfg(name)
    )


def register_tasks() -> None:
    for task_id, cfg_fn in TASKS.items():
        if task_id in registry:
            continue
        register(
            id=task_id,
            entry_point=ENTRY_POINT,
            max_episode_steps=None,
            disable_env_checker=True,
            kwargs={"cfg": None},
        )
        registry[task_id].kwargs["cfg"] = cfg_fn()


register_tasks()
