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

# Reward weights shared by every stage. One reward scale from Flat to Procedural keeps the
# VecNormalize return statistics and the value function restored by --resume valid when a
# run moves to the next stage, and keeps the gait terms worth as much there as on Flat.
LOCOMOTION_REWARDS = dict(
    progress=1.0, progress_per_second=True, clip_progress=4.0,
    waypoint_bonus=20.0, goal_bonus=200.0, fall_penalty=-30.0,
    # Per stride from the step terms: the flat_0.0.9 gallop (one foot leads 0.56 m, the other
    # catches up 0.19 m behind it) nets -13.2, the flat_0.0.10 skip (0.30 m lead, -0.02 m
    # catch-up) -1.7, an even 0.3 m walk +10 and an even 0.15 m shuffle +5. The 0.3 m margin
    # keeps paying for longer steps, which faster commands need.
    feet_air_time=2.0, feet_step_ahead=5.0, feet_step_ahead_margin=0.3,
    feet_step_symmetry=-10.0, feet_slide=-0.1,
    # Facing the waypoint: walking 33 deg sideways at 0.58 m/s costs ~0.17 per step.
    heading=0.3, lateral_velocity=-0.2,
)

# Keeps the height scan (flat => constant)
def flat_cfg() -> ParkourEnvCfg:
    return ParkourEnvCfg(
        terrain=TerrainCfg(
            kind="flat",
            difficulty=0.0,
            params={
                "course_length": 12.0, "waypoint_spacing": 3.0, "lateral_jitter": 1.25,
                "course_length_growth": 6.0, "lateral_jitter_growth": 1.25,
            },
            resample_every_n_resets=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
        ),
        episode_length_s=40.0,
        command_speed_range=(0.5, 1.5),
        command_speed_difficulty_shift=1.0,
        zero_command_prob=0.0,
        observation=ObservationCfg(speed_command=True, overhead_scan=True),
        reward=RewardCfg(
            # Per-step budget at a good 1 m/s walk: progress ~1, tracking ~1, gait ~0.5,
            # posture ~0.8, so no single term dominates the return.
            **LOCOMOTION_REWARDS,
            velocity_tracking=1.0, velocity_tracking_sigma=0.5, base_height=0.5,
            alive=0.15, upright=0.1,
            ctrl_cost=-0.01, action_rate=-0.02, angular_velocity=-0.05,
        ),
        termination=TerminationCfg(
            bad_height_scale=0.75, recovery_grace_steps=20,
            stall_steps=500, stall_distance=0.15,
        ),
        events=EventCfg(
            reset_noise_scale=0.01, reset_yaw_range=0.25,
            # 0.12-0.25 m/s kicks at difficulty 0, ramping to 0.3-0.6 m/s (Procedural's level) at 1.
            push_robot=True, push_interval_steps=200, push_velocity=0.25,
            push_velocity_difficulty_shift=0.35,
        ),
        debug_scan_markers=True,
    )


def rough_cfg() -> ParkourEnvCfg:
    cfg = ParkourEnvCfg(
        terrain=TerrainCfg(
            kind="rough",
            difficulty=0.0,
            # 16 m at difficulty 0 to 25 m at 1; difficulty also raises hills, deepens pits,
            # shortens bumps and adds rubble along the route (see make_rough).
            params={"course_length": 16.0, "course_length_growth": 9.0, "course_width": 12.0},
            resample_every_n_resets=5,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
        ),
        episode_length_s=60.0,
        command_speed_range=(0.5, 1.25),
        command_speed_difficulty_shift=1.0,
        zero_command_prob=0.0,
        observation=ObservationCfg(speed_command=True, overhead_scan=True),
        reward=RewardCfg(
            **LOCOMOTION_REWARDS,
            velocity_tracking=1.5, velocity_tracking_sigma=1, base_height=0.5,
            alive=0.15, upright=0.1,
            ctrl_cost=-0.01, action_rate=-0.02, angular_velocity=-0.05,
        ),
        termination=TerminationCfg(
            bad_height_scale=0.75, recovery_grace_steps=20,
            stall_steps=500, stall_distance=0.15,
        ),
        events=EventCfg(
            reset_noise_scale=0.02, reset_yaw_range=0.25, 
            randomize_friction=True, randomize_mass=True, 
            push_robot=True, push_interval_steps=200, 
            push_velocity=0.25, push_velocity_difficulty_shift=0.5,
        ),
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
            difficulty_range=(0.0, 1.0),
        ),
        episode_length_s=60.0,
        command_speed_range=(0.5, 2.0),
        command_speed_difficulty_shift=1.0,
        observation=ObservationCfg(speed_command=True, overhead_scan=True),
        reward=RewardCfg(
            **LOCOMOTION_REWARDS,
            velocity_tracking=1.5, base_height=0.5,
            alive=0.05, upright=0.1,
            ctrl_cost=-0.01, action_rate=-0.01, angular_velocity=-0.01,
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
