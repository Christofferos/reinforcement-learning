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
from .terrain.parkour import DEFAULT_SKILLS

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
    # rough_0.0.1 at d=0.5 had ~9 stumble steps per episode, and 8 of 19 falls came within 1 s
    # of one; never on flat ground. A trip of a few steps costs a few points against the -30 fall.
    feet_stumble=-1.0,
    # rough_0.0.2 at d=1 stood a foot within 5 cm of a block-top edge on ~25 foot-steps per
    # episode (never on hills alone), and 4 of 22 falls followed one within 1 s: about -13 per
    # episode, in line with the stumble penalty. On a Procedural beam it keeps feet off the sides.
    feet_edge=-0.5,
    # Standing upright (1.53 m head top) under a d=1 tunnel bar (1.2 m) is 0.38 m too tall,
    # -1.14 per step, while it is in reach of the upward rays: well under the ~3 per step a good
    # walk earns, so ducking pays but stopping short of the tunnel never does. Zero without one.
    ceiling_clearance=-3.0,
    # Facing the waypoint: walking 33 deg sideways at 0.58 m/s costs ~0.17 per step.
    heading=0.3, lateral_velocity=-0.2,
    # Per-step budget at a good 1 m/s walk: progress ~1, tracking ~1, gait ~0.5, posture ~0.8,
    # so no single term dominates the return. Procedural used to pay alive 0.05 and charge
    # action_rate and angular_velocity -0.01, which changed the return scale a resumed value
    # function expects.
    velocity_tracking_sigma=0.5, base_height=0.5, alive=0.15, upright=0.1,
    ctrl_cost=-0.01, action_rate=-0.02, angular_velocity=-0.05,
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
        reward=RewardCfg(**LOCOMOTION_REWARDS, velocity_tracking=1.0),
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
        command_speed_range=(1.0, 2.0),
        command_speed_difficulty_shift=0,
        zero_command_prob=0.0,
        observation=ObservationCfg(speed_command=True, overhead_scan=True),
        reward=RewardCfg(**LOCOMOTION_REWARDS, velocity_tracking=1.5),
        termination=TerminationCfg(
            bad_height_scale=0.75, recovery_grace_steps=20,
            stall_steps=500, stall_distance=0.15,
        ),
        events=EventCfg(
            reset_noise_scale=0.02, reset_yaw_range=0.25, 
            randomize_friction=True, randomize_mass=True, randomize_gear=True,
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
            # 4 modules at difficulty 0 to 8 at 1. The curriculum promotes on 80% of runs
            # finishing the whole course, which over 8 modules needs ~97% success on each;
            # rough_0.0.3 finished none of 40 eight-module courses at d=0.05.
            # Extra practice where the last run failed (each weighted module ~16% of courses' modules,
            # the rest ~5%). procedural_0.0.2 at d=0.38/0.6 failed hurdles 100%/100%, climbs 36%/71%
            # and stepping stones 22%/55%, but beams 0% and gaps 6%/20%; procedural_0.0.1 had failed
            # beam, stones and gap at d=0, which 3x practice there fixed. Every skill stays in the mix.
            params={
                "num_modules": 4, "num_modules_growth": 4,
                "skill_weights": {
                    **{skill: 1.0 for skill in DEFAULT_SKILLS},
                    "hurdle": 3.0, "climb": 3.0, "stepping_stones": 3.0, "gap": 1.5,
                },
            },
            resample_every_n_resets=1,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
        ),
        episode_length_s=75.0,
        # Rough's commands. (0.5, 2.0) plus a 1.0 difficulty shift asked for up to 3 m/s, far
        # beyond the ~1 m/s the walker reaches, which taught rough_0.0.1 to ignore the command.
        command_speed_range=(0.75, 1.5),
        command_speed_difficulty_shift=0.0,
        observation=ObservationCfg(
            speed_command=True, overhead_scan=True,
            # The same 9 x 5 rays at 0.2 m instead of 0.25 m: from 0.2 m behind to 1.4 m ahead and
            # 0.4 m to each side, so beam and stone edges are placed to within 0.1 m.
            scan_forward=(-0.2, 1.4), scan_lateral=(-0.4, 0.4),
            # Off-course rays read -0.5, not -1.0: on Rough's scale (std ~0.12 m) -1.0 normalised to
            # -7 std and saturated ~30% of the first tanh layer on beams, stones and gaps; at -0.5,
            # 12-26%. procedural_0.0.1's beam went from 0/20 to 4/20 at -0.5 (14/20 at -0.3, which
            # blurred pits with steps down). Ground 0.4 m or more below a robot at nominal height
            # (the reading counts from 0.1 m above the torso) now reads the same as a pit.
            scan_clip=(-0.5, 1.5),
        ),
        reward=RewardCfg(**LOCOMOTION_REWARDS, velocity_tracking=1.5),
        termination=TerminationCfg(bad_height_scale=0.70, recovery_grace_steps=20),
        events=EventCfg(
            reset_noise_scale=0.02,
            reset_yaw_range=0.25,
            randomize_mass=True,
            randomize_gear=True,
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
