"""MuJoCo parkour environment with procedural terrain and swappable MDP terms."""

from __future__ import annotations

import copy
from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.spaces import Box

from . import mdp, symmetry
from .env_cfg import ParkourEnvCfg, TerrainCfg
from .robots import get_robot
from .scene import SCAN_CLIPPED_RGBA, SCAN_MARKER_RGBA, build_scene_xml, default_build_dir
from .camera import EYE_MOUNT, LEGACY_NAMES, STABLE_EGOCENTRIC, StableEyeCamera, camera_frame
from .water import WaterEffects, WaterPhysics
from .terrain import build_terrain
from .terrain.parkour import DEFAULT_SKILLS

SUPPORT_NORMAL_Z = 0.5
"""A terrain contact whose normal has |z| above this supports weight (ground, a block top); at or
below it the contact is on a face steeper than 60 degrees (a block side, a stair riser)."""

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 6.0,
    "elevation": -20.0,
    "azimuth": 130.0,
}


def mean_curriculum_state(states, step: float = 0.0) -> float | dict[str, float]:
    """Training workers' curriculum states (``ParkourEnv.curriculum_state``) averaged into one,
    snapped to the ``step`` grid: a level, or each skill's level over the workers that have it."""
    snap = lambda level: round(round(level / step) * step, 6) if step > 0 else float(level)
    states = list(states)
    if states and all(isinstance(state, dict) for state in states):
        skills = sorted({skill for state in states for skill in state})
        return {skill: snap(np.mean([s[skill] for s in states if skill in s])) for skill in skills}
    return snap(float(np.mean(states)))


class ParkourEnv(gym.Env):
    """Single-agent parkour env.

    The scene is generated from a :class:`~g1_parkour.env_cfg.TerrainCfg`, so the same
    class serves the flat, rough and procedural-parkour tasks.  Every MDP component
    (observation terms, reward weights, terminations, domain randomisation) is a config
    field, which is what the ablation scripts sweep.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 66}

    def __init__(
        self,
        cfg: ParkourEnvCfg | None = None,
        render_mode: str | None = None,
        width: int = 1920,
        height: int = 1080,
        camera_name: str | None = None,
        default_camera_config: dict | None = None,
    ):
        self.cfg = copy.deepcopy(cfg) if cfg is not None else ParkourEnvCfg()
        if self.cfg.terrain.curriculum_window < 1 or not 0 < self.cfg.terrain.promote_success_rate <= 1:
            raise ValueError("curriculum window must be positive and success rate must be in (0, 1]")
        terrain = self.cfg.terrain
        if terrain.per_skill_difficulty and (
            terrain.kind != "parkour" or terrain.skill_window < 1
            or not 0 <= terrain.skill_demote_rate <= terrain.skill_promote_rate <= 1
        ):
            raise ValueError("per-skill difficulty needs parkour terrain, a positive window and "
                             "0 <= skill_demote_rate <= skill_promote_rate <= 1")
        if self.cfg.command_speed_range is not None:
            low, high = self.cfg.command_speed_range
            if not np.all(np.isfinite([low, high])) or low <= 0 or high < low:
                raise ValueError("command_speed_range must contain finite positive speeds in ascending order")
        if not np.isfinite(self.cfg.command_speed_difficulty_shift) or self.cfg.command_speed_difficulty_shift < 0:
            raise ValueError("command_speed_difficulty_shift must be finite and non-negative")
        if not 0.0 <= self.cfg.zero_command_prob <= 1.0:
            raise ValueError("zero_command_prob must be in [0, 1]")
        if self.cfg.control_mode not in ("position", "torque"):
            raise ValueError("control_mode must be 'position' or 'torque'")
        if not np.isfinite(self.cfg.position_action_scale) or self.cfg.position_action_scale <= 0:
            raise ValueError("position_action_scale must be finite and positive")
        self.render_mode = render_mode
        self._render_width = width
        self._render_height = height
        self._camera_name = LEGACY_NAMES.get(camera_name, camera_name)
        self._camera_config = default_camera_config or DEFAULT_CAMERA_CONFIG

        self.robot = get_robot(self.cfg.robot)
        self.build_dir: Path = default_build_dir(self.cfg.terrain.kind)
        self._np_random_seeded = np.random.default_rng()

        self._difficulty = float(self.cfg.terrain.difficulty)
        self._curriculum_successes = deque(maxlen=self.cfg.terrain.curriculum_window)
        self._curriculum_early_failures = deque(maxlen=self.cfg.terrain.curriculum_window)
        # Per-skill levels (empty unless cfg.terrain.per_skill_difficulty) and each skill's
        # recent module passes; this episode's module results wait for its end.
        self._skill_difficulty: dict[str, float] = {}
        self._skill_results: dict[str, deque] = {}
        if terrain.per_skill_difficulty:
            skills = terrain.params.get("skills", DEFAULT_SKILLS)
            self._skill_difficulty = {skill: self._clip_difficulty(terrain.difficulty) for skill in skills}
            self._skill_results = {skill: deque(maxlen=terrain.skill_window) for skill in skills}
            self.set_difficulty({**self._skill_difficulty, **(terrain.skill_difficulty or {})})
        self._built_skill_difficulty: dict[str, float] = {}
        self._module_results: list[tuple[str, bool]] = []
        self._episode_credited = False
        self._reset_count = 0
        self._mujoco_renderer = None
        self._viewer_camera_set = False
        self._stable_eye = StableEyeCamera()
        self._episode_terms: dict[str, float] = {}

        self._build_world(self._np_random_seeded)

        self.action_space = Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        obs = self._compute_observation()
        self.observation_space = Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float64)
        self.metadata = dict(self.metadata, render_fps=int(round(1.0 / self.dt)))

    # ---------------------------------------------------------------- construction

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.cfg.frame_skip

    @property
    def max_episode_steps(self) -> int:
        return int(self.cfg.episode_length_s / self.dt)

    def _build_world(self, rng: np.random.Generator) -> None:
        """(Re)generate terrain, compile the model and cache derived quantities."""
        terrain_cfg: TerrainCfg = self.cfg.terrain
        params = dict(terrain_cfg.params)
        if self._skill_difficulty:
            params["skill_difficulty"] = dict(self._skill_difficulty)
            params["skill_weights"] = self._skill_sampling_weights()
            if terrain_cfg.curriculum:  # practice around the levels is for training only
                params.update(review_share=terrain_cfg.skill_review_share,
                              preview_share=terrain_cfg.skill_preview_share,
                              preview_range=terrain_cfg.skill_preview_range)
        self.terrain = build_terrain(
            terrain_cfg.kind,
            rng,
            difficulty=self._difficulty,
            **params,
        )
        self._built_skill_difficulty = dict(self._skill_difficulty)
        num_markers = self.cfg.observation.num_scan_points if self.cfg.debug_scan_markers else 0
        num_overhead_markers = (
            len(self.cfg.observation.overhead_offsets)
            if self.cfg.debug_scan_markers and self.cfg.observation.overhead_scan else 0
        )
        xml_path = build_scene_xml(
            self.robot, self.terrain, self.build_dir, num_markers,
            num_overhead_markers=num_overhead_markers,
            visuals=self.render_mode is not None,
        )

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._ceiling_geom_ids = frozenset(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, box.name)
            for box in self.terrain.boxes if box.ceiling
        )
        self._wall_geom_ids = frozenset(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, box.name)
            for box in self.terrain.boxes if box.wall
        )

        self._nominal = {
            "geom_friction": self.model.geom_friction.copy(),
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "actuator_gear": self.model.actuator_gear.copy(),
        }
        self._root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.robot.root_body
        )
        if self._root_body_id < 0:
            raise RuntimeError(f"root body '{self.robot.root_body}' not found in the scene")
        self._water = (
            WaterPhysics(self.model, self.terrain.water, self._root_body_id)
            if self.terrain.water and self.cfg.water_physics else None
        )
        self._water_effects = None
        self._foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.robot.foot_bodies
        ]
        if any(body_id < 0 for body_id in self._foot_body_ids):
            raise RuntimeError("robot foot bodies not found in the scene")
        self._foot_geom_ids = [
            {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body_id}
            for body_id in self._foot_body_ids
        ]
        shin_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.robot.shin_bodies
        ]
        if any(body_id < 0 for body_id in shin_body_ids):
            raise RuntimeError("robot shin bodies not found in the scene")
        self._leg_geom_ids = set().union(*self._foot_geom_ids, *(
            {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body_id}
            for body_id in shin_body_ids
        ))
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        self._joint_range = self.model.jnt_range.copy()
        # Actuated hinge joints: qpos/qvel addresses, target limits and PD gains per actuator.
        act_joints = self.model.actuator_trnid[:, 0]
        self._act_qpos_adr = self.model.jnt_qposadr[act_joints]
        self._act_dof_adr = self.model.jnt_dofadr[act_joints]
        self._act_limited = self.model.jnt_limited[act_joints].astype(bool)
        self._act_joint_range = self.model.jnt_range[act_joints].copy()
        torque_limit = np.max(np.abs(self._ctrl_range), axis=1) * np.abs(self._nominal["actuator_gear"][:, 0])
        self._pd_kp = self.robot.pd_stiffness_per_torque * torque_limit
        self._pd_kd = self.robot.pd_damping_ratio * self._pd_kp
        # Raise torque ceilings only after the gains: same stiffness, more headroom.
        ceiling = np.ones(self.model.nu)
        for actuator in range(self.model.nu):
            name = self.model.actuator(actuator).name
            for suffix, scale in self.robot.torque_ceiling_scale.items():
                if name.endswith(suffix):
                    ceiling[actuator] = scale
        self.model.actuator_gear[:, 0] *= ceiling
        self._nominal["actuator_gear"][:, 0] *= ceiling
        # The robot's own joints below its free root, in qpos order. Terrain modules with moving
        # parts (a barrel's or seesaw's hinge, an elevator's slide) add joints after the robot's;
        # they are not observed, mirrored, randomised at reset or charged as joint limits.
        robot_tree = self.model.body_rootid[self._root_body_id]
        robot_joints = [
            j for j in range(self.model.njnt)
            if self.model.body_rootid[self.model.jnt_bodyid[j]] == robot_tree
        ]
        root_joint = robot_joints[0] if robot_joints else -1
        if root_joint < 0 or self.model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE or (
            self.model.jnt_qposadr[root_joint] != 0 or self.model.jnt_dofadr[root_joint] != 0
        ):
            raise RuntimeError("the robot's free root joint must come first in the scene")
        self._robot_joint_ids = robot_joints[1:]
        self._robot_geom_ids = frozenset(
            g for g in range(self.model.ngeom) if self.model.body_rootid[self.model.geom_bodyid[g]] == robot_tree
        )
        if any(self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in self._robot_joint_ids):
            raise RuntimeError("the robot's joints below its root must be hinges")
        self._robot_qpos_adr = self.model.jnt_qposadr[self._robot_joint_ids].astype(int)
        self._robot_dof_adr = self.model.jnt_dofadr[self._robot_joint_ids].astype(int)
        robot_qpos = set(range(7)) | set(self._robot_qpos_adr.tolist())
        robot_dofs = set(range(6)) | set(self._robot_dof_adr.tolist())
        self._terrain_qpos_adr = np.array([i for i in range(self.model.nq) if i not in robot_qpos], dtype=int)
        self._terrain_dof_adr = np.array([i for i in range(self.model.nv) if i not in robot_dofs], dtype=int)
        self._hinge_jnt_ids = np.array(
            [j for j in self._robot_joint_ids if self.model.jnt_limited[j]], dtype=int,
        )
        self._hinge_qpos_adr = self.model.jnt_qposadr[self._hinge_jnt_ids].astype(int)
        self._actors = list(self.terrain.actors)
        for actor in self._actors:
            actor.bind(self)
        self._site_target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_marker")
        self._site_next = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_marker_next"
        )
        self._scan_marker_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"scan_viz_{i}")
            for i in range(num_markers)
        ]
        self._overhead_marker_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"overhead_viz_{index}")
            for index in range(num_overhead_markers)
        ]

        self._init_qpos = self.model.qpos0.copy()
        for name, angle in self.robot.default_joint_pos.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"default_joint_pos joint '{name}' not found in the scene")
            self._init_qpos[self.model.jnt_qposadr[joint_id]] = angle
        self._default_act_pos = self._init_qpos[self._act_qpos_adr].copy()
        self._init_qvel = np.zeros(self.model.nv)
        self._waypoints = self._resolve_waypoints(self.terrain.waypoints)
        self._waypoint_speeds = np.zeros(len(self._waypoints))
        self._velocity_error_sum = 0.0
        self._episode_speed_sum = 0.0
        self._episode_height_sum = 0.0
        self._episode_height_samples = 0
        self._min_course_z = float(self._waypoints[:, 2].min()) if len(self._waypoints) else 0.0

        self._prev_action = np.zeros(self.model.nu)
        self._target_index = 0
        self._start_target_index = 0
        self._goal_reached = False
        self._step_count = 0
        self._waypoints_reached = 0
        self._stall_counter = 0
        self._bad_height_steps = 0
        self._bad_orientation_steps = 0
        self._steps_since_ceiling = 10**9
        self._steps_since_low_ceiling = 10**9
        self._height_floor = self.robot.healthy_height_range[0] * self.cfg.termination.bad_height_scale
        self._orientation_limit = self.cfg.termination.orientation_limit
        self._best_distance = float("inf")
        self._feet_air_time = np.zeros(len(self._foot_body_ids))
        self._feet_contact_time = np.zeros(len(self._foot_body_ids))
        self._feet_in_contact = np.zeros(len(self._foot_body_ids), dtype=bool)
        self._last_touchdown_foot = -1
        self._last_step_length = np.full(len(self._foot_body_ids), np.nan)
        self._next_push_step = -1
        self._pending_push = np.zeros(2)
        self._last_push: tuple[int, np.ndarray] | None = None
        self._zero_command = False
        self._action_buffer = deque(
            [np.zeros(self.model.nu)] * max(1, self.cfg.events.action_delay_steps + 1),
            maxlen=max(1, self.cfg.events.action_delay_steps + 1),
        )
        self._scan_buffer = deque(maxlen=max(1, self.cfg.observation.exteroceptive_delay + 1))
        self._scan_hits: np.ndarray | None = None
        self._scan_clipped = False
        self._ceiling_seen = False
        self._overhead_buffer = deque(maxlen=max(1, self.cfg.observation.exteroceptive_delay + 1))
        self._obs_history = deque(maxlen=max(1, self.cfg.observation.history_length))

        self._rebind_renderer()

    def _rebind_renderer(self) -> None:
        """Point an open renderer at the rebuilt model instead of closing it.

        Closing destroys the viewer window and terminates GLFW, and the next render initialises
        both again, which on macOS could leave the screen black after a reset (Procedural
        rebuilds on every one). The window, its camera and the viewer options survive; only
        the scene and the GPU context are made again for the new model.
        """
        renderer = self._mujoco_renderer
        if renderer is None:
            return
        renderer.model, renderer.data = self.model, self.data
        for viewer in renderer._viewers.values():
            viewer.make_context_current()
            viewer.con.free()
            viewer.model, viewer.data = self.model, self.data
            viewer.scn = mujoco.MjvScene(self.model, viewer.scn.maxgeom)
            viewer.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            viewer._set_mujoco_buffer()

    def _resolve_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3).copy()
        if not self.terrain.resolve_waypoint_z:
            return waypoints
        for i, point in enumerate(waypoints):
            dist = mdp.ground_ray(self.model, self.data, (point[0], point[1], 6.0), self._ceiling_geom_ids)
            waypoints[i, 2] = (6.0 - dist) + point[2] if dist >= 0 else point[2]
        return waypoints

    # ------------------------------------------------------------------- gym api

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        resample = self.cfg.terrain.resample_every_n_resets
        if (self._reset_count == 0 and seed is not None) or (
            self._reset_count > 0 and resample and self._reset_count % resample == 0
        ) or self.terrain.difficulty != self._difficulty or self._built_skill_difficulty != self._skill_difficulty:
            if self.cfg.terrain.kind == "flat" and self.terrain.difficulty == self._difficulty:
                self.terrain = build_terrain("flat", rng, difficulty=self._difficulty,
                                             **self.cfg.terrain.params)
                self._waypoints = self._resolve_waypoints(self.terrain.waypoints)
                self._waypoint_speeds = np.zeros(len(self._waypoints))
                self._min_course_z = float(self._waypoints[:, 2].min())
            else:
                self._build_world(rng)
        self._reset_count += 1

        mdp.randomize_model(self.model, rng, self.cfg.events, self._nominal)

        events = self.cfg.events
        qpos = self._init_qpos.copy()
        qvel = self._init_qvel.copy()
        noise = events.reset_noise_scale
        qpos += rng.uniform(-noise, noise, size=self.model.nq)
        qvel += rng.uniform(-noise, noise, size=self.model.nv)
        # Terrain parts start where their module put them, at rest.
        qpos[self._terrain_qpos_adr] = self._init_qpos[self._terrain_qpos_adr]
        qvel[self._terrain_dof_adr] = 0.0

        start_index = 0
        if events.reset_along_course and len(self._waypoints) > 2:
            start_index = int(rng.integers(0, len(self._waypoints) - 1))
            target = self._waypoints[start_index]
            qpos[0:3] = (target[0], target[1], target[2] + self.robot.spawn_height)
            start_index += 1
        if events.reset_yaw_range:
            yaw = rng.uniform(-events.reset_yaw_range, events.reset_yaw_range)
            qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.xfrc_applied[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self._stable_eye.time = None  # the view snaps to the new start instead of gliding there
        if self._water is not None:
            self._water.applying = False
        for actor in self._actors:
            actor.reset(self)
        mujoco.mj_forward(self.model, self.data)
        self._settle_feet_on_ground()

        self._target_index = start_index
        self._start_target_index = start_index
        self._vertical_debt = 0.0
        self._module_results = []
        self._episode_credited = False
        self._zero_command = False
        if self.cfg.command_speed_range is not None:
            shift = self.cfg.command_speed_difficulty_shift * float(np.clip(self.terrain.difficulty, 0.0, 1.0))
            low, high = self.cfg.command_speed_range
            self._waypoint_speeds = rng.uniform(low + shift, high + shift, size=len(self._waypoints))
            boost = self.terrain.waypoint_speed_boost
            if len(boost) == len(self._waypoint_speeds):
                self._waypoint_speeds += boost  # a module's sprint, such as a run
            if self.cfg.zero_command_prob and rng.random() < self.cfg.zero_command_prob:
                self._waypoint_speeds[:] = 0.0
                self._zero_command = True
        self._velocity_error_sum = 0.0
        self._episode_speed_sum = 0.0
        self._episode_height_sum = 0.0
        self._episode_height_samples = 0
        self._goal_reached = False
        self._step_count = 0
        self._waypoints_reached = 0
        self._prev_action = np.zeros(self.model.nu)
        self._action_buffer.clear()
        for _ in range(self._action_buffer.maxlen):
            self._action_buffer.append(np.zeros(self.model.nu))
        self._scan_buffer.clear()
        self._overhead_buffer.clear()
        self._obs_history.clear()
        self._best_distance = self._distance_to_target()
        self._stall_counter = 0
        self._bad_height_steps = 0
        self._bad_orientation_steps = 0
        self._steps_since_ceiling = 10**9
        self._steps_since_low_ceiling = 10**9
        self._height_floor = self.robot.healthy_height_range[0] * self.cfg.termination.bad_height_scale
        self._orientation_limit = self.cfg.termination.orientation_limit
        self._feet_air_time[:] = 0.0
        self._feet_contact_time[:] = 0.0
        self._feet_in_contact[:] = False
        self._last_touchdown_foot = -1
        self._last_step_length[:] = np.nan
        self._last_push = None
        if self.cfg.events.push_robot:
            self._next_push_step, self._pending_push = mdp.schedule_push(
                rng, self.cfg.events, 0, self.terrain.difficulty,
            )
        self._episode_terms = {}

        obs = self._compute_observation()
        return obs, self._info(reward_terms={})

    def _settle_feet_on_ground(self) -> None:
        """Drop/raise the root so the lowest foot rests just above the terrain under it."""
        clearance = []
        for geom_ids in self._foot_geom_ids:
            for geom_id in geom_ids:
                bottom = float(self.data.geom_xpos[geom_id, 2] - self.model.geom_rbound[geom_id])
                surface = self._terrain_height_below(self.data.geom_xpos[geom_id, :2], bottom + 2.0)
                if np.isfinite(surface):
                    clearance.append(bottom - surface)
        if clearance:
            self.data.qpos[2] -= min(clearance) - 0.01
            mujoco.mj_forward(self.model, self.data)

    def _apply_action(self, applied: np.ndarray) -> None:
        """Advance physics by one control step under the configured actuation mode."""
        if self._water is not None:
            # Held for the control step's substeps: water forces change far more slowly than
            # the PD torques, and once per step costs a fifth as much.
            self._water.apply(self.data)
        low, high = self._ctrl_range[:, 0], self._ctrl_range[:, 1]
        if self.cfg.control_mode == "torque":
            self.data.ctrl[:] = low + (applied + 1.0) * 0.5 * (high - low)
            mujoco.mj_step(self.model, self.data, nstep=self.cfg.frame_skip)
            return
        target = self._default_act_pos + self.cfg.position_action_scale * applied
        target = np.where(
            self._act_limited,
            np.clip(target, self._act_joint_range[:, 0], self._act_joint_range[:, 1]),
            target,
        )
        gear = self.model.actuator_gear[:, 0]
        for _ in range(self.cfg.frame_skip):
            q = self.data.qpos[self._act_qpos_adr]
            qd = self.data.qvel[self._act_dof_adr]
            torque = self._pd_kp * (target - q) - self._pd_kd * qd
            self.data.ctrl[:] = np.clip(torque / gear, low, high)
            mujoco.mj_step(self.model, self.data, nstep=1)

    def _foot_contacts(self) -> np.ndarray:
        """Per-foot flag for contact with anything other than the robot itself."""
        contacts = np.zeros(len(self._foot_geom_ids), dtype=bool)
        for contact in self.data.contact[: self.data.ncon]:
            for index, geom_ids in enumerate(self._foot_geom_ids):
                if contact.geom1 in geom_ids and contact.geom2 not in geom_ids:
                    contacts[index] = True
                elif contact.geom2 in geom_ids and contact.geom1 not in geom_ids:
                    contacts[index] = True
        return contacts

    def _stumbling(self) -> bool:
        """Whether a foot or shin is pressing on a near-vertical terrain face."""
        groups = self.model.geom_group
        for contact in self.data.contact[: self.data.ncon]:
            if abs(contact.frame[2]) > SUPPORT_NORMAL_Z:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in self._leg_geom_ids and groups[g2] == 1) or (g2 in self._leg_geom_ids and groups[g1] == 1):
                return True
        return False

    EDGE_SAMPLE_DIRECTIONS = np.stack(
        [np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], axis=1,
    )

    def _feet_at_edge(self) -> int:
        """Number of feet whose support centre is within ``feet_edge_margin`` of a drop deeper than
        ``feet_edge_drop``.

        The support centre is the mean of the foot's supporting contact points. For the
        humanoid's sphere foot that is its one contact point; for a flat sole touching at heel and
        toe it is mid-sole, so a toe over a stair nose with the foot well placed does not count,
        while a sole centred near the nose has about half its length unsupported, the "more
        than half the sole unsupported" rule of humanoid stair work. The ground is sampled at
        eight points ``feet_edge_margin`` around the centre; a step up beside the foot, a seam
        between flush platforms or a gentle slope is not an edge.
        """
        reward = self.cfg.reward
        groups = self.model.geom_group
        count = 0
        for geom_ids in self._foot_geom_ids:
            points = [
                contact.pos.copy()
                for contact in self.data.contact[: self.data.ncon]
                if abs(contact.frame[2]) > SUPPORT_NORMAL_Z and (
                    (contact.geom1 in geom_ids and groups[contact.geom2] == 1)
                    or (contact.geom2 in geom_ids and groups[contact.geom1] == 1)
                )
            ]
            if not points:
                continue
            centre = np.mean(points, axis=0)[:2]
            surface = max(float(point[2]) for point in points)
            samples = centre + reward.feet_edge_margin * self.EDGE_SAMPLE_DIRECTIONS
            if any(
                self._terrain_height_below(xy, surface + 1.0) < surface - reward.feet_edge_drop
                for xy in samples
            ):
                count += 1
        return count

    def _update_gait_state(self, travel_direction: np.ndarray | None = None) -> tuple[float, float, float, float]:
        """Advance per-foot swing/stance clocks; return (single-stance mode time, step-ahead, slide speed,
        step asymmetry).

        Steps are measured along ``travel_direction`` (unit vector toward the active waypoint), or the
        torso heading when none is given. Measuring on the torso heading let a gallop turn its body
        so the trailing foot's catch-up step projects to roughly zero instead of landing behind.
        """
        in_contact = self._foot_contacts()
        touchdown = np.flatnonzero(in_contact & ~self._feet_in_contact)
        self._feet_in_contact = in_contact
        self._feet_contact_time = np.where(in_contact, self._feet_contact_time + self.dt, 0.0)
        self._feet_air_time = np.where(in_contact, 0.0, self._feet_air_time + self.dt)
        mode_time = np.where(in_contact, self._feet_contact_time, self._feet_air_time)
        single_stance = int(in_contact.sum()) == 1
        air_time_reward = float(np.min(mode_time)) if single_stance else 0.0
        step_ahead = 0.0
        step_asymmetry = 0.0
        if len(touchdown) == 1 and len(self._foot_body_ids) == 2:
            foot = int(touchdown[0])
            if self._last_touchdown_foot not in (-1, foot):
                direction = mdp.quat_to_mat(self.root_quat)[:2, 0] if travel_direction is None else travel_direction
                offset = self.data.xpos[self._foot_body_ids[foot], :2] - self.data.xpos[self._foot_body_ids[1 - foot], :2]
                step = float(direction @ offset)
                step_ahead = float(np.clip(step / self.cfg.reward.feet_step_ahead_margin, -1.0, 1.0))
                if np.isfinite(self._last_step_length[1 - foot]):
                    step_asymmetry = abs(step - self._last_step_length[1 - foot])
                self._last_step_length[foot] = step
            self._last_touchdown_foot = foot
        slide = 0.0
        velocity = np.zeros(6)
        for index, body_id in enumerate(self._foot_body_ids):
            if in_contact[index]:
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
                slide += float(np.linalg.norm(velocity[3:5]))
        return air_time_reward, step_ahead, slide, step_asymmetry

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._action_buffer.append(action)
        applied = self._action_buffer[0]

        if self.cfg.events.push_robot and self._step_count == self._next_push_step:
            mdp.push(self.data, self._pending_push)
            self._last_push = (self._step_count, self._pending_push.copy())
            self._next_push_step, self._pending_push = mdp.schedule_push(
                self.np_random, self.cfg.events, self._step_count, self.terrain.difficulty,
            )

        for actor in self._actors:
            actor.before_step(self)

        active_target = self._target().copy()
        active_speed = self.command_speed
        command_origin = self.root_pos.copy()
        distance_before = self._distance_to_target()
        # vertical_progress: the gap below standing height over the active waypoint, before and
        # after the step at the same target and standing height, as progress measures distance.
        standing = self._vertical_gap_standing() if self.cfg.reward.vertical_progress else None
        deadband = self.cfg.reward.vertical_progress_deadband
        gap_before = (max(0.0, active_target[2] + standing - self.root_pos[2] - deadband)
                      if standing is not None else 0.0)
        self._apply_action(applied)
        self._step_count += 1

        distance_after = float(np.linalg.norm(active_target[:2] - self.root_pos[:2]))
        progress = distance_before - distance_after
        vertical_rate = 0.0
        if standing is not None:
            gap_after = max(0.0, active_target[2] + standing - self.root_pos[2] - deadband)
            vertical_rate = (gap_before - gap_after) / self.dt
            # Drops not yet climbed back: what an ending repays (the gap a higher waypoint opens
            # is never charged, so it is never repaid either).
            self._vertical_debt = max(0.0, self._vertical_debt + gap_after - gap_before)
        reached = self._advance_waypoint()

        reward, reward_terms, terminated, reason = self._evaluate(
            action, progress, reached, active_target=active_target, target_speed=active_speed,
            command_origin=command_origin, vertical_rate=vertical_rate,
        )
        truncated = self._step_count >= self.max_episode_steps
        if terminated and self._vertical_debt > 0.0:
            # An ending repays the drops charged and not climbed back, so a fall nets zero here
            # instead of a second fall penalty (~-50 for a torso dropping to the floor). A timeout
            # is no ending: training bootstraps its value.
            refund = self.cfg.reward.vertical_progress * self._vertical_debt / self.dt
            reward_terms["vertical_progress"] = reward_terms.get("vertical_progress", 0.0) + refund
            reward += refund
        obs = self._compute_observation()

        for key, value in reward_terms.items():
            self._episode_terms[key] = self._episode_terms.get(key, 0.0) + value

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, self._info(
            reward_terms, reason or ("timeout" if truncated else "")
        )

    def render(self):
        if self.render_mode is None:
            return None
        if self._mujoco_renderer is None:
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

            self._mujoco_renderer = MujocoRenderer(
                self.model,
                self.data,
                self._camera_config,
                self._render_width,
                self._render_height,
                camera_name=self._camera_name,
            )
        if self.render_mode == "human":
            viewer = self._mujoco_renderer._get_viewer(self.render_mode)
            if not self._viewer_camera_set:
                viewer.vopt.geomgroup[4] = 1
            viewer._add_marker_to_scene = lambda marker: self._add_waypoint_marker_to_scene(viewer, marker)
            for index, waypoint in enumerate(self._waypoints):
                completed = index < self._target_index or self._goal_reached
                active = index == self._target_index and not completed
                color = (0.2, 0.9, 0.3, 0.9) if completed else (
                    (1.0, 0.8, 0.0, 1.0) if active else (0.0, 0.8, 1.0, 0.8)
                )
                radius = 0.22 if active else 0.14
                label = ""
                if active:
                    label = f"WP {index + 1} (target)"
                    if self.cfg.command_speed_range is not None:
                        label += f" | {self._waypoint_speeds[index]:.2f} m/s"
                viewer.add_marker(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    pos=waypoint + np.array([0.0, 0.0, 0.3]),
                    size=np.full(3, radius),
                    rgba=color,
                    emission=0.6,
                    label=label,
                )
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_LABEL,
                pos=self.root_pos + np.array([0.0, 0.0, 0.3]),
                size=np.zeros(3),
                rgba=(1.0, 1.0, 1.0, 1.0),
                emission=0.0,
                label=f"Speed: {np.linalg.norm(self.data.qvel[:2]):.2f} m/s",
            )
            for marker in self._push_markers():
                viewer.add_marker(**marker)
        if (self.terrain.water or self._actors) and self.render_mode in ("human", "rgb_array"):
            viewer = self._mujoco_renderer._get_viewer(self.render_mode)
            viewer._add_marker_to_scene = lambda marker: self._add_waypoint_marker_to_scene(viewer, marker)
            if self.terrain.water and self._water_effects is None:
                self._water_effects = WaterEffects(self.model, self.terrain.water, self._root_body_id)
            markers = self._water_effects.markers(self.data) if self._water_effects is not None else []
            for actor in self._actors:
                markers += actor.markers(self)
            for marker in markers:
                viewer.add_marker(**marker)
        self._place_eye_camera()
        frame = self._mujoco_renderer.render(self.render_mode)
        if self.render_mode != "human" and self._mujoco_renderer._viewers.get(self.render_mode):
            # Gymnasium's window viewer drops its markers after each frame, the offscreen one
            # keeps them: without this they pile up to the scene's geom limit.
            self._mujoco_renderer._viewers[self.render_mode]._markers.clear()
        if self.render_mode == "human" and not self._viewer_camera_set:
            # Gymnasium's window viewer ignores camera_name for human rendering (it always
            # starts on the free camera), so point its interactive camera at the named fixed
            # camera once. TAB still cycles cameras afterwards.
            viewer = self._mujoco_renderer.viewer
            if self._camera_name:
                camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name)
                if camera_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = camera_id
            # Hide the on-screen help/overlay menu by default (same as pressing H).
            viewer._hide_menu = True
            self._viewer_camera_set = True
        return frame

    def _gaze_cues(self) -> dict[str, bool]:
        """The first-person view's cues, from the scans the policy just acted on: look down at a
        magenta (clipped) ground reading or ground more than ``LOOK_DOWN_RISE`` above the floor,
        up at a ceiling, and glance only while the scanned ground is flat."""
        floor = self._floor_z()
        hits = self._scan_hits
        rise = self._obstacle_ahead(floor, window=(-np.inf, np.inf), lateral=np.inf) if np.isfinite(floor) else 0.0
        flat = (
            hits is not None and not self._scan_clipped and not self._ceiling_seen
            and float(np.ptp(hits[:, 2])) <= StableEyeCamera.FLAT_TOLERANCE
        )
        return dict(
            look_down=self._scan_clipped or rise > StableEyeCamera.LOOK_DOWN_RISE,
            look_up=self._ceiling_seen, glance=flat,
        )

    def _place_eye_camera(self) -> None:
        """Put the first-person camera at the steadied eye (``StableEyeCamera``), whichever camera
        the viewer shows, so TAB finds it in place. Writes the camera's pose for this frame and
        its mocap mount for the next forward pass; nothing else in the simulation changes."""
        mount = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EYE_MOUNT)
        camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, STABLE_EGOCENTRIC)
        if mount < 0 or camera < 0:
            return
        lookat, distance, azimuth, elevation = self._stable_eye.update(
            self.data, self._root_body_id, self._course_heading(), **self._gaze_cues()
        )
        frame = camera_frame(azimuth, elevation, self._stable_eye.roll)
        eye = lookat - distance * -frame[:, 2]
        self.data.cam_xpos[camera] = eye
        self.data.cam_xmat[camera] = frame.ravel()
        mocap = self.model.body_mocapid[mount]
        quat = np.zeros(4)
        # The mount's frame is the camera's without its fixed xyaxes turn (look +x, up +z).
        mujoco.mju_mat2Quat(quat, (frame @ self.model.cam_mat0[camera].reshape(3, 3).T).ravel())
        self.data.mocap_pos[mocap], self.data.mocap_quat[mocap] = eye, quat

    PUSH_PREVIEW_STEPS = 20
    PUSH_BALL_SPEED_GAIN = 20.0
    """Incoming ball speed per m/s of push, so stronger pushes fly in faster."""
    PUSH_BALL_SPAWN_DISTANCE_GAIN = 10.0
    """Ball spawn distance beyond the impact point per m/s of push, so stronger pushes start farther out."""
    PUSH_BALL_IMPACT_DISTANCE = 0.1

    def _push_markers(self, schedule: tuple | None = None, origin: np.ndarray | None = None) -> list[dict]:
        """A ball flying in before a scheduled push and an arrow showing the kick just after it.

        ``schedule`` is ``(next_push_step, pending_push, last_push)``, the random pushes' own by
        default; a terrain module that pushes, like the wind turbine, passes its own and the
        ``origin`` its balls fly from instead of straight along the push.
        """
        if schedule is None:
            if not self.cfg.events.push_robot:
                return []
            schedule = (self._next_push_step, self._pending_push, self._last_push)
        next_push_step, pending_push, last_push = schedule
        markers = []
        incoming = next_push_step - self._step_count
        pending = float(np.linalg.norm(pending_push))
        # Both speed and spawn distance scale with the push, so their ratio sets how long the ball is in flight.
        flight_time = self.PUSH_BALL_SPAWN_DISTANCE_GAIN / self.PUSH_BALL_SPEED_GAIN
        time_to_impact = incoming * self.dt
        if 0 < incoming and time_to_impact <= flight_time + 1e-9 and pending > 1e-9:
            direction = np.append(pending_push / pending, 0.0)
            if origin is None:
                # Constant approach speed scaled by the push, arriving exactly on the push step.
                speed = self.PUSH_BALL_SPEED_GAIN * pending
                position = self.root_pos - direction * (self.PUSH_BALL_IMPACT_DISTANCE + speed * time_to_impact)
            else:
                impact = self.root_pos - direction * self.PUSH_BALL_IMPACT_DISTANCE
                position = impact + (np.asarray(origin) - impact) * (time_to_impact / flight_time)
            markers.append(dict(
                type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=position,
                size=np.full(3, 0.08), rgba=(1.0, 0.3, 0.1, 0.9), emission=0.5, label="",
            ))
        if last_push is not None and self._step_count - last_push[0] <= self.PUSH_PREVIEW_STEPS:
            kick = last_push[1]
            magnitude = float(np.linalg.norm(kick))
            if magnitude > 1e-9:
                z_axis = np.append(kick / magnitude, 0.0)
                y_axis = np.array([0.0, 0.0, 1.0])
                x_axis = np.cross(y_axis, z_axis)
                markers.append(dict(
                    type=mujoco.mjtGeom.mjGEOM_ARROW, pos=self.root_pos.copy(),
                    size=np.array([0.03, 0.03, 0.4 + magnitude]),
                    mat=np.column_stack([x_axis, y_axis, z_axis]).ravel(),
                    rgba=(1.0, 0.3, 0.1, 0.9), emission=0.5, label=f"Push {magnitude:.2f} m/s",
                ))
        return markers

    @staticmethod
    def _add_waypoint_marker_to_scene(viewer, marker):
        """Draw a marker if its geom group (4, the waypoints', unless it names another) is shown."""
        if not viewer.vopt.geomgroup[marker.get("group", 4)] or viewer.scn.ngeom >= viewer.scn.maxgeom:
            return
        geom = viewer.scn.geoms[viewer.scn.ngeom]
        mujoco.mjv_initGeom(
            geom, marker["type"], marker["size"], marker["pos"],
            np.asarray(marker.get("mat", np.eye(3).ravel()), dtype=np.float64),
            np.asarray(marker["rgba"], dtype=np.float32),
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.objid = -1
        geom.emission = marker["emission"]
        geom.label = marker["label"]
        viewer.scn.ngeom += 1

    def close(self):
        if self._mujoco_renderer is not None:
            self._mujoco_renderer.close()
            self._mujoco_renderer = None

    # ------------------------------------------------------------------ internals

    @property
    def root_pos(self) -> np.ndarray:
        return self.data.qpos[0:3]

    @property
    def root_quat(self) -> np.ndarray:
        return self.data.qpos[3:7]

    def _target(self, offset: int = 0) -> np.ndarray:
        index = min(self._target_index + offset, len(self._waypoints) - 1)
        return self._waypoints[index]

    COURSE_HEADING_BASELINE = 3.0
    """Metres of course ahead that ``_course_heading`` spans."""

    def _course_heading(self) -> float:
        """Direction (radians) of the course ahead: from the previous waypoint (or the spawn
        point) to the first waypoint at least ``COURSE_HEADING_BASELINE`` beyond it. A long
        baseline keeps waypoints that sit off the centre line, like stepping stones' up to
        0.2 m apart sideways, from swinging the direction from one to the next."""
        start = np.asarray(
            self._waypoints[self._target_index - 1] if self._target_index > 0 else self.terrain.spawn_pos
        )[:2]
        end = self._target()[:2]
        for waypoint in self._waypoints[self._target_index:]:
            end = waypoint[:2]
            if np.linalg.norm(end - start) >= self.COURSE_HEADING_BASELINE:
                break
        delta = end - start
        return float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-6 else 0.0

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self._target()[:2] - self.root_pos[:2]))

    @property
    def command_speed(self) -> float:
        return float(self._waypoint_speeds[min(self._target_index, len(self._waypoint_speeds) - 1)])

    def _advance_waypoint(self) -> bool:
        if self._target_index >= len(self._waypoints) - 1:
            reached = (
                self._distance_to_target() < self.cfg.waypoint_radius
                and self.root_pos[2] - self._target()[2] >= self._height_floor
                and -mdp.projected_gravity(self.root_quat)[2] >= self.cfg.termination.orientation_limit
            )
            if reached and not self._goal_reached:
                self._waypoints_reached += 1
            self._goal_reached = reached
            return reached
        if self._distance_to_target() < self.cfg.waypoint_radius:
            module = self._waypoint_module(self._target_index)
            if module > 0 and self._waypoint_module(self._target_index + 1) != module:
                self._module_results.append((self.terrain.modules[module], True, self._module_kind(module)))
            self._target_index += 1
            self._waypoints_reached += 1
            self._best_distance = self._distance_to_target()
            self._stall_counter = 0
            return True
        return False

    def _action_noise_scale(self) -> float:
        """Training's action-noise scale for the module the active waypoint belongs to
        (``TerrainCfg.skill_action_noise``); 1 off modules or for skills it leaves out."""
        scales = self.cfg.terrain.skill_action_noise
        module = self._waypoint_module(self._target_index) if scales else -1
        return float(scales.get(self.terrain.modules[module], 1.0)) if module > 0 else 1.0

    def _module_kind(self, module: int) -> str:
        """``"level"``, ``"review"`` or ``"preview"`` for module index ``module`` (``level`` when unknown)."""
        kinds = self.terrain.module_kinds
        return kinds[module] if 0 <= module < len(kinds) else "level"

    def _waypoint_module(self, index: int) -> int:
        """Index into ``terrain.modules`` of the module waypoint ``index`` belongs to: 0 for the
        start pad, -1 for the finish or a course without modules (or without per-skill levels)."""
        owners = self.terrain.waypoint_modules
        if not self._skill_difficulty or len(owners) != len(self._waypoints) or not 0 <= index < len(owners):
            return -1
        return int(owners[index])

    def _terrain_height_below(self, xy: np.ndarray, from_z: float) -> float:
        dist = mdp.ground_ray(self.model, self.data, (xy[0], xy[1], from_z), self._ceiling_geom_ids)
        return from_z - dist if dist >= 0 else -np.inf

    def _evaluate(self, action, progress, reached, *, active_target=None, target_speed=None,
                  command_origin=None, vertical_rate: float = 0.0):
        cfg = self.cfg
        gravity = mdp.projected_gravity(self.root_quat)
        upright = float(-gravity[2])
        rot = mdp.quat_to_mat(self.root_quat)
        lin_vel_local = rot.T @ self.data.qvel[0:3]
        ang_vel_local = self.data.qvel[3:6]

        target = self._target() if active_target is None else active_target
        speed = self.command_speed if target_speed is None else target_speed
        origin = self.root_pos if command_origin is None else command_origin
        to_target = target[:2] - origin[:2]
        norm = np.linalg.norm(to_target)
        heading_alignment = float(rot[:2, 0] @ (to_target / norm)) if norm > 1e-6 else 0.0
        desired_velocity = speed * to_target / norm if norm > 1e-6 else np.zeros(2)
        velocity_error_squared = float(np.sum(np.square(self.data.qvel[:2] - desired_velocity)))
        self._velocity_error_sum += np.sqrt(velocity_error_squared)

        surface_z = self._terrain_height_below(self.root_pos[:2], self.root_pos[2] + 0.1)
        height_above_terrain = self.root_pos[2] - surface_z if np.isfinite(surface_z) else 99.0
        self._episode_speed_sum += float(np.linalg.norm(self.data.qvel[:2]))
        if np.isfinite(surface_z):
            self._episode_height_sum += float(height_above_terrain)
            self._episode_height_samples += 1

        if self._hinge_jnt_ids.size:
            qpos = self.data.qpos[self._hinge_qpos_adr]
            low = self._joint_range[self._hinge_jnt_ids, 0]
            high = self._joint_range[self._hinge_jnt_ids, 1]
            violation = float(
                np.sum(np.clip(low - qpos, 0, None) + np.clip(qpos - high, 0, None))
            )
        else:
            violation = 0.0

        torque_power = float(np.sum(np.abs(self.data.actuator_force * self.data.actuator_velocity)))

        distance = self._distance_to_target()
        if distance < self._best_distance - cfg.termination.stall_distance:
            self._best_distance = distance
            self._stall_counter = 0
        elif self._zero_command:
            self._stall_counter = 0  # standing still is the task in zero-command episodes
        elif any(actor.holds_stall(self) for actor in self._actors):
            pass  # waiting is the task here, as for a moving platform to arrive
        else:
            self._stall_counter += 1

        reached_goal = self._goal_reached and reached
        self._height_floor = self._update_height_floor(surface_z)
        low_height = height_above_terrain < self._height_floor
        tilted = upright < self._orientation_limit
        self._bad_height_steps = self._bad_height_steps + 1 if low_height else 0
        self._bad_orientation_steps = self._bad_orientation_steps + 1 if tilted else 0
        air_time_reward, step_ahead, slide_speed, step_asymmetry = self._update_gait_state(
            to_target / norm if norm > 1e-6 else None
        )
        height_error = self._height_reward_error(surface_z)
        terminated, reason = mdp.check(
            cfg.termination,
            height_above_terrain=height_above_terrain,
            healthy_height_range=self.robot.healthy_height_range,
            upright=upright,
            position=tuple(self.root_pos),
            min_course_z=self._min_course_z,
            stall_counter=self._stall_counter,
            bad_height_steps=self._bad_height_steps,
            bad_orientation_steps=self._bad_orientation_steps,
            height_floor=self._height_floor,
            orientation_limit=self._orientation_limit,
        )
        if not terminated:  # a module's own rule, like a sprint's clock
            for actor in self._actors:
                reason = actor.terminates(self)
                if reason:
                    terminated = True
                    break
        fell = terminated and reason in ("bad_height", "bad_orientation", "fell_off_course")
        reached_goal = reached_goal and not terminated

        terms = mdp.terms(
            cfg.reward,
            dt=self.dt,
            progress=progress,
            heading_alignment=heading_alignment,
            upright=upright,
            height_error=height_error,
            height_deficit=max(0.0, -height_error),
            vertical_rate=vertical_rate,
            lateral_speed=float(lin_vel_local[1]),
            ang_vel=ang_vel_local,
            action=action,
            prev_action=self._prev_action,
            torque_power=torque_power,
            joint_limit_violation=violation,
            waypoints_reached=int(reached and not reached_goal),
            reached_goal=reached_goal,
            fell=fell,
            target_speed=speed if cfg.command_speed_range is not None else None,
            velocity_error_squared=velocity_error_squared,
            feet_air_time=air_time_reward,
            feet_step_ahead=step_ahead,
            feet_step_asymmetry=step_asymmetry,
            feet_slide_speed=slide_speed,
            stumbling=bool(cfg.reward.feet_stumble) and self._stumbling(),
            feet_at_edge=self._feet_at_edge() if cfg.reward.feet_edge else 0,
            feet_clearance=self._feet_clearance(surface_z, lin_vel_local[0]) if cfg.reward.feet_clearance else 0.0,
            ceiling_overlap=self._ceiling_overlap() if cfg.reward.ceiling_clearance else 0.0,
            ceiling_touch=self._touching_ceiling() if cfg.reward.ceiling_contact and self._ceiling_geom_ids else False,
            supported=not (low_height or tilted),
        )
        self._prev_action = action.copy()
        if reached_goal:
            terminated = True
            reason = "goal"
        return float(sum(terms.values())), terms, terminated, reason

    def _support_surface_z(self) -> float | None:
        """Height of the highest terrain contact holding the robot up, or None when airborne.

        Ceilings are not support: an arm pressing on a tunnel bar's underside made the posture
        reward read the floor at bar height and ask the robot to stand up into it.
        """
        groups = self.model.geom_group
        heights = [
            float(contact.pos[2])
            for contact in self.data.contact
            if contact.geom1 >= 0 and contact.geom2 >= 0
            and {int(groups[contact.geom1]), int(groups[contact.geom2])} == {0, 1}
            and contact.geom1 not in self._ceiling_geom_ids and contact.geom2 not in self._ceiling_geom_ids
            and contact.dist <= 0.0 and abs(contact.frame[2]) > SUPPORT_NORMAL_Z
            and contact.pos[2] < self.root_pos[2]
        ]
        return max(heights) if heights else None

    def _obstacle_ahead(self, floor_z: float, window: tuple[float, float] | None = None,
                        lateral: float = 0.2) -> float:
        """Height above ``floor_z`` of the highest ground the latest scan saw from ``window[0]``
        to ``window[1]`` m ahead (``feet_clearance_window`` by default) and within ``lateral``
        m to either side; 0 without a scan."""
        hits = self._scan_hits
        if hits is None:
            return 0.0
        points = self.cfg.observation.scan_points
        low, high = self.cfg.reward.feet_clearance_window if window is None else window
        near = (points[:, 0] >= low - 1e-9) & (points[:, 0] <= high + 1e-9) & (np.abs(points[:, 1]) <= lateral + 1e-9)
        return max(0.0, float(hits[near, 2].max()) - floor_z) if near.any() else 0.0

    def _floor_z(self, fallback: float | None = None) -> float:
        """Height of the floor the robot stands on: its highest supporting contact, or when
        airborne ``fallback`` (by default the terrain below the root)."""
        support = self._support_surface_z()
        if support is not None:
            return support
        if fallback is not None:
            return fallback
        return self._terrain_height_below(self.root_pos[:2], self.root_pos[2] + 0.1)

    def _feet_clearance(self, surface_z: float, forward_speed: float) -> float:
        """Summed swing-foot lift as a share of the obstacle ahead plus a margin (see
        ``RewardCfg.feet_clearance``); 0 without an obstacle ahead or forward motion."""
        reward = self.cfg.reward
        floor_z = self._floor_z(surface_z)
        if not np.isfinite(floor_z) or forward_speed < 0.2:
            return 0.0
        obstacle = self._obstacle_ahead(floor_z)
        if obstacle < reward.feet_clearance_min_obstacle:
            return 0.0
        target = obstacle + reward.feet_clearance_margin
        total = 0.0
        for foot, geoms in enumerate(self._foot_geom_ids):
            if self._feet_in_contact[foot]:
                continue
            bottom = min(float(self.data.geom_xpos[g, 2] - self.model.geom_rbound[g]) for g in geoms)
            total += float(np.clip((bottom - floor_z) / target, 0.0, 1.0))
        return total

    def _update_height_floor(self, surface_z: float) -> float:
        """Lowest healthy root height above the terrain this step, for the height termination
        and the progress reward's support gate alike: ``ceiling_bad_height_scale`` while the
        upward rays see a ceiling and for ``ceiling_relief_steps`` after, else ``bad_height_scale``.

        Under a crawl-height ceiling (``crawl_ceiling_clearance``, an air vent) and as long after,
        ``crawl_bad_height_scale`` applies instead and ``_orientation_limit`` is relaxed to
        ``crawl_orientation_limit``, for the tilt termination and the support gate alike.
        """
        termination = self.cfg.termination
        scale = termination.bad_height_scale
        self._orientation_limit = termination.orientation_limit
        crawl = termination.crawl_bad_height_scale is not None or termination.crawl_orientation_limit is not None
        if termination.ceiling_bad_height_scale is None and not crawl:
            return self.robot.healthy_height_range[0] * scale
        ceiling = self._ceiling_z(surface_z) if np.isfinite(surface_z) else None
        self._steps_since_ceiling = 0 if ceiling is not None else self._steps_since_ceiling + 1
        low = ceiling is not None and ceiling - surface_z <= termination.crawl_ceiling_clearance
        self._steps_since_low_ceiling = 0 if low else self._steps_since_low_ceiling + 1
        if termination.ceiling_bad_height_scale is not None and self._steps_since_ceiling <= termination.ceiling_relief_steps:
            scale = termination.ceiling_bad_height_scale
        if self._steps_since_low_ceiling <= termination.ceiling_relief_steps:
            if termination.crawl_bad_height_scale is not None:
                scale = termination.crawl_bad_height_scale
            if termination.crawl_orientation_limit is not None:
                self._orientation_limit = termination.crawl_orientation_limit
        return self.robot.healthy_height_range[0] * scale

    def _ceiling_z(self, surface_z: float) -> float | None:
        """World height of the lowest ceiling the upward rays see from ``surface_z``, or None."""
        observation = self.cfg.observation
        origin = self.root_pos.copy()
        origin[2] = surface_z + observation.overhead_origin_height
        depths, _ = mdp.overhead_scan(
            self.model, self.data, origin, mdp.yaw_from_quat(self.root_quat), observation,
        )
        if not len(depths) or np.min(depths) >= observation.overhead_max_depth:
            return None
        return float(origin[2] + np.min(depths))

    def _robot_top_z(self) -> float:
        """Highest point of the robot: exact for spheres and capsules, bounding sphere otherwise.

        A capsule's bounding sphere reaches its half-length above its centre whatever its
        angle, which put a level forearm ~0.1 m higher than it is.
        """
        robot = np.flatnonzero(self.model.geom_group == 0)
        size = self.model.geom_size[robot]
        top = self.data.geom_xpos[robot, 2] + self.model.geom_rbound[robot]
        capsule = self.model.geom_type[robot] == mujoco.mjtGeom.mjGEOM_CAPSULE
        axis_z = np.abs(self.data.geom_xmat[robot, 8])  # world z of each geom's local z axis
        top = np.where(capsule, self.data.geom_xpos[robot, 2] + size[:, 0] + size[:, 1] * axis_z, top)
        sphere = self.model.geom_type[robot] == mujoco.mjtGeom.mjGEOM_SPHERE
        top = np.where(sphere, self.data.geom_xpos[robot, 2] + size[:, 0], top)
        return float(np.max(top))

    def _height_reward_error(self, surface_z: float) -> float:
        """Torso height above the ground holding it up, less its standing height (``base_height``,
        ``base_height_low``); 0 when airborne."""
        reward = self.cfg.reward
        if not (reward.base_height or reward.base_height_low) or not np.isfinite(surface_z):
            return 0.0
        surface_z = self._support_surface_z()
        if surface_z is None:
            return 0.0
        return self.root_pos[2] - surface_z - self._standing_height(surface_z)

    def _standing_height(self, surface_z: float) -> float:
        """The torso height above ``surface_z`` the posture terms aim for: ``nominal_base_height``,
        lowered under a ceiling the upward rays see so the robot's top clears it."""
        target_height = self.cfg.nominal_base_height
        ceiling = self._ceiling_z(surface_z)
        if ceiling is not None:
            top_above_root = self._robot_top_z() - self.root_pos[2]
            target_height = min(target_height, max(
                0.0, ceiling - surface_z - top_above_root - self.cfg.reward.base_height_clearance_margin,
            ))
        return target_height

    def _vertical_gap_standing(self) -> float | None:
        """Standing height here (support surface, else the terrain below), for ``vertical_progress``."""
        surface_z = self._support_surface_z()
        if surface_z is None:
            surface_z = self._terrain_height_below(self.root_pos[:2], self.root_pos[2] + 0.1)
        return self._standing_height(surface_z) if np.isfinite(surface_z) else None

    def _touching_ceiling(self) -> bool:
        """Whether any robot geom is in contact with a ceiling (a tunnel bar or roof)."""
        ceilings, robot = self._ceiling_geom_ids, self._robot_geom_ids
        return any(
            (c.geom1 in ceilings and c.geom2 in robot) or (c.geom2 in ceilings and c.geom1 in robot)
            for c in self.data.contact[: self.data.ncon]
        )

    def _ceiling_overlap(self) -> float:
        """Metres the robot's top is above the lowest ceiling the upward rays see, less the margin."""
        surface_z = self._support_surface_z()
        if surface_z is None:
            surface_z = self._terrain_height_below(self.root_pos[:2], self.root_pos[2] + 0.1)
            if not np.isfinite(surface_z):
                return 0.0
        ceiling = self._ceiling_z(surface_z)
        if ceiling is None:
            return 0.0
        return max(0.0, self._robot_top_z() - (ceiling - self.cfg.reward.base_height_clearance_margin))

    def _compute_observation(self) -> np.ndarray:
        cfg = self.cfg.observation
        rng = self.np_random
        rot = mdp.quat_to_mat(self.root_quat)
        parts: list[np.ndarray] = []

        if cfg.base_lin_vel:
            parts.append(rot.T @ self.data.qvel[0:3] + rng.normal(0, cfg.noise_lin_vel, 3))
        if cfg.base_ang_vel:
            parts.append(self.data.qvel[3:6] + rng.normal(0, cfg.noise_ang_vel, 3))
        if cfg.projected_gravity:
            parts.append(mdp.projected_gravity(self.root_quat) + rng.normal(0, cfg.noise_gravity, 3))
        if cfg.base_height:
            surface = self._terrain_height_below(self.root_pos[:2], self.root_pos[2] + 0.1)
            height = self.root_pos[2] - surface if np.isfinite(surface) else cfg.scan_max_depth
            parts.append(np.array([np.clip(height, -5.0, 5.0)]))
        if cfg.joint_pos:
            parts.append(
                self.data.qpos[self._robot_qpos_adr]
                + rng.normal(0, cfg.noise_joint_pos, len(self._robot_qpos_adr))
            )
        if cfg.joint_vel:
            parts.append(
                self.data.qvel[self._robot_dof_adr]
                + rng.normal(0, cfg.noise_joint_vel, len(self._robot_dof_adr))
            )
        if cfg.last_action:
            parts.append(self._prev_action)
        if cfg.waypoint_command:
            command = []
            for offset in range(self.cfg.waypoint_lookahead):
                delta = self._target(offset) - self.root_pos
                command.append(rot.T @ delta)
            parts.append(np.concatenate(command))
        if cfg.speed_command:
            parts.append(np.array([self.command_speed]))
        if cfg.height_scan:
            parts.append(self._height_scan_obs())
        if cfg.overhead_scan:
            overhead = self._overhead_scan_obs()
            if cfg.overhead_scan_observation:
                parts.append(overhead)

        obs = np.concatenate(parts) if parts else np.zeros(1)
        if cfg.history_length > 1:
            if not self._obs_history:
                for _ in range(cfg.history_length):
                    self._obs_history.append(obs)
            else:
                self._obs_history.append(obs)
            return np.concatenate(list(self._obs_history))
        return obs

    def observation_mirror(self) -> symmetry.Mirror:
        """Left/right mirror of :meth:`_compute_observation`, term by term in the same order."""
        cfg = self.cfg.observation
        joints = symmetry.hinge_mirror(self.model, self._robot_joint_ids)
        scalar = symmetry.Mirror.identity(1)
        terms = [
            (cfg.base_lin_vel, symmetry.VECTOR),
            (cfg.base_ang_vel, symmetry.PSEUDOVECTOR),
            (cfg.projected_gravity, symmetry.VECTOR),
            (cfg.base_height, scalar),
            (cfg.joint_pos, joints),
            (cfg.joint_vel, joints),
            (cfg.last_action, self.action_mirror()),
        ]
        terms += [(cfg.waypoint_command, symmetry.VECTOR)] * self.cfg.waypoint_lookahead
        terms += [
            (cfg.speed_command, scalar),
            (cfg.height_scan, symmetry.lateral_mirror(cfg.scan_points)),
            (cfg.overhead_scan and cfg.overhead_scan_observation,
             symmetry.Mirror.identity(len(cfg.overhead_offsets))),
        ]
        parts = [mirror for enabled, mirror in terms if enabled] or [scalar]
        return symmetry.Mirror.concat([symmetry.Mirror.concat(parts)] * cfg.history_length)

    def action_mirror(self) -> symmetry.Mirror:
        """Left/right mirror of the action vector (see :func:`symmetry.actuator_mirror`)."""
        mirror = symmetry.actuator_mirror(self.model)
        if self.cfg.control_mode == "position" and not np.allclose(
            mirror(self._default_act_pos), self._default_act_pos
        ):
            raise ValueError("position control needs a mirror-symmetric default pose")
        return mirror

    def _height_scan_obs(self) -> np.ndarray:
        cfg = self.cfg.observation
        yaw = mdp.yaw_from_quat(self.root_quat)
        origin = self.root_pos.copy()
        origin[2] += 0.1
        depths, hits = mdp.height_scan(self.model, self.data, origin, yaw, cfg, self._ceiling_geom_ids,
                                       self._wall_geom_ids)
        self._scan_hits = hits  # this step's, undelayed: what feet_clearance treats as ahead
        current = self.cfg.nominal_base_height - depths
        clipped = (current < cfg.scan_clip[0]) | (current > cfg.scan_clip[1])
        self._scan_clipped = bool(clipped.any())  # magenta dots: the first-person view looks down
        self._scan_buffer.append(depths)
        delayed = self._scan_buffer[0]
        relative = np.clip(
            self.cfg.nominal_base_height - delayed, cfg.scan_clip[0], cfg.scan_clip[1]
        )
        if cfg.noise_height_scan:
            relative = relative + self.np_random.normal(0, cfg.noise_height_scan, relative.shape)
        if self._scan_marker_ids:
            # Draw each dot where the policy reads the ground, not where the ray hit: a ray off the
            # course falls into the pit but reads scan_clip[0], so its dot sits that far down and
            # turns magenta.
            seen = np.clip(current, cfg.scan_clip[0], cfg.scan_clip[1])
            shown = hits.copy()
            shown[:, 2] = origin[2] - (self.cfg.nominal_base_height - seen)
            for geom_id, position, clipped in zip(self._scan_marker_ids, shown, clipped):
                if geom_id >= 0:
                    self.model.geom_pos[geom_id] = position
                    self.data.geom_xpos[geom_id] = position
                    self.model.geom_rgba[geom_id] = SCAN_CLIPPED_RGBA if clipped else SCAN_MARKER_RGBA
        return relative

    def _overhead_scan_obs(self) -> np.ndarray:
        cfg = self.cfg.observation
        floor_ray_z = float(self.data.xpos[self._foot_body_ids, 2].min()) + 0.2
        surface = self._terrain_height_below(self.root_pos[:2], floor_ray_z)
        if not np.isfinite(surface):
            surface = self.root_pos[2] - self.cfg.nominal_base_height
        origin = self.root_pos.copy()
        origin[2] = surface + cfg.overhead_origin_height
        depths, hits = mdp.overhead_scan(
            self.model, self.data, origin, mdp.yaw_from_quat(self.root_quat), cfg,
        )
        self._overhead_buffer.append(depths)
        self._ceiling_seen = bool(np.any(depths < cfg.overhead_max_depth))  # the first-person view looks up
        for geom_id, depth, hit in zip(self._overhead_marker_ids, depths, hits):
            position = hit if depth < cfg.overhead_max_depth else np.array([0.0, 0.0, -50.0])
            self.model.geom_pos[geom_id] = position
            self.data.geom_xpos[geom_id] = position
        return self._overhead_buffer[0].copy()

    def _info(self, reward_terms: dict, reason: str = "") -> dict:
        if self._site_target >= 0:
            self.model.site_pos[self._site_target] = self._target()
        if self._site_next >= 0:
            self.model.site_pos[self._site_next] = self._target(1)
        total = max(len(self._waypoints) - self._start_target_index, 1)
        info = {
            "x_position": float(self.root_pos[0]),
            "y_position": float(self.root_pos[1]),
            "waypoints_reached": self._waypoints_reached,
            "course_completion": self._waypoints_reached / total,
            "difficulty": self._difficulty,
            "terrain_modules": self.terrain.modules,
            "termination": reason,
            "is_success": reason == "goal",
            "zero_command": self._zero_command,
            "action_noise_scale": self._action_noise_scale(),
        }
        if self.cfg.command_speed_range is not None:
            info["command_speed"] = self.command_speed
            info["velocity_tracking_error"] = float(self._velocity_error_sum / max(self._step_count, 1))
        info.update({f"reward/{k}": v for k, v in reward_terms.items()})
        if reason:
            info["episode_reward_terms"] = dict(self._episode_terms)
            info["avg_speed"] = self._episode_speed_sum / max(self._step_count, 1)
            if self._episode_height_samples:
                info["avg_height"] = self._episode_height_sum / self._episode_height_samples
            if self._skill_difficulty:
                info["skill_difficulty"] = dict(self._skill_difficulty)
                if not self._episode_credited:
                    module = self._waypoint_module(self._target_index)
                    if reason != "goal" and module > 0:  # it ended short of this module's end
                        self._module_results.append((self.terrain.modules[module], False, self._module_kind(module)))
                    # At the skills' levels (what promotion counts), and the review/preview rest.
                    info["module_results"] = [(s, p) for s, p, k in self._module_results if k == "level"]
                    info["module_practice_results"] = [r for r in self._module_results if r[2] != "level"]
            if not self._zero_command and not (self._skill_difficulty and self._episode_credited):
                self._update_curriculum(info["course_completion"], success=info["is_success"])
            self._episode_credited = True  # each module result counts once per episode
        return info

    def _clip_difficulty(self, difficulty: float) -> float:
        low, high = self.cfg.terrain.difficulty_range
        return round(float(np.clip(difficulty, low, high)), 6)

    def _mean_skill_difficulty(self) -> float:
        """The course's difficulty under per-skill levels: their mean, weighted by the task's
        ``skill_weights`` (not the level-scaled ones courses sample by, so a skill's promotion
        always raises it). It sets the course length and is what ``difficulty`` logs show."""
        weights = self.cfg.terrain.params.get("skill_weights")
        skills = list(self._skill_difficulty)
        w = np.array([max(weights.get(s, 0.0), 0.0) if weights else 1.0 for s in skills])
        if w.sum() <= 0:
            w = np.ones(len(skills))
        return float(np.dot(w, [self._skill_difficulty[s] for s in skills]) / w.sum())

    def _skill_sampling_weights(self) -> dict[str, float] | None:
        """The course's skill weights: ``skill_weights`` (1 each without them), each scaled down
        by its skill's level when ``skill_maintain_share`` is set."""
        base = self.cfg.terrain.params.get("skill_weights")
        share = self.cfg.terrain.skill_maintain_share
        if share is None:
            return base
        return {
            skill: (base.get(skill, 0.0) if base else 1.0) * (1.0 - (1.0 - share) * level)
            for skill, level in self._skill_difficulty.items()
        }

    def set_difficulty(self, difficulty: float | dict[str, float]) -> None:
        """Pin the terrain difficulty; the next reset rebuilds the course if it changed.

        With per-skill levels, a number sets every skill (as an old one-level checkpoint
        restores) and a dict sets the skills it names, leaving the rest; skills outside this
        course's mix are ignored. Without them only a number is accepted.
        """
        if self._skill_difficulty:
            levels = difficulty if isinstance(difficulty, dict) else dict.fromkeys(self._skill_difficulty, difficulty)
            for skill, level in levels.items():
                if skill in self._skill_difficulty:
                    self._skill_difficulty[skill] = self._clip_difficulty(level)
            for window in self._skill_results.values():
                window.clear()
            self._difficulty = self._mean_skill_difficulty()
        elif isinstance(difficulty, dict):
            raise ValueError("per-skill levels need TerrainCfg.per_skill_difficulty")
        else:
            low, high = self.cfg.terrain.difficulty_range
            self._difficulty = float(np.clip(difficulty, low, high))
        self._curriculum_successes.clear()
        self._curriculum_early_failures.clear()

    def curriculum_state(self) -> float | dict[str, float]:
        """What a checkpoint saves and ``set_difficulty`` restores: the course's level, or each skill's."""
        return dict(self._skill_difficulty) if self._skill_difficulty else self._difficulty

    def _update_skill_levels(self) -> None:
        """Move each skill whose window of module attempts is full: up a step at
        ``skill_promote_rate`` passes or more, down one below ``skill_demote_rate``."""
        cfg = self.cfg.terrain
        for skill, passed, kind in self._module_results:
            if kind != "level":
                continue  # review and preview modules are practice, not the level's test
            window = self._skill_results.get(skill)
            if window is None:
                continue
            window.append(passed)
            if len(window) < cfg.skill_window:
                continue
            rate = sum(window) / len(window)
            level = self._skill_difficulty[skill]
            if rate >= cfg.skill_promote_rate:
                level = self._clip_difficulty(level + cfg.difficulty_step)
            elif rate < cfg.skill_demote_rate:
                level = self._clip_difficulty(level - cfg.difficulty_step)
            if level != self._skill_difficulty[skill]:
                self._skill_difficulty[skill] = level
                window.clear()
        self._difficulty = self._mean_skill_difficulty()

    def _update_curriculum(self, completion: float, *, success: bool = False) -> None:
        cfg = self.cfg.terrain
        if not cfg.curriculum:
            return
        if self._skill_difficulty:
            self._update_skill_levels()
            return
        self._curriculum_successes.append(
            success and completion >= cfg.promote_completion and self._start_target_index == 0
        )
        self._curriculum_early_failures.append(completion <= cfg.demote_completion)
        if len(self._curriculum_successes) < cfg.curriculum_window:
            return
        low, high = cfg.difficulty_range
        difficulty = self._difficulty
        if sum(self._curriculum_successes) / cfg.curriculum_window >= cfg.promote_success_rate:
            difficulty = min(high, self._difficulty + cfg.difficulty_step)
        elif sum(self._curriculum_early_failures) / cfg.curriculum_window >= cfg.promote_success_rate:
            difficulty = max(low, self._difficulty - cfg.difficulty_step)
        if difficulty != self._difficulty:
            self._difficulty = difficulty
            self._curriculum_successes.clear()
            self._curriculum_early_failures.clear()
