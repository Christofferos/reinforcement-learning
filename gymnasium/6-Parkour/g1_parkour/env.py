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
from .scene import build_scene_xml, default_build_dir
from .terrain import build_terrain

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 6.0,
    "elevation": -20.0,
    "azimuth": 130.0,
}


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
        self._camera_name = camera_name
        self._camera_config = default_camera_config or DEFAULT_CAMERA_CONFIG

        self.robot = get_robot(self.cfg.robot)
        self.build_dir: Path = default_build_dir(self.cfg.terrain.kind)
        self._np_random_seeded = np.random.default_rng()

        self._difficulty = float(self.cfg.terrain.difficulty)
        self._curriculum_successes = deque(maxlen=self.cfg.terrain.curriculum_window)
        self._curriculum_early_failures = deque(maxlen=self.cfg.terrain.curriculum_window)
        self._reset_count = 0
        self._mujoco_renderer = None
        self._viewer_camera_set = False
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
        self.terrain = build_terrain(
            terrain_cfg.kind,
            rng,
            difficulty=self._difficulty,
            **terrain_cfg.params,
        )
        num_markers = self.cfg.observation.num_scan_points if self.cfg.debug_scan_markers else 0
        num_overhead_markers = (
            len(self.cfg.observation.overhead_offsets)
            if self.cfg.debug_scan_markers and self.cfg.observation.overhead_scan else 0
        )
        xml_path = build_scene_xml(
            self.robot, self.terrain, self.build_dir, num_markers,
            num_overhead_markers=num_overhead_markers,
        )

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

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
        self._hinge_qpos_adr = np.array(
            [
                self.model.jnt_qposadr[j]
                for j in range(self.model.njnt)
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                and self.model.jnt_limited[j]
            ],
            dtype=int,
        )
        self._hinge_jnt_ids = np.array(
            [
                j
                for j in range(self.model.njnt)
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                and self.model.jnt_limited[j]
            ],
            dtype=int,
        )
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
        self._overhead_buffer = deque(maxlen=max(1, self.cfg.observation.exteroceptive_delay + 1))
        self._obs_history = deque(maxlen=max(1, self.cfg.observation.history_length))

        if self._mujoco_renderer is not None:
            self._mujoco_renderer.close()
            self._mujoco_renderer = None
            self._viewer_camera_set = False

    def _resolve_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3).copy()
        if not self.terrain.resolve_waypoint_z:
            return waypoints
        direction = np.array([0.0, 0.0, -1.0])
        geom_id = np.zeros(1, dtype=np.int32)
        for i, point in enumerate(waypoints):
            start = np.array([point[0], point[1], 6.0])
            dist = mujoco.mj_ray(
                self.model, self.data, start, direction, mdp.observations.TERRAIN_RAY_MASK,
                1, -1, geom_id,
            )
            waypoints[i, 2] = (6.0 - dist) + point[2] if dist >= 0 else point[2]
        return waypoints

    # ------------------------------------------------------------------- gym api

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        resample = self.cfg.terrain.resample_every_n_resets
        if (self._reset_count == 0 and seed is not None) or (
            self._reset_count > 0 and resample and self._reset_count % resample == 0
        ) or self.terrain.difficulty != self._difficulty:
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
        mujoco.mj_forward(self.model, self.data)
        self._settle_feet_on_ground()

        self._target_index = start_index
        self._start_target_index = start_index
        self._zero_command = False
        if self.cfg.command_speed_range is not None:
            shift = self.cfg.command_speed_difficulty_shift * float(np.clip(self.terrain.difficulty, 0.0, 1.0))
            low, high = self.cfg.command_speed_range
            self._waypoint_speeds = rng.uniform(low + shift, high + shift, size=len(self._waypoints))
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

        active_target = self._target().copy()
        active_speed = self.command_speed
        command_origin = self.root_pos.copy()
        distance_before = self._distance_to_target()
        self._apply_action(applied)
        self._step_count += 1

        distance_after = float(np.linalg.norm(active_target[:2] - self.root_pos[:2]))
        progress = distance_before - distance_after
        reached = self._advance_waypoint()

        reward, reward_terms, terminated, reason = self._evaluate(
            action, progress, reached, active_target=active_target, target_speed=active_speed,
            command_origin=command_origin,
        )
        truncated = self._step_count >= self.max_episode_steps
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
        frame = self._mujoco_renderer.render(self.render_mode)
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

    PUSH_PREVIEW_STEPS = 20
    PUSH_BALL_SPEED_GAIN = 20.0
    """Incoming ball speed per m/s of push, so stronger pushes fly in faster."""
    PUSH_BALL_SPAWN_DISTANCE_GAIN = 10.0
    """Ball spawn distance beyond the impact point per m/s of push, so stronger pushes start farther out."""
    PUSH_BALL_IMPACT_DISTANCE = 0.1

    def _push_markers(self) -> list[dict]:
        """A ball flying in before a scheduled push and an arrow showing the kick just after it."""
        if not self.cfg.events.push_robot:
            return []
        markers = []
        incoming = self._next_push_step - self._step_count
        pending = float(np.linalg.norm(self._pending_push))
        # Both speed and spawn distance scale with the push, so their ratio sets how long the ball is in flight.
        flight_time = self.PUSH_BALL_SPAWN_DISTANCE_GAIN / self.PUSH_BALL_SPEED_GAIN
        time_to_impact = incoming * self.dt
        if 0 < incoming and time_to_impact <= flight_time + 1e-9 and pending > 1e-9:
            direction = np.append(self._pending_push / pending, 0.0)
            # Constant approach speed scaled by the push, arriving exactly on the push step.
            speed = self.PUSH_BALL_SPEED_GAIN * pending
            distance = self.PUSH_BALL_IMPACT_DISTANCE + speed * time_to_impact
            markers.append(dict(
                type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=self.root_pos - direction * distance,
                size=np.full(3, 0.08), rgba=(1.0, 0.3, 0.1, 0.9), emission=0.5, label="",
            ))
        if self._last_push is not None and self._step_count - self._last_push[0] <= self.PUSH_PREVIEW_STEPS:
            kick = self._last_push[1]
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
        if not viewer.vopt.geomgroup[4] or viewer.scn.ngeom >= viewer.scn.maxgeom:
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

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self._target()[:2] - self.root_pos[:2]))

    @property
    def command_speed(self) -> float:
        return float(self._waypoint_speeds[min(self._target_index, len(self._waypoint_speeds) - 1)])

    def _advance_waypoint(self) -> bool:
        if self._target_index >= len(self._waypoints) - 1:
            reached = (
                self._distance_to_target() < self.cfg.waypoint_radius
                and self.root_pos[2] - self._target()[2]
                >= self.robot.healthy_height_range[0] * self.cfg.termination.bad_height_scale
                and -mdp.projected_gravity(self.root_quat)[2] >= self.cfg.termination.orientation_limit
            )
            if reached and not self._goal_reached:
                self._waypoints_reached += 1
            self._goal_reached = reached
            return reached
        if self._distance_to_target() < self.cfg.waypoint_radius:
            self._target_index += 1
            self._waypoints_reached += 1
            self._best_distance = self._distance_to_target()
            self._stall_counter = 0
            return True
        return False

    def _terrain_height_below(self, xy: np.ndarray, from_z: float) -> float:
        geom_id = np.zeros(1, dtype=np.int32)
        start = np.array([xy[0], xy[1], from_z])
        dist = mujoco.mj_ray(
            self.model, self.data, start, np.array([0.0, 0.0, -1.0]),
            mdp.observations.TERRAIN_RAY_MASK, 1, -1, geom_id,
        )
        return from_z - dist if dist >= 0 else -np.inf

    def _evaluate(self, action, progress, reached, *, active_target=None, target_speed=None,
                  command_origin=None):
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
        else:
            self._stall_counter += 1

        reached_goal = self._goal_reached and reached
        low_height = height_above_terrain < self.robot.healthy_height_range[0] * cfg.termination.bad_height_scale
        tilted = upright < cfg.termination.orientation_limit
        self._bad_height_steps = self._bad_height_steps + 1 if low_height else 0
        self._bad_orientation_steps = self._bad_orientation_steps + 1 if tilted else 0
        air_time_reward, step_ahead, slide_speed, step_asymmetry = self._update_gait_state(
            to_target / norm if norm > 1e-6 else None
        )
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
        )
        fell = terminated and reason in ("bad_height", "bad_orientation", "fell_off_course")
        reached_goal = reached_goal and not terminated

        terms = mdp.terms(
            cfg.reward,
            dt=self.dt,
            progress=progress,
            heading_alignment=heading_alignment,
            upright=upright,
            height_error=self._height_reward_error(surface_z),
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
            supported=not (low_height or tilted),
        )
        self._prev_action = action.copy()
        if reached_goal:
            terminated = True
            reason = "goal"
        return float(sum(terms.values())), terms, terminated, reason

    def _height_reward_error(self, surface_z: float) -> float:
        if not self.cfg.reward.base_height or not np.isfinite(surface_z):
            return 0.0
        groups = self.model.geom_group
        support_contacts = [
            contact
            for contact in self.data.contact
            if contact.geom1 >= 0 and contact.geom2 >= 0
            and {int(groups[contact.geom1]), int(groups[contact.geom2])} == {0, 1}
            and contact.dist <= 0.0 and abs(contact.frame[2]) > 0.5
            and contact.pos[2] < self.root_pos[2]
        ]
        if not support_contacts:
            return 0.0
        surface_z = max(float(contact.pos[2]) for contact in support_contacts)
        height_above_terrain = self.root_pos[2] - surface_z
        observation = self.cfg.observation
        origin = self.root_pos.copy()
        origin[2] = surface_z + observation.overhead_origin_height
        depths, _ = mdp.overhead_scan(
            self.model, self.data, origin, mdp.yaw_from_quat(self.root_quat), observation,
        )
        target_height = self.cfg.nominal_base_height
        if len(depths) and np.min(depths) < observation.overhead_max_depth:
            robot_geoms = groups == 0
            top_above_root = float(np.max(
                self.data.geom_xpos[robot_geoms, 2] + self.model.geom_rbound[robot_geoms]
            ) - self.root_pos[2])
            clearance = observation.overhead_origin_height + float(np.min(depths))
            target_height = min(target_height, max(
                0.0, clearance - top_above_root - self.cfg.reward.base_height_clearance_margin,
            ))
        return height_above_terrain - target_height

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
            parts.append(self.data.qpos[7:] + rng.normal(0, cfg.noise_joint_pos, self.model.nq - 7))
        if cfg.joint_vel:
            parts.append(self.data.qvel[6:] + rng.normal(0, cfg.noise_joint_vel, self.model.nv - 6))
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
        joints = symmetry.hinge_mirror(self.model)
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
        depths, hits = mdp.height_scan(self.model, self.data, origin, yaw, cfg)
        self._scan_buffer.append(depths)
        delayed = self._scan_buffer[0]
        relative = np.clip(
            self.cfg.nominal_base_height - delayed, cfg.scan_clip[0], cfg.scan_clip[1]
        )
        if cfg.noise_height_scan:
            relative = relative + self.np_random.normal(0, cfg.noise_height_scan, relative.shape)
        for geom_id, hit in zip(self._scan_marker_ids, hits):
            if geom_id >= 0:
                self.model.geom_pos[geom_id] = hit
                self.data.geom_xpos[geom_id] = hit
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
            if not self._zero_command:
                self._update_curriculum(info["course_completion"], success=info["is_success"])
        return info

    def set_difficulty(self, difficulty: float) -> None:
        """Pin the terrain difficulty; the next reset rebuilds the course if it changed."""
        low, high = self.cfg.terrain.difficulty_range
        self._difficulty = float(np.clip(difficulty, low, high))
        self._curriculum_successes.clear()
        self._curriculum_early_failures.clear()

    def _update_curriculum(self, completion: float, *, success: bool = False) -> None:
        cfg = self.cfg.terrain
        if not cfg.curriculum:
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
