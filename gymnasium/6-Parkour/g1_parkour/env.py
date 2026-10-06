"""MuJoCo parkour environment with procedural terrain and swappable MDP terms."""

from __future__ import annotations

import copy
from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.spaces import Box

from . import mdp
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
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        self._joint_range = self.model.jnt_range.copy()
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
        self._init_qvel = np.zeros(self.model.nv)
        self._waypoints = self._resolve_waypoints(self.terrain.waypoints)
        self._waypoint_speeds = np.full(len(self._waypoints), self.cfg.reward.preferred_speed)
        self._velocity_error_sum = 0.0
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
        ) or (
            self.cfg.terrain.curriculum and self.terrain.difficulty != self._difficulty
        ):
            if self.cfg.terrain.kind == "flat":
                self.terrain = build_terrain("flat", rng, difficulty=self._difficulty,
                                             **self.cfg.terrain.params)
                self._waypoints = self._resolve_waypoints(self.terrain.waypoints)
                self._waypoint_speeds = np.full(len(self._waypoints), self.cfg.reward.preferred_speed)
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

        self._target_index = start_index
        self._start_target_index = start_index
        if self.cfg.command_speed_range is not None:
            self._waypoint_speeds = rng.uniform(*self.cfg.command_speed_range, size=len(self._waypoints))
        self._velocity_error_sum = 0.0
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
        self._episode_terms = {}

        obs = self._compute_observation()
        return obs, self._info(reward_terms={})

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._action_buffer.append(action)
        applied = self._action_buffer[0]

        low, high = self._ctrl_range[:, 0], self._ctrl_range[:, 1]
        self.data.ctrl[:] = low + (applied + 1.0) * 0.5 * (high - low)

        if self.cfg.events.push_robot and self._step_count and (
            self._step_count % self.cfg.events.push_interval_steps == 0
        ):
            mdp.push(self.data, self.np_random, self.cfg.events)

        active_target = self._target().copy()
        distance_before = self._distance_to_target()
        mujoco.mj_step(self.model, self.data, nstep=self.cfg.frame_skip)
        self._step_count += 1

        distance_after = float(np.linalg.norm(active_target[:2] - self.root_pos[:2]))
        progress = distance_before - distance_after
        reached = self._advance_waypoint()

        reward, reward_terms, terminated, reason = self._evaluate(action, progress, reached)
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
                    speed = (
                        self._waypoint_speeds[index]
                        if self.cfg.command_speed_range is not None
                        else self.cfg.reward.preferred_speed
                    )
                    label = f"WP {index + 1} (target) | {speed:.2f} m/s"
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

    @staticmethod
    def _add_waypoint_marker_to_scene(viewer, marker):
        if not viewer.vopt.geomgroup[4] or viewer.scn.ngeom >= viewer.scn.maxgeom:
            return
        geom = viewer.scn.geoms[viewer.scn.ngeom]
        mujoco.mjv_initGeom(
            geom, marker["type"], marker["size"], marker["pos"],
            np.eye(3).ravel(), np.asarray(marker["rgba"], dtype=np.float32),
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

    def _evaluate(self, action, progress, reached):
        cfg = self.cfg
        gravity = mdp.projected_gravity(self.root_quat)
        upright = float(-gravity[2])
        rot = mdp.quat_to_mat(self.root_quat)
        lin_vel_local = rot.T @ self.data.qvel[0:3]
        ang_vel_local = rot.T @ self.data.qvel[3:6]

        to_target = self._target()[:2] - self.root_pos[:2]
        norm = np.linalg.norm(to_target)
        heading_alignment = float(rot[:2, 0] @ (to_target / norm)) if norm > 1e-6 else 0.0
        desired_velocity = self.command_speed * to_target / norm if norm > 1e-6 else np.zeros(2)
        velocity_error_squared = float(np.sum(np.square(self.data.qvel[:2] - desired_velocity)))
        self._velocity_error_sum += np.sqrt(velocity_error_squared)

        surface_z = self._terrain_height_below(self.root_pos[:2], self.root_pos[2] + 0.1)
        height_above_terrain = self.root_pos[2] - surface_z if np.isfinite(surface_z) else 99.0

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
        else:
            self._stall_counter += 1

        reached_goal = self._goal_reached and reached
        low_height = height_above_terrain < self.robot.healthy_height_range[0] * cfg.termination.bad_height_scale
        self._bad_height_steps = self._bad_height_steps + 1 if low_height else 0
        self._bad_orientation_steps = self._bad_orientation_steps + 1 if upright < cfg.termination.orientation_limit else 0
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
            height_error=height_above_terrain - cfg.nominal_base_height,
            lateral_speed=float(lin_vel_local[1]),
            ang_vel=ang_vel_local,
            action=action,
            prev_action=self._prev_action,
            torque_power=torque_power,
            joint_limit_violation=violation,
            waypoints_reached=int(reached and not reached_goal),
            reached_goal=reached_goal,
            fell=fell,
            forward_speed=float(lin_vel_local[0]),
            target_speed=self.command_speed if cfg.command_speed_range is not None else None,
            velocity_error_squared=velocity_error_squared,
        )
        self._prev_action = action.copy()
        if reached_goal:
            terminated = True
            reason = "goal"
        return float(sum(terms.values())), terms, terminated, reason

    def _compute_observation(self) -> np.ndarray:
        cfg = self.cfg.observation
        rng = self.np_random
        rot = mdp.quat_to_mat(self.root_quat)
        parts: list[np.ndarray] = []

        if cfg.base_lin_vel:
            parts.append(rot.T @ self.data.qvel[0:3] + rng.normal(0, cfg.noise_lin_vel, 3))
        if cfg.base_ang_vel:
            parts.append(rot.T @ self.data.qvel[3:6] + rng.normal(0, cfg.noise_ang_vel, 3))
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
        }
        if self.cfg.command_speed_range is not None:
            info["command_speed"] = self.command_speed
            info["velocity_tracking_error"] = float(self._velocity_error_sum / max(self._step_count, 1))
        info.update({f"reward/{k}": v for k, v in reward_terms.items()})
        if reason:
            info["episode_reward_terms"] = dict(self._episode_terms)
            self._update_curriculum(info["course_completion"], success=info["is_success"])
        return info

    def _update_curriculum(self, completion: float, *, success: bool = False) -> None:
        cfg = self.cfg.terrain
        if not cfg.curriculum:
            return
        self._curriculum_successes.append(
            success and completion >= cfg.promote_completion and self._start_target_index == 0
        )
        low, high = cfg.difficulty_range
        difficulty = self._difficulty
        if completion <= cfg.demote_completion:
            difficulty = max(low, self._difficulty - cfg.difficulty_step)
        elif (
            len(self._curriculum_successes) == cfg.curriculum_window
            and sum(self._curriculum_successes) / cfg.curriculum_window >= cfg.promote_success_rate
        ):
            difficulty = min(high, self._difficulty + cfg.difficulty_step)
        if difficulty != self._difficulty:
            self._difficulty = difficulty
            self._curriculum_successes.clear()
