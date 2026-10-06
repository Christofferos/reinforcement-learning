"""Smoke tests: every task compiles, steps and reports sane metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour import tasks  # noqa: E402
from g1_parkour.terrain import build_terrain  # noqa: E402


def test_inline_heightfield_preserves_sample_orientation():
    import mujoco
    from g1_parkour.terrain.core import HeightField

    samples = np.array([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]])
    hfield = HeightField("test_heightfield", samples, 2.0, 1.0, 0.5)
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><asset>{hfield.asset_xml()}</asset></mujoco>"
    )
    np.testing.assert_allclose(model.hfield_data.reshape(samples.shape), samples, atol=1e-7)


def test_rough_runtime_does_not_write_heightfield_pngs(monkeypatch, tmp_path):
    import g1_parkour.env as env_module

    def reject_write(*args, **kwargs):
        pytest.fail("runtime terrain must not write PNG assets")

    monkeypatch.setattr(Path, "write_bytes", reject_write)
    monkeypatch.setattr(env_module, "default_build_dir", lambda tag: tmp_path)
    cfg = tasks.rough_cfg()
    cfg.terrain.resample_every_n_resets = 1
    env = env_module.ParkourEnv(cfg)
    try:
        for seed in (0, 1):
            obs, _ = env.reset(seed=seed)
            assert np.all(np.isfinite(obs))
            assert env.model.nhfield == 1
            samples = (np.clip(env.terrain.heightfields[0].data, 0.0, 1.0) * 255.0).astype(np.uint8)
            expected = (samples.astype(float) - samples.min()) / (float(samples.max()) - samples.min())
            np.testing.assert_allclose(env.model.hfield_data.reshape(samples.shape), expected, atol=1e-7)
            _, reward, _, _, _ = env.step(np.zeros(env.model.nu))
            assert np.isfinite(reward)
        assert not list(tmp_path.rglob("*.png"))
    finally:
        env.close()


@pytest.mark.parametrize("cfg_factory", [
    pytest.param(tasks.flat_cfg, id="flat"),
    pytest.param(tasks.rough_cfg, id="rough"),
    pytest.param(tasks.parkour_cfg, id="procedural"),
])
def test_human_render_waypoints_update_each_frame(cfg_factory):
    from types import SimpleNamespace
    import mujoco
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(cfg_factory(), render_mode="human")
    markers = []
    frames = []
    speed_frames = []
    viewer = SimpleNamespace(
        add_marker=lambda **marker: markers.append(marker),
        scn=mujoco.MjvScene(env.model, maxgeom=100),
        vopt=mujoco.MjvOption(),
    )
    viewer.vopt.geomgroup[4] = 1

    def render(mode):
        assert mode == "human"
        viewer.scn.ngeom = 0
        for marker in markers:
            viewer.scn.geoms[viewer.scn.ngeom].objid = 7
            viewer._add_marker_to_scene(marker)
        visible_markers = markers if viewer.vopt.geomgroup[4] else []
        assert viewer.scn.ngeom == len(visible_markers)
        for index, marker in enumerate(visible_markers):
            assert viewer.scn.geoms[index].objid == -1
            assert viewer.scn.geoms[index].label == marker["label"]
            np.testing.assert_allclose(viewer.scn.geoms[index].pos, marker["pos"])
        frames.append([marker for marker in visible_markers if marker["type"] == mujoco.mjtGeom.mjGEOM_SPHERE])
        speed_frames.append([marker for marker in visible_markers if marker["label"].startswith("Speed:")])
        markers.clear()

    env._mujoco_renderer = SimpleNamespace(
        _get_viewer=lambda mode: viewer, render=render, close=lambda: None,
    )
    env._viewer_camera_set = True
    try:
        env._waypoints = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.5], [4.0, 0.0, 1.0]])
        env._waypoint_speeds = np.array([0.5, 0.75, 1.25])
        expected_speeds = [0.5, 0.75, 1.25] if env.cfg.command_speed_range is not None else [1.2] * 3
        env._target_index = 1
        env.data.qvel[:3] = [3.0, 4.0, 10.0]
        env.render()
        assert len(frames[0]) == 3
        assert len(speed_frames[0]) == 1
        assert speed_frames[0][0]["label"] == "Speed: 5.00 m/s"
        assert speed_frames[0][0]["type"] == mujoco.mjtGeom.mjGEOM_LABEL
        np.testing.assert_allclose(speed_frames[0][0]["pos"], env.root_pos + [0.0, 0.0, 0.3])
        np.testing.assert_allclose(frames[0][1]["pos"], [2.0, 0.0, 0.8])
        assert [marker["label"] for marker in frames[0]] == [
            "", f"WP 2 (target) | {expected_speeds[1]:.2f} m/s", "",
        ]
        assert frames[0][0]["rgba"] == (0.2, 0.9, 0.3, 0.9)
        assert frames[0][1]["rgba"] == (1.0, 0.8, 0.0, 1.0)
        assert frames[0][2]["rgba"] == (0.0, 0.8, 1.0, 0.8)
        env._target_index = 2
        env.data.qvel[:3] = [0.3, 0.4, -10.0]
        env.data.qpos[:3] += [1.0, 2.0, 0.5]
        env.render()
        assert len(frames[1]) == 3
        assert speed_frames[1][0]["label"] == "Speed: 0.50 m/s"
        np.testing.assert_allclose(speed_frames[1][0]["pos"], env.root_pos + [0.0, 0.0, 0.3])
        assert frames[1][1]["rgba"] == (0.2, 0.9, 0.3, 0.9)
        assert [marker["label"] for marker in frames[1]] == [
            "", "", f"WP 3 (target) | {expected_speeds[2]:.2f} m/s",
        ]
        env._goal_reached = True
        env.render()
        assert all(marker["rgba"] == (0.2, 0.9, 0.3, 0.9) for marker in frames[2])
        assert [marker["label"] for marker in frames[2]] == [""] * 3
        viewer.vopt.geomgroup[4] = 0
        env.render()
        assert frames[3] == []
        assert speed_frames[3] == []
        viewer.vopt.geomgroup[4] = 1
        env.data.qvel[:2] = [1.23456, 0.0]
        env.render()
        assert len(frames[4]) == 3
        assert speed_frames[4][0]["label"] == "Speed: 1.23 m/s"
        env.data.qvel[:2] = 0.0
        env.render()
        assert speed_frames[5][0]["label"] == "Speed: 0.00 m/s"
    finally:
        env.close()


@pytest.mark.parametrize("cfg_factory", [tasks.flat_cfg, tasks.rough_cfg, tasks.parkour_cfg])
def test_height_scan_debug_markers_follow_ground_hits(cfg_factory):
    import mujoco
    from g1_parkour.env import ParkourEnv
    from g1_parkour.mdp import observations

    env = ParkourEnv(cfg_factory())
    try:
        env.reset(seed=0)
        assert len(env._scan_marker_ids) == env.cfg.observation.num_scan_points
        assert all(geom_id >= 0 for geom_id in env._scan_marker_ids)
        for offset in (0.0, 0.5):
            env.data.qpos[0] += offset
            mujoco.mj_forward(env.model, env.data)
            obs = env._compute_observation()
            assert obs.shape == env.observation_space.shape
            origin = env.root_pos.copy()
            origin[2] += 0.1
            _, hits = observations.height_scan(
                env.model, env.data, origin,
                observations.yaw_from_quat(env.root_quat), env.cfg.observation,
            )
            np.testing.assert_allclose(env.data.geom_xpos[env._scan_marker_ids], hits)
    finally:
        env.close()


@pytest.mark.parametrize("task_id", list(tasks.TASKS))
def test_task_steps(task_id):
    env = gym.make(task_id)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()


@pytest.mark.parametrize("kind", ["flat", "rough", "parkour"])
def test_terrain_is_monotonic_along_x(kind):
    rng = np.random.default_rng(0)
    terrain = build_terrain(kind, rng, difficulty=0.8)
    xs = terrain.waypoints[:, 0]
    assert len(xs) >= 2
    assert np.all(np.diff(xs) > -1e-6), "route waypoints must progress along +x"


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("direction", [-1, 1], ids=["descending", "ascending"])
def test_stair_surfaces_match_waypoint_heights(difficulty, direction):
    from dataclasses import replace
    import mujoco
    from g1_parkour.terrain.parkour import Cursor, mod_stairs
    from g1_parkour.mdp.observations import TERRAIN_RAY_MASK

    cursor = Cursor(x=5.0, y=1.0, z=2.0)
    stairs = mod_stairs(np.random.default_rng(0), difficulty, cursor, direction=direction)
    geoms = "".join(replace(box, material=None).to_xml() for box in stairs.boxes)
    model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{geoms}</worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    surface_heights = []
    for index, waypoint in enumerate(stairs.waypoints):
        origin = np.array([waypoint[0], waypoint[1], 5.0])
        if index == len(stairs.waypoints) - 1:
            origin[0] -= 0.01
        geom_id = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            model, data, origin, np.array([0.0, 0.0, -1.0]),
            TERRAIN_RAY_MASK, 1, -1, geom_id,
        )
        assert distance >= 0
        surface_heights.append(origin[2] - distance)
    np.testing.assert_allclose(surface_heights, np.asarray(stairs.waypoints)[:, 2], atol=1e-4)
    assert np.all(direction * np.diff(surface_heights[:-1]) > 0)
    assert cursor.z == pytest.approx(surface_heights[-1])


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("yaw", [0.0, np.pi / 2])
def test_overhead_scan_hits_tunnel_underside(difficulty, yaw):
    from dataclasses import replace
    import mujoco
    from g1_parkour.mdp.observations import ObservationCfg, overhead_scan
    from g1_parkour.terrain.parkour import Cursor, mod_tunnel

    cfg = ObservationCfg(overhead_scan=True)
    tunnel = mod_tunnel(np.random.default_rng(0), difficulty, Cursor(z=2.0))
    geoms = "".join(replace(box, material=None).to_xml() for box in tunnel.boxes)
    robot_geom = '<geom type="sphere" size=".1" pos="1.75 0 2.7" group="0"/>'
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body quat="{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}">'
        f'{geoms}{robot_geom}</body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    clearance = 1.45 - 0.4 * difficulty
    for index, offset in enumerate(cfg.overhead_offsets):
        origin = np.array([0.0, 0.0, 2.0 + cfg.overhead_origin_height])
        origin[:2] = (1.75 - offset) * forward
        depths, hits = overhead_scan(model, data, origin, yaw, cfg)
        assert depths[index] == pytest.approx(clearance - cfg.overhead_origin_height)
        np.testing.assert_allclose(hits[index, :2], 1.75 * forward, atol=1e-8)
        assert hits[index, 2] == pytest.approx(2.0 + clearance)
    origin[:2] = -2.0 * forward
    depths, _ = overhead_scan(model, data, origin, yaw, cfg)
    np.testing.assert_allclose(depths, cfg.overhead_max_depth)


@pytest.mark.parametrize("debug_markers", [False, True])
@pytest.mark.parametrize("delay", [0, 2])
@pytest.mark.parametrize("include_observation", [False, True])
def test_overhead_observations_and_markers(monkeypatch, debug_markers, delay, include_observation):
    from dataclasses import replace
    import mujoco
    import g1_parkour.env as env_module
    from g1_parkour.mdp import EventCfg
    from g1_parkour.terrain.core import TerrainSpec
    from g1_parkour.terrain.parkour import Cursor, mod_tunnel

    tunnel = mod_tunnel(np.random.default_rng(0), 1.0, Cursor())
    terrain = TerrainSpec(
        name="tunnel_test", boxes=tunnel.boxes,
        waypoints=np.asarray(tunnel.waypoints), spawn_pos=(0.75, 0.0, 0.0),
    )
    monkeypatch.setattr(env_module, "build_terrain", lambda *args, **kwargs: terrain)
    cfg = tasks.parkour_cfg()
    cfg.debug_scan_markers = debug_markers
    cfg.events = EventCfg(reset_noise_scale=0.0)
    cfg.terrain = replace(cfg.terrain, curriculum=False, resample_every_n_resets=0)
    cfg.observation = replace(
        cfg.observation, exteroceptive_delay=delay,
        overhead_scan_observation=include_observation,
    )
    env = env_module.ParkourEnv(cfg)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == ((116,) if include_observation else (113,))
        np.testing.assert_allclose(env._overhead_buffer[0], [2.0, 2.0, 0.55])
        if include_observation:
            np.testing.assert_allclose(obs[-3:], [2.0, 2.0, 0.55])
        assert len(env._overhead_marker_ids) == (3 if debug_markers else 0)
        if debug_markers:
            np.testing.assert_allclose(
                env.data.geom_xpos[env._overhead_marker_ids],
                [[0.0, 0.0, -50.0], [0.0, 0.0, -50.0], [1.75, 0.0, 1.05]],
            )
            np.testing.assert_allclose(
                env.model.geom_pos[env._overhead_marker_ids],
                env.data.geom_xpos[env._overhead_marker_ids],
            )
        env.data.qpos[0] = -2.0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(delay + 1):
            obs = env._compute_observation()
        np.testing.assert_allclose(env._overhead_buffer[0], [2.0, 2.0, 2.0])
        obs, _ = env.reset(seed=0)
        assert len(env._overhead_buffer) == 1
        np.testing.assert_allclose(env._overhead_buffer[0], [2.0, 2.0, 0.55])
        env.cfg.observation.overhead_scan = False
        legacy_obs = env._compute_observation()
        assert legacy_obs.shape == (113,)
        if not include_observation:
            np.testing.assert_allclose(obs, legacy_obs)
    finally:
        env.close()


@pytest.mark.parametrize("seed", [0, 11, 42])
def test_flat_routes_turn_and_resample(seed):
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.flat_cfg())
    try:
        env.reset(seed=seed)
        model = env.model
        route = env._waypoints.copy()
        segments = np.diff(np.vstack([np.zeros(3), route]), axis=0)
        headings = np.arctan2(segments[:, 1], segments[:, 0])
        assert np.max(np.abs(np.diff(headings))) > np.deg2rad(5.0)
        assert np.all(np.abs(route[:, 1]) <= 1.25)
        assert np.all(np.diff(route[:, 0]) > 0)
        env.reset()
        assert env.model is model
        assert not np.array_equal(route, env._waypoints)
        env.reset(seed=seed)
        np.testing.assert_array_equal(route, env._waypoints)
    finally:
        env.close()


@pytest.mark.parametrize("cfg_factory", [tasks.flat_cfg, tasks.rough_cfg])
def test_speed_is_observed_seeded_and_changes_at_waypoints(cfg_factory):
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(cfg_factory())
    legacy_cfg = cfg_factory()
    legacy_cfg.command_speed_range = None
    legacy_cfg.observation.speed_command = False
    legacy = ParkourEnv(legacy_cfg)
    replica = ParkourEnv(cfg_factory())
    try:
        observation, info = env.reset(seed=42)
        assert env.cfg.reward.velocity_tracking > 0
        low, high = env.cfg.command_speed_range
        assert env.cfg.episode_length_s >= env.terrain.course_length / low
        assert observation.shape == (legacy.observation_space.shape[0] + 1,)
        speeds = env._waypoint_speeds.copy()
        assert np.all((speeds >= low) & (speeds <= high))
        assert len(np.unique(speeds)) > 1
        assert observation[-49] == pytest.approx(speeds[0])
        assert info["command_speed"] == pytest.approx(speeds[0])
        env.data.qpos[:2] = env._waypoints[0, :2]
        assert env._advance_waypoint()
        assert env.command_speed == pytest.approx(speeds[1])
        assert env._compute_observation()[-49] == pytest.approx(speeds[1])
        replica.reset(seed=42)
        np.testing.assert_array_equal(replica._waypoint_speeds, speeds)
        env.reset()
        assert not np.array_equal(env._waypoint_speeds, speeds)
    finally:
        env.close()
        legacy.close()
        replica.close()


def test_regular_presets_share_observation_layout_and_policy():
    from dataclasses import fields
    from stable_baselines3 import PPO
    from g1_parkour.env import ParkourEnv
    from g1_parkour.mdp import ObservationCfg

    configs = [tasks.flat_cfg(), tasks.rough_cfg(), tasks.parkour_cfg()]
    for cfg in configs:
        for field in fields(ObservationCfg):
            np.testing.assert_equal(
                getattr(cfg.observation, field.name),
                getattr(configs[0].observation, field.name),
            )
    environments = []
    try:
        for cfg in configs:
            env = ParkourEnv(cfg)
            environments.append(env)
            obs, info = env.reset(seed=0)
            assert env.observation_space.shape == obs.shape == (116,)
            assert env.action_space == environments[0].action_space
            assert np.all(np.isfinite(obs))
            assert obs[-49] == pytest.approx(info["command_speed"])
            np.testing.assert_allclose(obs[-3:], env._overhead_buffer[0])
        policy = PPO(
            "MlpPolicy", environments[0], n_steps=8, batch_size=8,
            n_epochs=1, policy_kwargs={"net_arch": [16]}, device="cpu", seed=0,
        )
        for env in environments:
            policy.set_env(env)
            policy.learn(total_timesteps=8, reset_num_timesteps=False)
            obs, _ = env.reset(seed=0)
            action, _ = policy.predict(obs, deterministic=True)
            assert action.shape == env.action_space.shape
    finally:
        for env in environments:
            env.close()


@pytest.mark.parametrize("cfg_factory", [tasks.flat_cfg, tasks.rough_cfg])
def test_reward_tracks_speed_and_direction(cfg_factory):
    import mujoco
    from g1_parkour.env import ParkourEnv

    cfg = cfg_factory()
    cfg.command_speed_range = (0.75, 0.75)
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        direction = env._target()[:2] - env.root_pos[:2]
        direction /= np.linalg.norm(direction)
        tracking_rewards = []
        for velocity in (0.75 * direction, np.zeros(2), -0.75 * direction, 1.5 * direction):
            env.data.qvel[:2] = velocity
            mujoco.mj_forward(env.model, env.data)
            _, reward_terms, _, _ = env._evaluate(np.zeros(env.model.nu), 0.0, False)
            tracking_rewards.append(reward_terms.get("velocity_tracking", 0.0))
        assert tracking_rewards[0] == pytest.approx(cfg.reward.velocity_tracking)
        assert all(tracking_rewards[0] > reward for reward in tracking_rewards[1:])
        assert tracking_rewards[1] > tracking_rewards[2]
        _, _, _, _, info = env.step(np.zeros(env.model.nu))
        assert np.isfinite(info["velocity_tracking_error"])
    finally:
        env.close()


@pytest.mark.parametrize("speeds", [(0.0, 1.0), (1.0, 0.5), (0.5, float("inf"))])
def test_flat_rejects_invalid_speed_ranges(speeds):
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.command_speed_range = speeds
    with pytest.raises(ValueError, match="command_speed_range"):
        ParkourEnv(cfg)


def test_parkour_difficulty_changes_geometry():
    easy = build_terrain("parkour", np.random.default_rng(3), difficulty=0.0, num_modules=8)
    hard = build_terrain("parkour", np.random.default_rng(3), difficulty=1.0, num_modules=8)
    assert easy.modules == hard.modules
    assert hard.course_length != easy.course_length


def test_curriculum_promotes_on_reliable_full_runs():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.parkour_cfg()
    cfg.terrain.curriculum = True
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        start = env._difficulty
        for _ in range(15):
            env._update_curriculum(1.0, success=True)
            env.reset()
        for _ in range(4):
            env._update_curriculum(0.75)
        assert env._difficulty == start
        env._update_curriculum(1.0, success=True)
        assert env._difficulty == pytest.approx(start + cfg.terrain.difficulty_step)
        assert not env._curriculum_successes
        env._update_curriculum(1.0, success=True)
        assert env._difficulty == pytest.approx(start + cfg.terrain.difficulty_step)
        for _ in range(cfg.terrain.curriculum_window - 1):
            env._update_curriculum(1.0, success=True)
        assert env._difficulty == pytest.approx(start + 2 * cfg.terrain.difficulty_step)
        assert not env._curriculum_successes
    finally:
        env.close()


@pytest.mark.parametrize("difficulty, completion, success", [
    (0.1, 0.0, False), (1.0, 1.0, True),
])
def test_curriculum_respects_difficulty_bounds(difficulty, completion, success):
    from g1_parkour.env import ParkourEnv

    cfg = tasks.parkour_cfg()
    cfg.terrain.difficulty = difficulty
    env = ParkourEnv(cfg)
    try:
        for _ in range(cfg.terrain.curriculum_window):
            env._update_curriculum(completion, success=success)
        assert env._difficulty == difficulty
    finally:
        env.close()


@pytest.mark.parametrize("reason, start_index", [
    ("bad_height", 0), ("timeout", 0), ("goal", 1),
])
def test_curriculum_rejects_failed_or_partial_start_finishes(reason, start_index):
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.parkour_cfg())
    try:
        env.reset(seed=0)
        start = env._difficulty
        env._start_target_index = start_index
        env._waypoints_reached = len(env._waypoints) - start_index
        for _ in range(env.cfg.terrain.curriculum_window):
            info = env._info({}, reason)
            assert info["course_completion"] == 1.0
        assert env._difficulty == start
        assert not any(env._curriculum_successes)
    finally:
        env.close()


def test_curriculum_applies_new_level_before_rough_resample_interval():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.rough_cfg()
    cfg.terrain.curriculum_window = 2
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        start = env.terrain.difficulty
        env._update_curriculum(1.0, success=True)
        env._update_curriculum(1.0, success=True)
        env.reset()
        assert env.terrain.difficulty == pytest.approx(start + cfg.terrain.difficulty_step)
        assert env.terrain.difficulty == env._difficulty
        env._update_curriculum(1.0, success=True)
        env._update_curriculum(0.0)
        assert env._difficulty == pytest.approx(start)
        assert not env._curriculum_successes
    finally:
        env.close()


def test_curriculum_disabled_keeps_difficulty_fixed():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.parkour_cfg()
    cfg.terrain.curriculum = False
    env = ParkourEnv(cfg)
    try:
        start = env._difficulty
        for _ in range(cfg.terrain.curriculum_window):
            env._update_curriculum(1.0, success=True)
        env._update_curriculum(0.0)
        assert env._difficulty == start
        assert not env._curriculum_successes
    finally:
        env.close()


def test_goal_requires_reaching_final_waypoint():
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.flat_cfg())
    try:
        env.reset(seed=0)
        env._waypoints = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        env._target_index = 1
        env._waypoints_reached = 1
        env.data.qpos[:2] = [2.0, 0.0]
        reached = env._advance_waypoint()
        _, _, terminated, reason = env._evaluate(np.zeros(env.model.nu), 0.0, reached)
        assert not terminated
        assert reason != "goal"
        assert env._info({})["course_completion"] == pytest.approx(2 / 3)
        env.data.qpos[:2] = [4.0, 0.0]
        reached = env._advance_waypoint()
        _, _, terminated, reason = env._evaluate(np.zeros(env.model.nu), 0.0, reached)
        assert terminated and reason == "goal"
        assert env._info({})["course_completion"] == 1.0
    finally:
        env.close()


def test_height_recovery_grace_and_pit_termination():
    from g1_parkour.mdp.terminations import TerminationCfg, check

    cfg = TerminationCfg(recovery_grace_steps=20)
    state = dict(height_above_terrain=0.5, healthy_height_range=(0.8, 2.2),
                 upright=1.0, position=(0.0, 0.0, 1.0), min_course_z=0.0, stall_counter=0)
    assert check(cfg, **state, bad_height_steps=20) == (False, "")
    assert check(cfg, **state, bad_height_steps=21) == (True, "bad_height")
    state["position"] = (0.0, 0.0, -2.0)
    assert check(cfg, **state, bad_height_steps=1) == (True, "fell_off_course")


def test_timeout_updates_curriculum_and_reports_episode_terms():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.parkour_cfg()
    cfg.episode_length_s = 0.015
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        difficulty = env._difficulty
        _, _, _, truncated, info = env.step(np.zeros(env.model.nu))
        assert truncated
        assert info["termination"] == "timeout"
        assert "episode_reward_terms" in info
        assert env._difficulty < difficulty
    finally:
        env.close()


def test_progress_reward_prefers_controlled_speed():
    from g1_parkour.mdp.rewards import terms

    cfg = tasks.parkour_cfg().reward
    state = dict(dt=0.015, heading_alignment=1.0, upright=1.0, height_error=0.0,
                 lateral_speed=0.0, ang_vel=np.zeros(3), action=np.zeros(17),
                 prev_action=np.zeros(17), torque_power=0.0, joint_limit_violation=0.0,
                 waypoints_reached=0, reached_goal=False, fell=False)
    controlled = terms(cfg, progress=1.2 * state["dt"], forward_speed=1.2, **state)
    fast = terms(cfg, progress=3.0 * state["dt"], forward_speed=3.0, **state)
    assert controlled["progress"] == pytest.approx(20.0 * 1.2 * state["dt"])
    assert sum(controlled.values()) > sum(fast.values())
    active_state = dict(state, action=np.ones(17), prev_action=np.ones(17))
    active = terms(cfg, progress=1.2 * state["dt"], forward_speed=1.2, **active_state)
    assert sum(active.values()) > 0.0


def test_first_seeded_reset_reproduces_terrain():
    from g1_parkour.env import ParkourEnv

    first = ParkourEnv(tasks.parkour_cfg())
    second = ParkourEnv(tasks.parkour_cfg())
    try:
        first.reset(seed=42)
        second.reset(seed=42)
        assert first.terrain.modules == second.terrain.modules
        np.testing.assert_array_equal(first._waypoints, second._waypoints)
    finally:
        first.close()
        second.close()


def test_waypoint_switch_does_not_invent_progress(monkeypatch):
    from g1_parkour.env import ParkourEnv
    from g1_parkour import env as env_module

    env = ParkourEnv(tasks.flat_cfg())
    try:
        env.reset(seed=0)
        env._waypoints = np.array([[3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        env.data.qpos[:2] = [2.25, 0.0]
        measured = {}

        def advance(model, data, nstep):
            data.qpos[0] += 0.1

        def evaluate(action, progress, reached):
            measured.update(progress=progress, reached=reached)
            return 0.0, {}, False, ""

        monkeypatch.setattr(env_module.mujoco, "mj_step", advance)
        monkeypatch.setattr(env, "_evaluate", evaluate)
        env.step(np.zeros(env.model.nu))
        assert measured["reached"]
        assert measured["progress"] == pytest.approx(0.1)
        assert env._target_index == 1
    finally:
        env.close()


def test_stall_distance_configuration_is_used():
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.flat_cfg())
    try:
        env.reset(seed=0)
        env._waypoints = np.array([[1.0, 0.0, 0.0]])
        env.data.qpos[:2] = [0.2, 0.0]
        env._best_distance = 1.0
        env.cfg.termination.stall_distance = 0.5
        env._evaluate(np.zeros(env.model.nu), 0.0, False)
        assert env._stall_counter == 1
        env.cfg.termination.stall_distance = 0.1
        env._evaluate(np.zeros(env.model.nu), 0.0, False)
        assert env._stall_counter == 0
    finally:
        env.close()


def test_goal_cannot_be_reached_below_finish_platform():
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.flat_cfg())
    try:
        env.reset(seed=0)
        env._waypoints = np.array([[1.0, 0.0, 0.0]])
        env.data.qpos[:3] = [1.0, 0.0, 0.3]
        env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        assert not env._advance_waypoint()
        assert env._waypoints_reached == 0
        env.data.qpos[2] = 1.4
        assert env._advance_waypoint()
        assert env._advance_waypoint()
        assert env._waypoints_reached == 1
    finally:
        env.close()


def test_training_wall_clock_limit(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts import train

    callback = train.ProgressCallback(tmp_path, None, eval_every=1000, hours=10.0)
    callback.model = SimpleNamespace(num_timesteps=0)
    callback.locals = {"dones": [], "infos": []}
    callback.n_calls = 1
    monkeypatch.setattr(callback, "_evaluate", lambda: None)
    monkeypatch.setattr(train.time, "monotonic", lambda: 100.0)
    callback._on_training_start()
    assert callback.deadline == 36100.0
    assert callback._on_step()
    monkeypatch.setattr(train.time, "monotonic", lambda: 36100.0)
    assert not callback._on_step()


def test_training_evaluations_do_not_save_models(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts import train

    stats = dict(success_rate=1.0, completion_mean=1.0, endings={"goal": 20})
    monkeypatch.setattr(train, "evaluate_progress", lambda model, env: dict(stats))
    callback = train.ProgressCallback(tmp_path, None, eval_every=1)
    callback.model = SimpleNamespace(
        num_timesteps=0, logger=SimpleNamespace(record=lambda *args: None),
    )
    callback.locals = {"dones": [], "infos": []}
    callback._on_training_start()
    callback.model.num_timesteps = 1
    callback.n_calls = 1
    assert callback._on_step()
    callback._on_training_end()
    assert len((tmp_path / "evaluations.jsonl").read_text().splitlines()) == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == ["evaluations.jsonl"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_training_saves_only_final_or_interrupted(monkeypatch, tmp_path, interrupted):
    from types import SimpleNamespace
    from stable_baselines3.common import env_util
    from g1_parkour import env as env_module
    from scripts import train

    cfg = tasks.flat_cfg()
    fake_env = SimpleNamespace(get_attr=lambda name: [cfg] if name == "cfg" else [0.15], close=lambda: None)
    saved = []
    options = []

    def build_model(*args, **kwargs):
        options.append(kwargs)
        return model

    def learn(**kwargs):
        assert isinstance(kwargs["callback"], train.ProgressCallback)
        kwargs["callback"].init_callback(model)
        if interrupted:
            raise KeyboardInterrupt

    model = SimpleNamespace(num_timesteps=32, learn=learn, save=lambda path: saved.append(Path(path).name))
    model.get_env = lambda: fake_env
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "build_algo", lambda name: build_model)
    monkeypatch.setattr(env_util, "make_vec_env", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(env_module, "ParkourEnv", lambda cfg: fake_env)
    monkeypatch.setattr(sys, "argv", ["train.py", "Parkour-Flat-v0", "--run-name", "test", "--num-envs", "1",
                                    "--policy-hidden-sizes", "256", "256", "--ent-coef", "0.005"])
    if interrupted:
        with pytest.raises(SystemExit) as error:
            train.main()
        assert error.value.code == 130
    else:
        train.main()
    expected = "interrupted" if interrupted else "final"
    assert options[0]["policy_kwargs"] == {"net_arch": [256, 256]}
    assert options[0]["ent_coef"] == 0.005
    assert saved == [expected]
    assert (tmp_path / "models" / "test" / f"{expected}.state.json").exists()
