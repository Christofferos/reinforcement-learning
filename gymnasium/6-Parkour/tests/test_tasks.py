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


def test_rough_difficulty_grows_course_hills_pits_and_rubble_on_the_route():
    params = tasks.rough_cfg().terrain.params
    stats = []
    for difficulty in (0.0, 0.5, 1.0):
        lengths, blocks, relief = [], [], []
        for seed in range(5):
            terrain = build_terrain("rough", np.random.default_rng(seed), difficulty=difficulty, **params)
            field = terrain.heightfields[0]
            ground = field.data * field.elevation + field.pos[2]
            xs = np.linspace(field.pos[0] - field.radius_x, field.pos[0] + field.radius_x, ground.shape[1])
            ys = np.linspace(-field.radius_y, field.radius_y, ground.shape[0])
            assert ground[np.argmin(np.abs(ys)), np.argmin(np.abs(xs))] == pytest.approx(0.0)  # spawn level
            route_x = np.concatenate([[0.0], terrain.waypoints[:, 0]])
            route_y = np.concatenate([[0.0], terrain.waypoints[:, 1]])
            for box in terrain.boxes:
                (x, y, z), (hx, hy, hz) = box.pos, box.size
                assert abs(y - np.interp(x, route_x, route_y)) <= 1.0  # rubble lies on the route
                under = ground[np.abs(ys - y) <= hy][:, np.abs(xs - x) <= hx]
                assert z + hz >= under.max() + 0.05 - 1e-9  # never buried by a hill
            lengths.append(terrain.course_length)
            blocks.append(len(terrain.boxes))
            relief.append((ground.max(), ground.min()))
        hill, pit = np.mean(relief, axis=0)
        stats.append((np.mean(lengths), np.mean(blocks), hill, pit))
    (length0, blocks0, hill0, pit0), _, (length1, blocks1, hill1, pit1) = stats
    assert (length0, length1) == pytest.approx((16.0, 25.0))
    assert [s[0] for s in stats] == sorted(s[0] for s in stats)
    assert blocks0 == 0 and blocks1 >= 10 and [s[1] for s in stats] == sorted(s[1] for s in stats)
    assert pit0 < 0.0 < hill0 and hill1 > 2.5 * hill0 and pit1 < 2.5 * pit0


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
    # The bar is never between rays while any ray could reach it.
    for ahead in np.arange(-0.1, cfg.overhead_offsets[-1] + 0.1, 0.01):
        origin[:2] = (1.75 - ahead) * forward
        depths, _ = overhead_scan(model, data, origin, yaw, cfg)
        assert depths.min() == pytest.approx(clearance - cfg.overhead_origin_height), ahead


@pytest.mark.parametrize("module", ["hurdle", "climb"])
def test_overhead_scan_ignores_obstacles_taller_than_the_ray_origin(module):
    from dataclasses import replace
    import mujoco
    from g1_parkour.mdp.observations import ObservationCfg, overhead_scan
    from g1_parkour.terrain.parkour import MODULES, Cursor

    cfg = ObservationCfg(overhead_scan=True)
    built = MODULES[module](np.random.default_rng(0), 1.0, Cursor())
    obstacle = max(built.boxes, key=lambda box: box.pos[2] + box.size[2])
    assert obstacle.pos[2] + obstacle.size[2] > cfg.overhead_origin_height  # the ray starts inside it
    geoms = "".join(replace(box, material=None).to_xml() for box in built.boxes)
    model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{geoms}</worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    near_edge = obstacle.pos[0] - obstacle.size[0]
    origin = np.array([near_edge - cfg.overhead_offsets[-1] + 0.05, 0.0, cfg.overhead_origin_height])
    depths, _ = overhead_scan(model, data, origin, 0.0, cfg)
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
    # Spawn so only the farthest ray reaches the bar centre at x = 1.75.
    spawn_x = 1.75 - tasks.parkour_cfg().observation.overhead_offsets[-1]
    terrain = TerrainSpec(
        name="tunnel_test", boxes=tunnel.boxes,
        waypoints=np.asarray(tunnel.waypoints), spawn_pos=(spawn_x, 0.0, 0.0),
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
        shift = env.cfg.command_speed_difficulty_shift * env.terrain.difficulty
        low, high = low + shift, high + shift
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


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_flat_difficulty_scales_route_and_speed_window(difficulty):
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.terrain.difficulty = difficulty
    cfg.zero_command_prob = 0.0
    env = ParkourEnv(cfg)
    try:
        obs, _ = env.reset(seed=42)
        base_low, base_high = cfg.command_speed_range
        shift = cfg.command_speed_difficulty_shift * difficulty
        low, high = base_low + shift, base_high + shift
        assert high - low == pytest.approx(base_high - base_low)
        assert np.all((env._waypoint_speeds >= low) & (env._waypoint_speeds <= high))
        assert env.terrain.course_length == pytest.approx(12.0 + 6.0 * difficulty)
        assert env._waypoints[-1, 0] == pytest.approx(env.terrain.course_length)
        assert np.all(np.abs(env._waypoints[:, 1]) <= 1.25 + 1.25 * difficulty)
        easy = build_terrain("flat", np.random.default_rng(42), difficulty=0.0, **cfg.terrain.params)
        np.testing.assert_allclose(env._waypoints[0, 1], easy.waypoints[0, 1] * (1.0 + difficulty))
        route_length = np.linalg.norm(np.diff(np.vstack([np.zeros(3), env._waypoints]), axis=0), axis=1).sum()
        assert env.cfg.episode_length_s >= 1.25 * route_length / low
        lane_id = env.model.geom("lane_1").id
        assert env.model.geom_pos[lane_id, 0] == pytest.approx(env.terrain.course_length / 2.0)
        assert obs.shape == env.observation_space.shape
        speeds = env._waypoint_speeds.copy()
        env.reset(seed=42)
        np.testing.assert_array_equal(env._waypoint_speeds, speeds)
    finally:
        env.close()


def test_flat_curriculum_rebuilds_track_and_resamples_at_new_level():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.terrain.curriculum_window = 2
    cfg.terrain.difficulty_step = 0.5
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        model = env.model
        env._update_curriculum(1.0, success=True)
        env._update_curriculum(1.0, success=True)
        env.reset(seed=1)
        assert env._difficulty == pytest.approx(0.5)
        assert env.terrain.difficulty == pytest.approx(0.5)
        assert env.terrain.course_length == pytest.approx(15.0)
        assert env.model is not model
        assert np.all((env._waypoint_speeds >= 1.0) & (env._waypoint_speeds <= 2.0))
        model = env.model
        env.reset(seed=2)
        assert env.model is model
        env._update_curriculum(0.0)
        env.reset(seed=3)
        assert env.terrain.difficulty == pytest.approx(0.5), "one early failure must not demote"
        env._update_curriculum(0.0)
        env.reset(seed=3)
        assert env.terrain.difficulty == pytest.approx(0.0)
        assert env.terrain.course_length == pytest.approx(12.0)
        assert np.all((env._waypoint_speeds >= 0.5) & (env._waypoint_speeds <= 1.5))
    finally:
        env.close()


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


@pytest.mark.parametrize("cfg_factory", [tasks.flat_cfg, tasks.rough_cfg, tasks.parkour_cfg])
def test_reward_tracks_speed_and_direction(cfg_factory):
    import mujoco
    from g1_parkour.env import ParkourEnv

    cfg = cfg_factory()
    cfg.command_speed_range = (0.75, 0.75)
    cfg.command_speed_difficulty_shift = 0.0
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


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_angular_velocity_uses_mujoco_body_frame(axis):
    import mujoco
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.flat_cfg())
    try:
        env.reset(seed=0)
        env.data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, 0.0]
        env.data.qpos[4 + axis] = np.sin(np.pi / 4)
        env.data.qvel[3:6] = [0.4, 0.7, 1.2]
        mujoco.mj_forward(env.model, env.data)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY,
                                env._root_body_id, velocity, 0)
        local_angular_velocity = env.data.xmat[env._root_body_id].reshape(3, 3).T @ velocity[:3]
        obs = env._compute_observation()
        np.testing.assert_allclose(obs[3:6], local_angular_velocity, atol=1e-7)
        _, rewards, _, _ = env._evaluate(np.zeros(env.model.nu), 0.0, False)
        assert rewards["angular_velocity"] == pytest.approx(
            env.cfg.reward.angular_velocity * np.square(local_angular_velocity[:2]).sum()
        )
    finally:
        env.close()


@pytest.mark.parametrize("height, clearance, supported, surface_z, expected", [
    (1.25, None, True, 0.0, 0.0),
    (0.75, None, True, 0.0, -0.5),
    (0.72, 1.05, True, 0.0, 0.0),
    (1.25, 1.05, True, 0.0, 0.53),
    (2.0, None, False, 0.0, 0.0),
    (0.75, None, False, 0.0, 0.0),
    (0.75, None, True, -np.inf, 0.0),
    (1.25, None, True, -4.0, 0.0),
])
def test_height_reward_allows_tunnel_crouching_and_airborne_motion(
    monkeypatch, height, clearance, supported, surface_z, expected,
):
    from types import SimpleNamespace
    from g1_parkour.env import ParkourEnv
    from g1_parkour import mdp

    env = ParkourEnv.__new__(ParkourEnv)
    env.cfg = tasks.parkour_cfg()
    assert env.cfg.reward.base_height > 0.0
    env.model = SimpleNamespace(geom_group=np.array([0, 1]), geom_rbound=np.array([0.09, 0.0]))
    contact = SimpleNamespace(geom1=0, geom2=1, dist=-0.001,
                              frame=np.array([0.0, 0.0, 1.0]), pos=np.zeros(3))
    env.data = SimpleNamespace(
        qpos=np.array([0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0]),
        geom_xpos=np.array([[0.0, 0.0, height + 0.19], [0.0, 0.0, 0.0]]),
        contact=[contact] if supported else [],
    )
    depth = (env.cfg.observation.overhead_max_depth if clearance is None
             else clearance - env.cfg.observation.overhead_origin_height)
    monkeypatch.setattr(mdp, "overhead_scan", lambda *args: (np.full(3, depth), np.zeros((3, 3))))
    assert env._height_reward_error(surface_z) == pytest.approx(expected)


@pytest.mark.parametrize("overshoot", [False, True])
def test_waypoint_crossing_rewards_the_incoming_command(monkeypatch, overshoot):
    import mujoco
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.events.reset_noise_scale = 0.0
    cfg.events.reset_yaw_range = 0.0
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        env._waypoints = np.array([[0.1, 0.0, 0.0], [0.1, 3.0, 0.0]])
        env._waypoint_speeds = np.array([1.0, 4.0])
        env.data.qvel[:2] = [1.0, 0.0]

        def simulate_step(model, data, nstep):
            if overshoot:
                data.qpos[0] = 0.2
                mujoco.mj_forward(model, data)

        monkeypatch.setattr(mujoco, "mj_step", simulate_step)
        _, _, _, _, info = env.step(np.zeros(env.model.nu))
        assert env._target_index == 1
        assert info["command_speed"] == 4.0
        assert info["velocity_tracking_error"] == pytest.approx(0.0)
        assert info["reward/velocity_tracking"] == pytest.approx(cfg.reward.velocity_tracking)
        assert info["reward/heading"] == pytest.approx(cfg.reward.heading)
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
        window = cfg.terrain.curriculum_window
        needed = int(np.ceil(cfg.terrain.promote_success_rate * window))
        for _ in range(needed - 1):
            env._update_curriculum(1.0, success=True)
            env.reset()
        for _ in range(window - needed):
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


@pytest.mark.parametrize("cfg_factory", [tasks.rough_cfg, tasks.parkour_cfg])
@pytest.mark.parametrize("difficulty, completion, success", [
    (0.0, 0.0, False), (1.0, 1.0, True),
])
def test_curriculum_respects_difficulty_bounds(cfg_factory, difficulty, completion, success):
    from g1_parkour.env import ParkourEnv

    cfg = cfg_factory()
    assert cfg.terrain.curriculum
    assert cfg.terrain.difficulty_range == (0.0, 1.0)
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
        assert env._difficulty == pytest.approx(start + cfg.terrain.difficulty_step)
        env._update_curriculum(0.0)
        assert env._difficulty == pytest.approx(start)
        assert not env._curriculum_successes
    finally:
        env.close()


def test_curriculum_demotion_is_windowed_like_promotion():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.terrain.difficulty = 0.5
    env = ParkourEnv(cfg)
    try:
        window = cfg.terrain.curriculum_window
        # 87 % success with early falls sprinkled in: the old rule demoted on every fall.
        pattern = [(1.0, True)] * 7 + [(0.0, False)]
        for index in range(5 * window):
            completion, success = pattern[index % len(pattern)]
            env._update_curriculum(completion, success=success)
            assert env._difficulty >= 0.5
        assert env._difficulty > 0.5
        env.set_difficulty(0.5)
        # Mostly early failures demote, but only once a full window has been collected.
        for _ in range(window - 1):
            env._update_curriculum(0.0)
        assert env._difficulty == pytest.approx(0.5)
        env._update_curriculum(1.0, success=True)
        assert env._difficulty == pytest.approx(0.45)
        assert not env._curriculum_successes and not env._curriculum_early_failures
        # A mixed window (half successes, half early falls) holds the level.
        for index in range(window):
            if index % 2:
                env._update_curriculum(1.0, success=True)
            else:
                env._update_curriculum(0.0)
        assert env._difficulty == pytest.approx(0.45)
    finally:
        env.close()


def test_set_difficulty_rebuilds_without_curriculum():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.terrain.curriculum = False
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        env.set_difficulty(0.5)
        env.reset(seed=1)
        assert env.terrain.difficulty == pytest.approx(0.5)
        assert env.terrain.course_length == pytest.approx(15.0)
        env.set_difficulty(2.0)
        assert env._difficulty == pytest.approx(1.0)
        env._update_curriculum(0.0)
        assert env._difficulty == pytest.approx(1.0)
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


@pytest.mark.parametrize("cfg_factory", [tasks.flat_cfg, tasks.rough_cfg])
@pytest.mark.parametrize("height,tilt_degrees,expected_reason", [
    (0.70, 0.0, "bad_height"),
    (0.72, 0.0, ""),
    (1.0, 66.0, ""),
    (1.0, 67.0, "bad_orientation"),
])
def test_locomotion_terminates_unhealthy_pose_after_grace(monkeypatch, cfg_factory, height, tilt_degrees, expected_reason):
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(cfg_factory())
    try:
        env.reset(seed=0)
        env._waypoint_speeds[:] = 1.0
        monkeypatch.setattr(env, "_terrain_height_below", lambda *args: 0.0)
        env.data.qpos[2] = height
        half_angle = np.deg2rad(tilt_degrees) / 2.0
        env.data.qpos[3:7] = [np.cos(half_angle), np.sin(half_angle), 0.0, 0.0]
        grace = env.cfg.termination.recovery_grace_steps
        assert grace == 20
        assert env.cfg.termination.bad_height_scale * env.robot.healthy_height_range[0] == pytest.approx(0.7125)
        for _ in range(grace):
            _, terms, terminated, _ = env._evaluate(np.zeros(env.model.nu), 0.05, False)
            assert not terminated
            # Unhealthy poses earn no task reward even before the grace window runs out.
            if expected_reason:
                assert terms.get("velocity_tracking", 0.0) == 0.0 and terms["progress"] == 0.0
            else:
                assert terms["velocity_tracking"] > 0.0 and terms["progress"] > 0.0
        _, _, terminated, reason = env._evaluate(np.zeros(env.model.nu), 0.0, False)
        assert terminated == bool(expected_reason)
        assert reason == expected_reason
    finally:
        env.close()


def test_position_control_resets_to_standing_pose_and_tracks_targets():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.events.reset_noise_scale = 0.0
    cfg.events.reset_yaw_range = 0.0
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        knee = env.model.joint("right_knee")
        knee_angle = env.data.qpos[knee.qposadr[0]]
        low, high = knee.range
        assert low < knee_angle < high
        assert knee_angle == pytest.approx(np.radians(-30.0))
        feet_bottom = env.data.xpos[env._foot_body_ids, 2] + 0.1 - 0.075
        assert feet_bottom.min() == pytest.approx(0.01, abs=5e-3)
        assert env._foot_contacts().tolist() == [False, False]
        for _ in range(10):
            env.step(np.zeros(env.model.nu))
        assert env._foot_contacts().any()
        # Zero action holds the standing pose: knees stay close to their default target.
        for _ in range(20):
            env.step(np.zeros(env.model.nu))
        assert abs(env.data.qpos[knee.qposadr[0]] - np.radians(-30.0)) < np.radians(8.0)
        # A negative knee action drives the joint further into flexion, within torque limits.
        action = np.zeros(env.model.nu)
        action[env.model.actuator("right_knee").id] = -0.5
        before = env.data.qpos[knee.qposadr[0]]
        for _ in range(10):
            env.step(action)
        assert env.data.qpos[knee.qposadr[0]] < before
        assert np.all(np.abs(env.data.ctrl) <= np.abs(env._ctrl_range).max() + 1e-9)
    finally:
        env.close()


def test_zero_command_episode_rewards_standing_and_skips_curriculum():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.zero_command_prob = 1.0
    cfg.terrain.curriculum_window = 1
    env = ParkourEnv(cfg)
    try:
        obs, info = env.reset(seed=0)
        assert info["zero_command"]
        assert np.all(env._waypoint_speeds == 0.0)
        assert obs[-49] == pytest.approx(0.0)
        env._best_distance = 0.0  # no progress possible
        _, terms, _, _ = env._evaluate(np.zeros(env.model.nu), 0.1, False)
        assert env._stall_counter == 0
        assert terms["progress"] == 0.0
        assert terms["velocity_tracking"] > 0.9 * cfg.reward.velocity_tracking
        assert "feet_air_time" not in terms
        difficulty = env._difficulty
        env._info({}, reason="timeout")
        assert env._difficulty == difficulty
        cfg.zero_command_prob = 0.0
        env = ParkourEnv(cfg)
        _, info = env.reset(seed=0)
        assert not info["zero_command"] and np.all(env._waypoint_speeds > 0.0)
    finally:
        env.close()


def test_gait_terms_reward_single_stance_and_penalise_sliding():
    from g1_parkour.mdp.rewards import RewardCfg, terms

    cfg = RewardCfg(feet_air_time=2.0, feet_air_time_threshold=0.4, feet_slide=-0.1, velocity_tracking=1.0)
    state = dict(dt=0.015, progress=0.0, heading_alignment=0.0, upright=1.0, height_error=0.0,
                 lateral_speed=0.0, ang_vel=np.zeros(3), action=np.zeros(2), prev_action=np.zeros(2),
                 torque_power=0.0, joint_limit_violation=0.0, waypoints_reached=0, reached_goal=False,
                 fell=False, target_speed=1.0)
    assert terms(cfg, feet_air_time=0.3, **state)["feet_air_time"] == pytest.approx(0.6)
    assert terms(cfg, feet_air_time=0.9, **state)["feet_air_time"] == pytest.approx(0.8)
    assert "feet_air_time" not in terms(cfg, feet_air_time=0.3, **dict(state, target_speed=0.0))
    assert terms(cfg, feet_slide_speed=0.5, **state)["feet_slide"] == pytest.approx(-0.05)
    moving = dict(state, progress=0.03, velocity_error_squared=0.0)
    gated = terms(cfg, supported=False, **moving)
    assert gated["progress"] == 0.0 and "velocity_tracking" not in gated
    cfg.gate_on_support = False
    ungated = terms(cfg, supported=False, **moving)
    assert ungated["progress"] > 0.0 and ungated["velocity_tracking"] == pytest.approx(1.0)
    cfg.feet_step_ahead = 5.0
    assert terms(cfg, feet_step_ahead=0.6, **state)["feet_step_ahead"] == pytest.approx(3.0)
    assert terms(cfg, feet_step_ahead=-1.0, **state)["feet_step_ahead"] == pytest.approx(-5.0)
    assert "feet_step_ahead" not in terms(cfg, feet_step_ahead=1.0, **dict(state, target_speed=0.0))
    cfg.feet_step_symmetry = -5.0
    assert terms(cfg, feet_step_asymmetry=0.75, **state)["feet_step_symmetry"] == pytest.approx(-3.75)
    assert "feet_step_symmetry" not in terms(cfg, feet_step_asymmetry=0.75, **dict(state, target_speed=0.0))


def _place_feet(env, contacts, positions):
    """Fake foot contacts and planar foot positions: an x offset (robot facing +x) or an (x, y) pair."""
    env._foot_contacts = lambda: np.array(contacts, dtype=bool)
    for body_id, position in zip(env._foot_body_ids, positions):
        env.data.xpos[body_id, :2] = np.append(np.atleast_1d(position), 0.0)[:2]


def test_step_ahead_rewards_alternating_forward_steps_not_skipping():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.events.reset_yaw_range = 0.0
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        margin = cfg.reward.feet_step_ahead_margin
        _place_feet(env, [True, True], [0.0, 0.0])
        assert env._update_gait_state()[1] == 0.0  # simultaneous landing: no alternation info
        _place_feet(env, [True, False], [0.0, 0.0])
        env._update_gait_state()
        _place_feet(env, [True, True], [0.0, 0.3])
        assert env._update_gait_state()[1] == 0.0  # first touchdown only seeds the history
        assert env._last_touchdown_foot == 1
        # Walking: left swings and lands 0.3 m ahead of the right foot.
        _place_feet(env, [False, True], [0.0, 0.3])
        env._update_gait_state()
        _place_feet(env, [True, True], [0.6, 0.3])
        assert env._update_gait_state()[1] == pytest.approx(1.0)
        # Then right swings and lands ahead again -> full reward.
        _place_feet(env, [True, False], [0.6, 0.3])
        env._update_gait_state()
        _place_feet(env, [True, True], [0.6, 0.9])
        assert env._update_gait_state()[1] == pytest.approx(1.0)
        # Skip: left catches up but lands behind the right foot -> negative.
        _place_feet(env, [False, True], [0.6, 0.9])
        env._update_gait_state()
        _place_feet(env, [True, True], [0.9 - margin / 2, 0.9])
        assert env._update_gait_state()[1] == pytest.approx(-0.5)
        # Hop: the same foot lands twice in a row -> nothing.
        _place_feet(env, [False, True], [0.9 - margin / 2, 0.9])
        env._update_gait_state()
        _place_feet(env, [True, True], [1.5, 0.9])
        assert env._update_gait_state()[1] == 0.0
        # Reset clears the touchdown history.
        env.reset(seed=1)
        assert env._last_touchdown_foot == -1
    finally:
        env.close()


def test_every_stage_shares_the_locomotion_reward_scale():
    for cfg in (tasks.flat_cfg(), tasks.rough_cfg(), tasks.parkour_cfg()):
        for name, weight in tasks.LOCOMOTION_REWARDS.items():
            assert getattr(cfg.reward, name) == weight, (cfg.terrain.kind, name)


def test_gallop_is_penalised_along_the_travel_direction_whichever_way_the_torso_faces():
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.flat_cfg())
    try:
        env.reset(seed=0)
        yaw = np.radians(-33.0)  # torso turned toward the trailing foot, as the trained gallop does
        env.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        travel = np.array([1.0, 0.0])
        _place_feet(env, [False, True], [(0.0, 0.0), (0.0, 0.0)])
        env._update_gait_state(travel)  # right lands first: seeds the history
        _place_feet(env, [True, True], [(0.56, 0.0), (0.0, 0.0)])
        _, step_ahead, _, asymmetry = env._update_gait_state(travel)
        assert step_ahead == pytest.approx(1.0) and asymmetry == 0.0  # no right step to compare yet
        # The right foot catches up 0.19 m behind and 0.22 m to the side of the leading left foot.
        # On the turned torso's axis that offset is only -0.04 m, which the old projection paid
        # as a near-neutral step.
        trailing = (0.56 - 0.19, -0.22)
        torso_x = np.array([np.cos(yaw), np.sin(yaw)])
        assert torso_x @ (np.array(trailing) - [0.56, 0.0]) == pytest.approx(-0.04, abs=0.01)
        _place_feet(env, [True, False], [(0.56, 0.0), (0.0, 0.0)])
        env._update_gait_state(travel)
        _place_feet(env, [True, True], [(0.56, 0.0), trailing])
        _, step_ahead, _, asymmetry = env._update_gait_state(travel)
        # Behind along the travel direction, despite the yaw.
        assert step_ahead == pytest.approx(max(-0.19 / env.cfg.reward.feet_step_ahead_margin, -1.0))
        assert step_ahead < 0.0
        assert asymmetry == pytest.approx(0.75)  # 0.56 m lead step vs -0.19 m catch-up step
        # An even walk: each foot passes the other by 0.4 m.
        left = (trailing[0] + 0.4, 0.0)
        _place_feet(env, [False, True], [(0.56, 0.0), trailing])
        env._update_gait_state(travel)
        _place_feet(env, [True, True], [left, trailing])
        env._update_gait_state(travel)
        _place_feet(env, [True, False], [left, trailing])
        env._update_gait_state(travel)
        _place_feet(env, [True, True], [left, (left[0] + 0.4, -0.2)])
        _, step_ahead, _, asymmetry = env._update_gait_state(travel)
        assert step_ahead == pytest.approx(1.0) and asymmetry == pytest.approx(0.0)
        env.reset(seed=1)
        assert np.all(np.isnan(env._last_step_length))
    finally:
        env.close()


def test_scheduled_pushes_are_randomised_and_visualised():
    import mujoco
    from g1_parkour import mdp
    from g1_parkour.env import ParkourEnv
    from g1_parkour.mdp import EventCfg

    rng = np.random.default_rng(0)
    cfg = EventCfg(push_robot=True, push_interval_steps=200, push_velocity=0.4)
    steps, kicks = zip(*(mdp.schedule_push(rng, cfg, 10) for _ in range(200)))
    assert min(steps) >= 110 and max(steps) <= 310 and len(set(steps)) > 20
    magnitudes = np.linalg.norm(kicks, axis=1)
    assert np.all((magnitudes >= 0.2 - 1e-9) & (magnitudes <= 0.4 + 1e-9))
    assert magnitudes.min() < 0.25 and magnitudes.max() > 0.35
    assert np.ptp(np.arctan2(*np.array(kicks).T[::-1])) > 5.0
    cfg.push_velocity_difficulty_shift = 0.6
    hard = np.linalg.norm([mdp.schedule_push(rng, cfg, 0, 1.0)[1] for _ in range(200)], axis=1)
    assert np.all((hard >= 0.5 - 1e-9) & (hard <= 1.0 + 1e-9)) and hard.max() > 0.9
    half = np.linalg.norm([mdp.schedule_push(rng, cfg, 0, 0.5)[1] for _ in range(200)], axis=1)
    assert half.max() <= 0.7 + 1e-9 and half.max() > 0.6
    assert np.linalg.norm(mdp.schedule_push(rng, cfg, 0, 5.0)[1]) <= 1.0 + 1e-9

    env = ParkourEnv(tasks.flat_cfg())
    try:
        assert env.cfg.events.push_robot
        assert env.cfg.events.push_velocity_difficulty_shift > 0.0
        env.reset(seed=0)
        assert np.linalg.norm(env._pending_push) <= env.cfg.events.push_velocity + 1e-9
        env.set_difficulty(1.0)
        kicks = []
        for seed in range(30):
            env.reset(seed=seed)
            kicks.append(np.linalg.norm(env._pending_push))
        assert env.terrain.difficulty == pytest.approx(1.0)
        assert max(kicks) > env.cfg.events.push_velocity
        assert max(kicks) <= env.cfg.events.push_velocity + env.cfg.events.push_velocity_difficulty_shift + 1e-9
        env.set_difficulty(0.0)
        env.reset(seed=0)
        first_push = env._next_push_step
        assert 100 <= first_push <= 300
        assert env._push_markers() == []
        env._step_count = first_push - 5
        markers = env._push_markers()
        assert len(markers) == 1 and markers[0]["type"] == mujoco.mjtGeom.mjGEOM_SPHERE
        direction = env._pending_push / np.linalg.norm(env._pending_push)
        offset = markers[0]["pos"] - env.root_pos
        assert offset[2] == 0.0 and offset[:2] @ direction < 0.0
        env._step_count = first_push - 4
        step_travel = np.linalg.norm(offset) - np.linalg.norm(env._push_markers()[0]["pos"] - env.root_pos)
        expected = env.PUSH_BALL_SPEED_GAIN * np.linalg.norm(env._pending_push) * env.dt
        assert step_travel == pytest.approx(expected)
        pending = env._pending_push.copy()
        env._pending_push = 2.0 * pending
        far = np.linalg.norm(env._push_markers()[0]["pos"] - env.root_pos)
        env._step_count = first_push - 5
        assert np.linalg.norm(env._push_markers()[0]["pos"] - env.root_pos) - far == pytest.approx(2.0 * expected)
        env._pending_push = pending
        spawn = env.PUSH_BALL_IMPACT_DISTANCE + env.PUSH_BALL_SPAWN_DISTANCE_GAIN * np.linalg.norm(pending)
        distances = []
        for steps_left in range(1, first_push + 1):
            env._step_count = first_push - steps_left
            balls = [m for m in env._push_markers() if m["type"] == mujoco.mjtGeom.mjGEOM_SPHERE]
            if balls:
                distances.append(np.linalg.norm(balls[0]["pos"] - env.root_pos))
        assert max(distances) <= spawn + 1e-9
        assert max(distances) > spawn - expected
        env._step_count = first_push
        before = env.data.qvel[:2].copy()
        kick = env._pending_push.copy()
        env.step(np.zeros(env.model.nu))
        assert env._last_push[0] == first_push
        np.testing.assert_allclose(env._last_push[1], kick)
        assert env._next_push_step > first_push + 100
        assert not np.allclose(env.data.qvel[:2], before)
        markers = env._push_markers()
        arrows = [marker for marker in markers if marker["type"] == mujoco.mjtGeom.mjGEOM_ARROW]
        assert len(arrows) == 1
        assert arrows[0]["label"] == f"Push {np.linalg.norm(kick):.2f} m/s"
        mat = arrows[0]["mat"].reshape(3, 3)
        np.testing.assert_allclose(mat[:, 2], np.append(kick / np.linalg.norm(kick), 0.0), atol=1e-9)
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-9)
        viewer = type("Viewer", (), {})()
        viewer.vopt = mujoco.MjvOption()
        viewer.vopt.geomgroup[4] = 1
        viewer.scn = mujoco.MjvScene(env.model, maxgeom=4)
        env._add_waypoint_marker_to_scene(viewer, arrows[0])
        geom = viewer.scn.geoms[0]
        np.testing.assert_allclose(geom.mat.reshape(3, 3), mat, atol=1e-6)
        assert geom.type == mujoco.mjtGeom.mjGEOM_ARROW
        env._step_count = first_push + env.PUSH_PREVIEW_STEPS + 1
        assert all(marker["type"] != mujoco.mjtGeom.mjGEOM_ARROW for marker in env._push_markers())
        env.reset(seed=1)
        assert env._last_push is None
    finally:
        env.close()


def _rubble_test_model():
    import mujoco

    xml = """
    <mujoco>
      <asset>
        <material name="MatIce" rgba="1 1 1 1" specular="0.9" shininess="0.9" reflectance="0.4"/>
        <material name="MatMud" rgba="1 1 1 1" specular="0" shininess="0.02" reflectance="0"/>
      </asset>
      <worldbody>
        <geom name="ground" type="plane" size="5 5 0.1" friction="1 0.1 0.1"/>
        <geom name="rubble_run_0" type="box" size="1 1 0.1" pos="0 0 0" friction="1 0.1 0.1"/>
        <geom name="rubble_0_0" type="box" size="0.2 0.2 0.1" pos="0 0 0.2" friction="1 0.1 0.1"/>
        <geom name="rubble_1_0" type="box" size="0.2 0.2 0.1" pos="1 0 0.2" friction="1 0.1 0.1"/>
        <geom name="rough_block_2" type="box" size="0.3 0.3 0.1" pos="2 0 0.1" friction="1 0.1 0.1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    nominal = {
        "geom_friction": model.geom_friction.copy(),
        "body_mass": model.body_mass.copy(),
        "body_inertia": model.body_inertia.copy(),
        "actuator_gear": model.actuator_gear.copy(),
    }
    return model, nominal


def test_rubble_friction_randomization_tints_sections_ice_or_mud():
    import mujoco
    from g1_parkour.mdp import events

    model, nominal = _rubble_test_model()
    gid = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in range(model.ngeom)
    }
    ice_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatIce")
    mud_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatMud")

    cfg = events.EventCfg(randomize_friction=True)
    section0, section1 = [], []
    for seed in range(30):
        events.randomize_model(model, np.random.default_rng(seed), cfg, nominal)
        run, block0 = gid["rubble_run_0"], gid["rubble_0_0"]
        assert model.geom_friction[run, 0] == pytest.approx(model.geom_friction[block0, 0])
        scale0 = model.geom_friction[run, 0]
        scale1 = model.geom_friction[gid["rubble_1_0"], 0]
        assert 0.5 - 1e-9 <= scale0 <= 1.5 + 1e-9
        section0.append(scale0)
        section1.append(scale1)
        t = float(np.clip(scale0 - 0.5, 0.0, 1.0))
        expected = np.tile(t * events.MUD_RGBA + (1.0 - t) * events.ICE_RGBA, (2, 1))
        np.testing.assert_allclose(model.geom_rgba[[run, block0]], expected, atol=1e-6)
        assert model.geom_matid[run] == (mud_id if t >= 0.5 else ice_id)
        assert model.geom_matid[gid["rough_block_2"]] in (ice_id, mud_id)
    assert len(set(np.round(section0, 6))) > 10 and len(set(np.round(section1, 6))) > 10

    class _FixedRng:
        def __init__(self, value):
            self.value = value

        def uniform(self, *args, **kwargs):
            return self.value

    events.randomize_rubble_sections(model, _FixedRng(0.5), cfg, nominal)
    assert model.geom_matid[gid["rubble_0_0"]] == ice_id
    assert model.geom_rgba[gid["rubble_0_0"], 2] == pytest.approx(events.ICE_RGBA[2])
    events.randomize_rubble_sections(model, _FixedRng(1.5), cfg, nominal)
    assert model.geom_matid[gid["rubble_0_0"]] == mud_id
    assert model.geom_rgba[gid["rubble_0_0"], 2] == pytest.approx(events.MUD_RGBA[2])


def test_rubble_friction_tint_applies_in_rough_and_procedural_envs():
    from dataclasses import replace

    import mujoco
    from g1_parkour.env import ParkourEnv
    from g1_parkour.mdp import events

    cfg = tasks.parkour_cfg()
    cfg.terrain = replace(cfg.terrain, params={"num_modules": 2, "skills": ("rubble",)})
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        nominal = env._nominal["geom_friction"][:, 0]
        sections = {}
        for g in range(env.model.ngeom):
            key = events.rubble_section_key(
                mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            if key:
                sections.setdefault(key, []).append(g)
        assert len(sections) == 2  # one section per rubble module instance
        for ids in sections.values():
            scales = env.model.geom_friction[ids, 0] / nominal[ids]
            np.testing.assert_allclose(scales, scales[0])
            assert 0.5 - 1e-9 <= scales[0] <= 1.5 + 1e-9
            assert env.model.geom_matid[ids[0]] >= 0
    finally:
        env.close()

    env = ParkourEnv(tasks.rough_cfg())
    try:
        env.reset(seed=0)
        field = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "rough_terrain_geom")
        assert field >= 0
        scale = env.model.geom_friction[field, 0] / env._nominal["geom_friction"][field, 0]
        assert 0.5 - 1e-9 <= scale <= 1.5 + 1e-9
        assert env.model.geom_matid[field] >= 0
        assert not np.allclose(env.model.geom_rgba[field], (0.5, 0.5, 0.5, 1.0))
    finally:
        env.close()


def test_pushes_disabled_leave_dynamics_untouched():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.events.push_robot = False
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        assert env._next_push_step == -1
        assert env._push_markers() == []
        for _ in range(3):
            env.step(np.zeros(env.model.nu))
        assert env._last_push is None
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
    cfg.terrain.curriculum_window = 1
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


@pytest.mark.parametrize("missing_height", [False, True])
def test_episode_average_speed_and_height_reset(monkeypatch, missing_height):
    from g1_parkour.env import ParkourEnv
    from g1_parkour import env as env_module

    cfg = tasks.flat_cfg()
    cfg.episode_length_s = 0.03
    cfg.command_speed_range = None
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)

        def advance(model, data, nstep):
            data.qvel[:2] = [3.0, 4.0] if env._step_count == 0 else [0.0, 10.0]

        def surface_height(xy, from_z):
            if missing_height and env._step_count == 2:
                return -np.inf
            return env.root_pos[2] - (1.2 if env._step_count == 1 else 0.8)

        monkeypatch.setattr(env_module.mujoco, "mj_step", advance)
        monkeypatch.setattr(env, "_terrain_height_below", surface_height)
        action = np.zeros(env.model.nu)
        env.step(action)
        _, _, _, truncated, info = env.step(action)
        assert truncated
        assert info["avg_speed"] == pytest.approx(7.5)
        assert info["avg_height"] == pytest.approx(1.2 if missing_height else 1.0)
        env.reset(seed=1)
        env.step(action)
        info = env._info({}, reason="timeout")
        assert info["avg_speed"] == pytest.approx(5.0)
        assert info["avg_height"] == pytest.approx(1.2)
    finally:
        env.close()


def test_motion_logging_preserves_observations_and_rewards():
    from g1_parkour.env import ParkourEnv

    cfg = tasks.flat_cfg()
    cfg.episode_length_s = 0.03
    env = ParkourEnv(cfg)
    reference = ParkourEnv(cfg)
    try:
        observation, _ = env.reset(seed=42)
        baseline, _ = reference.reset(seed=42)
        np.testing.assert_array_equal(observation, baseline)
        assert env.observation_space.shape == reference.observation_space.shape == (116,)
        assert env.action_space.shape == reference.action_space.shape
        reference._episode_speed_sum = 1000.0
        reference._episode_height_sum = 1000.0
        reference._episode_height_samples = 1000
        for _ in range(2):
            result = env.step(np.full(env.model.nu, 0.1))
            expected = reference.step(np.full(reference.model.nu, 0.1))
            np.testing.assert_array_equal(result[0], expected[0])
            assert result[1:4] == expected[1:4]
            assert isinstance(result[1], float)
        assert result[4]["episode_reward_terms"] == expected[4]["episode_reward_terms"]
        assert result[4]["avg_speed"] != expected[4]["avg_speed"]
    finally:
        env.close()
        reference.close()


def test_tracking_reward_prefers_commanded_speed():
    from g1_parkour.mdp.rewards import terms

    cfg = tasks.parkour_cfg().reward
    state = dict(dt=0.015, heading_alignment=1.0, upright=1.0, height_error=0.0,
                 lateral_speed=0.0, ang_vel=np.zeros(3), action=np.zeros(17),
                 prev_action=np.zeros(17), torque_power=0.0, joint_limit_violation=0.0,
                 waypoints_reached=0, reached_goal=False, fell=False, target_speed=1.2)
    controlled = terms(cfg, progress=1.2 * state["dt"], velocity_error_squared=0.0, **state)
    fast = terms(cfg, progress=3.0 * state["dt"], velocity_error_squared=1.8**2, **state)
    per_step = 1.0 if cfg.progress_per_second else state["dt"]
    assert controlled["progress"] == pytest.approx(cfg.progress * 1.2 * per_step)
    assert sum(controlled.values()) > sum(fast.values())
    active_state = dict(state, action=np.ones(17), prev_action=np.ones(17))
    active = terms(cfg, progress=1.2 * state["dt"], **active_state)
    assert sum(active.values()) > 0.0


@pytest.mark.parametrize("cfg_factory", [tasks.flat_cfg, tasks.rough_cfg, tasks.parkour_cfg])
def test_progress_and_tracking_follow_waypoint_speed(cfg_factory):
    from g1_parkour.mdp.rewards import terms

    cfg = cfg_factory()
    assert not hasattr(cfg.reward, "overspeed")
    assert cfg.command_speed_range[1] + cfg.command_speed_difficulty_shift <= 4.0
    state = dict(dt=0.015, heading_alignment=1.0, upright=1.0, height_error=0.0,
                 lateral_speed=0.0, ang_vel=np.zeros(3), action=np.zeros(17),
                 prev_action=np.zeros(17), torque_power=0.0, joint_limit_violation=0.0,
                 waypoints_reached=0, reached_goal=False, fell=False, target_speed=4.0)
    slow = terms(cfg.reward, progress=1.5 * state["dt"],
                 velocity_error_squared=2.5**2, **state)
    matched = terms(cfg.reward, progress=4.0 * state["dt"],
                    velocity_error_squared=0.0, **state)
    assert matched["progress"] > slow["progress"]
    assert matched["velocity_tracking"] == pytest.approx(cfg.reward.velocity_tracking)
    assert matched["velocity_tracking"] > slow["velocity_tracking"]
    state["target_speed"] = 1.0
    overshooting = terms(cfg.reward, progress=4.0 * state["dt"], velocity_error_squared=3.0**2, **state)
    per_step = 1.0 if cfg.reward.progress_per_second else state["dt"]
    assert overshooting["progress"] == pytest.approx(cfg.reward.progress * per_step)
    assert "overspeed" not in overshooting
    assert overshooting["velocity_tracking"] < matched["velocity_tracking"]
    state["target_speed"] = None
    uncommanded = terms(cfg.reward, progress=4.0 * state["dt"], **state)
    assert uncommanded.get("velocity_tracking", 0.0) == 0.0
    assert "overspeed" not in uncommanded


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

    env = ParkourEnv(tasks.flat_cfg().replace(control_mode="torque"))
    try:
        env.reset(seed=0)
        env._waypoints = np.array([[3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        env.data.qpos[:2] = [2.25, 0.0]
        measured = {}

        def advance(model, data, nstep):
            data.qpos[0] += 0.1

        def evaluate(action, progress, reached, **command_context):
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
    callback.model = SimpleNamespace(num_timesteps=0, logger=SimpleNamespace(dump=lambda **kwargs: None))
    callback.locals = {"dones": [], "infos": []}
    callback.n_calls = 1
    monkeypatch.setattr(callback, "_evaluate", lambda: None)
    monkeypatch.setattr(train.time, "monotonic", lambda: 100.0)
    callback._on_training_start()
    assert callback.deadline == 36100.0
    assert callback._on_step()
    monkeypatch.setattr(train.time, "monotonic", lambda: 36100.0)
    assert not callback._on_step()


def test_resume_restores_difficulty_into_the_real_environments(tmp_path):
    import json
    from types import SimpleNamespace
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from scripts import train

    venv = VecNormalize(make_vec_env("Flat", n_envs=2, seed=0, vec_env_cls=DummyVecEnv))
    try:
        venv.reset()
        checkpoint = tmp_path / "checkpoint_100_steps.zip"
        train.state_path(checkpoint).write_text(json.dumps({"num_timesteps": 100, "difficulties": [0.4, 0.6]}))
        assert train.restore_difficulties(venv, checkpoint, num_envs=2) == [0.4, 0.6]
        real = [env.unwrapped for env in venv.venv.envs]
        assert [env._difficulty for env in real] == [0.4, 0.6]
        # Once the curriculum promotes a worker, saved state must follow the real environment.
        for _ in range(real[0].cfg.terrain.curriculum_window):
            real[0]._update_curriculum(1.0, success=True)
        assert real[0]._difficulty == pytest.approx(0.45)
        callback = train.ProgressCallback(tmp_path, None, eval_every=1000)
        callback.model = SimpleNamespace(num_timesteps=200, get_env=lambda: venv)
        callback.save_state("checkpoint_200_steps")
        saved = json.loads(train.state_path(tmp_path / "checkpoint_200_steps.zip").read_text())
        assert saved["difficulties"] == pytest.approx([0.45, 0.6])
        assert train.restore_difficulties(venv, tmp_path / "missing.zip", num_envs=2) is None
        with pytest.raises(ValueError):
            train.restore_difficulties(venv, checkpoint, num_envs=3)
    finally:
        venv.close()


def test_repair_state_rewrites_only_stale_files_from_the_log(tmp_path):
    import json
    from scripts import repair_state

    log = tmp_path / "progress.csv"
    log.write_text("time/total_timesteps,progress/difficulty\n1000,0.0\n2000,\n3000,0.52\n")
    stale = tmp_path / "checkpoint_3000_steps.state.json"
    stale.write_text(json.dumps({"num_timesteps": 3010, "difficulties": [0.0, 0.0]}))
    fine = tmp_path / "checkpoint_1000_steps.state.json"
    fine.write_text(json.dumps({"num_timesteps": 1000, "difficulties": [0.0, 0.05]}))
    preview = repair_state.repair_run(tmp_path, log, dry_run=True)
    assert [(path.name, level) for path, _, level in preview] == [("checkpoint_3000_steps.state.json", 0.5)]
    assert json.loads(stale.read_text())["difficulties"] == [0.0, 0.0]
    repair_state.repair_run(tmp_path, log)
    assert json.loads(stale.read_text()) == {"num_timesteps": 3010, "difficulties": [0.5, 0.5]}
    assert json.loads(fine.read_text())["difficulties"] == [0.0, 0.05]


def test_play_fixes_difficulty_from_flag_then_checkpoint_then_task(tmp_path):
    import json
    from g1_parkour.env import ParkourEnv
    from scripts import play

    checkpoint = tmp_path / "final.zip"
    cfg = tasks.TASKS["Procedural"]()
    assert play.configure_difficulty(cfg, None, str(checkpoint)) == "task default"
    assert cfg.terrain.difficulty == pytest.approx(tasks.TASKS["Procedural"]().terrain.difficulty)
    assert cfg.terrain.curriculum is False

    checkpoint.with_suffix(".state.json").write_text(json.dumps({"num_timesteps": 1, "difficulties": [0.4, 0.52]}))
    cfg = tasks.TASKS["Procedural"]()
    assert play.configure_difficulty(cfg, None, str(checkpoint)) == "saved with final.zip"
    assert cfg.terrain.difficulty == pytest.approx(0.45)

    cfg = tasks.TASKS["Procedural"]()
    assert play.configure_difficulty(cfg, 0.3, str(checkpoint)) == "--difficulty"
    assert cfg.terrain.difficulty == pytest.approx(0.3)
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        assert env.terrain.difficulty == pytest.approx(0.3)
        for _ in range(cfg.terrain.curriculum_window):
            env._update_curriculum(1.0, success=True)
        assert env._difficulty == pytest.approx(0.3)
    finally:
        env.close()


def test_training_saves_checkpoints_every_ten_minutes(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from scripts import train

    now = [100.0]
    saved = []
    callback = train.ProgressCallback(tmp_path, None, eval_every=1000)
    callback.model = SimpleNamespace(
        num_timesteps=5000, save=lambda path: saved.append(Path(path).name),
        get_env=lambda: SimpleNamespace(get_attr=lambda name: [0.15, 0.25]),
        logger=SimpleNamespace(dump=lambda **kwargs: None),
    )
    callback.locals = {"dones": [], "infos": []}
    monkeypatch.setattr(callback, "_evaluate", lambda: None)
    monkeypatch.setattr(train.time, "monotonic", lambda: now[0])
    callback._on_training_start()
    now[0] = 699.9
    assert callback._on_step()
    assert saved == []
    now[0] = 700.0
    assert callback._on_step()
    assert saved == ["checkpoint_5000_steps"]
    state = json.loads((tmp_path / "checkpoint_5000_steps.state.json").read_text())
    assert state == {"num_timesteps": 5000, "difficulties": [0.15, 0.25]}
    assert callback._on_step()
    now[0] = 1299.9
    assert callback._on_step()
    assert saved == ["checkpoint_5000_steps"]
    callback.model.num_timesteps = 6000
    now[0] = 1300.0
    assert callback._on_step()
    assert saved == ["checkpoint_5000_steps", "checkpoint_6000_steps"]
    assert (tmp_path / "checkpoint_6000_steps.state.json").exists()


def test_training_evaluations_do_not_save_models(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts import train

    stats = dict(success_rate=1.0, completion_mean=1.0, endings={"goal": 20})
    monkeypatch.setattr(train, "evaluate_progress", lambda model, env: dict(stats))
    eval_env = SimpleNamespace(
        cfg=SimpleNamespace(terrain=SimpleNamespace(difficulty_step=0.05)),
        set_difficulty=lambda difficulty: None,
    )
    callback = train.ProgressCallback(tmp_path, eval_env, eval_every=1)
    flushed_steps = []
    callback.model = SimpleNamespace(
        num_timesteps=0, logger=SimpleNamespace(
            record=lambda *args: None, dump=lambda step: flushed_steps.append(step),
        ),
        get_env=lambda: SimpleNamespace(get_attr=lambda name: [0.0]),
    )
    callback.locals = {"dones": [], "infos": []}
    callback._on_training_start()
    callback.model.num_timesteps = 1
    callback.n_calls = 1
    assert callback._on_step()
    callback._on_training_end()
    assert flushed_steps == [0, 1]
    assert len((tmp_path / "evaluations.jsonl").read_text().splitlines()) == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == ["evaluations.jsonl"]


def test_evaluation_follows_training_difficulty(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts import train

    pinned = []
    eval_env = SimpleNamespace(
        cfg=SimpleNamespace(terrain=SimpleNamespace(difficulty_step=0.05)),
        set_difficulty=pinned.append,
    )
    monkeypatch.setattr(train, "evaluate_progress", lambda model, env: dict(success_rate=1.0, endings={}))
    model = SimpleNamespace(
        num_timesteps=0,
        logger=SimpleNamespace(record=lambda *args: None, dump=lambda step: None),
        get_env=lambda: SimpleNamespace(get_attr=lambda name: [0.2, 0.3, 0.3, 0.35]),
    )
    callback = train.ProgressCallback(tmp_path, eval_env, eval_every=1)
    callback.model = model
    callback._evaluate()
    assert pinned == [pytest.approx(0.3)]


def test_training_logs_reward_totals_and_per_step_means(tmp_path):
    from types import SimpleNamespace
    from scripts import train

    recorded = {}
    callback = train.ProgressCallback(tmp_path, None, eval_every=1000)
    callback.model = SimpleNamespace(logger=SimpleNamespace(record=recorded.__setitem__))
    for length, tracking, speed, height in [(10, 15.0, 1.0, 0.8), (20, 10.0, 3.0, 1.2)]:
        callback.episodes.append({
            "episode": {"l": length}, "course_completion": 1.0, "difficulty": 0.0,
            "is_success": True, "termination": "goal", "velocity_tracking_error": 0.2,
            "episode_reward_terms": {"velocity_tracking": tracking},
            "avg_speed": speed, "avg_height": height,
        })
    callback._on_rollout_end()
    assert recorded["reward_terms/velocity_tracking"] == pytest.approx(12.5)
    assert recorded["reward_terms_per_step/velocity_tracking"] == pytest.approx(1.0)
    assert recorded["termination/goal"] == 1.0
    assert recorded["progress/velocity_tracking_error"] == pytest.approx(0.2)
    assert recorded["progress/avg_speed"] == pytest.approx(2.0)
    assert recorded["progress/avg_height"] == pytest.approx(1.0)
    callback.episodes.append({
        "course_completion": 0.0, "difficulty": 0.0, "is_success": False,
        "termination": "bad_height", "episode_reward_terms": {},
    })
    callback._on_rollout_end()
    assert recorded["reward_terms_per_step/velocity_tracking"] == pytest.approx(1.0)
    assert recorded["progress/avg_speed"] == pytest.approx(2.0)
    assert recorded["progress/avg_height"] == pytest.approx(1.0)


@pytest.mark.parametrize("interrupted", [False, True])
def test_short_training_saves_final_or_interrupted(monkeypatch, tmp_path, interrupted):
    import json
    from types import SimpleNamespace
    from stable_baselines3.common import env_util
    from g1_parkour import env as env_module
    from scripts import train

    cfg = tasks.flat_cfg()
    fake_env = SimpleNamespace(get_attr=lambda name: [cfg] if name == "cfg" else [0.15], close=lambda: None,
                               observation_mirror=lambda: "observation mirror",
                               action_mirror=lambda: "action mirror")
    saved = []
    options = []

    def build_model(*args, **kwargs):
        options.append(kwargs)
        return model

    def learn(**kwargs):
        assert isinstance(kwargs["callback"], train.ProgressCallback)
        kwargs["callback"].init_callback(model)
        model.logger.record("train/value_loss", 239.0)
        model.logger.dump(step=model.num_timesteps)
        if interrupted:
            model.logger.record("train/entropy_loss", -3.06)
            raise KeyboardInterrupt

    model = SimpleNamespace(num_timesteps=32, learn=learn, save=lambda path: saved.append(Path(path).name))
    model.set_logger = lambda logger: setattr(model, "logger", logger)
    model.get_env = lambda: fake_env
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "build_algo", lambda name: build_model)
    monkeypatch.setattr(env_util, "make_vec_env", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(env_module, "ParkourEnv", lambda cfg: fake_env)
    monkeypatch.setattr(sys, "argv", ["train.py", "Flat", "--run-name", "test", "--num-envs", "1",
                                    "--log-dir", str(tmp_path / "logs"), "--no-normalize",
                                    "--policy-hidden-sizes", "256", "256", "--ent-coef", "0.005"])
    if interrupted:
        with pytest.raises(SystemExit) as error:
            train.main()
        assert error.value.code == 130
    else:
        train.main()
    expected = "interrupted" if interrupted else "final"
    assert options[0]["policy_kwargs"] == {"net_arch": [256, 256], "log_std_init": -1.0}
    assert options[0]["ent_coef"] == 0.005
    assert (options[0]["symmetry"], options[0]["symmetry_coef"]) == ("both", 1.0)
    assert options[0]["mirror"] == ("observation mirror", "action mirror")
    assert saved == [expected]
    assert (tmp_path / "models" / "test" / f"{expected}.state.json").exists()
    log_dir = tmp_path / "logs" / "test"
    assert "train/value_loss" in (log_dir / "progress.csv").read_text()
    records = [json.loads(line) for line in (log_dir / "progress.json").read_text().splitlines()]
    assert records[0]["train/value_loss"] == 239.0
    if interrupted:
        assert records[-1]["train/entropy_loss"] == -3.06
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(log_dir)).Reload()
    value_loss = events.Scalars("train/value_loss")
    assert len(value_loss) == 1
    assert value_loss[0].step == 32
    assert value_loss[0].value == 239.0
    if interrupted:
        assert events.Scalars("train/entropy_loss")[0].value == pytest.approx(-3.06)
