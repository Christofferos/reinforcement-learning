"""Water tank module: terrain, scans, water forces and render-only effects."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour import tasks  # noqa: E402
from g1_parkour.env import ParkourEnv  # noqa: E402
from g1_parkour.terrain.parkour import (  # noqa: E402
    DEFAULT_SKILLS, MODULES, SAND_SOLREF, WATER_TANK_EASE, Cursor, make_parkour, mod_water_tank,
)
from g1_parkour.water import WaterEffects, submerged_fraction  # noqa: E402


def tank_env(difficulty=0.5, render_mode=None, **cfg_fields) -> ParkourEnv:
    cfg = tasks.parkour_cfg()
    cfg.terrain.params = {"num_modules": 1, "skills": ("water_tank",)}
    cfg.terrain.curriculum = False
    cfg.terrain.difficulty = difficulty
    for name, value in cfg_fields.items():
        setattr(cfg, name, value)
    return ParkourEnv(cfg, render_mode=render_mode, width=320, height=240)


def wade(env: ParkourEnv, into: float = 1.2) -> None:
    """Put the robot on the tank floor ``into`` metres past the rim, standing still."""
    water = env.terrain.water[0]
    env.data.qpos[0] = water.center[0] - water.half_size[0] + into
    env.data.qpos[1] = water.center[1]
    env.data.qpos[2] += 2.0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env._settle_feet_on_ground()


def test_water_tank_is_in_the_training_mix():
    assert "water_tank" in MODULES and "water_tank" in DEFAULT_SKILLS  # in the mix since 2026-10-17
    assert tasks.parkour_cfg().terrain.params["skill_weights"]["water_tank"] == 0.3
    tanks = 0
    for seed in range(20):
        course = make_parkour(np.random.default_rng(seed), difficulty=1.0, num_modules=8)
        assert len(course.water) == course.modules.count("water_tank")  # one volume per tank
        tanks += course.modules.count("water_tank")
    assert tanks > 0
    course = make_parkour(np.random.default_rng(0), num_modules=1, skills=("water_tank",))
    assert course.modules == ["start_pad", "water_tank"] and len(course.water) == 1


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_tank_drops_into_water_over_sand_and_ramps_back_to_the_rim(difficulty):
    cursor = Cursor(x=5.0, z=1.0)
    built = mod_water_tank(np.random.default_rng(0), difficulty, cursor)
    water = built.water[0]
    floor = next(b for b in built.boxes if b.name.startswith("water_tank_floor"))
    banks = [b for b in built.boxes if "bank" in b.name]
    eased = difficulty * WATER_TANK_EASE  # d = 0.9 builds what d = 0.3 did before 2026-10-17
    drop = 0.3 + 0.15 * eased
    assert floor.top_z == pytest.approx(1.0 - drop)
    # Shallower than the ground scan's -0.5 m clip, so the floor never reads as falling off.
    assert drop <= -tasks.parkour_cfg().observation.scan_clip[0] - 0.05
    assert water.surface_z == pytest.approx(0.9) and water.floor_z == pytest.approx(floor.top_z)
    assert len(banks) == int(2 + 2 * eased)  # the module floors it
    for bank in banks:
        assert bank.shape == "cylinder" and bank.solref == SAND_SOLREF
        crest = bank.pos[2] + bank.size[0] - floor.top_z
        assert crest == pytest.approx(0.05 + 0.2 * eased)
        # Gentle flanks: where a bank meets the floor its slope is at most 35 degrees.
        assert np.degrees(np.arccos(1.0 - crest / bank.size[0])) == pytest.approx(35.0)
    assert floor.solref == SAND_SOLREF
    assert cursor.z == pytest.approx(1.0) and built.waypoints[-1][0] == pytest.approx(cursor.x)


@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_tank_starts_on_its_own_deck_without_overlapping_the_previous_module(difficulty):
    from g1_parkour.terrain.parkour import PLATFORM_THICKNESS

    cursor = Cursor(x=5.0, z=1.0)
    built = mod_water_tank(np.random.default_rng(0), difficulty, cursor)
    deck = next(b for b in built.boxes if b.name.startswith("water_tank_deck"))
    floor = next(b for b in built.boxes if b.name.startswith("water_tank_floor"))
    # Nothing reaches back over the previous module, which ends at x = 5.
    assert min(b.pos[0] - b.size[0] for b in built.boxes if b.shape == "box" and not any(b.euler)) \
        == pytest.approx(5.0)
    assert deck.top_z == pytest.approx(1.0) and deck.pos[0] + deck.size[0] == pytest.approx(5.8)
    # One block from the rim to the tank's base: its end face is the whole riser.
    assert deck.pos[2] - deck.size[2] == pytest.approx(floor.top_z - PLATFORM_THICKNESS)
    assert floor.pos[0] - floor.size[0] == pytest.approx(5.8)


def test_drawn_water_stays_clear_of_the_tank_walls_and_floor():
    from g1_parkour.scene import WATER_INSET

    env = tank_env(render_mode="rgb_array")
    try:
        env.reset(seed=0)
        water = env.terrain.water[0]
        geom = env.model.geom(water.name)
        np.testing.assert_allclose(
            geom.size[:2], np.array(water.half_size) - WATER_INSET, atol=1e-4,
        )
        assert geom.pos[2] + geom.size[2] == pytest.approx(water.surface_z, abs=1e-4)
        assert geom.pos[2] - geom.size[2] == pytest.approx(water.floor_z + WATER_INSET, abs=1e-4)
    finally:
        env.close()


def test_wave_crests_are_sparse_varied_and_inside_the_tank():
    env = tank_env(difficulty=1.0)
    try:
        env.reset(seed=0)
        effects = WaterEffects(env.model, env.terrain.water, env._root_body_id)
        water = env.terrain.water[0]
        counts, headings, speeds = [], set(), set()
        for _ in range(600):  # 30 s at 20 frames per second
            env.data.time += 0.05
            effects.markers(env.data)
            counts.append(len(effects.crests))
            headings.update(np.round(effects.crests[:, 5], 3))
            speeds.update(np.round(effects.crests[:, 3], 3))
            assert np.all(np.abs(effects.crests[:, 1] - water.center[0]) <= water.half_size[0])
        span = 2.0 * water.half_size[0]
        expected = WaterEffects.WAVE_RATE * span * np.mean(WaterEffects.WAVE_LIFETIME)
        assert 0.5 * expected < np.mean(counts) < 1.5 * expected < 5
        assert len(headings) > 10 and len(speeds) > 10
    finally:
        env.close()


def test_only_soft_terrain_writes_contact_parameters():
    built = mod_water_tank(np.random.default_rng(0), 0.5, Cursor())
    floor = next(b for b in built.boxes if b.name.startswith("water_tank_floor"))
    assert 'solref="0.05 1.5"' in floor.to_xml() and 'solimp=' in floor.to_xml()
    run = MODULES["run"](np.random.default_rng(0), 0.5, Cursor()).boxes[0]
    assert "solref" not in run.to_xml() and "solimp" not in run.to_xml()


def test_tank_keeps_the_procedural_observation_and_action_sizes():
    env = tank_env()
    try:
        obs, _ = env.reset(seed=0)
        procedural = ParkourEnv(tasks.parkour_cfg())
        assert obs.shape == procedural.observation_space.shape
        assert env.action_space.shape == procedural.action_space.shape
        procedural.close()
    finally:
        env.close()


@pytest.mark.parametrize("render_mode", [None, "rgb_array"])
def test_ground_scan_reads_the_tank_floor_not_the_water(render_mode):
    env = tank_env(render_mode=render_mode)
    try:
        env.reset(seed=0)
        has_water_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.terrain.water[0].name) >= 0
        assert has_water_geom == (render_mode is not None)  # drawn only when rendering
        wade(env, into=0.3)
        floor = next(b for b in env.terrain.boxes if b.name.startswith("water_tank_floor"))
        under_robot = env._terrain_height_below(env.root_pos[:2], env.root_pos[2] + 0.1)
        assert under_robot <= env.terrain.water[0].surface_z - 0.1
        assert under_robot == pytest.approx(floor.top_z, abs=0.26)  # the floor, or a bank on it
    finally:
        env.close()


def test_overhead_rays_starting_inside_a_sand_bank_are_misses():
    from dataclasses import replace
    from g1_parkour.mdp.observations import ObservationCfg, overhead_scan

    cfg = ObservationCfg(overhead_scan=True)
    bank = next(b for b in mod_water_tank(np.random.default_rng(0), 1.0, Cursor()).boxes if "bank" in b.name)
    bank = replace(bank, pos=(0.0, 0.0, bank.pos[2]), euler=(90.0, 0.0, 0.0), material=None)
    model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{bank.to_xml()}</worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    crest = bank.pos[2] + bank.size[0]
    # Every ray starts below the crest, inside the bank: no ceiling.
    origin = np.array([-0.2, 0.0, crest - 0.1])
    depths, _ = overhead_scan(model, data, origin, 0.0, cfg)
    np.testing.assert_allclose(depths, cfg.overhead_max_depth)


def test_buoyancy_and_drag_follow_the_submerged_part_of_each_geom():
    env = tank_env()
    try:
        env.reset(seed=0)
        wade(env)
        water = env._water
        geoms = water.geoms
        pos, extent, _ = geoms.state(env.data)
        fraction = submerged_fraction(env.terrain.water[0], pos, extent)
        assert fraction.max() == 1.0 and fraction.min() == 0.0  # feet under, head above

        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        water.apply(env.data)
        lift = env.data.xfrc_applied[:, 2].sum()
        expected = env.terrain.water[0].density * 9.81 * np.sum(geoms.volume * fraction)
        assert lift == pytest.approx(expected)
        np.testing.assert_allclose(env.data.xfrc_applied[:, :2], 0.0, atol=1e-9)  # still: no drag

        # Wading forward at 1 m/s: drag pushes back, only on what is in the water.
        env.data.qvel[0] = 1.0
        mujoco.mj_forward(env.model, env.data)
        water.apply(env.data)
        drag = env.data.xfrc_applied[:, 0]
        assert drag.sum() < 0
        dry_bodies = {int(b) for b, f in zip(geoms.body_ids, fraction) if f == 0.0} - {
            int(b) for b, f in zip(geoms.body_ids, fraction) if f > 0.0
        }
        assert dry_bodies and all(drag[b] == 0.0 for b in dry_bodies)

        # Out of the tank, the forces are cleared.
        env.data.qpos[0] = 0.0
        mujoco.mj_forward(env.model, env.data)
        water.apply(env.data)
        np.testing.assert_array_equal(env.data.xfrc_applied, 0.0)
    finally:
        env.close()


def test_drag_slows_a_swinging_foot_only_below_the_surface():
    env = tank_env()
    try:
        env.reset(seed=0)
        wade(env)
        water, geoms = env._water, env._water.geoms
        foot = env._foot_body_ids[0]
        index = int(np.flatnonzero(geoms.body_ids == foot)[0])
        env.data.qvel[0] = 1.5
        mujoco.mj_forward(env.model, env.data)
        water.apply(env.data)
        submerged = env.data.xfrc_applied[foot, 0]
        # Lift the whole robot so the foot clears the surface: a high step pays no drag.
        surface = env.terrain.water[0].surface_z
        pos, extent, _ = geoms.state(env.data)
        env.data.qpos[2] += surface - (pos[index, 2] - extent[index]) + 0.01
        mujoco.mj_forward(env.model, env.data)
        water.apply(env.data)
        assert submerged < 0 and env.data.xfrc_applied[foot, 0] == 0.0
    finally:
        env.close()


def test_water_physics_off_or_dry_courses_leave_forces_untouched():
    env = tank_env(water_physics=False)
    try:
        env.reset(seed=0)
        wade(env)
        assert env._water is None
        env.step(np.zeros(env.action_space.shape))
        np.testing.assert_array_equal(env.data.xfrc_applied, 0.0)
    finally:
        env.close()
    env = ParkourEnv(tasks.parkour_cfg())
    try:
        env.reset(seed=0)
        assert env._water is None and not env.terrain.water
    finally:
        env.close()


def test_reset_clears_water_forces():
    env = tank_env()
    try:
        env.reset(seed=0)
        wade(env)
        env.step(np.zeros(env.action_space.shape))
        assert np.abs(env.data.xfrc_applied).sum() > 0
        env.cfg.terrain.resample_every_n_resets = 0  # reset without rebuilding the model
        env.reset()
        np.testing.assert_array_equal(env.data.xfrc_applied, 0.0)
    finally:
        env.close()


def test_sand_lets_feet_sink_where_concrete_does_not():
    env = tank_env(water_physics=False)
    try:
        sinking = {}
        for name, x in (("concrete", 1.0), ("sand", None)):
            env.reset(seed=0)
            if x is None:
                wade(env)
            else:
                env.data.qpos[0] = x
                mujoco.mj_forward(env.model, env.data)
                env._settle_feet_on_ground()
            feet = set().union(*env._foot_geom_ids)
            quat = env.data.qpos[3:7].copy()
            deepest = []
            for _ in range(600):  # 1.8 s held upright and still, sinking under its own weight
                env.data.qpos[3:7] = quat
                env.data.qvel[[0, 1, 3, 4, 5]] = 0.0
                mujoco.mj_step(env.model, env.data)
                deepest.append(max(
                    (-c.dist for c in env.data.contact[: env.data.ncon] if c.geom1 in feet or c.geom2 in feet),
                    default=np.nan,
                ))
            sinking[name] = float(np.nanmean(deepest[-100:]))
        assert sinking["concrete"] < 0.005 and 0.02 < sinking["sand"] < 0.06
    finally:
        env.close()


def test_effects_splash_on_fast_entry_and_expire():
    env = tank_env()
    try:
        env.reset(seed=0)
        effects = WaterEffects(env.model, env.terrain.water, env._root_body_id)
        water = env.terrain.water[0]
        env.data.qpos[0] = water.center[0] - water.half_size[0] + 1.2
        env.data.qpos[2] = water.surface_z + 1.5  # in the air above the water
        mujoco.mj_forward(env.model, env.data)
        assert not [m for m in effects.markers(env.data) if m["type"] == mujoco.mjtGeom.mjGEOM_SPHERE]
        env.data.qpos[2] = water.surface_z + 1.0  # feet now under, falling fast
        env.data.qvel[2] = -2.0
        env.data.time += 0.02
        mujoco.mj_forward(env.model, env.data)
        markers = effects.markers(env.data)
        assert [m for m in markers if m["type"] == mujoco.mjtGeom.mjGEOM_SPHERE]  # droplets
        assert all(m["group"] == WaterEffects.MARKER_GROUP for m in markers)
        env.data.time += WaterEffects.DROPLET_LIFETIME + 0.1
        assert not [m for m in effects.markers(env.data) if m["type"] == mujoco.mjtGeom.mjGEOM_SPHERE]
    finally:
        env.close()


def test_offscreen_renders_do_not_accumulate_markers():
    env = tank_env(render_mode="rgb_array")
    try:
        env.reset(seed=0)
        wade(env)
        try:
            env.render()
        except Exception as error:  # no OpenGL context on this machine
            pytest.skip(f"offscreen rendering unavailable: {error}")
        for _ in range(5):
            env.step(np.zeros(env.action_space.shape))
            env.render()
        assert env._mujoco_renderer._viewers["rgb_array"]._markers == []
    finally:
        env.close()


def test_the_eased_tank_at_d09_is_the_old_tank_at_d03():
    old = mod_water_tank(np.random.default_rng(0), 0.3, Cursor())  # built without easing below
    import g1_parkour.terrain.parkour as parkour

    at_09 = mod_water_tank(np.random.default_rng(0), 0.9, Cursor())
    original = parkour.WATER_TANK_EASE
    try:
        parkour.WATER_TANK_EASE = 1.0
        reference = mod_water_tank(np.random.default_rng(0), 0.3, Cursor())
    finally:
        parkour.WATER_TANK_EASE = original
    assert [b.pos for b in at_09.boxes] == [b.pos for b in reference.boxes]
    assert [b.size for b in at_09.boxes] == [b.size for b in reference.boxes]
