"""Experimental obstacle modules: moving parts, special contacts, wind, and the robot-only joint view."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour import mdp, tasks  # noqa: E402
from g1_parkour.env import ParkourEnv  # noqa: E402
from g1_parkour.terrain.actors import SEVEN_SEGMENTS  # noqa: E402
from g1_parkour.mdp.observations import TERRAIN_RAY_MASK  # noqa: E402
from g1_parkour.scene import SHARED_ASSETS  # noqa: E402
from g1_parkour.terrain.parkour import (  # noqa: E402
    DEFAULT_SKILLS, EXPERIMENTAL_SKILLS, INTRODUCED_SKILLS, MODULES, TRAMPOLINE_RAMP_ANGLE, TRAMPOLINE_SLOT, TRAMPOLINE_TILT, Cursor,
    make_parkour,
)

NEW_MODULES = (
    "spinning_barrel", "icy_slide", "seesaw", "trampoline", "wind_turbine", "sliding_elevator",
    "crumbling_bridge", "tilt_passage", "ball_balance", "air_vent",
)


def module_env(skill, difficulty=0.5, render_mode=None) -> ParkourEnv:
    cfg = tasks.parkour_cfg()
    cfg.terrain.params = {"num_modules": 1, "skills": (skill,)}
    cfg.terrain.curriculum = False
    cfg.terrain.difficulty = difficulty
    cfg.events.reset_yaw_range = 0.0
    return ParkourEnv(cfg, render_mode=render_mode, width=320, height=240)


def crate_world(skill, difficulty, crate_at, seed=0):
    """The module's terrain alone with a free 40 kg crate in place of the robot."""
    out = MODULES[skill](np.random.default_rng(seed), difficulty, Cursor())
    jointed = [b for b in out.bodies if b.joint]
    xml = f"""<mujoco><option timestep="0.003" integrator="RK4"/><asset>{SHARED_ASSETS}</asset><worldbody>
      {''.join(b.to_xml() for b in out.boxes)}
      <body name="crate" pos="{crate_at[0]} {crate_at[1]} {crate_at[2]}"><freejoint/>
        <geom type="box" size="0.15 0.15 0.15" mass="40" friction="1 0.1 0.1"/></body>
      {''.join(b.to_xml(False) for b in out.bodies)}
      </worldbody><contact>{''.join(f'<exclude body1="world" body2="{b.name}"/>' for b in jointed)}</contact>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return out, model, data


def joint_state(model, data, body):
    joint = model.joint(body.joint_name).id
    return model.jnt_qposadr[joint], model.jnt_dofadr[joint]


def test_the_newest_modules_stay_experimental_and_the_rest_were_introduced():
    for skill in NEW_MODULES:
        assert skill in MODULES
        assert (skill in EXPERIMENTAL_SKILLS) != (skill in DEFAULT_SKILLS)
    assert set(EXPERIMENTAL_SKILLS) == {"air_vent", "trampoline"}
    assert set(INTRODUCED_SKILLS) <= set(DEFAULT_SKILLS)
    weights = tasks.parkour_cfg().terrain.params["skill_weights"]
    assert {"tilt_passage", "ball_balance"} <= set(INTRODUCED_SKILLS)  # in training since 2026-10-17
    assert all(weights[skill] == 0.3 for skill in INTRODUCED_SKILLS)
    assert not set(weights) & set(EXPERIMENTAL_SKILLS)
    for seed in range(20):
        course = make_parkour(np.random.default_rng(seed), difficulty=1.0, num_modules=8)
        assert not set(course.modules) & set(EXPERIMENTAL_SKILLS)


@pytest.mark.parametrize("skill", NEW_MODULES)
@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_modules_keep_the_procedural_sizes_and_draw_decoration_only_when_rendering(skill, difficulty):
    procedural = ParkourEnv(tasks.parkour_cfg())
    sizes = procedural.observation_space.shape, procedural.action_space.shape
    procedural.close()
    decoration = None
    for render_mode in (None, "rgb_array"):
        env = module_env(skill, difficulty, render_mode)
        try:
            obs, _ = env.reset(seed=0)
            assert (obs.shape, env.action_space.shape) == sizes
            for _ in range(5):
                obs, reward, *_ = env.step(np.zeros(env.action_space.shape))
                assert np.all(np.isfinite(obs)) and np.isfinite(reward)
            names = {env.model.geom(g).name for g in range(env.model.ngeom)}
            decoration = {
                geom.name for body in env.terrain.bodies for geom in body.geoms if geom.visual
            } | {box.name for box in env.terrain.boxes if box.visual}
            assert (decoration <= names) == (render_mode is not None) or not decoration
            assert env.model.jnt_qposadr[env._robot_joint_ids[0]] == 7  # robot first
        finally:
            env.close()


def test_observation_ignores_terrain_joints():
    env = module_env("spinning_barrel", 1.0)
    try:
        env.observation_mirror()  # the barrels' hinges have no mirror partner, and need none
        env.reset(seed=0)
        assert env.model.nq > 7 + len(env._robot_joint_ids)  # the barrels' hinges
        barrel_adr = [env.model.jnt_qposadr[env.model.joint(b.joint_name).id] for b in env.terrain.bodies]
        assert not set(barrel_adr) & set(env._robot_qpos_adr.tolist())
        before = env._compute_observation()
        env.data.qpos[barrel_adr] += 1.0
        env.data.qvel[[env.model.jnt_dofadr[env.model.joint(b.joint_name).id] for b in env.terrain.bodies]] = 3.0
        np.testing.assert_array_equal(env._compute_observation()[:60], before[:60])  # proprioception unchanged
    finally:
        env.close()


def test_joint_limit_penalty_ignores_a_seesaw_at_its_stop():
    env = module_env("seesaw", 1.0)
    try:
        env.reset(seed=0)
        seesaw = env.model.joint(env.terrain.bodies[0].joint_name).id
        assert seesaw not in set(env._hinge_jnt_ids.tolist())
    finally:
        env.close()


def test_moving_parts_never_touch_static_terrain():
    env = module_env("trampoline", 0.5)
    try:
        xml = next(Path(env.build_dir).glob("*.xml")).read_text()
        assert f'<exclude body1="world" body2="{env.terrain.bodies[0].name}"/>' in xml
    finally:
        env.close()


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_icy_slide_is_slippery_but_never_steep_enough_to_slide_on_its_own(difficulty):
    out = MODULES["icy_slide"](np.random.default_rng(0), difficulty, Cursor())
    angle = np.radians(abs(out.boxes[0].euler[1]))
    assert np.degrees(angle) <= 9.0 + 1e-9
    assert out.boxes[0].friction[0] > np.tan(angle)
    x = 0.8
    _, model, data = crate_world("icy_slide", difficulty, (x, 0.0, -x * np.tan(angle) + 0.16 / np.cos(angle)))
    for _ in range(int(2.0 / 0.003)):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[0] - x) < 0.1  # at most a creep: MuJoCo's soft contacts, at d = 1


def test_slide_ice_is_glassier_the_less_friction_it_has():
    from g1_parkour.terrain.parkour import ICE_SLIDE_REFLECTANCE

    env = module_env("icy_slide", 0.0, render_mode=None)
    try:
        model = env.model
        reflectance = []
        for difficulty in np.linspace(0.0, 1.0, 9):
            slab = MODULES["icy_slide"](np.random.default_rng(0), difficulty, Cursor()).boxes[0]
            reflectance.append(model.mat_reflectance[model.material(slab.material).id])
        assert reflectance[0] == pytest.approx(ICE_SLIDE_REFLECTANCE[0])
        assert reflectance[-1] == pytest.approx(ICE_SLIDE_REFLECTANCE[-1])
        assert all(a <= b for a, b in zip(reflectance, reflectance[1:]))
    finally:
        env.close()


def test_barrel_spins_under_an_off_centre_load():
    out, model, data = crate_world("spinning_barrel", 1.0, (0.0, 0.0, 0.0))
    barrel = out.bodies[0]
    qadr, dadr = joint_state(model, data, barrel)
    data.qpos[0:3] = (barrel.pos[0] + 0.15, 0.0, barrel.pos[2] + 0.35 + 0.2)
    turned = 0.0
    for _ in range(150):  # the crate rolls it, then wedges between two barrels
        mujoco.mj_step(model, data)
        turned = max(turned, abs(data.qpos[qadr]))
    assert turned > np.radians(5)


@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_seesaw_tips_only_once_the_load_is_past_the_pivot(difficulty):
    for offset, tips in ((-0.8, False), (1.0, True)):
        out, model, data = crate_world("seesaw", difficulty, (0.0, 0.0, 0.0))
        seesaw = out.bodies[0]
        qadr, _ = joint_state(model, data, seesaw)
        # A 40 kg crate only tips it far past the pivot: put it 1 m out, or 0.8 m short.
        data.qpos[0:3] = (seesaw.pos[0] + offset, 0.0, seesaw.pos[2] + 0.3)
        for _ in range(800):
            mujoco.mj_step(model, data)
        full = model.jnt_range[model.joint(seesaw.joint_name).id][1]
        assert (data.qpos[qadr] > 0.9 * full) == tips, (offset, data.qpos[qadr], full)


def test_trampoline_throws_a_dropped_load_back_up():
    out, model, data = crate_world("trampoline", 0.5, (0.0, 0.0, 0.0))
    bed = out.bodies[0]
    qadr, _ = joint_state(model, data, bed)
    data.qpos[0:3] = (bed.pos[0] + 0.6, 0.0, bed.pos[2] + 0.15 + 0.5)
    lowest, rebound_speed = 0.0, 0.0
    for _ in range(400):
        mujoco.mj_step(model, data)
        lowest = min(lowest, data.qpos[qadr])
        rebound_speed = max(rebound_speed, data.qvel[2])
    assert lowest < -0.15  # the bed gives
    assert rebound_speed > 1.5  # and throws the crate back up (it landed at ~3.1 m/s)


def test_elevator_carries_a_rider_across_and_follows_its_schedule():
    out, model, data = crate_world("sliding_elevator", 1.0, (0.0, 0.0, 0.0))
    platform, actor = out.bodies[0], out.actors[0]
    qadr, dadr = joint_state(model, data, platform)
    data.qpos[0:3] = (platform.pos[0] + 0.5, 0.0, platform.pos[2] + 0.16)
    x0 = data.qpos[0]
    for step in range(int(actor.period / 2.0 / 0.015)):  # one crossing, out to the far side
        data.qpos[qadr], data.qvel[dadr] = actor.state(step * 0.015)
        for _ in range(5):
            mujoco.mj_step(model, data)
    assert actor.state(0.0) == (0.0, 0.0)
    assert actor.state(actor.period / 2.0)[0] == pytest.approx(actor.travel)
    assert data.qpos[qadr] > 0.95 * actor.travel
    assert data.qpos[0] - x0 == pytest.approx(data.qpos[qadr], abs=0.05)


def test_elevator_runs_only_near_and_pauses_the_stall_clock_at_the_boarding_edge():
    env = module_env("sliding_elevator", 1.0)
    try:
        env.reset(seed=0)
        actor = env._actors[0]
        env.data.qpos[0] = actor.x_range[0] - actor.NEAR_MARGIN - 0.5
        frozen = env.data.qpos[actor.qpos_adr]
        actor.before_step(env)
        assert actor.time == pytest.approx(actor.time) and env.data.qpos[actor.qpos_adr] == frozen
        env.data.qpos[0] = actor.boarding[1] - 0.3
        actor.time = actor.period / 2.0  # platform at the far side
        actor.before_step(env)
        assert env.data.qpos[actor.qpos_adr] > 0.05 and actor.holds_stall(env)
        env._stall_counter = 7
        env._best_distance = 0.0  # no progress possible
        env._evaluate(np.zeros(env.action_space.shape), 0.0, False)
        assert env._stall_counter == 7
    finally:
        env.close()


def test_bridge_planks_fall_a_crumble_delay_after_first_contact():
    env = module_env("crumbling_bridge", 0.5)
    try:
        env.reset(seed=0)
        actor = env._actors[0]
        plank = 2
        body = env.terrain.bodies[plank]
        home_z = actor.home_pos[plank][2]
        env.data.qpos[0] = body.pos[0]
        env.data.qpos[2] += 1.0
        mujoco.mj_forward(env.model, env.data)
        env._settle_feet_on_ground()
        heights, times = [], []
        for _ in range(int(3.0 / env.dt)):
            env.step(np.zeros(env.action_space.shape))
            heights.append(env.data.mocap_pos[actor.mocap_ids[plank]][2])
            times.append(env.data.time)
        touched = actor.touched[plank]
        assert np.isfinite(touched)
        heights, times = np.array(heights), np.array(times)
        assert np.all(heights[times < touched + actor.delay - 0.02] == home_z)
        assert heights[-1] < home_z - 1.0
        env.reset(seed=1)  # and a reset puts every plank back
        np.testing.assert_allclose(env.data.mocap_pos[actor.mocap_ids], actor.home_pos)
    finally:
        env.close()


def test_wind_pushes_only_on_its_module_and_away_from_the_cannon():
    env = module_env("wind_turbine", 1.0)
    try:
        env.reset(seed=0)
        actor = env._actors[0]
        for _ in range(100):  # on the start pad
            actor.before_step(env)
            env._step_count += 1
        assert actor.last_push is None
        env.data.qpos[0] = (actor.x_range[0] + actor.x_range[1]) / 2.0
        kicks = []
        for _ in range(int(6.0 / env.dt)):
            env.data.qvel[:] = 0.0
            actor.before_step(env)
            if actor.last_push is not None and (not kicks or kicks[-1][0] != actor.last_push[0]):
                kicks.append(actor.last_push)
                assert np.sign(env.data.qvel[1]) == -actor.side  # pushed away from the cannon
            env._step_count += 1
        assert 3 <= len(kicks) <= 12
        assert all(np.linalg.norm(kick) <= actor.strength + 1e-9 for _, kick in kicks)
    finally:
        env.close()


def test_bridge_planks_fall_their_own_random_ways():
    env = module_env("crumbling_bridge", 1.0)
    try:
        env.reset(seed=0)
        actor = env._actors[0]
        first = (actor.drift.copy(), actor.tumble_axis.copy(), actor.tumble_rate.copy())
        assert len(np.unique(np.sign(actor.drift[:, 1]))) == 2  # off both edges
        assert len(np.unique(np.sign(actor.tumble_rate))) == 2  # tumbling both ways
        assert np.ptp(np.arctan2(actor.tumble_axis[:, 1], actor.tumble_axis[:, 0])) > 1.0
        env.reset(seed=1)
        actor = env._actors[0]  # each reset builds a new course, with its own actors
        assert not np.allclose(actor.drift[:2], first[0][:2]) and not np.allclose(actor.tumble_rate[:2], first[2][:2])
        # A falling plank follows its drift and drops under gravity.
        actor.touched[0] = env.data.time - actor.delay - 0.5
        env.data.qpos[0] = actor.x_range[0]
        actor.before_step(env)
        moved = env.data.mocap_pos[actor.mocap_ids[0]] - actor.home_pos[0]
        np.testing.assert_allclose(moved[:2], actor.drift[0, :2] * 0.5)
        assert moved[2] == pytest.approx(-0.5 * 9.81 * 0.25)
    finally:
        env.close()


@pytest.mark.parametrize("difficulty, barrels", [(0.0, 1), (0.24, 1), (0.25, 2), (0.5, 2), (0.75, 3), (1.0, 3)])
def test_barrel_count_rounds_halves_up(difficulty, barrels):
    assert len(MODULES["spinning_barrel"](np.random.default_rng(0), difficulty, Cursor()).bodies) == barrels


def test_courses_build_every_module_at_the_requested_difficulty():
    # Eight d=1 barrel modules along the whole course, not 0.43 rising to 1 as courses once ramped.
    cfg = tasks.parkour_cfg()
    params = {**cfg.terrain.params, "skills": ("spinning_barrel",)}
    course = make_parkour(np.random.default_rng(0), difficulty=1.0, **params)
    counts = {}
    for body in course.bodies:
        module = body.name.split("_")[2]
        counts[module] = counts.get(module, 0) + 1
    assert len(counts) == 8 and set(counts.values()) == {3}


def test_icy_slide_has_a_snowman_in_a_run_out_corner_drawn_only_when_rendering():
    corners = set()
    for seed in range(12):
        out = MODULES["icy_slide"](np.random.default_rng(seed), 0.5, Cursor())
        runout = next(b for b in out.boxes if "runout" in b.name)
        snowman = [b for b in out.boxes if "snowman" in b.name]
        assert snowman and all(b.visual for b in snowman)
        carrot = next(b for b in snowman if b.name.endswith("carrot"))
        base = next(b for b in snowman if b.name.endswith("snow_0"))
        assert abs(base.pos[1]) > 0.6  # off the robot's line
        assert base.pos[2] == pytest.approx(runout.top_z + 0.15)
        corners.add((base.pos[0] > runout.pos[0], base.pos[1] > 0))
        assert carrot.shape == "ellipsoid" and carrot.rgba[:3] == (1.0, 0.5, 0.08)
    assert len(corners) == 4


def _seesaw_tip(difficulty, past_pivot, seconds=3.0):
    """Seconds from a 42 kg load landing ``past_pivot`` m beyond the pivot until the plank is
    down at the exit, or None if it never tips."""
    out = MODULES["seesaw"](np.random.default_rng(0), difficulty, Cursor())
    seesaw = out.bodies[0]
    xml = f"""<mujoco><option timestep="0.003" integrator="RK4"/><asset>{SHARED_ASSETS}</asset><worldbody>
      {''.join(b.to_xml() for b in out.boxes)}
      <body name="load" pos="{seesaw.pos[0] + past_pivot} 0 {seesaw.pos[2] + 0.6}"><freejoint/>
        <geom type="box" size="0.12 0.12 0.12" mass="42" friction="1 0.1 0.1"/></body>
      {seesaw.to_xml(False)}</worldbody><contact><exclude body1="world" body2="{seesaw.name}"/></contact></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joint = model.joint(seesaw.joint_name).id
    plank = model.body(seesaw.name).id
    landed = None
    for _ in range(int(seconds / 0.003)):
        mujoco.mj_step(model, data)
        if landed is None and any(plank in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]) for c in data.contact[: data.ncon]):
            landed = data.time
        if landed is not None and data.qpos[model.jnt_qposadr[joint]] > 0.95 * model.jnt_range[joint][1]:
            return data.time - landed
    return None


def test_seesaw_tips_earlier_and_faster_at_higher_difficulty():
    assert _seesaw_tip(1.0, 0.1) is None and _seesaw_tip(1.0, 0.2) is not None  # tips from ~0.15 m
    assert _seesaw_tip(0.0, 0.45) is None and _seesaw_tip(0.0, 0.6) is not None  # from ~0.5 m
    assert _seesaw_tip(1.0, 0.8) == pytest.approx(0.5, abs=0.05)
    assert _seesaw_tip(0.0, 0.8) > 0.85


def test_trampoline_launches_a_robot_that_drops_onto_it():
    env = module_env("trampoline", 0.5)
    try:
        env.reset(seed=0)
        actor = env._actors[0]
        bed = env.terrain.bodies[0]
        env.data.qpos[0] = bed.pos[0] + 0.6
        env.data.qpos[2] += 2.0
        mujoco.mj_forward(env.model, env.data)
        env._settle_feet_on_ground()
        env.data.qpos[2] += 0.4  # dropped from 0.4 m
        mujoco.mj_forward(env.model, env.data)
        rising = 0.0
        for _ in range(60):
            env.step(np.zeros(env.action_space.shape))
            rising = max(rising, env.data.qvel[2])
        assert actor.launches == 1 and rising >= actor.launch - 0.3
        # Along the tilted bed's normal, so it also throws forward (it dropped straight down).
        assert env.data.qvel[0] > 0.6 * actor.launch * actor.normal[0]
        # Standing on it does not launch (a fall onto it would, as any fast landing does).
        env.reset(seed=0)
        actor = env._actors[0]
        env.data.qpos[0] = bed.pos[0] + 0.6
        env.data.qpos[2] += 2.0
        mujoco.mj_forward(env.model, env.data)
        env._settle_feet_on_ground()
        for _ in range(10):
            env.step(np.zeros(env.action_space.shape))
        assert actor.launches == 0 and env.data.qvel[2] > -actor.ARM_SPEED
    finally:
        env.close()


@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_trampoline_reads_as_a_v_with_a_pit_at_the_bottom(difficulty):
    """Its scan signature: bed sloping down, a full-row pit, then the landing sloping up."""
    out, model, data = crate_world("trampoline", difficulty, (-5.0, 0.0, 5.0))
    actor = out.actors[0]
    gid = np.array([-1], dtype=np.int32)

    def ground(x, y=0.0):
        dist = mujoco.mj_ray(model, data, np.array([x, y, 3.0]), np.array([0.0, 0.0, -1.0]), TERRAIN_RAY_MASK, 1, -1, gid)
        return None if dist < 0 else 3.0 - dist

    bed_x, front_x = actor.x_range
    slope = np.tan(np.radians(TRAMPOLINE_TILT))
    assert ground(bed_x + 0.2) - ground(bed_x + 0.8) == pytest.approx(0.6 * slope, abs=0.005)
    slot = [x for x in np.arange(front_x - 0.1, front_x + 0.4, 0.002) if ground(x) is None]
    assert slot[-1] - slot[0] >= 0.2 - 0.005  # any 0.2 m scan grid puts a ray in it
    for y in (-0.4, 0.4):  # across the whole scan width
        assert ground(front_x + 0.1, y) is None
    land_x = slot[-1] + 0.05
    assert ground(land_x + 0.6) - ground(land_x) == pytest.approx(0.6 * np.tan(np.radians(TRAMPOLINE_RAMP_ANGLE)), abs=0.005)


@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_trampoline_bounces_a_robot_holding_its_pose_over_the_slot(difficulty):
    env = module_env("trampoline", difficulty)
    try:
        env.reset(seed=0)
        actor = env._actors[0]
        bed = env.terrain.bodies[0]
        deck = env.terrain.boxes[0]
        env.data.qpos[0] = bed.pos[0] + 0.3
        env.data.qpos[2] += 2.0
        mujoco.mj_forward(env.model, env.data)
        env._settle_feet_on_ground()
        env.data.qpos[2] += deck.pos[2] + deck.size[2] - bed.pos[2]  # as if dropped from the deck
        mujoco.mj_forward(env.model, env.data)
        for _ in range(120):
            _, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape))
            assert not terminated
        assert actor.launches >= 1 and env.data.qpos[0] > actor.x_range[1] + TRAMPOLINE_SLOT[1]
    finally:
        env.close()


def test_wind_blows_steadily_only_on_its_module_at_higher_difficulty():
    for difficulty, steady in ((0.0, 0.0), (1.0, 10.0)):
        env = module_env("wind_turbine", difficulty)
        try:
            env.reset(seed=0)
            actor = env._actors[0]
            env.data.qpos[0] = (actor.x_range[0] + actor.x_range[1]) / 2.0
            actor.before_step(env)
            assert env.data.qfrc_applied[1] == pytest.approx(-actor.side * steady)
            env.data.qpos[0] = actor.x_range[1] + 1.0
            actor.before_step(env)
            assert env.data.qfrc_applied[1] == 0.0
        finally:
            env.close()


@pytest.mark.parametrize("difficulty, delay", [(0.0, 1.0), (0.5, 0.75), (1.0, 0.5)])
def test_bridge_planks_hold_for_less_time_at_higher_difficulty(difficulty, delay):
    actor = MODULES["crumbling_bridge"](np.random.default_rng(0), difficulty, Cursor()).actors[0]
    assert actor.delay == pytest.approx(delay)


def _wall_contact(env, yaw_deg, straight_legs=False):
    """Whether the robot, centred in a tilt passage turned ``yaw_deg``, penetrates a wall."""
    m, d = env.model, env.data
    if straight_legs:
        for side in ("left", "right"):
            for joint, value in (("hip_x", 0.0), ("hip_y", 0.0), ("hip_z", 0.0), ("knee", -2.0)):
                d.qpos[m.jnt_qposadr[m.joint(f"{side}_{joint}").id]] = np.radians(value)
    walls = {m.geom(b.name).id for b in env.terrain.boxes if b.wall}
    wall = next(b for b in env.terrain.boxes if b.wall)
    yaw = np.radians(yaw_deg)
    d.qpos[0], d.qpos[1] = wall.pos[0], 0.0
    d.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    mujoco.mj_forward(m, d)
    env._settle_feet_on_ground()
    mujoco.mj_forward(m, d)
    return any((c.geom1 in walls) != (c.geom2 in walls) and c.dist < 0 for c in d.contact[: d.ncon])


def test_the_narrowest_tilt_passage_fits_the_robot_only_sideways():
    env = module_env("tilt_passage", 1.0)
    try:
        for straight in (False, True):
            env.reset(seed=0)
            assert _wall_contact(env, 0, straight)  # facing forward, its hips do not fit
        env.reset(seed=0)
        assert not _wall_contact(env, 90, straight_legs=True)
    finally:
        env.close()
    wide = module_env("tilt_passage", 0.0)
    try:
        wide.reset(seed=0)
        assert not _wall_contact(wide, 0)  # at d = 0 it walks through facing forward
    finally:
        wide.close()


def test_tilt_passage_walls_read_as_walls_even_from_a_ray_started_inside():
    env = module_env("tilt_passage", 1.0)
    try:
        env.reset(seed=0)
        wall = next(b for b in env.terrain.boxes if b.wall)
        inside = (wall.pos[0], wall.pos[1], wall.pos[2] + 0.4)  # under the wall top, as a crouching robot's ray
        assert mdp.ground_ray(env.model, env.data, inside, env._ceiling_geom_ids, env._wall_geom_ids) == 0.0
        assert mdp.ground_ray(env.model, env.data, inside, env._ceiling_geom_ids) > 0.4  # the floor, unmarked
    finally:
        env.close()


@pytest.mark.parametrize("difficulty, count, radius", [(0.0, 2, 0.675), (0.5, 3, 0.5375), (1.0, 4, 0.4)])
def test_balls_float_over_the_pit_with_their_tops_level_with_the_deck(difficulty, count, radius):
    out = MODULES["ball_balance"](np.random.default_rng(0), difficulty, Cursor())
    balls = [b for b in out.boxes if b.shape == "sphere"]
    assert len(balls) == count and all(b.size[0] == pytest.approx(radius) for b in balls)
    assert all(b.pos[2] + b.size[0] == pytest.approx(0.0) for b in balls)  # crests at deck height
    assert not [b for b in out.boxes if b.shape == "box" and any(
        abs(b.pos[0] - ball.pos[0]) < b.size[0] and b.pos[2] + b.size[2] <= ball.pos[2] for ball in balls)]
    strides = np.diff([b.pos[0] for b in balls])
    assert np.all(strides <= 0.8 + 1e-9)  # crest to crest stays a stride however big the balls
    gaps = strides - 2 * radius
    if difficulty == 0.0:
        assert np.all(gaps < 0)  # big balls overlap into a bumpy walkway
    if difficulty == 1.0:
        assert np.allclose(gaps, 0.0)  # the smallest just touch


@pytest.mark.parametrize("difficulty, clearance", [(0.0, 1.2), (1.0, 0.95)])
def test_air_vent_allows_a_crawl_under_its_roof_but_tunnels_do_not(difficulty, clearance):
    for skill, crawl in (("air_vent", True), ("tunnel", False)):
        env = module_env(skill, difficulty)
        try:
            env.reset(seed=0)
            roof = next(b for b in env.terrain.boxes if b.ceiling and b.name.split("_")[1] in ("roof", "bar"))
            env.data.qpos[0] = roof.pos[0]
            mujoco.mj_forward(env.model, env.data)
            surface = env._terrain_height_below(env.root_pos[:2], env.root_pos[2] + 0.1)
            floor = env._update_height_floor(surface)
            term = env.cfg.termination
            if crawl:
                assert env._ceiling_z(surface) - surface == pytest.approx(clearance, abs=0.005)
                assert floor == pytest.approx(env.robot.healthy_height_range[0] * term.crawl_bad_height_scale)
                assert env._orientation_limit == term.crawl_orientation_limit
            else:
                assert floor == pytest.approx(env.robot.healthy_height_range[0] * term.ceiling_bad_height_scale)
                assert env._orientation_limit == term.orientation_limit
        finally:
            env.close()


@pytest.fixture
def _run_clock_on(monkeypatch):
    import g1_parkour.terrain.parkour as parkour

    assert parkour.RUN_CLOCK_SHARE is None  # off in training for now
    monkeypatch.setattr(parkour, "RUN_CLOCK_SHARE", (0.5, 0.65))


def _run_env(difficulty=0.5):
    env = module_env("run", difficulty, render_mode=None)
    env.reset(seed=0)
    clock = env._actors[0]
    return env, clock


def test_the_run_clock_allows_the_commanded_time_over_the_required_share(_run_clock_on):
    env, clock = _run_env(1.0)
    try:
        x0, x1 = clock.x_range
        idx = [i for i, x in enumerate(env._waypoints[:, 0]) if x0 < x <= x1 + 1e-6]
        assert len(idx) == 2 and np.all(env._waypoint_speeds[idx] > 1.5)  # sprint boost included
        mid = env._waypoints[idx[0], 0]
        expected = (mid - x0) / env._waypoint_speeds[idx[0]] + (x1 - mid) / env._waypoint_speeds[idx[1]]
        assert clock.required_share == pytest.approx(0.65)
        assert clock.allowed_time(env) == pytest.approx(expected / 0.65 + clock.GRACE)
        env._waypoint_speeds[:] = 0.0  # a zero-command episode has no clock
        assert clock.allowed_time(env) is None
    finally:
        env.close()


def test_the_run_clock_ends_a_slow_crossing_but_not_a_fast_one(_run_clock_on):
    env, clock = _run_env(0.5)
    try:
        x0, x1 = clock.x_range
        env.data.qpos[0] = x0 + 0.01
        clock.before_step(env)
        limit = clock.limit
        assert clock.start_step == env._step_count and limit > clock.GRACE
        env._step_count += int(limit / env.dt) - 1
        assert clock.terminates(env) == ""  # still inside the limit
        env._step_count += 3
        assert clock.terminates(env) == "too_slow"
        # A crossing finished in time stops the clock for good.
        clock.reset(env)
        env.data.qpos[0] = x0 + 0.01
        clock.before_step(env)
        env.data.qpos[0] = x1 + 0.01
        clock.before_step(env)
        env._step_count += 10_000
        assert clock.finished and clock.terminates(env) == ""
    finally:
        env.close()


def test_a_too_slow_ending_terminates_the_episode_without_a_fall_penalty(monkeypatch, _run_clock_on):
    env, clock = _run_env(0.5)
    try:
        monkeypatch.setattr(clock, "terminates", lambda env: "too_slow")
        _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))
        assert terminated and info["termination"] == "too_slow"
        assert info.get("reward/fall_penalty", 0.0) == 0.0
    finally:
        env.close()


def test_the_run_clock_shows_the_remaining_seconds_in_red_seven_segments(_run_clock_on):
    env, clock = _run_env(0.5)
    try:
        clock.limit, clock.start_step = 1.5, env._step_count
        env.data.qpos[0] = (clock.x_range[0] + clock.x_range[1]) / 2.0
        markers = clock.markers(env)
        panel, lit = markers[0], markers[1:]
        assert panel["rgba"][:3] == (0.02, 0.02, 0.02)
        assert all(m["rgba"][0] == 1.0 and m["rgba"][1] < 0.1 for m in lit)
        per_side = len(SEVEN_SEGMENTS["1"]) + len(SEVEN_SEGMENTS["5"]) + 1  # "1.5" with its point
        assert len(lit) == 4 * per_side  # on all four sides
        env.data.qpos[0] = clock.x_range[1] + 20.0
        assert clock.markers(env) == []  # far away, nothing drawn
    finally:
        env.close()


@pytest.mark.parametrize("difficulty, rise, clearance", [(0.0, 0.15, 1.2), (1.0, 0.25, 0.95)])
def test_air_vent_ramps_up_from_its_inlet_under_a_roof_that_follows_the_floor(difficulty, rise, clearance):
    import mujoco

    env = module_env("air_vent", difficulty)
    try:
        env.reset(seed=0)
        boxes = {b.name.rsplit("_", 1)[0] if b.name.count("_") > 2 else b.name: b for b in env.terrain.boxes}
        inlet = next(b for b in env.terrain.boxes if b.name.startswith("vent_in_"))
        ramp = next(b for b in env.terrain.boxes if b.name.startswith("vent_ramp"))
        exit_floor = next(b for b in env.terrain.boxes if b.name.startswith("vent_exit"))
        inlet_end = inlet.pos[0] + inlet.size[0] * np.cos(np.radians(inlet.euler[1]))
        ramp_start = ramp.pos[0] - ramp.size[0] * np.cos(np.radians(ramp.euler[1]))
        assert ramp_start == pytest.approx(inlet_end, abs=0.03)  # the ramp starts where the inlet ends
        assert ramp.euler[1] < 0 and exit_floor.pos[2] + exit_floor.size[2] == pytest.approx(rise)
        assert not [b for b in env.terrain.boxes if b.name.startswith("vent_out")]  # no opening funnel
        # Along the ramp the roof stays the same height above the floor.
        ramp_end = ramp.pos[0] + ramp.size[0] * np.cos(np.radians(ramp.euler[1]))
        for t in (0.15, 0.5, 0.85):
            env.data.qpos[0] = ramp_start + t * (ramp_end - ramp_start)
            mujoco.mj_forward(env.model, env.data)
            surface = env._terrain_height_below(env.root_pos[:2], env.root_pos[2] + 0.1)
            assert env._ceiling_z(surface) - surface == pytest.approx(clearance, abs=0.01)
            assert surface == pytest.approx(rise * t, abs=0.01)
    finally:
        env.close()


def test_the_eased_elevator_at_d09_is_the_old_elevator_at_d05():
    import g1_parkour.terrain.parkour as parkour

    at_09 = MODULES["sliding_elevator"](np.random.default_rng(0), 0.9, Cursor())
    original = parkour.SLIDING_ELEVATOR_EASE
    try:
        parkour.SLIDING_ELEVATOR_EASE = 1.0
        reference = parkour.MODULES["sliding_elevator"](np.random.default_rng(0), 0.5, Cursor())
    finally:
        parkour.SLIDING_ELEVATOR_EASE = original
    a, b = at_09.actors[0], reference.actors[0]
    assert (a.travel, a.move_time, a.dwell) == pytest.approx((b.travel, b.move_time, b.dwell))
    assert [box.size for box in at_09.boxes] == [box.size for box in reference.boxes]
