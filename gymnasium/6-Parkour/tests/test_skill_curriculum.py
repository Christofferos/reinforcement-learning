"""Per-skill difficulty: each module at its skill's level, credited passes and failures, and
levels that move, save, restore and play back per skill. Also the tunnel's ceiling contact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour import tasks  # noqa: E402
from g1_parkour.env import ParkourEnv, mean_curriculum_state  # noqa: E402
from g1_parkour.terrain.actors import WindActor  # noqa: E402
from g1_parkour.terrain.parkour import TUNNEL_POST_LENGTH, TUNNEL_POST_WIDTH, make_parkour  # noqa: E402


def skill_env(skills=("hurdle", "climb"), num_modules=4, **terrain) -> ParkourEnv:
    cfg = tasks.parkour_cfg()
    cfg.terrain.params = {"num_modules": num_modules, "skills": skills}
    for key, value in terrain.items():
        setattr(cfg.terrain, key, value)
    return ParkourEnv(cfg)


def test_procedural_uses_per_skill_levels_starting_at_the_task_level():
    cfg = tasks.parkour_cfg()
    assert cfg.terrain.per_skill_difficulty and cfg.terrain.curriculum
    env = ParkourEnv(cfg)
    try:
        state = env.curriculum_state()
        assert set(state) == set(cfg.terrain.params["skill_weights"])
        assert set(state.values()) == {cfg.terrain.difficulty}
    finally:
        env.close()


def test_every_module_is_built_at_its_skills_level_along_the_whole_course():
    course = make_parkour(np.random.default_rng(0), difficulty=0.5, num_modules=8, skills=("hurdle", "climb"),
                          skill_difficulty={"hurdle": 1.0, "climb": 0.0})
    hurdles = [b for b in course.boxes if b.name.startswith("hurdle_") and not b.name.startswith("hurdle_run")]
    climbs = [b for b in course.boxes if b.name.startswith("climb_")]
    assert hurdles and climbs
    assert all(2 * b.size[2] == pytest.approx(0.5) for b in hurdles)  # 0.2 + 0.3 d², d = 1, first to last
    # Waypoints belong to their module: 0 the start pad, then each module in turn, -1 the finish.
    owners = course.waypoint_modules
    assert owners[0] == 0 and owners[-1] == -1 and len(owners) == len(course.waypoints)
    assert list(dict.fromkeys(owners[1:-1])) == list(range(1, len(course.modules)))


def test_a_module_is_passed_at_its_last_waypoint_and_failed_where_the_episode_ends():
    env = skill_env(curriculum=False)
    try:
        env.reset(seed=0)
        owners = env.terrain.waypoint_modules
        first_end = int(np.flatnonzero(owners == 1)[-1])
        for index in range(1, first_end + 2):  # through module 1 to module 2's first waypoint
            env.data.qpos[0:2] = env._waypoints[env._target_index][:2]
            mujoco.mj_forward(env.model, env.data)
            env._advance_waypoint()
        assert env._module_results == [(env.terrain.modules[1], True, "level")]
        info = env._info({}, "bad_height")
        assert info["module_results"] == [(env.terrain.modules[1], True), (env.terrain.modules[2], False)]
        assert "module_results" not in env._info({}, "bad_height")  # counted once per episode
    finally:
        env.close()


def test_skills_move_one_step_on_their_own_windows():
    env = skill_env(skill_window=4, difficulty=0.5)
    try:
        env.reset(seed=0)
        assert (env.cfg.terrain.skill_promote_rate, env.cfg.terrain.skill_demote_rate) == (0.75, 0.5)
        level = [("hurdle", True, "level")] * 3 + [("hurdle", False, "level")]  # 75%: promotes
        env._module_results = level + [("climb", False, "level")] * 3 + [("climb", True, "level")]
        env._update_curriculum(0.0)
        assert env.curriculum_state() == {"hurdle": 0.55, "climb": 0.45}
        assert not env._skill_results["hurdle"] and not env._skill_results["climb"]
        # 2 of 4 passed (50%) sits between the demote and promote rates: the level holds. Review
        # and preview modules do not count at all, passed or failed.
        env._module_results = [("hurdle", True, "level"), ("hurdle", False, "level")] * 2 + [
            ("hurdle", False, "preview")] * 4 + [("hurdle", True, "review")] * 4
        env._update_curriculum(0.0)
        assert env.curriculum_state()["hurdle"] == 0.55 and len(env._skill_results["hurdle"]) == 4
        assert env._difficulty == pytest.approx(0.5)
        env.reset()
        assert env._built_skill_difficulty == {"hurdle": 0.55, "climb": 0.45}  # the next course follows
    finally:
        env.close()


def test_set_difficulty_takes_one_level_for_every_skill_or_named_skills():
    env = skill_env()
    try:
        env.set_difficulty(0.7)  # an old one-level checkpoint
        assert env.curriculum_state() == {"hurdle": 0.7, "climb": 0.7}
        env.set_difficulty({"hurdle": 0.3, "water_tank": 0.9})  # skills outside the mix are ignored
        assert env.curriculum_state() == {"hurdle": 0.3, "climb": 0.7}
        assert env._difficulty == pytest.approx(0.5)
    finally:
        env.close()
    flat = ParkourEnv(tasks.flat_cfg())
    try:
        assert flat.curriculum_state() == flat._difficulty
        with pytest.raises(ValueError):
            flat.set_difficulty({"hurdle": 0.3})
    finally:
        flat.close()


def test_worker_states_average_per_skill_on_the_grid():
    assert mean_curriculum_state([0.4, 0.52], 0.05) == pytest.approx(0.45)
    states = [{"hurdle": 0.3, "climb": 0.5}, {"hurdle": 0.4, "climb": 0.5, "beam": 0.2}]
    assert mean_curriculum_state(states, 0.05) == {"beam": 0.2, "climb": 0.5, "hurdle": 0.35}


def test_checkpoints_save_and_restore_per_skill_levels(tmp_path):
    from types import SimpleNamespace
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from scripts import train

    venv = make_vec_env("Procedural", n_envs=2, seed=0, vec_env_cls=DummyVecEnv)
    try:
        real = [env.unwrapped for env in venv.envs]
        checkpoint = tmp_path / "checkpoint_100_steps.zip"
        train.state_path(checkpoint).write_text(json.dumps({"num_timesteps": 100, "difficulties": [0.7, 0.75]}))
        assert train.restore_difficulties(venv, checkpoint, num_envs=2) == [0.7, 0.75]
        assert set(real[0].curriculum_state().values()) == {0.7} and set(real[1].curriculum_state().values()) == {0.75}
        real[0].set_difficulty({"hurdle": 0.55})
        callback = train.ProgressCallback(tmp_path, None, eval_every=1000)
        callback.model = SimpleNamespace(num_timesteps=200, get_env=lambda: venv)
        callback.save_state("checkpoint_200_steps")
        saved = json.loads(train.state_path(tmp_path / "checkpoint_200_steps.zip").read_text())["difficulties"]
        assert saved[0]["hurdle"] == 0.55 and saved[0]["beam"] == 0.7 and saved[1]["hurdle"] == 0.75
        venv.env_method("set_difficulty", 0.0)
        train.restore_difficulties(venv, tmp_path / "checkpoint_200_steps.zip", num_envs=2)
        assert real[0].curriculum_state() == saved[0]
    finally:
        venv.close()


def test_play_uses_each_skills_saved_level_and_the_task_level_for_new_skills(tmp_path):
    from scripts import play

    checkpoint = tmp_path / "final.zip"
    checkpoint.with_suffix(".state.json").write_text(json.dumps({"num_timesteps": 1, "difficulties": [
        {"hurdle": 0.3, "climb": 0.6}, {"hurdle": 0.4, "climb": 0.6},
    ]}))
    cfg = tasks.parkour_cfg()
    cfg.terrain.params = {**cfg.terrain.params, "skills": ("hurdle", "climb", "water_tank")}
    assert play.configure_difficulty(cfg, None, str(checkpoint)) == "saved with final.zip"
    env = ParkourEnv(cfg)
    try:
        assert env.curriculum_state() == {"hurdle": 0.35, "climb": 0.6, "water_tank": cfg.terrain.difficulty}
    finally:
        env.close()
    cfg = tasks.parkour_cfg()
    assert play.configure_difficulty(cfg, 0.9, str(checkpoint)) == "--difficulty"
    env = ParkourEnv(cfg)
    try:
        assert set(env.curriculum_state().values()) == {0.9}
    finally:
        env.close()


def tunnel_env(difficulty=1.0) -> ParkourEnv:
    return skill_env(skills=("tunnel",), num_modules=1, curriculum=False, difficulty=difficulty)


def test_touching_a_tunnel_roof_costs_a_step_penalty():
    env = tunnel_env()
    try:
        env.reset(seed=0)
        bar = next(b for b in env.terrain.boxes if b.name.startswith("tunnel_bar"))
        assert not env._touching_ceiling()
        env.data.qpos[0] = bar.pos[0]  # standing tall under a 1.25 m roof
        mujoco.mj_forward(env.model, env.data)
        assert env._touching_ceiling()
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
        assert env.cfg.reward.ceiling_contact < 0
        assert info["reward/ceiling_contact"] == pytest.approx(env.cfg.reward.ceiling_contact)
    finally:
        env.close()


@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_tunnel_posts_are_thin_pillars_at_the_course_edges(difficulty):
    env = tunnel_env(difficulty)
    try:
        posts = [b for b in env.terrain.boxes if b.name.startswith("tunnel_post")]
        bar = next(b for b in env.terrain.boxes if b.name.startswith("tunnel_bar"))
        assert len(posts) == 2
        assert all(2 * p.size[0] == pytest.approx(min(TUNNEL_POST_LENGTH, 2 * bar.size[0])) for p in posts)
        assert all(2 * p.size[1] == pytest.approx(TUNNEL_POST_WIDTH) for p in posts)
        # On the floor (not over the pit beside it), under the roof, at its edges clear of the route.
        floor = next(b for b in env.terrain.boxes if b.name.startswith("tunnel_run"))
        for p in posts:
            assert abs(p.pos[1] - floor.pos[1]) + p.size[1] <= floor.size[1] + 1e-9
            assert abs(p.pos[1] - floor.pos[1]) - p.size[1] >= floor.size[1] - TUNNEL_POST_WIDTH - 1e-9
            assert p.pos[2] - p.size[2] == pytest.approx(floor.pos[2] + floor.size[2])  # resting on it
            assert bar.pos[2] - bar.size[2] < p.pos[2] + p.size[2] <= bar.pos[2] + bar.size[2]  # into the roof
    finally:
        env.close()


def _hazard_end(out, exit_z: float, module_end: float) -> float:
    """Where a module's last hazard ends along its route: the last x where the top surface
    under the robot's width (0.3 m either side of the route) is missing or off the exit height,
    or where a roof, moving part, water or wind lies."""
    path = np.vstack([[0.0, 0.0, 0.0], np.asarray(out.waypoints)])
    solid = [b for b in out.boxes if not b.visual]
    # Flat-topped boxes, a yaw-turned rubble block by its bounding square; slopes have no flat
    # top, so the ground under them reads as missing.
    flat = []
    for b in solid:
        if b.ceiling or b.shape != "box" or b.quat is not None or b.euler[0] or b.euler[1]:
            continue  # a ball's or slope's top is not flat ground
        half = np.hypot(b.size[0], b.size[1]) if b.euler[2] else None
        flat.append((b.pos, (half, half, b.size[2]) if half else b.size))
    extents = [(b.pos[0] - b.size[0], b.pos[0] + b.size[0]) for b in solid if b.ceiling]
    # Moving parts by their x half-size (a box's, or a cylinder's radius: barrels lie across the route).
    extents += [(body.pos[0] + g.pos[0] - g.size[0], body.pos[0] + g.pos[0] + g.size[0])
                for body in out.bodies for g in body.geoms if not g.visual]
    extents += [(w.center[0] - w.half_size[0], w.center[0] + w.half_size[0]) for w in out.water]
    extents += [a.x_range for a in out.actors if isinstance(a, WindActor)]  # it blows over its walkway
    end = 0.0
    for x in np.arange(0.01, module_end - 0.01, 0.02):
        y = float(np.interp(x, path[:, 0], path[:, 1]))
        tops = [
            max((pos[2] + size[2] for pos, size in flat
                 if abs(x - pos[0]) <= size[0] and abs(y + dy - pos[1]) <= size[1]), default=None)
            for dy in (-0.3, 0.0, 0.3)
        ]
        if any(t is None or abs(t - exit_z) > 1e-6 for t in tops) or any(lo <= x <= hi for lo, hi in extents):
            end = x
    return end


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_every_module_counts_as_passed_only_a_metre_past_its_last_hazard(difficulty):
    """The last waypoint's 0.7 m radius must not credit a pass inside a module, or a fall while
    recovering from it (standing up after a tunnel) would count against the next module."""
    from g1_parkour.terrain.parkour import MODULES, Cursor

    radius = tasks.parkour_cfg().waypoint_radius
    for name, module in MODULES.items():
        for seed in range(3):
            cursor = Cursor()
            out = module(np.random.default_rng(seed), difficulty, cursor)
            last = np.asarray(out.waypoints[-1])
            assert last[0] == pytest.approx(cursor.x), name  # the module's last waypoint is its end
            margin = last[0] - radius - _hazard_end(out, cursor.z, cursor.x)
            assert margin >= 1.0, (name, difficulty, seed, round(margin, 2))


def test_mastered_skills_are_kept_up_with_fewer_modules():
    env = skill_env(skills=("hurdle", "climb"), num_modules=8, curriculum=False, skill_maintain_share=0.25)
    try:
        env.set_difficulty({"hurdle": 1.0, "climb": 0.0})
        assert env._skill_sampling_weights() == pytest.approx({"hurdle": 0.25, "climb": 1.0})
        counts = {"hurdle": 0, "climb": 0}
        for seed in range(40):
            env.reset(seed=seed)
            for module in env.terrain.modules[1:]:
                counts[module] += 1
        assert counts["climb"] / sum(counts.values()) == pytest.approx(0.8, abs=0.05)
        assert env._difficulty == pytest.approx(0.5)  # the mean level ignores the practice scaling
    finally:
        env.close()
    plain = skill_env(skills=("hurdle", "climb"), skill_maintain_share=None)
    try:
        assert plain._skill_sampling_weights() is None  # no share: the task's weights as given
    finally:
        plain.close()


def test_procedural_scales_practice_by_level():
    cfg = tasks.parkour_cfg()
    assert cfg.terrain.skill_maintain_share == 0.25
    env = ParkourEnv(cfg)
    try:
        env.set_difficulty({"hurdle": 0.6, "ramp": 1.0})
        weights, base = env._skill_sampling_weights(), cfg.terrain.params["skill_weights"]
        assert weights["hurdle"] == pytest.approx(base["hurdle"] * 0.55)
        assert weights["ramp"] == pytest.approx(base["ramp"] * 0.25)
    finally:
        env.close()


def test_training_courses_mix_review_and_preview_modules_around_the_levels():
    from collections import Counter

    levels = {"hurdle": 0.6, "climb": 0.5}
    course = make_parkour(np.random.default_rng(0), num_modules=400, skills=tuple(levels), skill_difficulty=levels,
                          review_share=0.2, preview_share=0.1, preview_range=0.1)
    kinds = Counter(course.module_kinds[1:])
    assert kinds["level"] / 400 == pytest.approx(0.7, abs=0.06)
    assert kinds["review"] / 400 == pytest.approx(0.2, abs=0.05)
    assert kinds["preview"] / 400 == pytest.approx(0.1, abs=0.04)
    hurdles = [b for b in course.boxes if b.name.startswith("hurdle_") and not b.name.startswith("hurdle_run")]
    heights = {round(2 * b.size[2], 4) for b in hurdles}
    level_height = round(0.2 + 0.3 * 0.6 ** 2, 4)
    assert level_height in heights
    assert min(heights) < level_height < max(heights) <= round(0.2 + 0.3 * 0.7 ** 2, 4) + 1e-9
    # Without the shares no draw is made, so a course is exactly as before.
    plain = make_parkour(np.random.default_rng(0), num_modules=8, skills=tuple(levels), skill_difficulty=levels)
    again = make_parkour(np.random.default_rng(0), num_modules=8, skills=tuple(levels), skill_difficulty=levels,
                         review_share=0.0, preview_share=0.0)
    assert [b.pos for b in plain.boxes] == [b.pos for b in again.boxes] and set(plain.module_kinds[1:]) == {"level"}


def test_review_and_preview_apply_in_training_only():
    cfg = tasks.parkour_cfg()
    assert (cfg.terrain.skill_review_share, cfg.terrain.skill_preview_share) == (0.2, 0.1)
    for curriculum, mixed in ((True, True), (False, False)):
        cfg = tasks.parkour_cfg()
        cfg.terrain.curriculum = curriculum
        env = ParkourEnv(cfg)
        try:
            kinds = set()
            for seed in range(30):
                env.reset(seed=seed)
                kinds |= set(env.terrain.module_kinds[1:])
            assert (kinds != {"level"}) == mixed  # evaluation and playback build exact levels
        finally:
            env.close()


def _scripted(skill, difficulty=0.8):
    """A one-module env whose step sets the torso height to the next scripted value (no physics)."""
    import mujoco

    env = skill_env(skills=(skill,), num_modules=1, curriculum=False, difficulty=difficulty)
    env.reset(seed=0)
    heights = []

    def place(applied):
        env.data.qpos[2] = heights.pop(0)
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

    env._apply_action = place
    return env, heights


def _vertical_total(env, heights, path, end=False):
    """Sum of vertical_progress over the scripted torso heights; ``end`` makes the last step an ending."""
    original = env._evaluate
    total, last = 0.0, len(path) - 1
    heights.extend(path)
    for k in range(len(path)):
        def evaluate(*args, _k=k, **kwargs):
            reward, terms, terminated, reason = original(*args, **kwargs)
            return (reward, terms, True, "bad_height") if end and _k == last else (reward, terms, False, "")
        env._evaluate = evaluate
        _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))
        total += info.get("reward/vertical_progress", 0.0)
    env._evaluate = original
    return total


def test_vertical_progress_pays_rising_toward_a_higher_waypoint_only():
    env, heights = _scripted("climb")
    try:
        top = env.terrain.waypoints[1][2]  # the climb's first waypoint, on its top
        env._target_index = 1
        start = env.data.qpos[2]
        standing = env.cfg.nominal_base_height
        deadband = env.cfg.reward.vertical_progress_deadband
        # Push the torso from the floor up to standing height over the top, in ten steps.
        path = list(np.linspace(start, top + standing, 11)[1:])
        paid = _vertical_total(env, heights, path)
        expected = (top + standing - start - deadband) / env.dt
        assert paid == pytest.approx(expected, rel=0.02) and paid > 0
    finally:
        env.close()
    flat, heights = _scripted("run")
    try:
        z = flat.data.qpos[2]
        assert _vertical_total(flat, heights, [z - 0.05, z, z - 0.08, z]) == pytest.approx(0.0, abs=1e-6)  # gait bobbing
    finally:
        flat.close()


@pytest.mark.parametrize("skill", ["run", "climb"])
def test_a_crouch_or_fall_and_its_recovery_or_ending_nets_zero(skill):
    env, heights = _scripted(skill)
    try:
        env._target_index = 1
        z = env.data.qpos[2]
        assert _vertical_total(env, heights, [z - 0.6, z - 0.3, z]) == pytest.approx(0.0, abs=1e-6)  # crouch, recover
        assert _vertical_total(env, heights, [z - 0.3, z - 0.9], end=True) == pytest.approx(0.0, abs=1e-6)  # fall, end
    finally:
        env.close()


def test_a_failed_climb_keeps_only_what_it_rose():
    env, heights = _scripted("climb")
    try:
        env._target_index = 1
        z = env.data.qpos[2]
        rose = _vertical_total(env, heights, [z + 0.05, z + 0.1, z - 0.7], end=True)  # up 0.1 m, then falls
        assert rose == pytest.approx(0.1 / env.dt, rel=0.02)  # not the rest of the climb
    finally:
        env.close()


def test_a_crouched_gait_costs_every_step_but_not_under_a_roof():
    reward = tasks.parkour_cfg().reward
    assert reward.base_height_low == 1.0 and reward.vertical_progress == 1.0
    for skill, roofed in (("run", False), ("tunnel", True)):
        env, heights = _scripted(skill, difficulty=1.0)
        try:
            if roofed:  # stand under the middle of the roof
                import mujoco

                bar = next(b for b in env.terrain.boxes if b.name.startswith("tunnel_bar"))
                env.data.qpos[0] = bar.pos[0]
                mujoco.mj_forward(env.model, env.data)
            for drop in (0.10, 0.25):
                heights.append(env.data.qpos[2] - drop)
                _, _, _, _, info = env.step(np.zeros(env.action_space.shape))  # this step is crouched
                surface = env._support_surface_z()
                deficit = env._standing_height(surface) - (env.root_pos[2] - surface)
                expected = 0.0 if roofed else max(0.0, deficit - reward.base_height_low_deadband)
                assert -info.get("reward/base_height_low", 0.0) == pytest.approx(expected, abs=1e-6)
                if not roofed:
                    assert expected > 0.0  # the crouch costs on open ground, every step it lasts
                heights.append(env.data.qpos[2] + drop)
                env.step(np.zeros(env.action_space.shape))
        finally:
            env.close()


def test_playback_courses_can_draw_every_module():
    from collections import Counter
    from scripts import play
    from g1_parkour.terrain.parkour import EXPERIMENTAL_SKILLS, MODULES

    cfg = tasks.parkour_cfg()
    play.configure_difficulty(cfg, 0.5, None)
    assert "all" in play.configure_skills(cfg, None, training_mix=False)
    env = ParkourEnv(cfg)
    try:
        seen = Counter()
        for seed in range(60):
            env.reset(seed=seed)
            seen.update(env.terrain.modules[1:])
        assert set(seen) == set(MODULES)  # experimental modules included
        assert set(EXPERIMENTAL_SKILLS) <= set(env.curriculum_state())
        assert set(env.curriculum_state().values()) == {0.5}
    finally:
        env.close()
    mix = tasks.parkour_cfg()
    trained = set(mix.terrain.params["skill_weights"])
    play.configure_skills(mix, None, training_mix=True)
    assert set(mix.terrain.params["skills"]) == trained and not set(trained) & set(EXPERIMENTAL_SKILLS)
    assert set(mix.terrain.params["skill_weights"].values()) == {1.0} and mix.terrain.skill_maintain_share is None
    named = tasks.parkour_cfg()
    play.configure_skills(named, ["air_vent"], training_mix=False)
    assert named.terrain.params["skills"] == ("air_vent",)
    flat = tasks.flat_cfg()
    assert play.configure_skills(flat, None, training_mix=False) == ""


def test_equal_mix_is_for_playback_only():
    from scripts import train
    import inspect

    assert "equal_module_mix" not in inspect.getsource(train.main)  # training and its evaluation as before
    cfg = tasks.parkour_cfg()
    base = dict(cfg.terrain.params["skill_weights"])
    skills = tasks.equal_module_mix(cfg)
    assert set(skills) == set(base) and set(cfg.terrain.params["skill_weights"].values()) == {1.0}
    assert tasks.parkour_cfg().terrain.params["skill_weights"] == base  # training's own mix is untouched
    assert tasks.equal_module_mix(tasks.flat_cfg()) == ()
