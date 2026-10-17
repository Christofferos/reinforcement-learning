"""The stabilised first-person camera (render-only)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour.camera import EYE_MOUNT, STABLE_EGOCENTRIC, StableEyeCamera, camera_frame  # noqa: E402

DT = 0.015


def body(time, position, yaw=0.0, pitch=0.0, roll=0.0):
    """A one-body ``data`` stand-in: world position and orientation of body 0 at ``time``."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return SimpleNamespace(time=time, xpos=np.array([position], dtype=float), xmat=(rz @ ry @ rx).reshape(1, 9))


def eye(camera, data):
    lookat, distance, azimuth, elevation = camera.update(data, 0)
    a, e = np.radians(azimuth), np.radians(elevation)
    return lookat - distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]), azimuth, elevation


def test_first_frame_puts_the_eye_in_front_of_the_head_looking_ahead_and_down():
    position, azimuth, elevation = eye(StableEyeCamera(), body(0.0, (1.0, 2.0, 1.3), yaw=np.pi / 2))
    forward, _, up = StableEyeCamera.EYE_OFFSET
    np.testing.assert_allclose(position, (1.0, 2.0 + forward, 1.3 + up), atol=1e-9)
    assert azimuth == pytest.approx(90.0) and elevation == pytest.approx(-StableEyeCamera.LOOK_DOWN)


def test_forward_motion_is_followed_exactly():
    camera = StableEyeCamera()
    for k in range(200):
        position, *_ = eye(camera, body(k * DT, (1.0 * k * DT, 0.0, 1.3)))
    assert position[0] - 1.0 * 199 * DT == pytest.approx(StableEyeCamera.EYE_OFFSET[0], abs=1e-9)


def _swing(axis, frequency, amplitude=0.02):
    """Amplitude of the eye's steady swing along ``axis`` while the torso swings ``amplitude``."""
    camera, track = StableEyeCamera(), []
    for k in range(600):
        t = k * DT
        offset = np.zeros(3)
        offset[axis] = amplitude * np.sin(2 * np.pi * frequency * t)
        position, *_ = eye(camera, body(t, np.array([0.0, 0.0, 1.3]) + offset))
        track.append(position[axis])
    return np.ptp(track[300:]) / 2.0


def test_footstep_bounce_is_exaggerated_not_damped():
    # A 2 Hz bob: the eye bounces more than the torso, up to the 1.3x gain.
    ratio = _swing(2, 2.0) / 0.02
    assert 1.05 < ratio <= StableEyeCamera.BOB_GAIN + 0.05


def test_sideways_sway_is_mostly_kept():
    # One smoothing stage keeps 1 / sqrt(1 + (2 pi tau)^2) of a 1 Hz sway: 0.88 at 0.09 s.
    tau = StableEyeCamera.SWAY_TIME_CONSTANT
    assert _swing(1, 1.0) / 0.02 == pytest.approx(1.0 / np.sqrt(1.0 + (2 * np.pi * tau) ** 2), abs=0.04)

def test_the_view_takes_a_share_of_the_torso_pitch():
    camera = StableEyeCamera()
    for k in range(100):
        _, _, elevation = eye(camera, body(k * DT, (0.0, 0.0, 1.3), pitch=np.radians(10.0), roll=np.radians(8.0)))
    assert elevation == pytest.approx(-(StableEyeCamera.LOOK_DOWN + StableEyeCamera.PITCH_SHARE * 10.0), abs=0.01)
    assert StableEyeCamera.LOOK_DOWN == pytest.approx(30.0)


def test_the_eye_is_leashed_to_the_head_on_a_drop():
    camera, worst = StableEyeCamera(), 0.0
    for k in range(200):
        z = 1.3 if k < 50 else 0.9  # a 0.4 m step down
        position, *_ = eye(camera, body(k * DT, (0.0, 0.0, z)))
        worst = max(worst, abs(position[2] - (z + StableEyeCamera.EYE_OFFSET[2])))
    assert worst <= StableEyeCamera.MAX_BOB_LAG + 1e-9

def test_yaw_follows_turns_but_smooths_twitches():
    camera, azimuths = StableEyeCamera(), []
    for k in range(400):
        twitch = np.radians(6.0) * np.sin(2 * np.pi * 2.0 * k * DT)  # 2 Hz, ±6 degrees
        _, azimuth, _ = eye(camera, body(k * DT, (0.0, 0.0, 1.3), yaw=np.radians(30.0) + twitch))
        azimuths.append(azimuth)
    assert np.mean(azimuths[-100:]) == pytest.approx(30.0, abs=0.5)
    assert np.ptp(azimuths[-100:]) / 2.0 < 2.0  # under a third of the twitch


def test_a_reset_snaps_instead_of_gliding():
    camera = StableEyeCamera()
    eye(camera, body(5.0, (0.0, 0.0, 1.3)))
    position, *_ = eye(camera, body(0.0, (10.0, 3.0, 2.0)))  # time ran backwards: a new episode
    np.testing.assert_allclose(position, np.array([10.0, 3.0, 2.0]) + StableEyeCamera.EYE_OFFSET, atol=1e-9)


def test_tab_reaches_the_steady_first_person_camera_on_its_mount():
    import mujoco
    from g1_parkour import tasks
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.parkour_cfg())
    try:
        model = env.model
        camera = model.camera(STABLE_EGOCENTRIC).id
        assert model.cam_bodyid[camera] == model.body(EYE_MOUNT).id  # not fixed to the torso
        assert model.body_mocapid[model.body(EYE_MOUNT).id] >= 0
        names = [model.camera(c).name for c in range(model.ncam)]
        assert names.index(STABLE_EGOCENTRIC) == names.index("front_view") + 1  # TAB order kept
        assert not any(model.geom_bodyid[g] == model.body(EYE_MOUNT).id for g in range(model.ngeom))
        env.reset(seed=0)
        qpos = env.data.qpos.copy()
        env._place_eye_camera()
        lookat, distance, azimuth, elevation = env._stable_eye.update(env.data, env._root_body_id, env._course_heading())
        frame = camera_frame(azimuth, elevation)
        np.testing.assert_allclose(env.data.cam_xpos[camera], lookat + distance * frame[:, 2], atol=1e-9)
        np.testing.assert_allclose(env.data.cam_xmat[camera].reshape(3, 3) @ [0, 0, -1], -frame[:, 2], atol=1e-9)
        assert elevation < -25.0  # looking down
        np.testing.assert_array_equal(env.data.qpos, qpos)  # nothing in the simulation moved
        # After a forward pass the mount puts the camera in the same place.
        mujoco.mj_forward(model, env.data)
        np.testing.assert_allclose(env.data.cam_xpos[camera], lookat + distance * frame[:, 2], atol=1e-9)
        np.testing.assert_allclose(env.data.cam_xmat[camera], frame.ravel(), atol=1e-9)
    finally:
        env.close()


def test_the_old_camera_name_still_works():
    from g1_parkour import tasks
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.flat_cfg(), camera_name="egocentric_stable")
    try:
        assert env._camera_name == STABLE_EGOCENTRIC
    finally:
        env.close()

def test_a_twisting_torso_barely_moves_the_eye_sideways():
    # The walking humanoid's torso twists about ±9° once a stride; its eye sits 0.2 m ahead.
    camera, lateral = StableEyeCamera(), []
    for k in range(600):
        twist = np.radians(9.0) * np.sin(2 * np.pi * 1.0 * k * DT)
        position, *_ = eye(camera, body(k * DT, (0.0, 0.0, 1.3), yaw=twist))
        lateral.append(position[1])
    # Turned with the raw twist, the offset alone swung the eye ±3 cm; now under ±5 mm.
    assert np.ptp(lateral[300:]) / 2.0 < 0.005



def test_looking_along_the_course_takes_a_share_of_the_torso_twist():
    camera, azimuths = StableEyeCamera(), []
    for k in range(400):
        twist = np.radians(9.0) * np.sin(2 * np.pi * 1.27 * k * DT)
        _, _, azimuth, _ = camera.update(body(k * DT, (0.0, 0.0, 1.3), yaw=twist), 0, heading=np.radians(20.0))
        azimuths.append(azimuth)
    steady = np.array(azimuths[200:])
    # The torso faces 0° on a 20° course: the view turns TWIST_SHARE of the way towards it, and
    # takes that share of its ±9° twist too, smoothed by the yaw filter.
    share, tau = StableEyeCamera.TWIST_SHARE, StableEyeCamera.YAW_TIME_CONSTANT
    assert np.mean(steady) == pytest.approx(20.0 + share * (0.0 - 20.0), abs=0.3)
    expected = share * 9.0 / np.sqrt(1.0 + (2 * np.pi * 1.27 * tau) ** 2)
    assert np.ptp(steady) / 2.0 == pytest.approx(expected, rel=0.3)


def test_the_picture_tilts_with_a_share_of_the_torso_lean():
    camera = StableEyeCamera()
    for k in range(100):
        camera.update(body(k * DT, (0.0, 0.0, 1.3), roll=np.radians(-10.0)), 0, heading=0.0)  # left side down
    assert camera.roll == pytest.approx(StableEyeCamera.ROLL_SHARE * 10.0, abs=0.01)
    frame = camera_frame(0.0, 0.0, camera.roll)
    assert frame[1, 1] > 0  # the picture's up leans to the left (+y), as the head does

def test_course_heading_spans_a_few_metres_of_waypoints():
    from g1_parkour import tasks
    from g1_parkour.env import ParkourEnv

    env = ParkourEnv(tasks.flat_cfg())
    try:
        env.reset(seed=0)
        # Stepping-stone-like waypoints: 0.5 m apart, alternating 0.2 m to either side.
        env._waypoints = np.array([[1.0 + 0.5 * k, 0.2 * (-1) ** k, 0.0] for k in range(12)])
        headings = []
        for index in range(1, 6):
            env._target_index = index
            headings.append(np.degrees(env._course_heading()))
        assert max(abs(h) for h in headings) < 9.0  # next-waypoint aim would swing ±39°
    finally:
        env.close()


def test_random_glances_stay_in_range_and_ease_back():
    camera, glances, gazes = StableEyeCamera(seed=3), [], []
    for k in range(int(60.0 / DT)):  # a minute of walking straight
        camera.update(body(k * DT, (k * DT, 0.0, 1.3)), 0, heading=0.0)
        gazes.append(camera.gaze)
        if not glances or glances[-1][0] != camera.glance_until:
            if np.isfinite(camera.glance_until):
                glances.append((camera.glance_until, camera.glance))
    assert 4 <= len(glances) <= 20  # every 3-8 s, plus the glance itself
    low, high = StableEyeCamera.GLANCE_ANGLE
    assert all(low <= angle <= high for _, angle in glances)  # only ever up
    assert min(gazes) >= -1e-9 and max(gazes) <= high + 1e-9
    # Between glances the view eases back to its usual look-down.
    assert sum(abs(g) < 0.5 for g in gazes) > 0.3 * len(gazes)


def test_scan_cues_turn_the_view_and_the_ceiling_wins():
    for cues, expected in ((dict(look_down=True), -15.0), (dict(look_up=True), 35.0),
                           (dict(look_down=True, look_up=True), 35.0)):
        camera = StableEyeCamera()
        for k in range(int(2.5 / DT)):  # before any random glance (the first comes after 3 s)
            camera.update(body(k * DT, (0.0, 0.0, 1.3)), 0, heading=0.0, **cues)
        assert camera.gaze == pytest.approx(expected, abs=0.1)


def test_rendering_draws_no_randomness_from_the_environment():
    camera = StableEyeCamera()
    assert camera.rng is not np.random  # its own generator, so watching a run cannot change it


def test_env_cues_for_a_gap_ahead_and_a_tunnel_bar():
    import mujoco
    from g1_parkour import tasks
    from g1_parkour.env import ParkourEnv

    for skill, attribute in (("gap", "_scan_clipped"), ("tunnel", "_ceiling_seen")):
        cfg = tasks.parkour_cfg()
        cfg.terrain.params = {"num_modules": 1, "skills": (skill,)}
        cfg.terrain.curriculum = False
        cfg.terrain.difficulty = 1.0
        env = ParkourEnv(cfg)
        try:
            env.reset(seed=0)
            assert not getattr(env, attribute)  # on the start pad: nothing to look at
            if skill == "gap":
                landing = next(b for b in env.terrain.boxes if b.name.startswith("gap_land"))
                env.data.qpos[0] = landing.pos[0] - landing.size[0] - 0.9 - 0.3  # 0.3 m before the edge
            else:
                bar = next(b for b in env.terrain.boxes if b.ceiling)
                env.data.qpos[0] = bar.pos[0] - 0.1
            mujoco.mj_forward(env.model, env.data)
            env._compute_observation()
            assert getattr(env, attribute)
        finally:
            env.close()



def test_glances_start_only_on_flat_ground_and_finish_before_any_cue():
    camera, gazes = StableEyeCamera(seed=3), []
    for k in range(int(30.0 / DT)):
        camera.update(body(k * DT, (k * DT, 0.0, 1.3)), 0, heading=0.0, glance=False)
        gazes.append(camera.gaze)
    assert max(abs(g) for g in gazes) == 0.0  # never flat: never a glance
    # A glance under way holds against a ceiling cue until its time is up.
    camera = StableEyeCamera(seed=3)
    k, started = 0, None
    while started is None or k * DT < camera.glance_until + 1.5:
        t = k * DT
        under_way = np.isfinite(camera.glance_until) and t < camera.glance_until
        if started is None and under_way:
            started = (camera.glance, camera.glance_until)
        camera.update(body(t, (t, 0.0, 1.3)), 0, heading=0.0, glance=started is None,
                      look_up=started is not None)
        if started is not None and t < started[1]:
            assert camera.gaze <= started[0] + 1e-9  # easing into the glance, not the 25° cue
        k += 1
    assert camera.gaze == pytest.approx(StableEyeCamera.CUE_UP_ANGLE, abs=0.5)  # then the cue

def test_env_looks_down_at_tall_ground_and_glances_only_on_the_flat():
    import mujoco
    from g1_parkour import tasks
    from g1_parkour.env import ParkourEnv

    cfg = tasks.parkour_cfg()
    cfg.terrain.params = {"num_modules": 1, "skills": ("hurdle",)}
    cfg.terrain.curriculum = False
    cfg.terrain.difficulty = 1.0  # a 0.5 m hurdle
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        env.data.qpos[0] = 1.0  # on the start pad: flat, nothing to look at
        mujoco.mj_forward(env.model, env.data)
        env._compute_observation()
        assert env._gaze_cues() == dict(look_down=False, look_up=False, glance=True)
        hurdle = next(b for b in env.terrain.boxes if b.name.startswith("hurdle_") and "run" not in b.name)
        env.data.qpos[0] = hurdle.pos[0] - 1.0  # the hurdle in the scan, 1 m ahead
        mujoco.mj_forward(env.model, env.data)
        env._compute_observation()
        cues = env._gaze_cues()
        assert cues["look_down"] and not cues["glance"]
    finally:
        env.close()



def test_looking_up_lasts_only_while_the_upward_scan_sees_the_ceiling():
    camera, gazes = StableEyeCamera(), []
    for k in range(int(4.0 / DT)):  # the ceiling is seen from 0.2 s to 2.0 s
        t = k * DT
        camera.update(body(t, (0.0, 0.0, 1.3)), 0, heading=0.0, look_up=0.2 <= t < 2.0, glance=False)
        gazes.append((t, camera.gaze))
    assert max(g for t, g in gazes if t < 2.0) == pytest.approx(StableEyeCamera.CUE_UP_ANGLE, abs=1.0)
    after = [g for t, g in gazes if 2.0 + 5 * StableEyeCamera.GAZE_TIME_CONSTANT <= t]
    assert max(after) < 1.0  # back to normal once the ceiling is gone, no hold
