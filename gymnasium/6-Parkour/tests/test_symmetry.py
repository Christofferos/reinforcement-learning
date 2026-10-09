"""Left/right mirror maps and symmetric PPO."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour import symmetry, tasks  # noqa: E402
from g1_parkour.env import ParkourEnv  # noqa: E402


def _random_state(env, rng, height=1.3):
    """An arbitrary tilted, moving pose with every joint inside its range."""
    qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
    qpos[0:3] = (0.4, 0.3, height)
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([0.3, -0.2, 0.7]), "xyz")
    qpos[3:7] = quat
    hinges = range(1, env.model.njnt)
    low, high = env.model.jnt_range[list(hinges)].T
    qpos[7:] = rng.uniform(low, high)
    qvel[:] = rng.normal(0.0, 1.0, env.model.nv)
    return qpos, qvel


def _mirror_state(env, qpos, qvel):
    """Reflect the simulation in the world x-z plane (y -> -y)."""
    joints = symmetry.hinge_mirror(env.model)
    qpos, qvel = qpos.copy(), qvel.copy()
    qpos[1] = -qpos[1]
    qpos[3:7] = qpos[3:7] * (1.0, -1.0, 1.0, -1.0)  # R -> REFLECTION @ R @ REFLECTION
    qpos[7:] = joints(qpos[7:])
    qvel[0:3] = qvel[0:3] * (1.0, -1.0, 1.0)  # world-frame linear velocity
    qvel[3:6] = qvel[3:6] * (-1.0, 1.0, -1.0)  # body-frame angular velocity
    qvel[6:] = joints(qvel[6:])
    return qpos, qvel


def _set_state(env, qpos, qvel):
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.parametrize("task_id", ["Flat", "Rough", "Procedural", "Parkour-Ablation-ObsHistory-v0",
                                     "Parkour-Ablation-Blind-v0"])
def test_mirrors_cover_the_spaces_and_undo_themselves(task_id):
    env = gym.make(task_id).unwrapped
    try:
        observation, action = env.observation_mirror(), env.action_mirror()
        assert len(observation) == env.observation_space.shape[0]
        assert len(action) == env.action_space.shape[0]
        x = np.random.default_rng(0).normal(size=len(observation))
        assert np.array_equal(observation(observation(x)), x)
    finally:
        env.close()


def test_humanoid_joint_signs_follow_the_joint_axes():
    env = gym.make("Flat").unwrapped
    try:
        model = env.model
        mirror = symmetry.hinge_mirror(model)
        names = [model.joint(j).name for j in range(1, model.njnt)]
        signs = {name: int(sign) for name, sign in zip(names, mirror.sign)}
        partners = {name: names[index] for name, index in zip(names, mirror.perm)}
        # Yaw and roll of the spine flip; its pitch does not.
        assert (signs["abdomen_z"], signs["abdomen_y"], signs["abdomen_x"]) == (-1, 1, -1)
        # Leg joints have mirrored axes in the XML, so equal angles are mirror images.
        for joint in ("hip_x", "hip_z", "hip_y", "knee"):
            assert partners[f"left_{joint}"] == f"right_{joint}" and signs[f"left_{joint}"] == 1
        # The shoulders share axes up to the reflection, so their angles flip (ranges -85..60 <-> -60..85).
        assert signs["left_shoulder1"] == signs["left_shoulder2"] == -1 and signs["left_elbow"] == 1
    finally:
        env.close()


def test_observation_mirror_matches_a_mirrored_simulation():
    cfg = tasks.flat_cfg()
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=3)
        rng = np.random.default_rng(1)
        qpos, qvel = _random_state(env, rng)
        waypoints = env._waypoints.copy()
        waypoints[:, 1] += rng.uniform(-1.0, 1.0, len(waypoints))
        action = rng.uniform(-1.0, 1.0, env.model.nu)

        _set_state(env, qpos, qvel)
        env._waypoints, env._prev_action = waypoints.copy(), action.copy()
        obs = env._compute_observation()
        hits = env.model.geom_pos[env._scan_marker_ids].copy()

        _set_state(env, *_mirror_state(env, qpos, qvel))
        env._waypoints = waypoints * (1.0, -1.0, 1.0)
        env._prev_action = env.action_mirror()(action)
        mirrored_obs = env._compute_observation()
        mirrored_hits = env.model.geom_pos[env._scan_marker_ids].copy()

        mirror = env.observation_mirror()
        np.testing.assert_allclose(mirrored_obs, mirror(obs), atol=1e-9)
        # Flat ground gives every scan point the same height, so check the scan geometry too:
        # mirrored point k samples the reflection of the point it takes its value from.
        scan = symmetry.lateral_mirror(cfg.observation.scan_points)
        np.testing.assert_allclose(mirrored_hits[:, :2], hits[scan.perm, :2] * (1.0, -1.0), atol=1e-9)
    finally:
        env.close()


@pytest.mark.parametrize("standing", [False, True])
def test_action_mirror_matches_mirrored_dynamics(standing):
    cfg = tasks.flat_cfg()
    cfg.events.push_robot = False
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(2)
        if standing:
            # Land on both feet, then move off the mirror plane (y = 0) so the reflection moves
            # the robot and the contacts.
            for _ in range(40):
                env._apply_action(rng.uniform(-0.3, 0.3, env.model.nu))
            qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
            qpos[1] = 0.3
            _set_state(env, qpos, qvel)
            assert env._foot_contacts().all()
        else:
            qpos, qvel = _random_state(env, rng, height=3.0)  # airborne: no contacts
        action = rng.uniform(-1.0, 1.0, env.model.nu)
        mirror = env.action_mirror()

        def step(state, applied):
            _set_state(env, *state)
            env._apply_action(applied)
            return env.data.qpos.copy(), env.data.qvel.copy()

        expected = _mirror_state(env, *step((qpos, qvel), action))
        mirrored_state = _mirror_state(env, qpos, qvel)
        mirrored = step(mirrored_state, mirror(action))
        unsigned = step(mirrored_state, action[mirror.perm])  # swap without the axis signs
        # In flight the mirror is exact to round-off; in contact the solver stops at its
        # tolerance, which leaves ~1e-7 (an asymmetric joint value alone gave ~1e-3).
        position_tol, velocity_tol = (1e-6, 1e-4) if standing else (1e-10, 1e-8)
        np.testing.assert_allclose(mirrored[0], expected[0], atol=position_tol)
        np.testing.assert_allclose(mirrored[1], expected[1], atol=velocity_tol)
        assert np.abs(unsigned[0] - expected[0]).max() > 1e-2
    finally:
        env.close()


def _flat_vec_env(seed=0, normalize=False):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_vec_env("Flat", n_envs=1, seed=seed)
    return VecNormalize(env, norm_reward=False) if normalize else env


def _maps(vec_env):
    base = vec_env.unwrapped.envs[0].unwrapped if hasattr(vec_env, "venv") else vec_env.envs[0].unwrapped
    return base.observation_mirror(), base.action_mirror()


def _ppo(cls, env, **kwargs):
    return cls("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=2, seed=0, device="cpu", **kwargs)


def test_normalised_observations_are_mirrored_in_raw_units():
    import torch as th
    from g1_parkour.symmetric_ppo import SymmetricPPO

    env = _flat_vec_env(normalize=True)
    try:
        maps = _maps(env)
        rng = np.random.default_rng(0)
        size = env.observation_space.shape[0]
        env.obs_rms.mean = rng.normal(size=size)  # deliberately asymmetric statistics
        env.obs_rms.var = rng.uniform(0.1, 4.0, size)
        model = _ppo(SymmetricPPO, env, symmetry="loss", mirror=maps)
        mirror_observations, mirror_actions = model.mirror_functions()
        raw = rng.normal(size=(5, size))
        mirrored = mirror_observations(th.as_tensor(env.normalize_obs(raw), dtype=th.float32))
        np.testing.assert_allclose(mirrored.numpy(), env.normalize_obs(maps[0](raw)), atol=1e-4)
        actions = rng.uniform(-1.0, 1.0, (5, env.action_space.shape[0]))
        np.testing.assert_allclose(mirror_actions(th.as_tensor(actions)).numpy(), maps[1](actions))
    finally:
        env.close()


def test_mirror_loss_with_zero_weight_trains_exactly_like_ppo():
    import torch as th
    from stable_baselines3 import PPO
    from g1_parkour.symmetric_ppo import SymmetricPPO

    def trained(cls, **kwargs):
        # Build and train one at a time: construction reseeds the global RNGs the rollout uses.
        env = _flat_vec_env()
        try:
            model = _ppo(cls, env, **kwargs)
            model.learn(64)
            return [parameter.detach().clone() for parameter in model.policy.parameters()]
        finally:
            env.close()

    probe = _flat_vec_env()
    maps = _maps(probe)
    probe.close()
    plain = trained(PPO)
    symmetric = trained(SymmetricPPO, symmetry="loss", symmetry_coef=0.0, mirror=maps)
    assert all(th.equal(a, b) for a, b in zip(plain, symmetric))


@pytest.mark.parametrize("mode", ["loss", "both"])
def test_mirror_loss_makes_an_asymmetric_policy_symmetric(mode):
    import torch as th
    from g1_parkour.symmetric_ppo import SymmetricPPO

    env = _flat_vec_env()
    try:
        model = _ppo(SymmetricPPO, env, symmetry=mode, symmetry_coef=10.0, learning_rate=1e-3,
                     mirror=_maps(env))
        with th.no_grad():  # lopsided start, like a policy that always leads with one leg
            model.policy.action_net.bias.copy_(th.linspace(-1.0, 1.0, env.action_space.shape[0]))
        observations = [env.reset()]
        for _ in range(63):
            observations.append(env.step(np.array([env.action_space.sample()]))[0])
        observations = th.as_tensor(np.concatenate(observations), dtype=th.float32)
        mirror_observations, mirror_actions = model.mirror_functions()

        def asymmetry() -> float:
            with th.no_grad():
                mean = model.policy.get_distribution(observations).mode()
                mirrored = model.policy.get_distribution(mirror_observations(observations)).mode()
            return th.mean((mirrored - mirror_actions(mean)) ** 2).item()

        before = asymmetry()
        model.learn(320)
        assert asymmetry() < 0.5 * before
    finally:
        env.close()


def test_augmentation_trains_on_mirrored_samples_and_logs_them():
    from g1_parkour.symmetric_ppo import SymmetricPPO

    env = _flat_vec_env(normalize=True)
    try:
        model = _ppo(SymmetricPPO, env, symmetry="augment", mirror=_maps(env))
        model.learn(64)
        logged = model.logger.name_to_value
        assert "train/mirrored_clip_fraction" in logged and "train/symmetry_loss" in logged
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["augment", "both"])
def test_augmentation_keeps_the_action_std_mirror_symmetric(mode):
    import torch as th
    from g1_parkour.symmetric_ppo import SymmetricPPO

    env = _flat_vec_env()
    try:
        maps = _maps(env)
        model = _ppo(SymmetricPPO, env, symmetry=mode, mirror=maps)
        log_std = model.policy.log_std
        perm = th.as_tensor(maps[1].perm)
        with th.no_grad():  # left/right std drifted apart, like flat_0.0.10's
            log_std.copy_(th.linspace(-1.2, -0.8, len(log_std)))
        assert not th.allclose(log_std, log_std[perm])
        model.learn(64)
        assert th.allclose(log_std, log_std[perm])
    finally:
        env.close()


def test_checkpoints_load_without_the_maps(tmp_path):
    from stable_baselines3 import PPO
    from g1_parkour.symmetric_ppo import SymmetricPPO

    env = _flat_vec_env()
    try:
        maps = _maps(env)
        model = _ppo(SymmetricPPO, env, symmetry="loss", symmetry_coef=2.0, mirror=maps)
        model.save(tmp_path / "model.zip")
        plain = PPO.load(tmp_path / "model.zip")  # play.py and ablate.py load checkpoints this way
        assert getattr(plain, "mirror", None) is None
        resumed = SymmetricPPO.load(tmp_path / "model.zip", env=env, mirror=maps)
        assert (resumed.symmetry, resumed.symmetry_coef, resumed.mirror) == ("loss", 2.0, maps)
    finally:
        env.close()
