"""Per-module action-noise scale: the environment reports it, SymmetricPPO samples and updates with it."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_parkour import tasks  # noqa: E402
from g1_parkour.env import ParkourEnv  # noqa: E402
from g1_parkour.symmetric_ppo import NoiseScaledRolloutBuffer, SymmetricPPO  # noqa: E402


def test_the_environment_reports_the_noise_scale_of_the_module_ahead():
    cfg = tasks.parkour_cfg()
    assert cfg.terrain.skill_action_noise == {"stepping_stones": 0.6}
    cfg.terrain.params = {"num_modules": 4, "skills": ("run", "stepping_stones")}
    cfg.terrain.curriculum = False
    env = ParkourEnv(cfg)
    try:
        env.reset(seed=0)
        modules, owners = env.terrain.modules, env.terrain.waypoint_modules
        assert "stepping_stones" in modules and "run" in modules
        for index, owner in enumerate(owners):
            env._target_index = index
            expected = 0.6 if owner > 0 and modules[owner] == "stepping_stones" else 1.0
            assert env._info({})["action_noise_scale"] == expected  # 1 on the start pad and the finish
    finally:
        env.close()
    cfg.terrain.skill_action_noise = {}
    plain = ParkourEnv(cfg)
    try:
        plain.reset(seed=0)
        assert {plain._info({})["action_noise_scale"]} == {1.0}
    finally:
        plain.close()


class _HalfNoise(gym.Wrapper):
    """Pendulum whose every state asks for half the policy's action noise."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, {**info, "action_noise_scale": 0.5}


def _model(**kwargs) -> SymmetricPPO:
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: _HalfNoise(gym.make("Pendulum-v1"))])
    return SymmetricPPO("MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=1, seed=0, device="cpu", **kwargs)


def _collect(model: SymmetricPPO) -> None:
    _, callback = model._setup_learn(model.n_steps, None)
    callback.on_training_start(locals(), globals())
    assert model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps)


def test_rollouts_sample_and_store_the_scaled_noise():
    model = _model()
    assert isinstance(model.rollout_buffer, NoiseScaledRolloutBuffer)
    _collect(model)
    buffer = model.rollout_buffer
    scales = buffer.noise_scales[:, 0]
    assert scales[0] == 1.0 and np.all(scales[1:] == 0.5)  # the first step follows a reset
    obs = th.as_tensor(buffer.observations[:, 0])
    actions = th.as_tensor(buffer.actions[:, 0])
    with th.no_grad():
        normal = model.policy.get_distribution(obs).distribution
        expected = th.distributions.Normal(normal.mean, normal.stddev * th.as_tensor(scales).reshape(-1, 1))
        np.testing.assert_allclose(buffer.log_probs[:, 0], expected.log_prob(actions).sum(dim=1).numpy(), atol=1e-5)
        spread = ((actions - normal.mean) / normal.stddev)[1:].std().item()
    assert spread == pytest.approx(0.5, abs=0.1)  # the actions really were sampled at half the noise


def test_scale_one_is_exactly_the_policys_own_evaluation():
    model = _model()
    obs = th.randn(16, model.observation_space.shape[0])
    actions = th.randn(16, model.action_space.shape[0])
    with th.no_grad():
        own = model.policy.evaluate_actions(obs, actions)
        same = model.evaluate_scaled_actions(obs, actions, th.ones(16))
        half = model.evaluate_scaled_actions(obs, actions, th.full((16,), 0.5))
        normal = model.policy.get_distribution(obs).distribution
        manual = th.distributions.Normal(normal.mean, 0.5 * normal.stddev).log_prob(actions).sum(dim=1)
    for a, b in zip(own, same):
        assert th.equal(a, b)
    np.testing.assert_allclose(half[1].numpy(), manual.numpy(), atol=1e-5)
    assert th.all(half[2] < own[2])  # less noise, less entropy


def test_a_scaled_rollout_trains_and_checkpoints_still_load_with_plain_ppo(tmp_path):
    from stable_baselines3 import PPO

    model = _model()
    model.learn(total_timesteps=256)
    assert np.isfinite(model.logger.name_to_value.get("train/approx_kl", 0.0))
    model.save(tmp_path / "scaled")
    assert isinstance(PPO.load(tmp_path / "scaled", device="cpu"), PPO)
    loaded = SymmetricPPO.load(tmp_path / "scaled", env=model.env, device="cpu")
    assert isinstance(loaded.rollout_buffer, NoiseScaledRolloutBuffer)


def test_the_bounds_loss_pulls_means_outside_the_action_space_back_to_its_edge():
    from stable_baselines3.common.vec_env import DummyVecEnv

    def model(bounds_coef):
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])  # actions in [-2, 2]
        ppo = SymmetricPPO("MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=4, seed=0, device="cpu",
                           learning_rate=3e-3, bounds_coef=bounds_coef)
        with th.no_grad():
            ppo.policy.action_net.bias.fill_(6.0)  # every mean far past the +2 bound
        return ppo

    obs = th.zeros(1, 3)
    pulled, free = model(1.0), model(0.0)
    for ppo in (pulled, free):
        ppo.learn(total_timesteps=128 * 6)
    mean = lambda ppo: ppo.policy.get_distribution(obs).mode().item()
    assert mean(free) > 5.0  # without the loss nothing moves a mean the clip makes irrelevant
    assert mean(pulled) < mean(free) - 1.5  # with it the mean heads back towards the +2 edge
    assert pulled.logger.name_to_value["train/mean_beyond_bounds"] >= 0.0
    assert "train/bounds_loss" in pulled.logger.name_to_value and "train/bounds_loss" not in free.logger.name_to_value
    with pytest.raises(ValueError):
        SymmetricPPO("MlpPolicy", "Pendulum-v1", bounds_coef=-1.0)


def test_train_script_defaults_to_a_moderate_bounds_loss():
    from scripts import train
    import inspect

    assert "--bounds-coef" in inspect.getsource(train.main) and 'default=0.02' in inspect.getsource(train.main)
