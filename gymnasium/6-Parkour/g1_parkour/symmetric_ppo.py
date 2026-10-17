"""PPO with left/right mirror symmetry.

``symmetry`` selects how the mirror maps from :mod:`g1_parkour.symmetry` enter training:

- ``"loss"``: a mirror loss pulling the policy mean at each mirrored state toward the mirrored
  mean at the original state (Yu, Turk & Liu 2018, also used for Digit by van Marum et al.
  2024). Abdolhosseini et al. (2019) found it the most consistent way to get symmetric gaits.
  It works from any starting policy, including an asymmetric one being resumed.
- ``"augment"``: every minibatch also trains on its mirrored copy, which keeps the original
  advantages, returns and old log-probabilities (Mittal et al. 2024). It learns fastest from a
  near-symmetric policy, but mirrored samples of a strongly asymmetric policy fall far outside
  the PPO clip range and teach almost nothing. Augmentation also keeps the action std mirror
  symmetric (see :meth:`SymmetricPPO.symmetrise_action_std`).
- ``"both"``, or ``"none"`` for plain PPO.

Action noise can be scaled per step: each environment's ``info["action_noise_scale"]`` (the
environment sets it from the module the robot is on, ``TerrainCfg.skill_action_noise``) scales
the policy's std for the next action it samples. The scale is stored with the step and the update
evaluates log-probabilities and entropy under the same scaled distribution, so the PPO ratio
stays exact. With every scale 1 training follows stable_baselines3's own code paths.

``bounds_coef`` adds an action-bounds loss (as rl_games' ``bounds_loss_coef``): the squared
amount by which each action's mean lies outside the action space ([-1, 1] here), summed over
actions and averaged over the minibatch. Actions are clipped to [-1, 1], so a mean of 5 acts like a mean of 1 but no noise
can ever move it, and nothing stopped the policy drifting there: procedural_0.0.4-0.0.7 had a
median |mean| of 2.2 and 88% of hip and knee means outside the bounds, which made their control
bang-bang. The loss pulls such means back to the bound without changing what the clipped action
does; inside it there is no loss at all.

Observations reach the policy normalised by ``VecNormalize``, whose statistics need not be
symmetric, so they are mirrored in raw units: un-normalise, mirror, normalise again.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from torch.nn import functional as F

from .symmetry import Mirror


class NoiseScaledSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    noise_scales: th.Tensor


class NoiseScaledRolloutBuffer(RolloutBuffer):
    """A rollout buffer that also keeps the action-noise scale each step was sampled with."""

    def reset(self) -> None:
        self.noise_scales = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(self, *args, noise_scale: np.ndarray | None = None, **kwargs) -> None:
        if noise_scale is not None:
            self.noise_scales[self.pos] = noise_scale
        super().add(*args, **kwargs)

    def get(self, batch_size: int | None = None):
        if not self.generator_ready:
            self.noise_scales = self.swap_and_flatten(self.noise_scales)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> NoiseScaledSamples:
        samples = super()._get_samples(batch_inds, env)
        return NoiseScaledSamples(*samples, self.to_torch(self.noise_scales[batch_inds].flatten()))


class SymmetricPPO(PPO):
    MODES = ("none", "loss", "augment", "both")

    def __init__(self, *args, symmetry: str = "none", symmetry_coef: float = 1.0,
                 mirror: tuple[Mirror, Mirror] | None = None, max_std: float | None = None,
                 bounds_coef: float = 0.0, **kwargs):
        if max_std is not None and not max_std > 0:
            raise ValueError("max_std must be positive")
        if not bounds_coef >= 0:
            raise ValueError("bounds_coef must be non-negative")
        self.bounds_coef = bounds_coef
        """Weight of the action-bounds loss; 0 leaves it out."""
        self.symmetry = symmetry
        self.symmetry_coef = symmetry_coef
        self.mirror = mirror
        """(observation mirror, action mirror), rebuilt from the environment on every run."""
        self.max_std = max_std
        """Ceiling on the learned action std; None leaves it free."""
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        # Also on load, where the checkpoint's saved buffer class would replace a constructor's.
        self.rollout_buffer_class = NoiseScaledRolloutBuffer
        super()._setup_model()

    def _setup_learn(self, *args, **kwargs):
        # Cap a resumed policy's std before the first rollout, not after the first update.
        result = super()._setup_learn(*args, **kwargs)
        self.cap_action_std()
        self._noise_scales = np.ones(self.env.num_envs, dtype=np.float32)
        return result

    def _excluded_save_params(self) -> list[str]:
        # Checkpoints stay loadable by plain PPO.load; the maps come from the environment.
        return super()._excluded_save_params() + ["mirror", "rollout_buffer_class", "_noise_scales"]

    def _scaled_distribution(self, obs: th.Tensor, scales: th.Tensor) -> th.distributions.Normal:
        """The behaviour distribution: the policy's Gaussian with its std times each row's scale."""
        normal = self.policy.get_distribution(obs).distribution
        return th.distributions.Normal(normal.mean, normal.stddev * scales.reshape(-1, 1))

    def evaluate_scaled_actions(self, obs: th.Tensor, actions: th.Tensor, scales: th.Tensor | None):
        """``policy.evaluate_actions`` under the distribution the actions were sampled from."""
        if scales is None or bool(th.all(scales == 1.0)):
            return self.policy.evaluate_actions(obs, actions)
        distribution = self._scaled_distribution(obs, scales)
        values = self.policy.predict_values(obs)
        return values, distribution.log_prob(actions).sum(dim=1), distribution.entropy().sum(dim=1)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        # Follows stable_baselines3 2.6.0 OnPolicyAlgorithm.collect_rollouts; lines marked "noise"
        # are additions. gSDE (use_sde) is not supported with noise scales.
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            scales = self._noise_scales.copy()  # noise
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if np.all(scales == 1.0) or self.use_sde:
                    actions, values, log_probs = self.policy(obs_tensor)
                else:  # noise: sample from the policy's Gaussian with each env's scaled std
                    distribution = self._scaled_distribution(obs_tensor, th.as_tensor(scales, device=self.device))
                    actions = distribution.sample()
                    log_probs = distribution.log_prob(actions).sum(dim=1)
                    values = self.policy.predict_values(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                noise_scale=scales,  # noise
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            # noise: the scale for each env's next state; a new episode starts at 1 (its start pad).
            self._noise_scales = np.array([
                1.0 if done else float(info.get("action_noise_scale", 1.0)) for done, info in zip(dones, infos)
            ], dtype=np.float32)

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def mirror_functions(self):
        """Torch functions mirroring a batch of policy observations and of actions."""
        observation, action = self.mirror
        if len(observation) != self.observation_space.shape[0] or len(action) != self.action_space.shape[0]:
            raise ValueError("mirror maps do not match the observation and action spaces")
        scale, offset, clip = observation.sign, np.zeros(len(observation)), float("inf")
        normalizer = self.get_vec_normalize_env()
        if normalizer is not None and normalizer.norm_obs:
            # normalised x_k = (raw_k - mean_k) / std_k and mirrored raw_k = sign_k * raw_perm(k).
            std = np.sqrt(normalizer.obs_rms.var + normalizer.epsilon)
            mean = normalizer.obs_rms.mean
            scale = observation.sign * std[observation.perm] / std
            offset = (observation.sign * mean[observation.perm] - mean) / std
            clip = normalizer.clip_obs

        def tensor(values, dtype=th.float32):
            return th.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

        obs_perm, obs_scale, obs_offset = tensor(observation.perm, th.long), tensor(scale), tensor(offset)
        act_perm, act_sign = tensor(action.perm, th.long), tensor(action.sign)

        def mirror_observations(obs: th.Tensor) -> th.Tensor:
            return th.clamp(obs[:, obs_perm] * obs_scale + obs_offset, -clip, clip)

        def mirror_actions(actions: th.Tensor) -> th.Tensor:
            return actions[:, act_perm] * act_sign

        return mirror_observations, mirror_actions

    def symmetrise_action_std(self) -> None:
        """Average each action's log std with its mirror partner's.

        The std is one learned value per action, the same in every state, and the mirror loss
        only matches means, so left and right std drift apart (by up to 0.055 on flat_0.0.10).
        Augmentation scores a mirrored action against the original's old log-probability, which
        assumes both are equally likely; unequal std alone kept 60% of flat_0.0.10's mirrored
        samples outside the clip range, even with its means made exactly mirrored.
        """
        log_std = getattr(self.policy, "log_std", None)
        if log_std is None:
            return
        perm = th.as_tensor(self.mirror[1].perm, dtype=th.long, device=log_std.device)
        with th.no_grad():
            log_std.copy_(0.5 * (log_std + log_std[..., perm]))

    def cap_action_std(self) -> None:
        """Clamp each action's std at ``max_std``.

        Actions are clipped to [-1, 1] before the robot gets them, so once the noise overshoots
        the limits a wider std changes nothing the robot does and nothing in the reward pushes
        back, while the entropy bonus keeps pushing up: rough_0.0.2's std drifted to 2.32, with
        66% of its training actions clipped. Below the cap the std is still learned.
        """
        log_std = getattr(self.policy, "log_std", None)
        if self.max_std is None or log_std is None:
            return
        with th.no_grad():
            log_std.clamp_(max=float(np.log(self.max_std)))

    def train(self) -> None:
        noise_scales = getattr(self.rollout_buffer, "noise_scales", np.ones(1))
        scaled = not np.all(noise_scales == 1.0)  # noise
        self.logger.record("train/noise_scaled_share", float(np.mean(noise_scales != 1.0)))  # noise
        bounds_coef = getattr(self, "bounds_coef", 0.0)  # bounds (absent from older checkpoints)
        if self.symmetry == "none" and not scaled and not bounds_coef:
            super().train()
            self.cap_action_std()
            return
        if self.symmetry not in self.MODES:
            raise ValueError(f"symmetry must be one of {self.MODES}")
        symmetric = self.symmetry != "none"
        if symmetric and (self.mirror is None or not isinstance(self.action_space, spaces.Box)):
            raise ValueError("symmetric PPO needs mirror maps and a continuous action space")
        if symmetric:
            mirror_observations, mirror_actions = self.mirror_functions()
        use_loss = self.symmetry in ("loss", "both")
        use_augment = self.symmetry in ("augment", "both")

        # Follows stable_baselines3 2.6.0 PPO.train; lines marked "symmetry", "max_std" or "noise"
        # are additions.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        symmetry_losses, mirrored_clip_fractions = [], []  # symmetry
        bounds_losses, beyond_shares = [], []  # bounds

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                scales = getattr(rollout_data, "noise_scales", None)  # noise
                values, log_prob, entropy = self.evaluate_scaled_actions(rollout_data.observations, actions, scales)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                def surrogate_loss(ratio: th.Tensor) -> th.Tensor:
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    return -th.min(policy_loss_1, policy_loss_2).mean()

                def critic_loss(values: th.Tensor) -> th.Tensor:
                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    return F.mse_loss(rollout_data.returns, values_pred)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = surrogate_loss(ratio)
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                value_loss = critic_loss(values)

                # symmetry: mirrored copies of the minibatch.
                if symmetric:
                    mirrored_obs = mirror_observations(rollout_data.observations)
                if use_augment:
                    # The mirrored action has the original's probability under the mirrored
                    # behaviour policy, so the ratio keeps the original old log-probability. It
                    # was sampled at the same noise scale (a mirror does not change the module).
                    mirrored_values, mirrored_log_prob, _ = self.evaluate_scaled_actions(
                        mirrored_obs, mirror_actions(actions), scales
                    )
                    mirrored_ratio = th.exp(mirrored_log_prob - rollout_data.old_log_prob)
                    policy_loss = 0.5 * (policy_loss + surrogate_loss(mirrored_ratio))
                    value_loss = 0.5 * (value_loss + critic_loss(mirrored_values.flatten()))
                    mirrored_clip_fractions.append(
                        th.mean((th.abs(mirrored_ratio - 1) > clip_range).float()).item()
                    )
                if symmetric:
                    with th.no_grad():
                        target = mirror_actions(self.policy.get_distribution(rollout_data.observations).mode())
                    with th.set_grad_enabled(use_loss):
                        mirrored_mean = self.policy.get_distribution(mirrored_obs).mode()
                        symmetry_loss = F.mse_loss(mirrored_mean, target)
                    symmetry_losses.append(symmetry_loss.item())

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if use_loss:
                    loss = loss + self.symmetry_coef * symmetry_loss  # symmetry
                if bounds_coef:  # bounds: means outside the action clip
                    mean = self.policy.get_distribution(rollout_data.observations).mode()
                    low = th.as_tensor(self.action_space.low, dtype=mean.dtype, device=mean.device)
                    high = th.as_tensor(self.action_space.high, dtype=mean.dtype, device=mean.device)
                    outside = th.relu(mean - high) + th.relu(low - mean)
                    bounds_loss = (outside ** 2).sum(dim=1).mean()
                    loss = loss + bounds_coef * bounds_loss
                    bounds_losses.append(bounds_loss.item())
                    beyond_shares.append((outside > 0).float().mean().item())

                # Early stopping on the original samples only: mirrored samples of an
                # asymmetric policy are far off-policy and would end every update at once.
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                if use_augment:
                    self.symmetrise_action_std()  # symmetry
                self.cap_action_std()  # max_std

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if symmetry_losses:
            self.logger.record("train/symmetry_loss", np.mean(symmetry_losses))  # symmetry
        if bounds_losses:  # bounds
            self.logger.record("train/bounds_loss", np.mean(bounds_losses))
            self.logger.record("train/mean_beyond_bounds", np.mean(beyond_shares))
        if mirrored_clip_fractions:
            self.logger.record("train/mirrored_clip_fraction", np.mean(mirrored_clip_fractions))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
