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

Observations reach the policy normalised by ``VecNormalize``, whose statistics need not be
symmetric, so they are mirrored in raw units: un-normalise, mirror, normalise again.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from .symmetry import Mirror


class SymmetricPPO(PPO):
    MODES = ("none", "loss", "augment", "both")

    def __init__(self, *args, symmetry: str = "none", symmetry_coef: float = 1.0,
                 mirror: tuple[Mirror, Mirror] | None = None, **kwargs):
        self.symmetry = symmetry
        self.symmetry_coef = symmetry_coef
        self.mirror = mirror
        """(observation mirror, action mirror), rebuilt from the environment on every run."""
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        # Checkpoints stay loadable by plain PPO.load; the maps come from the environment.
        return super()._excluded_save_params() + ["mirror"]

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

    def train(self) -> None:
        if self.symmetry == "none":
            return super().train()
        if self.symmetry not in self.MODES:
            raise ValueError(f"symmetry must be one of {self.MODES}")
        if self.mirror is None or not isinstance(self.action_space, spaces.Box):
            raise ValueError("symmetric PPO needs mirror maps and a continuous action space")
        mirror_observations, mirror_actions = self.mirror_functions()
        use_loss = self.symmetry in ("loss", "both")
        use_augment = self.symmetry in ("augment", "both")

        # Follows stable_baselines3 2.6.0 PPO.train; lines marked "symmetry" are additions.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        symmetry_losses, mirrored_clip_fractions = [], []  # symmetry

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
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
                mirrored_obs = mirror_observations(rollout_data.observations)
                if use_augment:
                    # The mirrored action has the original's probability under the mirrored
                    # behaviour policy, so the ratio keeps the original old log-probability.
                    mirrored_values, mirrored_log_prob, _ = self.policy.evaluate_actions(
                        mirrored_obs, mirror_actions(actions)
                    )
                    mirrored_ratio = th.exp(mirrored_log_prob - rollout_data.old_log_prob)
                    policy_loss = 0.5 * (policy_loss + surrogate_loss(mirrored_ratio))
                    value_loss = 0.5 * (value_loss + critic_loss(mirrored_values.flatten()))
                    mirrored_clip_fractions.append(
                        th.mean((th.abs(mirrored_ratio - 1) > clip_range).float()).item()
                    )
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
        self.logger.record("train/symmetry_loss", np.mean(symmetry_losses))  # symmetry
        if mirrored_clip_fractions:
            self.logger.record("train/mirrored_clip_fraction", np.mean(mirrored_clip_fractions))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
