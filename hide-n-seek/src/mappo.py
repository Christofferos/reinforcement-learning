"""
MAPPO (Multi-Agent PPO) for Hide and Seek
==========================================
Centralized Training with Decentralized Execution (CTDE).

Each *team* shares a single policy (hiders share one, seekers share another).
The critic receives the full global state; the actor sees only local observations.

References:
    Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2021)
    Baker et al. "Emergent Tool Use From Multi-Agent Autocurricula" (2019)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ──────────────────────────────────────────────────────────
# Networks
# ──────────────────────────────────────────────────────────

def _init_weights(module: nn.Module, gain: float = np.sqrt(2)):
    """Orthogonal initialization (recommended by MAPPO paper)."""
    if isinstance(module, (nn.Linear,)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ActorNetwork(nn.Module):
    """
    Gaussian policy for continuous actions.
    Decentralised — takes only the agent's local observation.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, use_feature_norm: bool = True):
        super().__init__()

        self.use_feature_norm = use_feature_norm
        if use_feature_norm:
            self.feature_norm = nn.LayerNorm(obs_dim)

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable per-dim

        self.apply(lambda m: _init_weights(m, gain=np.sqrt(2)))
        _init_weights(self.mean_head, gain=0.01)

    def forward(self, obs: torch.Tensor):
        if self.use_feature_norm:
            obs = self.feature_norm(obs)
        x = self.trunk(obs)
        mean = self.mean_head(x)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # Clip (not tanh) to [-1, 1] — avoids Jacobian correction issues
        action = torch.clamp(action, -1.0, 1.0)
        return action, log_prob

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self.forward(obs)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Centralised value function.
    Takes the full global state as input.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, use_feature_norm: bool = True):
        super().__init__()

        self.use_feature_norm = use_feature_norm
        if use_feature_norm:
            self.feature_norm = nn.LayerNorm(state_dim)

        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(lambda m: _init_weights(m, gain=np.sqrt(2)))
        _init_weights(self.value_head, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.use_feature_norm:
            state = self.feature_norm(state)
        x = self.trunk(state)
        return self.value_head(x)


# ──────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores transitions for one team across one rollout."""

    def __init__(self):
        self.obs = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, global_state, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def get_batches(self, advantages, returns, mini_batch_size: int):
        """Yield random mini-batches."""
        n = len(self.obs)
        indices = np.arange(n)
        np.random.shuffle(indices)

        obs = np.array(self.obs, dtype=np.float32)
        states = np.array(self.global_states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            idx = indices[start:end]
            yield (
                torch.FloatTensor(obs[idx]),
                torch.FloatTensor(states[idx]),
                torch.FloatTensor(actions[idx]),
                torch.FloatTensor(old_log_probs[idx]),
                torch.FloatTensor(advantages[idx]),
                torch.FloatTensor(returns[idx]),
            )

    def clear(self):
        self.obs.clear()
        self.global_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.obs)


# ──────────────────────────────────────────────────────────
# Team Policy (shared across agents in the same team)
# ──────────────────────────────────────────────────────────

class TeamPolicy:
    """
    A MAPPO policy shared by all agents in one team.

    Contains an actor (decentralised) and a critic (centralised).
    """

    def __init__(
        self,
        team_name: str,
        obs_dim: int,
        act_dim: int,
        global_state_dim: int,
        config: dict,
        device: str = "cpu",
    ):
        self.team_name = team_name
        self.device = torch.device(device)
        self.config = config

        # Hyperparameters
        self.gamma = config.get("gamma", 0.998)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.ppo_epochs = config.get("ppo_epochs", 15)
        self.mini_batch_size = config.get("mini_batch_size", 256)

        hidden_dim = config.get("hidden_dim", 256)
        n_layers = config.get("n_layers", 2)
        use_feature_norm = config.get("use_feature_norm", True)

        # Networks
        self.actor = ActorNetwork(obs_dim, act_dim, hidden_dim, n_layers, use_feature_norm).to(self.device)
        self.critic = CriticNetwork(global_state_dim, hidden_dim, n_layers, use_feature_norm).to(self.device)

        lr_actor = config.get("lr_actor", 3e-4)
        lr_critic = config.get("lr_critic", 1e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        # Per-agent rollout buffers (keys can be strings or tuples)
        self.buffers: dict = {}

    def init_buffers(self, agent_keys: list):
        """Initialise rollout buffers. Keys can be agent names (str) or (env_idx, name) tuples."""
        self.buffers = {key: RolloutBuffer() for key in agent_keys}

    def select_action(self, obs: np.ndarray, global_state: np.ndarray,
                      deterministic: bool = False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        state_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.actor.get_action(obs_t, deterministic=deterministic)
            value = self.critic(state_t)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze(0).item(),
            value.squeeze(0).item(),
        )

    def store_transition(self, buffer_key, obs, global_state, action,
                         log_prob, reward, done, value):
        self.buffers[buffer_key].add(obs, global_state, action, log_prob, reward, done, value)

    def update(self) -> dict[str, float]:
        """Run PPO update over all collected data from all team agents."""
        # Merge data from all agent buffers
        all_advantages = []
        all_returns = []

        for name, buf in self.buffers.items():
            if len(buf) == 0:
                continue
            # Compute last value
            last_obs = buf.obs[-1]
            last_state = buf.global_states[-1]
            with torch.no_grad():
                state_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                last_value = self.critic(state_t).item()
            advantages, returns = buf.compute_returns(last_value, self.gamma, self.gae_lambda)
            all_advantages.append(advantages)
            all_returns.append(returns)

        if not all_advantages:
            return {}

        # Normalise advantages globally
        all_adv = np.concatenate(all_advantages)
        adv_mean, adv_std = all_adv.mean(), all_adv.std() + 1e-8

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.ppo_epochs):
            # Iterate over each agent's buffer
            adv_offset = 0
            ret_offset = 0
            for name, buf in self.buffers.items():
                if len(buf) == 0:
                    continue
                n = len(buf)
                advantages = all_advantages[list(self.buffers.keys()).index(name)]
                returns = all_returns[list(self.buffers.keys()).index(name)]
                norm_advantages = (advantages - adv_mean) / adv_std

                for (b_obs, b_state, b_act, b_old_lp, b_adv, b_ret) in buf.get_batches(
                    norm_advantages, returns, self.mini_batch_size
                ):
                    b_obs = b_obs.to(self.device)
                    b_state = b_state.to(self.device)
                    b_act = b_act.to(self.device)
                    b_old_lp = b_old_lp.to(self.device)
                    b_adv = b_adv.unsqueeze(-1).to(self.device)
                    b_ret = b_ret.unsqueeze(-1).to(self.device)

                    # Actor loss
                    new_log_prob, entropy = self.actor.evaluate(b_obs, b_act)
                    ratio = (new_log_prob - b_old_lp.unsqueeze(-1)).exp()
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Critic loss
                    value_pred = self.critic(b_state)
                    value_loss = nn.functional.mse_loss(value_pred, b_ret)

                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                    # Update actor
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    n_updates += 1

        # Clear buffers
        for buf in self.buffers.values():
            buf.clear()

        if n_updates == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
