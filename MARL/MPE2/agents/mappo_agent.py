"""
MAPPO (Multi-Agent Proximal Policy Optimization) implementation.
Centralized Training with Decentralized Execution (CTDE).

References:
- Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2021)
- https://arxiv.org/abs/2103.01955
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from .base_agent import BaseAgent


class ActorNetwork(nn.Module):
    """Decentralized actor - uses only local observations."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits


class CriticNetwork(nn.Module):
    """Centralized critic - uses global state information."""
    
    def __init__(self, total_state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class RolloutBuffer:
    """Buffer for storing trajectories during rollout."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.global_states = []
        
    def add(self, state, action, log_prob, reward, done, value, global_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.global_states.append(global_state)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.global_states.clear()
        
    def __len__(self):
        return len(self.states)


class MAPPOAgent(BaseAgent):
    """
    MAPPO Agent with Centralized Training and Decentralized Execution.
    
    - Training: Centralized critic uses global state
    - Execution: Decentralized actor uses only local observations
    """
    
    def __init__(self, agent_id, observation_space, action_space, 
                 total_obs_dim, config=None):
        """
        Initialize MAPPO agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            observation_space: The observation space
            action_space: The action space (discrete)
            total_obs_dim (int): Total observation dimension across all agents
            config (dict): Configuration dictionary
        """
        super().__init__(agent_id, observation_space, action_space, config)
        
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.n  # Assuming discrete action space
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.lr_actor = config.get('lr_actor', 3e-4)
        self.lr_critic = config.get('lr_critic', 1e-3)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # Networks
        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic = CriticNetwork(total_obs_dim, self.hidden_dim)  # Centralized
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
    def select_action(self, observation, training=True, global_state=None):
        """
        Select action using actor network (decentralized execution).
        
        Args:
            observation: Local observation
            training (bool): Whether in training mode
            global_state: Global state for critic (only needed during training)
            
        Returns:
            action: Selected action
            log_prob: Log probability of action (if training)
            value: Value estimate (if training)
        """
        state = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            action_logits = self.actor(state)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            if training:
                log_prob = dist.log_prob(action)
                
                # Get value estimate from centralized critic
                if global_state is not None:
                    global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
                    value = self.critic(global_state_tensor)
                else:
                    value = None
                
                return action.item(), log_prob.item(), value.item() if value is not None else 0.0
        
        return action.item()
    
    def store_transition(self, state, action, log_prob, reward, done, value, global_state):
        """Store transition in rollout buffer."""
        self.buffer.add(state, action, log_prob, reward, done, value, global_state)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
            
        Returns:
            advantages: Computed advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update(self, next_global_state=None):
        """
        Update agent using PPO algorithm.
        
        Args:
            next_global_state: Global state for computing next value
        """
        if len(self.buffer) == 0:
            return
        
        # Compute next value for GAE
        if next_global_state is not None:
            with torch.no_grad():
                next_value = self.critic(
                    torch.FloatTensor(next_global_state).unsqueeze(0)
                ).item()
        else:
            next_value = 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            next_value
        )
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states))
        actions = torch.LongTensor(self.buffer.actions)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs)
        global_states = torch.FloatTensor(np.array(self.buffer.global_states))
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        dataset_size = len(states)
        
        for _ in range(self.ppo_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > dataset_size:
                    continue
                    
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_global_states = global_states[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor loss
                action_logits = self.actor(batch_states)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy
                
                # Critic loss
                values = self.critic(batch_global_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        
        # Clear buffer after update
        self.buffer.clear()
    
    def save(self, filepath):
        """Save agent parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load agent parameters."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class MAPPOController:
    """Controller for coordinating MAPPO agents."""
    
    def __init__(self, agents):
        """
        Initialize MAPPO controller.
        
        Args:
            agents (dict): Dictionary of MAPPOAgent objects
        """
        self.agents = agents
        self.agent_names = list(agents.keys())
        
    def update_all_agents(self, next_observations):
        """
        Update all agents after collecting a trajectory.
        
        Args:
            next_observations (dict): Next observations for computing next values
        """
        # Compute global state from next observations
        next_global_state = np.concatenate([next_observations[name] for name in self.agent_names])
        
        # Update each agent
        for agent_name, agent in self.agents.items():
            agent.update(next_global_state)
