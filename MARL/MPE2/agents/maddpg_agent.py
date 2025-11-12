"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) implementation.
Centralized Training with Decentralized Execution (CTDE).

References:
- Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (2017)
- https://arxiv.org/abs/1706.02275
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from .base_agent import BaseAgent


class Actor(nn.Module):
    """Actor network - decentralized, only uses local observations."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Tanh for continuous actions in [-1, 1], or use for discrete with Gumbel-Softmax
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic network - centralized, uses all agents' observations and actions."""
    
    def __init__(self, total_state_dim, total_action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MADDPGReplayBuffer:
    """Replay buffer for multi-agent experience."""
    
    def __init__(self, buffer_size, batch_size, n_agents):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.memory = deque(maxlen=buffer_size)
        
    def add(self, states, actions, rewards, next_states, dones):
        """
        Add experience to buffer.
        
        Args:
            states (dict): Dictionary of states for each agent
            actions (dict): Dictionary of actions for each agent
            rewards (dict): Dictionary of rewards for each agent
            next_states (dict): Dictionary of next states for each agent
            dones (dict): Dictionary of done flags for each agent
        """
        self.memory.append((states, actions, rewards, next_states, dones))
        
    def sample(self):
        """Sample a batch of experiences."""
        batch = random.sample(self.memory, k=self.batch_size)
        return batch
    
    def __len__(self):
        return len(self.memory)


class MADDPGAgent(BaseAgent):
    """
    MADDPG Agent with Centralized Training and Decentralized Execution.
    
    - Training: Critic uses global state (all agents' observations)
    - Execution: Actor uses only local observations
    """
    
    def __init__(self, agent_id, observation_space, action_space, 
                 n_agents, total_obs_dim, total_action_dim, config=None):
        """
        Initialize MADDPG agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            observation_space: The observation space
            action_space: The action space
            n_agents (int): Total number of agents
            total_obs_dim (int): Total observation dimension across all agents
            total_action_dim (int): Total action dimension across all agents
            config (dict): Configuration dictionary
        """
        super().__init__(agent_id, observation_space, action_space, config)
        
        self.n_agents = n_agents
        self.state_dim = observation_space.shape[0]
        
        # Handle discrete vs continuous action spaces
        if hasattr(action_space, 'n'):  # Discrete
            self.action_dim = action_space.n
            self.discrete = True
        else:  # Continuous
            self.action_dim = action_space.shape[0]
            self.discrete = False
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.95)
        self.tau = config.get('tau', 0.01)
        self.lr_actor = config.get('lr_actor', 1e-4)
        self.lr_critic = config.get('lr_critic', 1e-3)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # Networks
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Centralized critic
        self.critic = Critic(total_obs_dim, total_action_dim, self.hidden_dim)
        self.critic_target = Critic(total_obs_dim, total_action_dim, self.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Exploration noise
        self.noise_scale = config.get('noise_scale', 0.1)
        self.noise_decay = config.get('noise_decay', 0.9999)
        self.min_noise = config.get('min_noise', 0.01)
        
    def select_action(self, observation, training=True, add_noise=True):
        """
        Select action using actor network (decentralized execution).
        
        Args:
            observation: Local observation
            training (bool): Whether in training mode
            add_noise (bool): Whether to add exploration noise
            
        Returns:
            action: Selected action
        """
        state = torch.FloatTensor(observation).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        self.actor.train()
        
        # Add exploration noise during training
        if training and add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = action + noise
            action = np.clip(action, -1, 1)
        
        # Convert to discrete action if needed
        if self.discrete:
            # Use Gumbel-Softmax or argmax for discrete actions
            action_probs = self._to_action_probs(action)
            if training and add_noise:
                action = np.random.choice(self.action_dim, p=action_probs)
            else:
                action = np.argmax(action_probs)
        
        return action
    
    def _to_action_probs(self, action_values):
        """Convert continuous action values to discrete probabilities."""
        exp_values = np.exp(action_values - np.max(action_values))
        return exp_values / exp_values.sum()
    
    def update(self, experience, all_agents):
        """
        Update is handled by the MADDPG controller, not individual agents.
        See MADDPGController class below.
        """
        pass
    
    def soft_update(self, target, source):
        """Soft update of target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def decay_noise(self):
        """Decay exploration noise."""
        self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
    
    def save(self, filepath):
        """Save agent parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'noise_scale': self.noise_scale,
        }, filepath)
    
    def load(self, filepath):
        """Load agent parameters."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.noise_scale = checkpoint['noise_scale']


class MADDPGController:
    """
    Controller for coordinating MADDPG agents.
    Handles centralized training with shared replay buffer.
    """
    
    def __init__(self, agents, buffer_size=100000, batch_size=64):
        """
        Initialize MADDPG controller.
        
        Args:
            agents (dict): Dictionary of MADDPGAgent objects
            buffer_size (int): Size of replay buffer
            batch_size (int): Batch size for training
        """
        self.agents = agents
        self.agent_names = list(agents.keys())
        self.n_agents = len(agents)
        self.batch_size = batch_size
        
        # Shared replay buffer
        self.replay_buffer = MADDPGReplayBuffer(buffer_size, batch_size, self.n_agents)
        
    def store_experience(self, states, actions, rewards, next_states, dones):
        """Store experience in shared replay buffer."""
        self.replay_buffer.add(states, actions, rewards, next_states, dones)
    
    def update_all_agents(self):
        """Update all agents using centralized training."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample()
        
        for agent_name, agent in self.agents.items():
            self._update_agent(agent_name, agent, batch)
            agent.soft_update(agent.actor_target, agent.actor)
            agent.soft_update(agent.critic_target, agent.critic)
            agent.decay_noise()
    
    def _update_agent(self, agent_name, agent, batch):
        """Update a single agent using MADDPG algorithm."""
        # Prepare batch data
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for states, actions, rewards, next_states, dones in batch:
            # Concatenate all agents' observations for centralized critic
            all_states = np.concatenate([states[name] for name in self.agent_names])
            all_next_states = np.concatenate([next_states[name] for name in self.agent_names])
            
            # Concatenate all agents' actions
            all_actions = np.concatenate([
                self._to_action_vector(actions[name], self.agents[name]) 
                for name in self.agent_names
            ])
            
            states_batch.append(all_states)
            next_states_batch.append(all_next_states)
            actions_batch.append(all_actions)
            rewards_batch.append(rewards[agent_name])
            dones_batch.append(dones[agent_name])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states_batch))
        actions_tensor = torch.FloatTensor(np.array(actions_batch))
        rewards_tensor = torch.FloatTensor(rewards_batch).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(np.array(next_states_batch))
        dones_tensor = torch.FloatTensor(dones_batch).unsqueeze(1)
        
        # Get next actions from target actors
        next_actions = []
        for name in self.agent_names:
            next_obs = torch.FloatTensor(np.array([next_states[name] for _, _, _, next_states, _ in batch]))
            with torch.no_grad():
                next_action = self.agents[name].actor_target(next_obs)
                next_actions.append(next_action)
        next_actions_tensor = torch.cat(next_actions, dim=1)
        
        # Update Critic
        target_q = agent.critic_target(next_states_tensor, next_actions_tensor)
        target_value = rewards_tensor + agent.gamma * target_q * (1 - dones_tensor)
        
        current_q = agent.critic(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(current_q, target_value.detach())
        
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()
        
        # Update Actor
        # Get current actions from all agents
        current_actions = []
        for i, name in enumerate(self.agent_names):
            obs = torch.FloatTensor(np.array([states[name] for states, _, _, _, _ in batch]))
            if name == agent_name:
                action = agent.actor(obs)
            else:
                with torch.no_grad():
                    action = self.agents[name].actor(obs)
            current_actions.append(action)
        current_actions_tensor = torch.cat(current_actions, dim=1)
        
        actor_loss = -agent.critic(states_tensor, current_actions_tensor).mean()
        
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
    
    def _to_action_vector(self, action, agent):
        """Convert action to vector format for critic."""
        if agent.discrete:
            # One-hot encoding for discrete actions
            action_vec = np.zeros(agent.action_dim)
            action_vec[action] = 1.0
            return action_vec
        else:
            return np.array(action)
