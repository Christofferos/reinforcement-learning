"""
Deep Q-Network (DQN) agent implementation for Simple Tag environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from .base_agent import BaseAgent


class QNetwork(nn.Module):
    """Q-Network for DQN agent."""
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128]):
        """
        Initialize Q-Network.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            hidden_sizes (list): Sizes of hidden layers
        """
        super(QNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size, seed=42):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            seed (int): Random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to memory."""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in experiences]))
        actions = torch.LongTensor(np.array([e[1] for e in experiences]))
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences]))
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences]))
        dones = torch.FloatTensor(np.array([e[4] for e in experiences]))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.memory)


class DQNAgent(BaseAgent):
    """DQN agent for discrete action spaces."""
    
    def __init__(self, agent_id, observation_space, action_space, config=None):
        """
        Initialize DQN agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            observation_space: The observation space
            action_space: The action space (must be discrete)
            config (dict): Configuration dictionary
        """
        super().__init__(agent_id, observation_space, action_space, config)
        
        # Get dimensions
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.lr = config.get('learning_rate', 1e-4)
        self.update_every = config.get('update_every', 4)
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 100000)
        
        # Epsilon-greedy parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Q-Networks
        self.qnetwork_local = QNetwork(self.state_size, self.action_size)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Step counter
        self.t_step = 0
        
    def select_action(self, observation, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            training (bool): Whether in training mode
            
        Returns:
            action: Selected action
        """
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return self.action_space.sample()
        
        # Greedy action selection
        state = torch.FloatTensor(observation).unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        return action_values.argmax().item()
    
    def update(self, experience):
        """
        Update Q-network using experience.
        
        Args:
            experience (tuple): (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = experience
        
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Update every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Learn if enough samples in buffer
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self._learn(experiences)
    
    def _learn(self, experiences):
        """
        Update Q-network parameters using batch of experiences.
        
        Args:
            experiences (tuple): Batch of (s, a, r, s', done)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(1) + (self.gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._soft_update()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _soft_update(self):
        """Soft update target network parameters."""
        for target_param, local_param in zip(self.qnetwork_target.parameters(), 
                                             self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Save agent parameters."""
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
    
    def load(self, filepath):
        """Load agent parameters."""
        checkpoint = torch.load(filepath)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
