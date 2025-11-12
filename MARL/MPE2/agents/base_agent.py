"""
Base agent class for Simple Tag environment.
All agent implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, agent_id, observation_space, action_space, config=None):
        """
        Initialize the base agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            observation_space: The observation space of the environment
            action_space: The action space of the environment
            config (dict): Configuration dictionary for the agent
        """
        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or {}
        
    @abstractmethod
    def select_action(self, observation, training=True):
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current observation from the environment
            training (bool): Whether the agent is in training mode
            
        Returns:
            action: The selected action
        """
        pass
    
    @abstractmethod
    def update(self, experience):
        """
        Update the agent's policy based on experience.
        
        Args:
            experience (tuple): (state, action, reward, next_state, done)
        """
        pass
    
    def save(self, filepath):
        """Save agent parameters to file."""
        pass
    
    def load(self, filepath):
        """Load agent parameters from file."""
        pass
    
    def reset(self):
        """Reset agent's internal state if needed."""
        pass
