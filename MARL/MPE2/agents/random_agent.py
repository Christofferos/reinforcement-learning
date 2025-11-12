"""
Random agent implementation for baseline comparison.
"""

import numpy as np
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects random actions."""
    
    def __init__(self, agent_id, observation_space, action_space, config=None):
        """Initialize random agent."""
        super().__init__(agent_id, observation_space, action_space, config)
        
    def select_action(self, observation, training=True):
        """
        Select a random action from the action space.
        
        Args:
            observation: Current observation (unused for random agent)
            training (bool): Training mode flag (unused for random agent)
            
        Returns:
            action: A randomly sampled action
        """
        return self.action_space.sample()
    
    def update(self, experience):
        """
        Random agent doesn't learn, so update does nothing.
        
        Args:
            experience (tuple): (state, action, reward, next_state, done)
        """
        pass
