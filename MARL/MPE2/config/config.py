"""
Configuration file for Simple Tag environment and training parameters.
"""

# Environment Configuration
ENV_CONFIG = {
    'num_good': 1,
    'num_adversaries': 3,
    'num_obstacles': 2,
    'max_cycles': 25,
    'continuous_actions': False,
    'dynamic_rescaling': False,
    'render_mode': None  # 'human', 'rgb_array', or None
}

# Training Configuration
TRAINING_CONFIG = {
    'total_episodes': 10000,
    'max_steps_per_episode': 25,
    'batch_size': 64,
    'buffer_size': 100000,
    'learning_rate': 1e-4,
    'gamma': 0.99,  # Discount factor
    'tau': 0.005,  # Soft update parameter
    'update_every': 4,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'seed': 42
}

# Agent-specific Configuration
ADVERSARY_CONFIG = {
    'reward_collision': 10.0,  # Reward for hitting good agents
    'speed': 1.0,  # Slower than good agents
}

GOOD_AGENT_CONFIG = {
    'reward_collision': -10.0,  # Penalty for being hit
    'penalty_boundary': True,  # Penalize for leaving area
    'speed': 1.3,  # Faster than adversaries
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints',
    'save_frequency': 100,  # Save every N episodes
    'log_frequency': 10,  # Log every N episodes
}

# Evaluation Configuration
EVAL_CONFIG = {
    'num_episodes': 100,
    'render': True,
    'save_video': False,
    'video_dir': 'videos'
}
