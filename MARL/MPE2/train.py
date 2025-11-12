"""
Main training script for Simple Tag environment.
"""

import os
import sys
import numpy as np
from mpe2 import simple_tag_v3

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENV_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG
from agents import RandomAgent, DQNAgent
from utils import Logger, create_directories, set_seed, get_timestamp


def create_agents(env, agent_type='random', config=None):
    """
    Create agents for all agents in the environment.
    
    Args:
        env: The environment
        agent_type (str): Type of agent ('random', 'dqn')
        config (dict): Configuration for agents
        
    Returns:
        dict: Dictionary mapping agent names to agent objects
    """
    agents = {}
    
    for agent_name in env.possible_agents:
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        
        if agent_type == 'random':
            agents[agent_name] = RandomAgent(agent_name, obs_space, act_space)
        elif agent_type == 'dqn':
            agents[agent_name] = DQNAgent(agent_name, obs_space, act_space, config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agents


def run_episode(env, agents, training=True):
    """
    Run a single episode.
    
    Args:
        env: The environment
        agents (dict): Dictionary of agents
        training (bool): Whether agents should learn
        
    Returns:
        dict: Episode statistics
    """
    observations, infos = env.reset()
    episode_rewards = {agent_name: 0 for agent_name in env.possible_agents}
    episode_length = 0
    
    while env.agents:
        actions = {}
        
        # Get actions from all agents
        for agent_name in env.agents:
            obs = observations[agent_name]
            action = agents[agent_name].select_action(obs, training=training)
            actions[agent_name] = action
        
        # Step environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Update agents if training
        if training:
            for agent_name in env.agents:
                if agent_name in observations and agent_name in next_observations:
                    obs = observations[agent_name]
                    action = actions[agent_name]
                    reward = rewards[agent_name]
                    next_obs = next_observations[agent_name]
                    done = terminations[agent_name] or truncations[agent_name]
                    
                    agents[agent_name].update((obs, action, reward, next_obs, done))
        
        # Track rewards
        for agent_name, reward in rewards.items():
            episode_rewards[agent_name] += reward
        
        observations = next_observations
        episode_length += 1
    
    # Calculate statistics
    total_reward = sum(episode_rewards.values())
    adversary_reward = sum(r for name, r in episode_rewards.items() if 'adversary' in name)
    good_agent_reward = sum(r for name, r in episode_rewards.items() if 'agent' in name and 'adversary' not in name)
    
    return {
        'total_reward': total_reward,
        'adversary_reward': adversary_reward,
        'agent_reward': good_agent_reward,
        'episode_length': episode_length,
        'individual_rewards': episode_rewards
    }


def train(num_episodes=None, agent_type='dqn', experiment_name=None):
    """
    Main training loop.
    
    Args:
        num_episodes (int): Number of episodes to train
        agent_type (str): Type of agents to use
        experiment_name (str): Name for this experiment
    """
    # Setup
    num_episodes = num_episodes or TRAINING_CONFIG['total_episodes']
    seed = TRAINING_CONFIG['seed']
    set_seed(seed)
    
    # Create directories
    create_directories([
        LOGGING_CONFIG['log_dir'],
        LOGGING_CONFIG['checkpoint_dir']
    ])
    
    # Initialize logger
    logger = Logger(LOGGING_CONFIG['log_dir'], experiment_name)
    logger.log(f"Starting training with {agent_type} agents")
    logger.log(f"Environment config: {ENV_CONFIG}")
    logger.log(f"Training config: {TRAINING_CONFIG}")
    
    # Create environment
    env = simple_tag_v3.parallel_env(**ENV_CONFIG)
    env.reset(seed=seed)
    
    # Create agents
    agents = create_agents(env, agent_type=agent_type, config=TRAINING_CONFIG)
    logger.log(f"Created {len(agents)} agents: {list(agents.keys())}")
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        # Run episode
        stats = run_episode(env, agents, training=True)
        
        # Log metrics
        logger.record('total_reward', stats['total_reward'])
        logger.record('adversary_reward', stats['adversary_reward'])
        logger.record('agent_reward', stats['agent_reward'])
        logger.record('episode_length', stats['episode_length'])
        
        # Periodic logging
        if episode % LOGGING_CONFIG['log_frequency'] == 0:
            logger.log(
                f"Episode {episode}/{num_episodes} | "
                f"Total Reward: {stats['total_reward']:.2f} | "
                f"Adversary: {stats['adversary_reward']:.2f} | "
                f"Good Agent: {stats['agent_reward']:.2f} | "
                f"Length: {stats['episode_length']}"
            )
        
        # Print summary
        if episode % 100 == 0:
            logger.print_summary(episode)
        
        # Save checkpoints
        if episode % LOGGING_CONFIG['save_frequency'] == 0:
            checkpoint_dir = os.path.join(
                LOGGING_CONFIG['checkpoint_dir'],
                f"{experiment_name or 'default'}_{episode}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            for agent_name, agent in agents.items():
                checkpoint_path = os.path.join(checkpoint_dir, f"{agent_name}.pth")
                agent.save(checkpoint_path)
            
            logger.log(f"Saved checkpoint at episode {episode}")
    
    # Save final metrics
    logger.save_metrics()
    logger.log("Training completed!")
    
    env.close()
    
    return agents, logger


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train agents in Simple Tag environment')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes to train')
    parser.add_argument('--agent-type', type=str, default='dqn', choices=['random', 'dqn'], 
                       help='Type of agent to use')
    parser.add_argument('--experiment-name', type=str, default=None, 
                       help='Name for this experiment')
    
    args = parser.parse_args()
    
    if args.experiment_name is None:
        args.experiment_name = f"{args.agent_type}_{get_timestamp()}"
    
    train(
        num_episodes=args.episodes,
        agent_type=args.agent_type,
        experiment_name=args.experiment_name
    )
