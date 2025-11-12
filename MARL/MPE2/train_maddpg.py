"""
Training script for MADDPG (Multi-Agent Deep Deterministic Policy Gradient).
Implements Centralized Training with Decentralized Execution (CTDE).
"""

import os
import sys
import numpy as np
from mpe2 import simple_tag_v3

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENV_CONFIG, LOGGING_CONFIG
from agents import MADDPGAgent, MADDPGController
from utils import Logger, create_directories, set_seed, get_timestamp


# MADDPG-specific configuration
MADDPG_CONFIG = {
    'gamma': 0.95,              # Discount factor
    'tau': 0.01,                # Soft update rate for target networks
    'lr_actor': 1e-4,           # Actor learning rate
    'lr_critic': 1e-3,          # Critic learning rate (usually higher than actor)
    'hidden_dim': 128,          # Hidden layer size
    'noise_scale': 0.2,         # Initial exploration noise
    'noise_decay': 0.9999,      # Noise decay rate per step
    'min_noise': 0.01,          # Minimum noise level
    'buffer_size': 100000,      # Replay buffer size
    'batch_size': 64,           # Mini-batch size
    'warmup_steps': 1000,       # Random actions before training starts
}

TRAINING_CONFIG = {
    'total_episodes': 10000,
    'max_steps_per_episode': 25,
    'update_frequency': 1,      # Update every N steps
    'seed': 42
}


def create_maddpg_agents(env, config):
    """
    Create MADDPG agents for all agents in the environment.
    
    Args:
        env: The environment
        config (dict): Configuration for agents
        
    Returns:
        dict: Dictionary mapping agent names to MADDPGAgent objects
        MADDPGController: Controller for coordinating agents
    """
    agent_names = env.possible_agents
    
    # Calculate total dimensions for centralized critic
    total_obs_dim = sum(env.observation_space(name).shape[0] for name in agent_names)
    total_action_dim = sum(env.action_space(name).n for name in agent_names)
    
    print(f"\n{'='*60}")
    print("MADDPG Setup")
    print(f"{'='*60}")
    print(f"Number of agents: {len(agent_names)}")
    print(f"Agent names: {agent_names}")
    print(f"Total observation dim: {total_obs_dim}")
    print(f"Total action dim: {total_action_dim}")
    print(f"{'='*60}\n")
    
    agents = {}
    for agent_name in agent_names:
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        
        agents[agent_name] = MADDPGAgent(
            agent_id=agent_name,
            observation_space=obs_space,
            action_space=act_space,
            n_agents=len(agent_names),
            total_obs_dim=total_obs_dim,
            total_action_dim=total_action_dim,
            config=config
        )
        
        print(f"Created MADDPG agent: {agent_name}")
        print(f"  - Observation space: {obs_space.shape}")
        print(f"  - Action space: {act_space.n}")
    
    # Create controller for centralized training
    controller = MADDPGController(
        agents=agents,
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size']
    )
    
    print(f"\nMADDPG Controller created")
    print(f"  - Replay buffer size: {config['buffer_size']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"{'='*60}\n")
    
    return agents, controller


def run_episode(env, agents, controller, training=True, warmup=False):
    """
    Run a single episode with MADDPG agents.
    
    Args:
        env: The environment
        agents (dict): Dictionary of MADDPG agents
        controller: MADDPG controller
        training (bool): Whether to train agents
        warmup (bool): Whether in warmup phase (random actions)
        
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
            
            if warmup:
                # Random actions during warmup
                action = agents[agent_name].action_space.sample()
            else:
                # Use policy with exploration noise
                action = agents[agent_name].select_action(obs, training=training)
            
            actions[agent_name] = action
        
        # Step environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Store experience in shared replay buffer
        if training:
            dones = {name: terminations.get(name, False) or truncations.get(name, False) 
                    for name in env.possible_agents}
            controller.store_experience(observations, actions, rewards, next_observations, dones)
            
            # Update agents if not in warmup
            if not warmup:
                controller.update_all_agents()
        
        # Track rewards
        for agent_name, reward in rewards.items():
            episode_rewards[agent_name] += reward
        
        observations = next_observations
        episode_length += 1
    
    # Calculate statistics
    total_reward = sum(episode_rewards.values())
    adversary_reward = sum(r for name, r in episode_rewards.items() if 'adversary' in name)
    good_agent_reward = sum(r for name, r in episode_rewards.items() 
                           if 'agent' in name and 'adversary' not in name)
    
    return {
        'total_reward': total_reward,
        'adversary_reward': adversary_reward,
        'agent_reward': good_agent_reward,
        'episode_length': episode_length,
        'individual_rewards': episode_rewards
    }


def train(num_episodes=None, experiment_name=None):
    """
    Main training loop for MADDPG.
    
    Args:
        num_episodes (int): Number of episodes to train
        experiment_name (str): Name for this experiment
    """
    # Setup
    num_episodes = num_episodes or TRAINING_CONFIG['total_episodes']
    warmup_steps = MADDPG_CONFIG['warmup_steps']
    seed = TRAINING_CONFIG['seed']
    set_seed(seed)
    
    # Create directories
    create_directories([
        LOGGING_CONFIG['log_dir'],
        LOGGING_CONFIG['checkpoint_dir']
    ])
    
    # Initialize logger
    if experiment_name is None:
        experiment_name = f"maddpg_{get_timestamp()}"
    
    logger = Logger(LOGGING_CONFIG['log_dir'], experiment_name)
    logger.log(f"Starting MADDPG training")
    logger.log(f"Environment config: {ENV_CONFIG}")
    logger.log(f"MADDPG config: {MADDPG_CONFIG}")
    logger.log(f"Training config: {TRAINING_CONFIG}")
    
    # Create environment
    env = simple_tag_v3.parallel_env(**ENV_CONFIG)
    env.reset(seed=seed)
    
    # Create MADDPG agents and controller
    agents, controller = create_maddpg_agents(env, MADDPG_CONFIG)
    
    # Warmup phase
    logger.log(f"\n{'='*60}")
    logger.log(f"Warmup phase: {warmup_steps} steps with random actions")
    logger.log(f"{'='*60}\n")
    
    total_steps = 0
    warmup_episodes = 0
    
    while total_steps < warmup_steps:
        stats = run_episode(env, agents, controller, training=True, warmup=True)
        total_steps += stats['episode_length']
        warmup_episodes += 1
        
        if warmup_episodes % 10 == 0:
            logger.log(f"Warmup: {warmup_episodes} episodes, {total_steps}/{warmup_steps} steps")
    
    logger.log(f"Warmup completed! Replay buffer size: {len(controller.replay_buffer)}")
    logger.log(f"{'='*60}\n")
    
    # Training loop
    logger.log("Starting main training loop...\n")
    
    for episode in range(1, num_episodes + 1):
        # Run episode
        stats = run_episode(env, agents, controller, training=True, warmup=False)
        
        # Log metrics
        logger.record('total_reward', stats['total_reward'])
        logger.record('adversary_reward', stats['adversary_reward'])
        logger.record('agent_reward', stats['agent_reward'])
        logger.record('episode_length', stats['episode_length'])
        
        # Log noise scale (exploration)
        avg_noise = np.mean([agent.noise_scale for agent in agents.values()])
        logger.record('exploration_noise', avg_noise)
        
        # Periodic logging
        if episode % LOGGING_CONFIG['log_frequency'] == 0:
            logger.log(
                f"Episode {episode}/{num_episodes} | "
                f"Total: {stats['total_reward']:.2f} | "
                f"Adversary: {stats['adversary_reward']:.2f} | "
                f"Good Agent: {stats['agent_reward']:.2f} | "
                f"Length: {stats['episode_length']} | "
                f"Noise: {avg_noise:.4f} | "
                f"Buffer: {len(controller.replay_buffer)}"
            )
        
        # Print summary
        if episode % 100 == 0:
            logger.print_summary(episode)
        
        # Save checkpoints
        if episode % LOGGING_CONFIG['save_frequency'] == 0:
            checkpoint_dir = os.path.join(
                LOGGING_CONFIG['checkpoint_dir'],
                f"{experiment_name}_{episode}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            for agent_name, agent in agents.items():
                checkpoint_path = os.path.join(checkpoint_dir, f"{agent_name}.pth")
                agent.save(checkpoint_path)
            
            logger.log(f"✓ Saved checkpoint at episode {episode}")
    
    # Save final metrics
    logger.save_metrics()
    logger.log("\n" + "="*60)
    logger.log("Training completed!")
    logger.log("="*60)
    
    # Save final checkpoint
    final_checkpoint_dir = os.path.join(
        LOGGING_CONFIG['checkpoint_dir'],
        f"{experiment_name}_final"
    )
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    for agent_name, agent in agents.items():
        checkpoint_path = os.path.join(final_checkpoint_dir, f"{agent_name}.pth")
        agent.save(checkpoint_path)
    
    logger.log(f"✓ Saved final checkpoint")
    
    env.close()
    
    return agents, controller, logger


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MADDPG agents in Simple Tag environment')
    parser.add_argument('--episodes', type=int, default=None, 
                       help='Number of episodes to train (default: 10000)')
    parser.add_argument('--experiment-name', type=str, default=None, 
                       help='Name for this experiment (default: maddpg_TIMESTAMP)')
    parser.add_argument('--warmup', type=int, default=1000,
                       help='Warmup steps with random actions (default: 1000)')
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.warmup:
        MADDPG_CONFIG['warmup_steps'] = args.warmup
    
    train(
        num_episodes=args.episodes,
        experiment_name=args.experiment_name
    )
