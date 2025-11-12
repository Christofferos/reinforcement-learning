"""
Evaluation script for trained agents in Simple Tag environment.
"""

import os
import sys
import numpy as np
from mpe2 import simple_tag_v3

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENV_CONFIG, EVAL_CONFIG, TRAINING_CONFIG
from agents import RandomAgent, DQNAgent
from utils import create_directories


def load_agents(env, checkpoint_dir, agent_type='dqn', config=None):
    """
    Load trained agents from checkpoint.
    
    Args:
        env: The environment
        checkpoint_dir (str): Directory containing agent checkpoints
        agent_type (str): Type of agents
        config (dict): Agent configuration
        
    Returns:
        dict: Dictionary of loaded agents
    """
    # Use TRAINING_CONFIG as default if no config provided
    if config is None:
        config = TRAINING_CONFIG
    
    agents = {}
    
    for agent_name in env.possible_agents:
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        
        if agent_type == 'random':
            agents[agent_name] = RandomAgent(agent_name, obs_space, act_space)
        elif agent_type == 'dqn':
            agent = DQNAgent(agent_name, obs_space, act_space, config)
            
            # Load checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"{agent_name}.pth")
            if os.path.exists(checkpoint_path):
                agent.load(checkpoint_path)
                print(f"Loaded checkpoint for {agent_name}")
            else:
                print(f"Warning: No checkpoint found for {agent_name} at {checkpoint_path}")
            
            agents[agent_name] = agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agents


def evaluate_episode(env, agents, render=False, record_frames=False):
    """
    Evaluate agents for one episode.
    
    Args:
        env: The environment
        agents (dict): Dictionary of agents
        render (bool): Whether to render the environment
        record_frames (bool): Whether to record frames for video
        
    Returns:
        dict: Episode statistics
        list: Frames (if record_frames=True)
    """
    observations, infos = env.reset()
    episode_rewards = {agent_name: 0 for agent_name in env.possible_agents}
    episode_length = 0
    frames = [] if record_frames else None
    
    while env.agents:
        actions = {}
        
        # Get actions from all agents (evaluation mode)
        for agent_name in env.agents:
            obs = observations[agent_name]
            action = agents[agent_name].select_action(obs, training=False)
            actions[agent_name] = action
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Track rewards
        for agent_name, reward in rewards.items():
            episode_rewards[agent_name] += reward
        
        # Render or record frame
        if record_frames:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except:
                pass
        elif render:
            try:
                env.render()
                # Add small delay for human viewing
                import time
                time.sleep(0.05)
            except:
                pass
        
        episode_length += 1
    
    # Calculate statistics
    total_reward = sum(episode_rewards.values())
    adversary_reward = sum(r for name, r in episode_rewards.items() if 'adversary' in name)
    good_agent_reward = sum(r for name, r in episode_rewards.items() if 'agent' in name and 'adversary' not in name)
    
    stats = {
        'total_reward': total_reward,
        'adversary_reward': adversary_reward,
        'agent_reward': good_agent_reward,
        'episode_length': episode_length,
        'individual_rewards': episode_rewards
    }
    
    return stats, frames


def evaluate(checkpoint_dir, num_episodes=None, agent_type='dqn', render=False, save_video=False):
    """
    Evaluate trained agents.
    
    Args:
        checkpoint_dir (str): Directory containing agent checkpoints
        num_episodes (int): Number of episodes to evaluate
        agent_type (str): Type of agents
        render (bool): Whether to render episodes
        save_video (bool): Whether to save video of episodes
    """
    num_episodes = num_episodes or EVAL_CONFIG['num_episodes']
    
    # Setup environment
    env_config = ENV_CONFIG.copy()
    if save_video:
        env_config['render_mode'] = 'rgb_array'
    elif render:
        env_config['render_mode'] = 'human'
    
    env = simple_tag_v3.parallel_env(**env_config)
    env.reset()
    
    # Load agents
    agents = load_agents(env, checkpoint_dir, agent_type=agent_type)
    print(f"Loaded {len(agents)} agents from {checkpoint_dir}")
    
    # Evaluation loop
    all_stats = []
    all_frames = []
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        stats, frames = evaluate_episode(
            env, agents, 
            render=render, 
            record_frames=save_video
        )
        
        all_stats.append(stats)
        if save_video and frames:
            all_frames.extend(frames)
        
        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Total: {stats['total_reward']:.2f} | "
                f"Adversary: {stats['adversary_reward']:.2f} | "
                f"Good Agent: {stats['agent_reward']:.2f}"
            )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    metrics = ['total_reward', 'adversary_reward', 'agent_reward', 'episode_length']
    for metric in metrics:
        values = [s[metric] for s in all_stats]
        print(f"{metric:20s}: {np.mean(values):8.2f} ± {np.std(values):6.2f} "
              f"[{np.min(values):6.2f}, {np.max(values):6.2f}]")
    
    print("="*60)
    
    # Save video if requested
    if save_video and all_frames:
        try:
            from utils.visualizer import save_video_frames
            
            create_directories([EVAL_CONFIG['video_dir']])
            video_path = os.path.join(
                EVAL_CONFIG['video_dir'],
                f"evaluation_{agent_type}.mp4"
            )
            save_video_frames(all_frames, video_path)
        except Exception as e:
            print(f"Error saving video: {e}")
    
    env.close()
    
    return all_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained agents in Simple Tag')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing agent checkpoints')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to evaluate')
    parser.add_argument('--agent-type', type=str, default='dqn', 
                       choices=['random', 'dqn'],
                       help='Type of agent')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes during evaluation')
    parser.add_argument('--save-video', action='store_true',
                       help='Save video of evaluation episodes')
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_dir=args.checkpoint_dir,
        num_episodes=args.episodes,
        agent_type=args.agent_type,
        render=args.render,
        save_video=args.save_video
    )
