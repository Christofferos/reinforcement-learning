"""
Evaluation script for MADDPG agents in Simple Tag environment.
"""

import os
import sys
import numpy as np
from mpe2 import simple_tag_v3

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENV_CONFIG, EVAL_CONFIG
from agents import MADDPGAgent, MADDPGController
from utils import create_directories


# MADDPG config for evaluation (should match training)
MADDPG_CONFIG = {
    'gamma': 0.95,
    'tau': 0.01,
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'hidden_dim': 128,
    'noise_scale': 0.0,  # No exploration noise during evaluation
    'noise_decay': 1.0,
    'min_noise': 0.0,
}


def load_maddpg_agents(env, checkpoint_dir, config=None):
    """
    Load trained MADDPG agents from checkpoint.
    
    Args:
        env: The environment
        checkpoint_dir (str): Directory containing agent checkpoints
        config (dict): Agent configuration
        
    Returns:
        dict: Dictionary of loaded agents
        MADDPGController: Controller (for consistency, though not needed for eval)
    """
    if config is None:
        config = MADDPG_CONFIG
    
    agent_names = env.possible_agents
    
    # Calculate total dimensions
    total_obs_dim = sum(env.observation_space(name).shape[0] for name in agent_names)
    total_action_dim = sum(env.action_space(name).n for name in agent_names)
    
    agents = {}
    for agent_name in agent_names:
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        
        agent = MADDPGAgent(
            agent_id=agent_name,
            observation_space=obs_space,
            action_space=act_space,
            n_agents=len(agent_names),
            total_obs_dim=total_obs_dim,
            total_action_dim=total_action_dim,
            config=config
        )
        
        # Load checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{agent_name}.pth")
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            print(f"✓ Loaded checkpoint for {agent_name}")
        else:
            print(f"⚠ Warning: No checkpoint found for {agent_name} at {checkpoint_path}")
        
        agents[agent_name] = agent
    
    # Create controller (not used for evaluation but kept for consistency)
    controller = MADDPGController(agents, buffer_size=1000, batch_size=64)
    
    return agents, controller


def evaluate_episode(env, agents, render=False, record_frames=False):
    """
    Evaluate MADDPG agents for one episode.
    
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
        
        # Get actions from all agents (no exploration noise)
        for agent_name in env.agents:
            obs = observations[agent_name]
            action = agents[agent_name].select_action(obs, training=False, add_noise=False)
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
    good_agent_reward = sum(r for name, r in episode_rewards.items() 
                           if 'agent' in name and 'adversary' not in name)
    
    stats = {
        'total_reward': total_reward,
        'adversary_reward': adversary_reward,
        'agent_reward': good_agent_reward,
        'episode_length': episode_length,
        'individual_rewards': episode_rewards
    }
    
    return stats, frames


def evaluate(checkpoint_dir, num_episodes=None, render=False, save_video=False):
    """
    Evaluate trained MADDPG agents.
    
    Args:
        checkpoint_dir (str): Directory containing agent checkpoints
        num_episodes (int): Number of episodes to evaluate
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
    print(f"\nLoading MADDPG agents from {checkpoint_dir}...")
    agents, controller = load_maddpg_agents(env, checkpoint_dir)
    print(f"✓ Loaded {len(agents)} MADDPG agents\n")
    
    # Evaluation loop
    all_stats = []
    all_frames = []
    
    print(f"{'='*60}")
    print(f"Evaluating for {num_episodes} episodes...")
    print(f"{'='*60}\n")
    
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
                f"Episode {episode:3d}/{num_episodes} | "
                f"Total: {stats['total_reward']:7.2f} | "
                f"Adversary: {stats['adversary_reward']:7.2f} | "
                f"Good Agent: {stats['agent_reward']:7.2f} | "
                f"Length: {stats['episode_length']:2d}"
            )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    metrics = ['total_reward', 'adversary_reward', 'agent_reward', 'episode_length']
    for metric in metrics:
        values = [s[metric] for s in all_stats]
        print(f"{metric:20s}: {np.mean(values):8.2f} ± {np.std(values):6.2f} "
              f"[{np.min(values):7.2f}, {np.max(values):7.2f}]")
    
    print("="*60)
    
    # Print individual agent statistics
    print("\nIndividual Agent Performance:")
    print("-"*60)
    agent_names = list(all_stats[0]['individual_rewards'].keys())
    for agent_name in agent_names:
        agent_rewards = [s['individual_rewards'][agent_name] for s in all_stats]
        print(f"{agent_name:20s}: {np.mean(agent_rewards):8.2f} ± {np.std(agent_rewards):6.2f}")
    print("="*60 + "\n")
    
    # Save video if requested
    if save_video and all_frames:
        try:
            from utils.visualizer import save_video_frames
            
            create_directories([EVAL_CONFIG['video_dir']])
            video_path = os.path.join(
                EVAL_CONFIG['video_dir'],
                f"evaluation_maddpg_{os.path.basename(checkpoint_dir)}.mp4"
            )
            save_video_frames(all_frames, video_path)
        except Exception as e:
            print(f"Error saving video: {e}")
    
    env.close()
    
    return all_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained MADDPG agents in Simple Tag')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing agent checkpoints')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes during evaluation')
    parser.add_argument('--save-video', action='store_true',
                       help='Save video of evaluation episodes')
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_dir=args.checkpoint_dir,
        num_episodes=args.episodes,
        render=args.render,
        save_video=args.save_video
    )
