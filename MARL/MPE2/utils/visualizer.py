"""
Visualization utilities for training metrics and results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(metrics, save_path=None, window=100):
    """
    Plot training curves for various metrics.
    
    Args:
        metrics (dict): Dictionary of metrics to plot
        save_path (str): Path to save the plot
        window (int): Window size for moving average
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Define metrics to plot
    metric_configs = [
        ('total_reward', 'Total Reward per Episode', axes[0, 0]),
        ('episode_length', 'Episode Length', axes[0, 1]),
        ('adversary_reward', 'Adversary Reward', axes[1, 0]),
        ('agent_reward', 'Good Agent Reward', axes[1, 1])
    ]
    
    for metric_name, title, ax in metric_configs:
        if metric_name in metrics and len(metrics[metric_name]) > 0:
            data = metrics[metric_name]
            episodes = range(1, len(data) + 1)
            
            # Plot raw data
            ax.plot(episodes, data, alpha=0.3, label='Raw')
            
            # Plot moving average if enough data
            if len(data) >= window:
                moving_avg = moving_average(data, window)
                ax.plot(range(window, len(data) + 1), moving_avg, 
                       linewidth=2, label=f'{window}-episode MA')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_episode_rewards(episode_rewards, agent_names=None, save_path=None):
    """
    Plot rewards for different agents in an episode.
    
    Args:
        episode_rewards (dict): Dictionary mapping agent names to reward lists
        agent_names (list): List of agent names (optional)
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    if agent_names is None:
        agent_names = list(episode_rewards.keys())
    
    for agent_name in agent_names:
        if agent_name in episode_rewards:
            rewards = episode_rewards[agent_name]
            steps = range(len(rewards))
            plt.plot(steps, rewards, label=agent_name, marker='o', markersize=3)
    
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Episode Rewards by Agent')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Episode rewards plot saved to {save_path}")
    
    plt.show()


def moving_average(data, window_size):
    """
    Calculate moving average of data.
    
    Args:
        data (list): Input data
        window_size (int): Size of the moving window
        
    Returns:
        np.array: Moving average
    """
    if len(data) < window_size:
        return np.array(data)
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def save_video_frames(frames, output_path, fps=30):
    """
    Save frames as a video file.
    
    Args:
        frames (list): List of image frames
        output_path (str): Path to save the video
        fps (int): Frames per second
    """
    try:
        import imageio
        
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved to {output_path}")
    except ImportError:
        print("imageio not installed. Cannot save video.")
        print("Install with: pip install imageio[ffmpeg]")
