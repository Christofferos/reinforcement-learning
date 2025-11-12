"""
Logger class for tracking training progress and metrics.
"""

import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime


class Logger:
    """Logger for tracking and saving training metrics."""
    
    def __init__(self, log_dir, experiment_name=None):
        """
        Initialize logger.
        
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
        # Log file
        self.log_file = os.path.join(self.experiment_dir, 'training.log')
        self.metrics_file = os.path.join(self.experiment_dir, 'metrics.json')
        
    def log(self, message, print_msg=True):
        """
        Log a message to file and optionally print.
        
        Args:
            message (str): Message to log
            print_msg (bool): Whether to print the message
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        if print_msg:
            print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def record(self, key, value):
        """
        Record a metric value.
        
        Args:
            key (str): Metric name
            value: Metric value
        """
        self.metrics[key].append(value)
    
    def record_episode(self, episode_num, metrics_dict):
        """
        Record metrics for an episode.
        
        Args:
            episode_num (int): Episode number
            metrics_dict (dict): Dictionary of metrics
        """
        metrics_dict['episode'] = episode_num
        metrics_dict['timestamp'] = datetime.now().isoformat()
        
        for key, value in metrics_dict.items():
            self.episode_metrics[key].append(value)
    
    def get_stats(self, key, window=100):
        """
        Get statistics for a metric.
        
        Args:
            key (str): Metric name
            window (int): Window size for recent statistics
            
        Returns:
            dict: Statistics dictionary
        """
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return None
        
        values = self.metrics[key]
        recent_values = values[-window:] if len(values) > window else values
        
        return {
            'mean': np.mean(recent_values),
            'std': np.std(recent_values),
            'min': np.min(recent_values),
            'max': np.max(recent_values),
            'count': len(values)
        }
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        # Convert defaultdict to regular dict for JSON serialization
        metrics_data = {
            'metrics': dict(self.metrics),
            'episode_metrics': dict(self.episode_metrics)
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)
    
    def load_metrics(self):
        """Load metrics from JSON file."""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                self.metrics = defaultdict(list, data.get('metrics', {}))
                self.episode_metrics = defaultdict(list, data.get('episode_metrics', {}))
    
    def print_summary(self, episode, window=100):
        """
        Print summary of recent training progress.
        
        Args:
            episode (int): Current episode number
            window (int): Window size for statistics
        """
        summary = f"\n{'='*60}\n"
        summary += f"Episode {episode} Summary (last {window} episodes)\n"
        summary += f"{'='*60}\n"
        
        for key in ['total_reward', 'episode_length', 'success_rate']:
            stats = self.get_stats(key, window)
            if stats:
                summary += f"{key:20s}: {stats['mean']:8.2f} ± {stats['std']:6.2f} "
                summary += f"[{stats['min']:6.2f}, {stats['max']:6.2f}]\n"
        
        summary += f"{'='*60}\n"
        self.log(summary)
