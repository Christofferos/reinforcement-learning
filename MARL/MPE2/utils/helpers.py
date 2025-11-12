"""
Utility functions for Simple Tag environment.
"""

import numpy as np
import os
import json
from datetime import datetime


def create_directories(dirs):
    """
    Create directories if they don't exist.
    
    Args:
        dirs (list): List of directory paths to create
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def save_config(config, filepath):
    """
    Save configuration to JSON file.
    
    Args:
        config (dict): Configuration dictionary
        filepath (str): Path to save the config
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(filepath):
    """
    Load configuration from JSON file.
    
    Args:
        filepath (str): Path to the config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp():
    """
    Get current timestamp string.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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
