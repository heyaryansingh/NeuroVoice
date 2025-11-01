"""
Helper utility functions.
"""

import json
from pathlib import Path
from typing import Dict, Any


def ensure_dir(directory: Path):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
    """
    directory.mkdir(parents=True, exist_ok=True)


def save_config(config: Dict[str, Any], config_path: Path):
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

