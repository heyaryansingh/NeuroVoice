"""
Logging utilities for NeuroVoice project.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    name: str = "NeuroVoice",
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        name: Logger name
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

