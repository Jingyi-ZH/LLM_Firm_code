"""Logging configuration for the project.

This module provides standardized logging setup for all scripts and modules.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from .paths import get_logs_path


def setup_logging(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Set up logging with file and console handlers.

    Args:
        name: Logger name (usually script/module name)
        log_file: Optional log filename. If not provided, uses name.log
        level: Logging level (default: INFO)
        console: Whether to also log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Formatter
    formatter = logging.Formatter("%(message)s")

    # File handler
    if log_file is None:
        log_file = f"{name}.log"
    log_path = get_logs_path(log_file)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Suppress noisy loggers
    for noisy_logger in ["httpx", "urllib3", "openai"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Log start time
    logger.info(f"Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return logger


def get_experiment_logger(
    experiment_type: str,
    identifier: str,
) -> logging.Logger:
    """Get a logger for a specific experiment.

    Args:
        experiment_type: Type of experiment (e.g., 'basic', 'fixreal', 'top')
        identifier: Unique identifier (e.g., 'iPhone 16 Pro', '0_1000')

    Returns:
        Configured logger
    """
    # Sanitize identifier for filename
    safe_id = identifier.replace(" ", "_").replace("/", "_")
    log_file = f"{experiment_type}_{safe_id}.log"
    logger_name = f"{experiment_type}_{safe_id}"

    return setup_logging(logger_name, log_file)
