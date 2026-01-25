"""Utility functions and helpers.

This module provides:
    - paths: Centralized path management
    - attributes: iPhone attribute definitions
    - logging_setup: Logging configuration
"""

from .paths import (
    get_project_root,
    get_data_path,
    get_output_path,
    get_logs_path,
    get_plot_path,
    get_models_path,
)

__all__ = [
    "get_project_root",
    "get_data_path",
    "get_output_path",
    "get_logs_path",
    "get_plot_path",
    "get_models_path",
]
