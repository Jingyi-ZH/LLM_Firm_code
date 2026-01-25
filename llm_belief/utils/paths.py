"""Centralized path management to eliminate hardcoded paths.

This module provides functions to get absolute paths for various project
directories, using the configuration from config/config.yaml.

Example:
    from llm_belief.utils.paths import get_data_path, get_output_path

    # Get path to a data file
    profiles_path = get_data_path("profiles_shuffled.csv")

    # Get path to output directory
    output_dir = get_output_path()
"""

import sys
from pathlib import Path
from functools import lru_cache

# Add project root to path for config import
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import get_config


@lru_cache()
def get_project_root() -> Path:
    """Get project root directory.

    Returns:
        Path to project root directory.
    """
    return get_config().root


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path(filename: str = "") -> Path:
    """Get path within data directory.

    Args:
        filename: Optional filename to append.

    Returns:
        Absolute path to data directory or file.
    """
    base_path = get_config().get_path("data_dir")
    return base_path / filename if filename else base_path


def get_data_raw_path(filename: str = "") -> Path:
    """Get path within raw data directory."""
    base_path = get_config().get_path("data_raw_dir")
    return base_path / filename if filename else base_path


def get_data_interim_path(filename: str = "") -> Path:
    """Get path within interim data directory."""
    base_path = get_config().get_path("data_interim_dir")
    return base_path / filename if filename else base_path


def get_data_processed_path(filename: str = "") -> Path:
    """Get path within processed data directory."""
    base_path = get_config().get_path("data_processed_dir")
    return base_path / filename if filename else base_path


def get_output_path(filename: str = "") -> Path:
    """Get path within output directory.

    Args:
        filename: Optional filename to append.

    Returns:
        Absolute path to output directory or file.
    """
    base_path = get_config().get_path("output_dir")
    _ensure_dir(base_path)
    return base_path / filename if filename else base_path


def get_output_figures_path(filename: str = "") -> Path:
    """Get path within output figures directory."""
    base_path = get_config().get_path("output_figures_dir")
    _ensure_dir(base_path)
    return base_path / filename if filename else base_path


def get_output_tables_path(filename: str = "") -> Path:
    """Get path within output tables directory."""
    base_path = get_config().get_path("output_tables_dir")
    _ensure_dir(base_path)
    return base_path / filename if filename else base_path


def get_output_models_path(filename: str = "") -> Path:
    """Get path within output models directory."""
    base_path = get_config().get_path("output_models_dir")
    _ensure_dir(base_path)
    return base_path / filename if filename else base_path


def get_logs_path(filename: str = "") -> Path:
    """Get path within logs directory.

    Args:
        filename: Optional filename to append.

    Returns:
        Absolute path to logs directory or file.
    """
    base_path = get_config().get_path("logs_dir")
    _ensure_dir(base_path)
    return base_path / filename if filename else base_path


def get_plot_path(filename: str = "") -> Path:
    """Get path within plot directory.

    Args:
        filename: Optional filename to append.

    Returns:
        Absolute path to plot directory or file.
    """
    base_path = get_config().get_path("plot_dir")
    _ensure_dir(base_path)
    return base_path / filename if filename else base_path


def get_models_path(filename: str = "") -> Path:
    """Get path within models directory.

    Args:
        filename: Optional filename to append.

    Returns:
        Absolute path to models directory or file.
    """
    base_path = get_config().get_path("models_dir")
    _ensure_dir(base_path)
    return base_path / filename if filename else base_path
