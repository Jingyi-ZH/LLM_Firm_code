"""Configuration management module for LLM Belief Elicitation project."""

from pathlib import Path
import yaml
import os
from functools import lru_cache
from dotenv import load_dotenv


class Config:
    """Central configuration class for the project.

    This class handles loading and accessing configuration from config.yaml,
    as well as managing environment variables for API keys.
    """

    def __init__(self, config_path=None):
        """Initialize configuration.

        Args:
            config_path: Optional path to config.yaml. If not provided,
                        will auto-detect by traversing up from cwd.
        """
        if config_path is None:
            config_path = self._find_config()
        self._config_path = Path(config_path)
        self._root = self._config_path.parent.parent

        with open(config_path, 'r') as f:
            self._data = yaml.safe_load(f)

        # Load .env file from project root
        env_path = self._root / '.env'
        if env_path.exists():
            load_dotenv(env_path)

    @staticmethod
    def _find_config():
        """Find config.yaml by traversing up from current directory."""
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            config_path = parent / "config" / "config.yaml"
            if config_path.exists():
                return config_path
        raise FileNotFoundError(
            "config.yaml not found. Make sure you're running from within "
            "the project directory or a subdirectory."
        )

    @property
    def root(self) -> Path:
        """Get project root directory."""
        return self._root

    @property
    def data(self) -> dict:
        """Get raw configuration data."""
        return self._data

    def get(self, *keys, default=None):
        """Get nested configuration value.

        Args:
            *keys: Sequence of keys to traverse the config tree.
            default: Default value if key path not found.

        Returns:
            Configuration value or default.

        Example:
            config.get('openai', 'model')  # Returns 'gpt-5-nano'
            config.get('training', 'batch_size')  # Returns 64
        """
        value = self._data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

    def get_path(self, key: str) -> Path:
        """Get absolute path for a configured directory.

        Args:
            key: Key name in paths section (e.g., 'data_dir', 'output_dir')

        Returns:
            Absolute Path object.
        """
        rel_path = self.get('paths', key)
        if rel_path is None:
            raise KeyError(f"Path key '{key}' not found in config")
        return self._root / rel_path

    def get_api_key(self, env_var: str = None) -> str:
        """Get API key from environment.

        Args:
            env_var: Environment variable name. If not provided,
                    uses the one configured in openai.api_key_env_var.

        Returns:
            API key string.

        Raises:
            ValueError: If environment variable is not set.
        """
        if env_var is None:
            env_var = self.get('openai', 'api_key_env_var', default='OPENAI_API_KEY')

        key = os.getenv(env_var)
        if not key:
            raise ValueError(
                f"Environment variable {env_var} not set. "
                f"Please add it to your .env file or set it in your environment."
            )
        return key

    def get_attributes(self) -> dict:
        """Get iPhone attributes configuration.

        Returns:
            Dictionary of attribute configurations.
        """
        return self.get('attributes', default={})

    def get_real_profiles(self) -> dict:
        """Get real iPhone profile configurations.

        Returns:
            Dictionary of real iPhone profiles.
        """
        return self.get('real_profiles', default={})


# Global config instance cache
_config_instance = None


def get_config(config_path=None, force_reload=False) -> Config:
    """Get or create global config instance.

    Args:
        config_path: Optional path to config.yaml.
        force_reload: If True, reload config even if already loaded.

    Returns:
        Config instance.
    """
    global _config_instance

    if _config_instance is None or force_reload:
        _config_instance = Config(config_path)

    return _config_instance


def reset_config():
    """Reset the global config instance. Useful for testing."""
    global _config_instance
    _config_instance = None
