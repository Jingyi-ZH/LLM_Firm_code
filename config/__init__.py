"""Configuration management module for LLM Belief Elicitation project."""

from pathlib import Path
import yaml
import os
from functools import lru_cache
from dotenv import load_dotenv
from typing import Any, Dict, Optional


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

    def _load_yaml_file(self, path: Path) -> dict:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML must be a mapping: {path}")
        return data

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

    @staticmethod
    def _deep_merge_dicts(base: dict, override: dict) -> dict:
        """Deep merge override into base (dicts only), returning a new dict."""
        if not isinstance(base, dict):
            base = {}
        if not isinstance(override, dict):
            return dict(base)
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = Config._deep_merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    def get_app_spec(self) -> dict:
        """Return the active app spec dict.

        Backward compatible:
          - If top-level `attributes`/`real_profiles` exist, they are treated as the app spec.
          - Otherwise, loads a spec YAML referenced by `app.spec_path`.
        """
        # Legacy inline app spec (older config.yaml layout)
        has_inline_app = any(k in (self._data or {}) for k in ("attributes", "real_profiles", "prompting"))
        app_cfg = self.get("app", default={}) or {}
        env_spec_path = os.getenv("LLM_BELIEF_APP_SPEC_PATH")
        spec_path = env_spec_path or app_cfg.get("spec_path") or app_cfg.get("spec") or None

        spec: dict = {}
        if spec_path:
            spec_file = (self._root / spec_path).resolve()
            spec = self._load_yaml_file(spec_file)
            # allow specs to be nested under `app:` while still carrying siblings
            if "app" in spec and isinstance(spec.get("app"), dict):
                # keep the full mapping; callers can read `app` metadata if needed
                pass

        if has_inline_app:
            inline = {
                "attributes": self.get("attributes", default=None),
                "real_profiles": self.get("real_profiles", default=None),
                "prompting": self.get("prompting", default=None),
            }
            inline = {k: v for k, v in inline.items() if isinstance(v, dict)}
            spec = self._deep_merge_dicts(spec, inline)

        if not isinstance(spec, dict) or not spec:
            raise ValueError(
                "No app spec found. Provide either top-level `attributes`/`real_profiles` "
                "or set `app.spec_path` in config/config.yaml."
            )
        return spec

    def get_prompting(self) -> dict:
        """Get prompting configuration from the active app spec."""
        spec = self.get_app_spec()
        prompting = spec.get("prompting", {})
        prompting = prompting if isinstance(prompting, dict) else {}

        # Backward/lenient support: allow neutral criteria variants at the top level of the app spec.
        # Many specs naturally place these alongside `prompting:` rather than nested within.
        for k in ("neutral_criteria", "neutral_criteria_variants", "neutral_criteria_texts"):
            if k not in prompting and k in spec:
                prompting[k] = spec.get(k)

        return prompting

    def get_app_meta(self) -> dict:
        """Get optional app metadata (id/entity/etc.) from the active app spec."""
        spec = self.get_app_spec()
        app_meta = spec.get("app", {})
        if isinstance(app_meta, dict) and app_meta:
            return app_meta
        # fallback to base config app section
        app_cfg = self.get("app", default={}) or {}
        return app_cfg if isinstance(app_cfg, dict) else {}

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
        """Get active app attributes configuration.

        Returns:
            Dictionary of attribute configurations.
        """
        spec = self.get_app_spec()
        attrs = spec.get("attributes", {})
        return attrs if isinstance(attrs, dict) else {}

    def get_real_profiles(self) -> dict:
        """Get active app real profile configurations.

        Returns:
            Dictionary of real iPhone profiles.
        """
        spec = self.get_app_spec()
        real = spec.get("real_profiles", {})
        return real if isinstance(real, dict) else {}


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
