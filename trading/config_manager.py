"""
Configuration Management (Issue #49)
Loads config.yaml, applies environment variable overrides,
and provides typed access to all settings.
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
ENV_PREFIX = "STOCKAI_"


class ConfigManager:
    """
    Singleton-style config loader.
    Priority: env vars > config.yaml > defaults

    Usage:
        cfg = ConfigManager()
        capital = cfg.get("trading.initial_capital", default=100000)
        api_key = cfg.get("broker.alpaca_api_key")
    """

    _instance: Optional["ConfigManager"] = None

    def __new__(cls, config_path: str = CONFIG_PATH):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, config_path: str = CONFIG_PATH):
        if self._loaded:
            return
        self._config_path = config_path
        self._data: dict = {}
        self._load()
        self._loaded = True

    # ── Public API ────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-notation key.
        e.g. cfg.get("trading.initial_capital")
        Env var override: STOCKAI_TRADING_INITIAL_CAPITAL
        """
        # Check env var first
        env_key = ENV_PREFIX + key.upper().replace(".", "_")
        env_val = os.getenv(env_key)
        if env_val is not None:
            return self._cast(env_val, self._get_nested(key))

        val = self._get_nested(key)
        return val if val is not None else default

    def get_section(self, section: str) -> dict:
        """Return an entire config section as a dict."""
        return dict(self._data.get(section, {}))

    def set(self, key: str, value: Any) -> None:
        """Override a value at runtime (not persisted to file)."""
        parts = key.split(".")
        d = self._data
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    def reload(self) -> None:
        """Reload config from disk."""
        self._load()
        logger.info("ConfigManager: config reloaded from %s", self._config_path)

    def all(self) -> dict:
        """Return full config dict (without secrets)."""
        import copy
        safe = copy.deepcopy(self._data)
        for section in safe.values():
            if isinstance(section, dict):
                for k in list(section.keys()):
                    if any(s in k for s in ("key", "secret", "pass", "token")):
                        section[k] = "***"
        return safe

    # ── Private ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            import yaml
            if os.path.exists(self._config_path):
                with open(self._config_path) as f:
                    self._data = yaml.safe_load(f) or {}
                logger.info("ConfigManager: loaded %s", self._config_path)
            else:
                logger.warning("ConfigManager: %s not found — using defaults", self._config_path)
                self._data = {}
        except ImportError:
            logger.warning("ConfigManager: PyYAML not installed — falling back to env vars only")
            self._data = {}
        except Exception as e:
            logger.warning("ConfigManager: failed to load config: %s", e)
            self._data = {}

    def _get_nested(self, key: str) -> Any:
        parts = key.split(".")
        d = self._data
        for part in parts:
            if not isinstance(d, dict):
                return None
            d = d.get(part)
        return d

    def _cast(self, value: str, reference: Any) -> Any:
        """Cast env var string to the same type as the reference value."""
        if reference is None:
            return value
        try:
            if isinstance(reference, bool):
                return value.lower() in ("true", "1", "yes")
            if isinstance(reference, int):
                return int(value)
            if isinstance(reference, float):
                return float(value)
        except (ValueError, TypeError):
            pass
        return value


# Module-level singleton
_cfg: Optional[ConfigManager] = None


def get_config(config_path: str = CONFIG_PATH) -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _cfg
    if _cfg is None:
        _cfg = ConfigManager(config_path)
    return _cfg
