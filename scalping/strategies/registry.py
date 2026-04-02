"""
Strategy Registry — module-level singleton for strategy discovery.
"""

import logging
from typing import Dict, List, Type

from scalping.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_registry: Dict[str, Type[BaseStrategy]] = {}


def register(name: str, cls: Type[BaseStrategy]) -> None:
    """Register a strategy class under the given name."""
    _registry[name] = cls
    logger.debug("Registered strategy: %s", name)


def get(name: str, params: dict = None) -> BaseStrategy:
    """Instantiate and return a registered strategy by name."""
    if name not in _registry:
        available = list(_registry.keys())
        raise KeyError(f"Strategy '{name}' not registered. Available: {available}")
    return _registry[name](params=params or {})


def list_all() -> List[str]:
    """Return all registered strategy names."""
    return list(_registry.keys())
