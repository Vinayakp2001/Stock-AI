# Scalping Strategies
from scalping.strategies.ema_crossover import EMACrossoverStrategy
from scalping.strategies.vwap_strategy import VWAPStrategy
from scalping.strategies.rsi_scalp import RSIScalpStrategy
from scalping.strategies import registry

# Auto-register all three strategies
registry.register("EMA_9_21_Crossover", EMACrossoverStrategy)
registry.register("RSI_Scalp_35_65", RSIScalpStrategy)
registry.register("VWAP_Bounce", VWAPStrategy)

# ImprovedScalpingStrategy is imported lazily to avoid circular imports
__all__ = [
    "EMACrossoverStrategy",
    "VWAPStrategy",
    "RSIScalpStrategy",
    "ImprovedScalpingStrategy",
    "registry",
]


def __getattr__(name):
    if name == "ImprovedScalpingStrategy":
        from scalping.strategies.improved_strategy import ImprovedScalpingStrategy
        return ImprovedScalpingStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
