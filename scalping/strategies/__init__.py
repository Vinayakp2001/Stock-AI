# Scalping Strategies
from scalping.strategies.ema_crossover import EMACrossoverStrategy
from scalping.strategies.vwap_strategy import VWAPStrategy
from scalping.strategies.rsi_scalp import RSIScalpStrategy

# ImprovedScalpingStrategy is imported lazily to avoid circular imports
# (improved_strategy -> ensemble_scorer -> strategies -> improved_strategy)
# Import it directly: from scalping.strategies.improved_strategy import ImprovedScalpingStrategy

__all__ = ['EMACrossoverStrategy', 'VWAPStrategy', 'RSIScalpStrategy', 'ImprovedScalpingStrategy']


def __getattr__(name):
    if name == 'ImprovedScalpingStrategy':
        from scalping.strategies.improved_strategy import ImprovedScalpingStrategy
        return ImprovedScalpingStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
