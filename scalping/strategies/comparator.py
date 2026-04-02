"""
Strategy Comparator — runs all registered strategies against the same data
and ranks them by win rate.
"""

import logging
from typing import Any, Dict, List

from scalping.backtester import ScalpingBacktester
import scalping.strategies.registry as registry

logger = logging.getLogger(__name__)


class StrategyComparator:
    def __init__(self, initial_capital: float = 100_000, market: str = "NSE"):
        self.backtester = ScalpingBacktester(initial_capital=initial_capital, market=market)

    def compare(
        self, symbol: str, period: str = "7d", mode: str = "conservative"
    ) -> List[Dict[str, Any]]:
        """
        Run every registered strategy through the backtester.
        Returns list of result dicts sorted by win_rate descending.
        """
        results = []
        for name in registry.list_all():
            try:
                strategy = registry.get(name)
                result = self.backtester.run_backtest(strategy, symbol, period=period, mode=mode)
                results.append(
                    {
                        "strategy_name": name,
                        "win_rate": result.win_rate,
                        "profit_factor": result.profit_factor,
                        "total_trades": result.total_trades,
                        "avg_daily_return_pct": result.avg_daily_return_pct,
                        "validation_passed": result.validation_passed,
                    }
                )
            except Exception as exc:
                logger.warning("Strategy '%s' failed during comparison: %s", name, exc)

        results.sort(key=lambda r: r["win_rate"], reverse=True)
        return results

    def best(self, symbol: str, period: str = "7d", mode: str = "conservative") -> str:
        """Return the name of the highest win-rate strategy with >= 5 trades."""
        results = self.compare(symbol, period, mode)
        eligible = [r for r in results if r["total_trades"] >= 5]
        if not eligible:
            raise ValueError("No strategy produced >= 5 trades for the given symbol/period.")
        return eligible[0]["strategy_name"]

    def print_report(self, results: List[Dict[str, Any]]) -> None:
        """Print a formatted comparison table to stdout."""
        if not results:
            print("No results to display.")
            return
        header = f"{'Strategy':<30} {'Win Rate':>9} {'PF':>7} {'Trades':>7} {'Avg Daily':>10} {'Valid':>6}"
        print("\n" + "=" * len(header))
        print("STRATEGY COMPARISON REPORT")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for r in results:
            valid = "YES" if r["validation_passed"] else "NO"
            print(
                f"{r['strategy_name']:<30} "
                f"{r['win_rate']:>8.1%} "
                f"{r['profit_factor']:>7.2f} "
                f"{r['total_trades']:>7} "
                f"{r['avg_daily_return_pct']:>9.2%} "
                f"{valid:>6}"
            )
        print("=" * len(header))
