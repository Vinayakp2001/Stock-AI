"""
Batch Backtester
Runs all symbol × strategy combinations and produces a ranked summary.
"""

import json
import os
import logging
from typing import Any, Dict, List

from scalping.backtester import ScalpingBacktester, ScalpBacktestResult

logger = logging.getLogger(__name__)

BATCH_RESULTS_FILE = os.path.join("data", "scalping", "batch_results.json")


class BatchBacktester:
    """
    Iterates over every (symbol, strategy) combination, catches per-symbol
    errors so one bad ticker never aborts the whole run, and saves a combined
    JSON result file.
    """

    def __init__(self, initial_capital: float = 100000, market: str = "NSE"):
        self.initial_capital = initial_capital
        self.market = market
        self._backtester = ScalpingBacktester(initial_capital=initial_capital, market=market)

    def run_batch(
        self,
        symbols: List[str],
        strategies: Dict[str, Any],   # {name: strategy_instance}
        mode: str = "conservative",
        capital: float = None,
    ) -> Dict[str, Any]:
        """
        Run backtest for every symbol × strategy combo.

        Args:
            symbols:    List of ticker symbols
            strategies: Dict mapping strategy name → strategy instance
            mode:       'conservative' or 'aggressive'
            capital:    Override initial capital (uses __init__ value if None)

        Returns:
            Dict with per-combo results and a ranked summary list
        """
        if capital is not None:
            self._backtester.initial_capital = capital

        results: List[Dict[str, Any]] = []

        total = len(symbols) * len(strategies)
        done = 0

        for symbol in symbols:
            for strat_name, strategy in strategies.items():
                done += 1
                print(f"  [{done}/{total}] {symbol} | {strat_name} ...", end=" ", flush=True)
                try:
                    result: ScalpBacktestResult = self._backtester.run_backtest(
                        strategy, symbol, period="7d", interval="1m", mode=mode
                    )
                    results.append({
                        "symbol": symbol,
                        "strategy": strat_name,
                        "mode": mode,
                        "total_trades": result.total_trades,
                        "win_rate": round(result.win_rate, 4),
                        "profit_factor": round(result.profit_factor, 4),
                        "total_net_pnl": round(result.total_net_pnl, 2),
                        "total_return_pct": round(result.total_return_pct, 4),
                        "avg_daily_return_pct": round(result.avg_daily_return_pct, 4),
                        "max_drawdown_pct": round(result.max_drawdown_pct, 4),
                        "sharpe_ratio": round(result.sharpe_ratio, 4),
                        "meets_conservative": result.meets_conservative_target,
                        "meets_aggressive": result.meets_aggressive_target,
                        "validation_passed": result.validation_passed,
                        "status": "ok",
                        "error": None,
                    })
                    print(f"win_rate={result.win_rate:.1%}  trades={result.total_trades}")
                except Exception as exc:
                    logger.warning(f"Batch error {symbol}/{strat_name}: {exc}")
                    results.append({
                        "symbol": symbol,
                        "strategy": strat_name,
                        "mode": mode,
                        "status": "error",
                        "error": str(exc),
                        "win_rate": None,
                        "total_trades": 0,
                    })
                    print(f"ERROR: {exc}")

        # Rank valid results by win rate descending
        valid = [r for r in results if r["status"] == "ok"]
        ranked = sorted(valid, key=lambda r: r["win_rate"], reverse=True)

        batch_output = {
            "mode": mode,
            "total_combinations": total,
            "successful": len(valid),
            "failed": total - len(valid),
            "ranked_results": ranked,
            "all_results": results,
        }

        self._save(batch_output)
        return batch_output

    def print_summary_table(self, batch_result: Dict[str, Any]) -> None:
        """Print a ranked table of all batch results sorted by win rate."""
        ranked = batch_result.get("ranked_results", [])
        print("\n" + "=" * 80)
        print("BATCH BACKTEST SUMMARY — ranked by win rate")
        print(f"Mode: {batch_result.get('mode', '').upper()}  |  "
              f"{batch_result['successful']}/{batch_result['total_combinations']} succeeded")
        print("=" * 80)
        header = f"{'#':<4} {'Symbol':<18} {'Strategy':<16} {'Trades':>7} {'Win%':>7} "
        header += f"{'PF':>6} {'Net P&L':>10} {'DD%':>7} {'Gate':<6}"
        print(header)
        print("-" * 80)
        for i, r in enumerate(ranked, 1):
            gate = "PASS" if r.get("validation_passed") else "FAIL"
            print(
                f"{i:<4} {r['symbol']:<18} {r['strategy']:<16} "
                f"{r['total_trades']:>7} {r['win_rate']:>6.1%} "
                f"{r['profit_factor']:>6.2f} {r['total_net_pnl']:>10,.0f} "
                f"{r['max_drawdown_pct']:>6.1%} {gate:<6}"
            )

        # Show errors at the bottom
        errors = [r for r in batch_result.get("all_results", []) if r["status"] == "error"]
        if errors:
            print(f"\nFailed ({len(errors)}):")
            for r in errors:
                print(f"  {r['symbol']} / {r['strategy']}: {r['error']}")
        print("=" * 80)
        print(f"\nResults saved to: {BATCH_RESULTS_FILE}")

    def _save(self, batch_output: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(BATCH_RESULTS_FILE), exist_ok=True)

        def _default(obj):
            # Handle inf/nan floats and numpy bools
            if isinstance(obj, float):
                if obj == float('inf'):
                    return None
                if obj != obj:   # NaN
                    return None
                return obj
            if hasattr(obj, 'item'):   # numpy scalar
                return obj.item()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(BATCH_RESULTS_FILE, "w") as f:
            json.dump(batch_output, f, indent=2, default=_default)
        logger.info(f"Batch results saved to {BATCH_RESULTS_FILE}")
