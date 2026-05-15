"""
Performance Benchmarking System (Issue #52)
Runs all scalping strategies across multiple symbols and produces
a ranked comparison report with key metrics.

Usage:
    python trading/performance_benchmark.py
    python trading/performance_benchmark.py --symbols RELIANCE.NS TCS.NS --mode aggressive
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

logger = logging.getLogger(__name__)

BENCHMARK_DIR = os.path.join("data", "benchmarks")

DEFAULT_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
DEFAULT_STRATEGIES = ["ema", "vwap", "rsi", "improved"]
STRATEGY_NAMES = {
    "ema":      "EMA Crossover",
    "vwap":     "VWAP Bounce",
    "rsi":      "RSI Scalp",
    "improved": "Improved (5-layer)",
}


def _load_strategy(name: str):
    if name == "ema":
        from scalping.strategies.ema_crossover import EMACrossoverStrategy
        return EMACrossoverStrategy()
    if name == "vwap":
        from scalping.strategies.vwap_strategy import VWAPStrategy
        return VWAPStrategy()
    if name == "rsi":
        from scalping.strategies.rsi_scalp import RSIScalpStrategy
        return RSIScalpStrategy()
    if name == "improved":
        from scalping.strategies.improved_strategy import ImprovedScalpingStrategy
        return ImprovedScalpingStrategy()
    raise ValueError(f"Unknown strategy: {name}")


def run_benchmark(
    symbols: List[str] = None,
    strategies: List[str] = None,
    mode: str = "conservative",
    capital: float = 100_000,
    period: str = "7d",
    interval: str = "1m",
) -> Dict[str, Any]:
    """
    Run all strategies on all symbols and return a ranked results dict.
    """
    from scalping.backtester import ScalpingBacktester

    symbols = symbols or DEFAULT_SYMBOLS
    strategies = strategies or DEFAULT_STRATEGIES

    backtester = ScalpingBacktester(initial_capital=capital)
    results = []
    errors = []

    total = len(symbols) * len(strategies)
    done = 0

    for symbol in symbols:
        for strat_name in strategies:
            done += 1
            print(f"[{done}/{total}] {symbol} — {STRATEGY_NAMES[strat_name]} ...", end=" ", flush=True)
            try:
                strategy = _load_strategy(strat_name)
                result = backtester.run_backtest(
                    strategy, symbol, period=period, interval=interval, mode=mode
                )
                results.append({
                    "symbol":           symbol,
                    "strategy":         strat_name,
                    "strategy_name":    STRATEGY_NAMES[strat_name],
                    "mode":             mode,
                    "win_rate":         round(result.win_rate, 4),
                    "profit_factor":    round(result.profit_factor, 4),
                    "total_trades":     result.total_trades,
                    "net_pnl":          round(result.total_net_pnl, 2),
                    "return_pct":       round(result.total_return_pct, 4),
                    "avg_daily_return": round(result.avg_daily_return_pct, 4),
                    "max_drawdown":     round(result.max_drawdown_pct, 4),
                    "sharpe":           round(result.sharpe_ratio, 4),
                    "validation_passed": result.validation_passed,
                    "meets_conservative": result.meets_conservative_target,
                    "meets_aggressive":   result.meets_aggressive_target,
                })
                status = "✓" if result.validation_passed else "✗"
                print(f"{status} WR={result.win_rate:.1%} PF={result.profit_factor:.2f} T={result.total_trades}")
            except Exception as e:
                errors.append({"symbol": symbol, "strategy": strat_name, "error": str(e)})
                print(f"ERROR: {e}")

    # Rank by win rate, then profit factor
    results.sort(key=lambda r: (r["win_rate"], r["profit_factor"]), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Aggregate stats per strategy
    strategy_summary = {}
    for strat in strategies:
        strat_results = [r for r in results if r["strategy"] == strat]
        if strat_results:
            strategy_summary[strat] = {
                "strategy_name":    STRATEGY_NAMES[strat],
                "avg_win_rate":     round(np.mean([r["win_rate"] for r in strat_results]), 4),
                "avg_profit_factor": round(np.mean([r["profit_factor"] for r in strat_results]), 4),
                "avg_sharpe":       round(np.mean([r["sharpe"] for r in strat_results]), 4),
                "validation_pass_rate": round(
                    sum(1 for r in strat_results if r["validation_passed"]) / len(strat_results), 4
                ),
                "total_symbols":    len(strat_results),
            }

    report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "symbols": symbols, "strategies": strategies,
            "mode": mode, "capital": capital,
            "period": period, "interval": interval,
        },
        "results": results,
        "strategy_summary": strategy_summary,
        "errors": errors,
        "best_overall": results[0] if results else None,
    }

    _save_report(report)
    return report


def print_report(report: Dict[str, Any]) -> None:
    """Print a formatted benchmark report to console."""
    print("\n" + "=" * 72)
    print("  PERFORMANCE BENCHMARK REPORT")
    print(f"  Generated: {report['generated_at'][:19]}")
    print("=" * 72)

    cfg = report["config"]
    print(f"  Symbols: {', '.join(cfg['symbols'])}")
    print(f"  Mode: {cfg['mode']}  |  Capital: ₹{cfg['capital']:,.0f}  |  Period: {cfg['period']}")
    print("-" * 72)

    # Strategy summary
    print("\nSTRATEGY SUMMARY (averaged across symbols)")
    print(f"{'Strategy':<22} {'Avg WR':>8} {'Avg PF':>8} {'Avg Sharpe':>11} {'Pass Rate':>10}")
    print("-" * 62)
    for s in report["strategy_summary"].values():
        print(f"{s['strategy_name']:<22} {s['avg_win_rate']:>7.1%} {s['avg_profit_factor']:>8.2f}"
              f" {s['avg_sharpe']:>11.3f} {s['validation_pass_rate']:>9.0%}")

    # Full results table
    print("\nFULL RESULTS (ranked by win rate)")
    print(f"{'#':<4} {'Symbol':<15} {'Strategy':<22} {'WR':>7} {'PF':>6} {'Trades':>7}"
          f" {'Net P&L':>10} {'Sharpe':>7} {'Gate':>5}")
    print("-" * 85)
    for r in report["results"]:
        gate = "✓" if r["validation_passed"] else "✗"
        print(f"{r['rank']:<4} {r['symbol']:<15} {r['strategy_name']:<22}"
              f" {r['win_rate']:>6.1%} {r['profit_factor']:>6.2f} {r['total_trades']:>7}"
              f" {r['net_pnl']:>10,.0f} {r['sharpe']:>7.3f} {gate:>5}")

    if report.get("best_overall"):
        b = report["best_overall"]
        print(f"\nBest: {b['symbol']} — {b['strategy_name']} | WR={b['win_rate']:.1%}"
              f" | PF={b['profit_factor']:.2f} | Sharpe={b['sharpe']:.3f}")

    if report["errors"]:
        print(f"\nErrors: {len(report['errors'])} combinations failed")

    print("=" * 72 + "\n")


def _save_report(report: Dict[str, Any]) -> str:
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(BENCHMARK_DIR, f"benchmark_{date_str}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Benchmark report saved to %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Run performance benchmark")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--mode", default="conservative", choices=["conservative", "aggressive"])
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--period", default="7d")
    args = parser.parse_args()

    report = run_benchmark(
        symbols=args.symbols,
        strategies=args.strategies,
        mode=args.mode,
        capital=args.capital,
        period=args.period,
    )
    print_report(report)
