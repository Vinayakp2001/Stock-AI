"""
Scalping Module Runner
Quick way to run backtests and paper trading from command line
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalping.backtester import ScalpingBacktester
from scalping.paper_trader import PaperTrader
from scalping.strategies.ema_crossover import EMACrossoverStrategy
from scalping.strategies.vwap_strategy import VWAPStrategy
from scalping.strategies.rsi_scalp import RSIScalpStrategy
from scalping.strategies.improved_strategy import ImprovedScalpingStrategy
from scalping.config import RECOMMENDED_STOCKS


STRATEGIES = {
    "ema":      EMACrossoverStrategy(),
    "vwap":     VWAPStrategy(),
    "rsi":      RSIScalpStrategy(),
    "improved": ImprovedScalpingStrategy(),
}


def run_backtest(symbol, strategy_name, mode, capital):
    print(f"\nRunning backtest: {symbol} | {strategy_name.upper()} | {mode.upper()}")
    strategy = STRATEGIES[strategy_name]
    backtester = ScalpingBacktester(initial_capital=capital)
    result = backtester.run_backtest(strategy, symbol, period="7d", interval="1m", mode=mode)
    backtester.print_report(result)
    return result


def run_train_ml(symbol, capital):
    """Train the ML model for the improved strategy."""
    print(f"\nTraining ML model on {symbol} data...")
    from scalping.backtester import ScalpingBacktester
    backtester = ScalpingBacktester(initial_capital=capital)
    data = backtester.fetch_data(symbol, period="7d", interval="1m")
    strategy = STRATEGIES["improved"]
    accuracy = strategy.train_ml_model(data)
    if accuracy > 0:
        print(f"ML model trained. Test accuracy: {accuracy:.1%}")
    else:
        print("ML training failed or insufficient data (need 500+ samples).")


def run_paper(symbol, strategy_name, mode, capital, days):
    print(f"\nRunning paper trading: {symbol} | {strategy_name.upper()} | {days} days")

    # First run backtest to validate
    backtester = ScalpingBacktester(initial_capital=capital)
    strategy = STRATEGIES[strategy_name]
    backtest_result = backtester.run_backtest(strategy, symbol, period="7d", interval="1m", mode=mode)

    paper_trader = PaperTrader(capital=capital)
    gate = paper_trader.can_start_paper_trading(backtest_result)

    if not gate['allowed']:
        print(f"\nPaper trading BLOCKED: {gate['message']}")
        print("Fix these issues first:")
        for k, v in gate.get('what_failed', {}).items():
            print(f"  - {k}: {v}")
        return

    print(f"\nGate check passed. Starting paper trading...")
    result = paper_trader.run_paper_session(strategy, symbol, mode=mode, days=days)

    print(f"\nPaper Trading Results:")
    stats = result['stats']
    print(f"  Trades:       {stats['total_trades']}")
    print(f"  Win Rate:     {stats['win_rate']:.1%}")
    print(f"  Net P&L:      {stats['net_pnl']:,.2f}")
    print(f"  Return:       {stats['return_pct']:.2%}")
    print(f"  Avg Daily:    {stats['avg_daily_return_pct']:.2%}")

    print(f"\nLive Trading Status:")
    status = result['live_trading_status']
    print(f"  {status['message']}")
    print(f"  Days traded:  {status['days_traded']}/{status['days_required']}")
    print(f"  Trades done:  {status['total_paper_trades']}/{status['trades_required']}")
    print(f"  Win rate:     {status['current_win_rate']} (need {status['required_win_rate']})")


def run_validation(capital: float = 100000):
    """
    Run improved vs baseline backtest on 3 symbols and save report.
    Req 6.1-6.6
    """
    import json
    import os

    symbols   = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    report    = {"symbols": {}, "summary": {}}
    backtester = ScalpingBacktester(initial_capital=capital)

    improved_strategy = STRATEGIES["improved"]
    baseline_strategy = STRATEGIES["ema"]

    all_improved_wr = []
    all_baseline_wr = []

    print("\n" + "=" * 60)
    print("VALIDATION REPORT — Improved vs Baseline (EMA)")
    print("=" * 60)

    for symbol in symbols:
        print(f"\n[{symbol}]")
        try:
            imp = backtester.run_backtest(improved_strategy, symbol, period="7d", interval="1m")
            base = backtester.run_backtest(baseline_strategy, symbol, period="7d", interval="1m")

            delta_wr = imp.win_rate - base.win_rate
            print(f"  Improved  → trades={imp.total_trades:3d}  win_rate={imp.win_rate:.1%}  pf={imp.profit_factor:.2f}")
            print(f"  Baseline  → trades={base.total_trades:3d}  win_rate={base.win_rate:.1%}  pf={base.profit_factor:.2f}")
            print(f"  Delta win rate: {delta_wr:+.1%}")

            all_improved_wr.append(imp.win_rate)
            all_baseline_wr.append(base.win_rate)

            report["symbols"][symbol] = {
                "improved": {
                    "trades":       imp.total_trades,
                    "win_rate":     round(imp.win_rate, 4),
                    "profit_factor": round(imp.profit_factor, 3),
                    "sharpe":       round(imp.sharpe_ratio, 3),
                    "max_drawdown": round(imp.max_drawdown_pct, 4),
                    "avg_daily_return": round(imp.avg_daily_return_pct, 4),
                    "validated":    imp.validation_passed,
                },
                "baseline": {
                    "trades":       base.total_trades,
                    "win_rate":     round(base.win_rate, 4),
                    "profit_factor": round(base.profit_factor, 3),
                },
                "delta_win_rate": round(delta_wr, 4),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            report["symbols"][symbol] = {"error": str(e)}

    # Summary
    avg_improved = sum(all_improved_wr) / len(all_improved_wr) if all_improved_wr else 0
    avg_baseline = sum(all_baseline_wr) / len(all_baseline_wr) if all_baseline_wr else 0
    validated    = avg_improved >= 0.60

    report["summary"] = {
        "avg_improved_win_rate": round(avg_improved, 4),
        "avg_baseline_win_rate": round(avg_baseline, 4),
        "avg_delta":             round(avg_improved - avg_baseline, 4),
        "validated_for_paper_trading": validated,
        "target_win_rate": 0.60,
    }

    print("\n" + "=" * 60)
    print(f"AVERAGE IMPROVED WIN RATE: {avg_improved:.1%}  (target: 60%+)")
    print(f"AVERAGE BASELINE WIN RATE: {avg_baseline:.1%}")
    if validated:
        print("RESULT: PASSED — Strategy validated for paper trading")
    else:
        print("RESULT: FAILED — Win rate below 60% target")
        print("Recommendations:")
        print("  1. Collect more data (use longer symbols or multiple instruments)")
        print("  2. Tune ADX threshold (try ADX > 25 for stricter regime filter)")
        print("  3. Train ML model with --train-ml flag for more data")
    print("=" * 60)

    # Save report (Req 6.6)
    out_path = os.path.join("data", "scalping", "validation_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Scalping Module Runner")
    parser.add_argument("--mode", choices=["backtest", "paper", "validate"], default="backtest")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--strategy", choices=["ema", "vwap", "rsi", "improved"], default="ema")
    parser.add_argument("--trading-mode", choices=["conservative", "aggressive"], default="conservative")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--days", type=int, default=5, help="Days for paper trading (max 7)")
    parser.add_argument("--list-stocks", action="store_true")
    parser.add_argument("--train-ml", action="store_true", help="Train ML model for improved strategy")

    args = parser.parse_args()

    if args.list_stocks:
        print("\nRecommended stocks for scalping:")
        print("NSE (Indian):", ", ".join(RECOMMENDED_STOCKS['NSE']))
        print("NYSE (US):", ", ".join(RECOMMENDED_STOCKS['NYSE']))
        return

    if args.train_ml:
        run_train_ml(args.symbol, args.capital)
        return

    if args.mode == "backtest":
        run_backtest(args.symbol, args.strategy, args.trading_mode, args.capital)
    elif args.mode == "paper":
        run_paper(args.symbol, args.strategy, args.trading_mode, args.capital, args.days)
    elif args.mode == "validate":
        run_validation(args.capital)


if __name__ == "__main__":
    main()
