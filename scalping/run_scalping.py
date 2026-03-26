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
from scalping.config import RECOMMENDED_STOCKS


STRATEGIES = {
    "ema": EMACrossoverStrategy(),
    "vwap": VWAPStrategy(),
    "rsi": RSIScalpStrategy()
}


def run_backtest(symbol, strategy_name, mode, capital):
    print(f"\nRunning backtest: {symbol} | {strategy_name.upper()} | {mode.upper()}")
    strategy = STRATEGIES[strategy_name]
    backtester = ScalpingBacktester(initial_capital=capital)
    result = backtester.run_backtest(strategy, symbol, period="7d", interval="1m", mode=mode)
    backtester.print_report(result)
    return result


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


def main():
    parser = argparse.ArgumentParser(description="Scalping Module Runner")
    parser.add_argument("--mode", choices=["backtest", "paper"], default="backtest")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--strategy", choices=["ema", "vwap", "rsi"], default="ema")
    parser.add_argument("--trading-mode", choices=["conservative", "aggressive"], default="conservative")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--days", type=int, default=5, help="Days for paper trading (max 7)")
    parser.add_argument("--list-stocks", action="store_true")

    args = parser.parse_args()

    if args.list_stocks:
        print("\nRecommended stocks for scalping:")
        print("NSE (Indian):", ", ".join(RECOMMENDED_STOCKS['NSE']))
        print("NYSE (US):", ", ".join(RECOMMENDED_STOCKS['NYSE']))
        return

    if args.mode == "backtest":
        run_backtest(args.symbol, args.strategy, args.trading_mode, args.capital)
    elif args.mode == "paper":
        run_paper(args.symbol, args.strategy, args.trading_mode, args.capital, args.days)


if __name__ == "__main__":
    main()
