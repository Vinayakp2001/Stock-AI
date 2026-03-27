# Project Context for Scalping Win Rate Improvement

## What This Project Is
Stock AI - An autonomous trading bot being built at https://github.com/Vinayakp2001/Stock-AI

## Current State of the Scalping Module
The scalping module is already built and located in `scalping/` directory:
- `scalping/backtester.py` - Backtesting engine with realistic cost modeling
- `scalping/paper_trader.py` - Paper trading with validation gate
- `scalping/config.py` - Benchmark targets (conservative/aggressive)
- `scalping/run_scalping.py` - CLI runner
- `scalping/strategies/ema_crossover.py` - EMA 9/21 strategy
- `scalping/strategies/vwap_strategy.py` - VWAP bounce strategy
- `scalping/strategies/rsi_scalp.py` - RSI scalp strategy
- `scalping/dashboard.py` - Standalone Dash dashboard on port 8051

## Current Performance (Baseline to Beat)
- EMA strategy on RELIANCE.NS: 33% win rate, PF 1.40, -0.15% daily
- Target: 60-72% win rate, 1.5-2.5% daily net

## Benchmark Targets (Locked - Do Not Change)
Conservative:
- Win rate: 62%+
- Profit per trade: 0.2-0.3%
- Stop loss: 0.15%
- Daily net: 1-2% after costs
- Monthly: 20-40%

Aggressive:
- Win rate: 67%+
- Profit per trade: 0.3-0.5%
- Daily net: 2-3% after costs
- Monthly: 40-60%

## Data Source
Currently using yfinance (free, 1-min delayed, max 7 days of 1-min data).
Future: Will integrate Zerodha WebSocket + TrueData API for real-time tick data.
Do NOT change data source in this spec - keep yfinance for now.

## Key Design Decisions Already Made
1. Position sizing: 20-30% of capital per trade (not 2% - too small for costs)
2. Minimum hold: 3 candles before checking exit (prevents whipsaw stops)
3. Status based on exit reason (TAKE_PROFIT=WIN, STOP_LOSS=STOPPED) not net P&L
4. Transaction costs modeled: ₹20 brokerage + STT + slippage per trade
5. Validation gate: 60% win rate + 50 trades + 14 days paper trading to unlock live

## Development Workflow
For each task:
1. Implement the code
2. Run diagnostics to check for errors
3. Test with: python scalping/run_scalping.py --mode backtest --symbol RELIANCE.NS --strategy improved
4. Commit when working
5. Close GitHub issue when fully done:
   gh issue close 56 --repo Vinayakp2001/Stock-AI --comment "Implemented and tested."

## GitHub Repository
https://github.com/Vinayakp2001/Stock-AI
Issue being worked on: #56 - Improve Scalping Strategy Win Rate to 60%+

## Important: Do Not Break Existing Code
- Keep existing strategies (ema_crossover.py, vwap_strategy.py, rsi_scalp.py) working
- The improved strategy is ADDITIVE - new files only
- Existing backtester.py and paper_trader.py should still work unchanged
- Only update run_scalping.py and dashboard.py to ADD the new strategy option
