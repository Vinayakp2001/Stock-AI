# Stock AI - Project Context

## What This Project Is
An autonomous trading bot being built at https://github.com/Vinayakp2001/Stock-AI

The system has two main parts:
1. **Prediction System** - ML-based stock price prediction (already built)
2. **Scalping Module** - Automated intraday scalping (in progress)

---

## Current Development Phase

### Phase 1: Complete All 26 GitHub Issues
Working through issues in this order (scalping first, then rest):
- Issue #56: Improve Scalping Strategy Win Rate to 60%+ ← CURRENT
- Issue #48: Enhanced Backtesting and Paper Trading
- Issue #4: Risk Management Module
- Issue #5: Trading Strategy Framework
- Issue #3: Multi-Factor Scoring System
- Issues #1, #2, #6-#25: Remaining features

Full plan: See `SCALP_PLAN.md`

### Phase 2: Integrations (After All Issues Done)
1. TrueData API (tick data)
2. Zerodha WebSocket (real-time feed)
3. BANKNIFTY futures support
4. Auto trader (automated paper trading)
5. Cloud deployment
6. Live trading

---

## Development Workflow (Per Issue)
```
1. Pick issue from SCALP_PLAN.md
2. Create spec in .kiro/specs/<issue-name>/
3. Implement code
4. Test it works
5. Commit to GitHub
6. Close issue via CLI:
   gh issue close <number> --repo Vinayakp2001/Stock-AI --comment "Done."
7. Update SCALP_PLAN.md status to DONE
8. Move to next issue
```

---

## Project Structure
```
Stock-AI/
├── agents/                    # Data fetching and ML prediction
├── backtesting/               # Swing trading backtesting engine
├── scalping/                  # Scalping module (main focus)
│   ├── strategies/            # EMA, VWAP, RSI strategies
│   ├── backtester.py          # Scalping backtester
│   ├── paper_trader.py        # Paper trading engine
│   ├── config.py              # Benchmark targets
│   ├── run_scalping.py        # CLI runner
│   └── dashboard.py           # Standalone dashboard (port 8051)
├── app_fresh.py               # Main dashboard (port 8050)
├── SCALP_PLAN.md              # Master development plan
└── .kiro/specs/               # One spec folder per issue
```

---

## Scalping Benchmark Targets (Locked - Never Change)
```
Conservative:
- Win rate: 62%+, Daily net: 1-2%, Monthly: 20-40%

Aggressive:
- Win rate: 67%+, Daily net: 2-3%, Monthly: 40-60%

Validation Gate (to unlock live trading):
- 60%+ win rate, 50+ trades, 14 days paper trading
```

---

## Key Technical Decisions (Already Made)
- Data: yfinance (1-min, max 7 days) until Zerodha WebSocket integrated
- Position sizing: 20-30% of capital per trade (not 2% - too small for costs)
- Minimum hold: 3 candles before checking exit (prevents whipsaw stops)
- Trade status: Based on exit reason (TAKE_PROFIT=WIN) not net P&L
- Transaction costs: ₹20 brokerage + STT (0.025%) + slippage (0.03%)
- Target instrument: BANKNIFTY futures (after real-time data integration)

---

## GitHub Info
- Repo: https://github.com/Vinayakp2001/Stock-AI
- Issues: https://github.com/Vinayakp2001/Stock-AI/issues
- CLI path: C:\Program Files\GitHub CLI\gh.exe
- Set PATH: $env:PATH = "C:\Program Files\GitHub CLI" + [IO.Path]::PathSeparator + $env:PATH

---

## Important Rules
1. Never break existing working code - all improvements are additive
2. Always run getDiagnostics after writing code
3. Test with: python scalping/run_scalping.py --mode backtest --symbol RELIANCE.NS --strategy <name>
4. Commit after each issue is complete
5. Close GitHub issue via CLI when done
6. Update SCALP_PLAN.md status after each issue
