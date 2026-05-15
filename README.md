# Stock AI — Autonomous Trading Bot

An autonomous trading bot with ML-based price prediction, multi-factor scoring, scalping strategies, risk management, and a full broker API layer.

## Dashboards

| Dashboard | Port | Command |
|-----------|------|---------|
| Prediction & Analysis | 8050 | `python app_fresh.py` |
| Scalping Module | 8051 | `python scalping/dashboard.py` |
| Trading Bot | 8052 | `python trading/dashboard.py` |

## Quick Start

```bash
git clone https://github.com/Vinayakp2001/Stock-AI.git
cd Stock-AI
python -m venv stock_env
stock_env\Scripts\activate   # Windows
pip install -r requirements.txt
python app_fresh.py
```

## Project Structure

```
Stock-AI/
├── agents/
│   ├── data_agent.py           # Data fetching & preprocessing
│   ├── prediction_agent.py     # ML price prediction
│   ├── fundamental_agent.py    # Fundamental analysis (12 ratios, 0-100 score)
│   └── sentiment_agent.py      # News sentiment via yfinance + TextBlob
│
├── brokers/
│   ├── base_broker.py          # Abstract broker interface
│   ├── paper_broker.py         # Paper trading (no API key needed)
│   ├── zerodha_broker.py       # Zerodha Kite Connect
│   └── alpaca_broker.py        # Alpaca API
│
├── scalping/
│   ├── strategies/             # EMA, VWAP, RSI, Improved (5-layer filter)
│   ├── filters/                # Regime filter, session filter
│   ├── ml/                     # ML signal confirmer (RandomForest)
│   ├── risk/                   # Risk manager, adaptive stop-loss
│   ├── backtester.py           # Scalping backtester with cost modeling
│   ├── batch_backtester.py     # Multi-symbol batch backtesting
│   ├── paper_trader.py         # Paper trading with validation gate
│   ├── ensemble_scorer.py      # Multi-strategy ensemble scoring
│   └── run_scalping.py         # CLI entry point
│
├── trading/
│   ├── decision_engine.py      # 5-component weighted scoring (0-100)
│   ├── market_regime.py        # BULL/BEAR/SIDEWAYS/VOLATILE detection
│   ├── order_executor.py       # Bracket orders, retry, chunking
│   ├── position_manager.py     # Trailing stops, time exits, daily P&L halt
│   ├── portfolio_optimizer.py  # MPT — max Sharpe, min volatility, efficient frontier
│   ├── continuous_learner.py   # Auto-retrains ML model as trades accumulate
│   ├── alert_system.py         # Console + email alerts for signals & risk events
│   ├── safety_controller.py    # Kill switch + circuit breakers
│   ├── config_manager.py       # YAML config with env var overrides
│   └── dashboard.py            # Trading bot dashboard (port 8052)
│
├── tests/                      # pytest test suite (25 tests)
├── config.yaml                 # Central configuration
├── app_fresh.py                # Prediction dashboard (port 8050)
└── scalping/dashboard.py       # Scalping dashboard (port 8051)
```

## Scalping Module

### Run a Backtest
```bash
python scalping/run_scalping.py --mode backtest --symbol RELIANCE.NS --strategy ema
python scalping/run_scalping.py --mode backtest --symbol RELIANCE.NS --strategy improved
```

### Strategies
| Strategy | Description |
|----------|-------------|
| `ema` | EMA 9/21 crossover |
| `vwap` | VWAP bounce with volume confirmation |
| `rsi` | RSI 35/65 with momentum filter |
| `improved` | 5-layer filter: regime + session + ADX + volume + ML |

### Benchmark Targets
| Mode | Win Rate | Daily Net | Monthly |
|------|----------|-----------|---------|
| Conservative | 62%+ | 1-2% | 20-40% |
| Aggressive | 67%+ | 2-3% | 40-60% |
| Validation Gate | 60%+ | — | — |

## Trading Module

### Decision Engine
Combines 5 components into a 0-100 score:
- Technical (30%) — ensemble of scalping strategies
- Fundamental (25%) — 12 financial ratios
- Sentiment (15%) — news sentiment
- ML Prediction (20%) — RandomForest signal probability
- Risk Metrics (10%) — drawdown & consecutive loss state

### Portfolio Optimizer
```python
from trading.portfolio_optimizer import PortfolioOptimizer
opt = PortfolioOptimizer(['RELIANCE.NS', 'TCS.NS', 'INFY.NS'], period='1y')
result = opt.optimize()
opt.print_report(result)
```

### Alert System
```python
from trading.alert_system import AlertSystem
alerts = AlertSystem(email_to='you@example.com')  # optional email
alerts.trade_signal('RELIANCE.NS', 'BUY', 72.5, 1360, 1347, 1387)
alerts.risk_breach('DRAWDOWN', 4.5, 3.0)
alerts.regime_change('BULL', 'VOLATILE', 0.75, 'TRADE_CAUTIOUS')
```

### Safety Controller
```python
from trading.safety_controller import SafetyController
sc = SafetyController(initial_capital=100_000)
if sc.check_trade(capital, position_size):
    # place order
    sc.record_trade_result(pnl, new_capital)
```

## Configuration

Edit `config.yaml` or use environment variables:

```bash
# Override any config value
set STOCKAI_TRADING_INITIAL_CAPITAL=200000
set STOCKAI_BROKER_DEFAULT=alpaca
set STOCKAI_ALERTS_EMAIL_TO=you@example.com
```

## Brokers

| Broker | Env Vars Required |
|--------|-------------------|
| Paper (default) | None |
| Zerodha | `ZERODHA_API_KEY`, `ZERODHA_ACCESS_TOKEN` |
| Alpaca | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` |

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

25 tests covering: RiskManager, SafetyController, AlertSystem, ConfigManager, PortfolioOptimizer.

## Key Technical Decisions

- Data: yfinance (1-min, max 7 days) until real-time feed integrated
- Position sizing: 20-30% of capital per trade
- Minimum hold: 3 candles before checking exit
- Transaction costs: ₹20 brokerage + STT (0.025%) + slippage (0.03%)
- Target instrument: BANKNIFTY futures (after Zerodha WebSocket integration)

## Disclaimer

For educational and research purposes only. Not financial advice. Trading involves risk.

## License

MIT — see [LICENSE](LICENSE)
