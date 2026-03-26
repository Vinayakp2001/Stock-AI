"""
Scalping Module Configuration
Benchmarks locked based on AI-assisted trading targets
"""

# ─── Benchmark Targets ────────────────────────────────────────────────────────

CONSERVATIVE = {
    "mode": "conservative",
    "win_rate_target": 0.62,          # 62% minimum
    "profit_target_pct": 0.003,       # 0.3% per trade
    "stop_loss_pct": 0.0015,          # 0.15% stop loss
    "daily_net_target_pct": 0.015,    # 1.5% daily net
    "monthly_target_pct": 0.30,       # 30% monthly
    "max_risk_per_trade_pct": 0.02,   # 2% capital per trade
    "max_trades_per_day": 30,
    "min_risk_reward": 2.0,           # 1:2 minimum
}

AGGRESSIVE = {
    "mode": "aggressive",
    "win_rate_target": 0.67,          # 67% minimum
    "profit_target_pct": 0.004,       # 0.4% per trade
    "stop_loss_pct": 0.002,           # 0.2% stop loss
    "daily_net_target_pct": 0.025,    # 2.5% daily net
    "monthly_target_pct": 0.50,       # 50% monthly
    "max_risk_per_trade_pct": 0.03,   # 3% capital per trade
    "max_trades_per_day": 50,
    "min_risk_reward": 2.0,
}

# ─── Paper Trading Validation Gate ────────────────────────────────────────────
# Bot must pass these before live trading is unlocked

VALIDATION_GATE = {
    "min_paper_trading_days": 14,     # 2 weeks minimum
    "min_win_rate": 0.60,             # 60% win rate
    "min_profit_factor": 1.5,         # Gross profit / Gross loss
    "max_drawdown_pct": 0.10,         # Max 10% drawdown
    "min_trades_to_validate": 50,     # At least 50 trades
    "min_daily_net_pct": 0.01,        # 1% daily net minimum
}

# ─── Transaction Costs (Indian Markets - NSE) ─────────────────────────────────

COSTS_INDIA = {
    "brokerage_per_trade": 20,        # Zerodha flat ₹20
    "stt_pct": 0.00025,               # 0.025% STT
    "exchange_charges_pct": 0.0000325,
    "gst_on_brokerage": 0.18,         # 18% GST on brokerage
    "sebi_charges_pct": 0.000001,
    "slippage_pct": 0.0003,           # 0.03% average slippage
}

# ─── Transaction Costs (US Markets) ───────────────────────────────────────────

COSTS_US = {
    "brokerage_per_trade": 0,         # Alpaca commission-free
    "sec_fee_pct": 0.0000229,
    "finra_taf_per_share": 0.000119,
    "slippage_pct": 0.0002,           # 0.02% average slippage
}

# ─── Data Settings ─────────────────────────────────────────────────────────────

DATA_CONFIG = {
    "primary_interval": "1m",         # 1-minute candles
    "secondary_interval": "5m",       # 5-minute for trend confirmation
    "warmup_period": "5d",            # Data needed for indicators
    "backtest_period": "7d",            # Max 7 days for 1m data (yfinance limit)
    "cache_duration_seconds": 60,     # 1 min cache for intraday
}

# ─── Trading Hours ─────────────────────────────────────────────────────────────

TRADING_HOURS = {
    "NSE": {
        "open": "09:15",
        "close": "15:30",
        "best_window_start": "09:15",
        "best_window_end": "11:15",   # First 2 hours = best liquidity
        "avoid_last_minutes": 15,     # Avoid last 15 min (volatile close)
    },
    "NYSE": {
        "open": "09:30",
        "close": "16:00",
        "best_window_start": "09:30",
        "best_window_end": "11:30",
        "avoid_last_minutes": 15,
    }
}

# ─── Signal Thresholds ─────────────────────────────────────────────────────────

SIGNAL_CONFIG = {
    "min_signal_score": 70,           # Minimum score to enter trade (0-100)
    "min_volume_ratio": 1.5,          # Volume must be 1.5x average
    "rsi_oversold": 35,               # RSI oversold threshold for scalping
    "rsi_overbought": 65,             # RSI overbought threshold for scalping
    "ema_fast": 9,                    # Fast EMA period
    "ema_slow": 21,                   # Slow EMA period
    "vwap_deviation_pct": 0.002,      # 0.2% deviation from VWAP to trigger
}

# ─── Supported High-Liquidity Stocks ──────────────────────────────────────────

RECOMMENDED_STOCKS = {
    "NSE": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
        "INFY.NS", "ICICIBANK.NS", "SBIN.NS",
        "AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS",
        "^NSEI",   # NIFTY index
    ],
    "NYSE": [
        "AAPL", "MSFT", "GOOGL", "AMZN",
        "TSLA", "NVDA", "META", "SPY",
    ]
}
