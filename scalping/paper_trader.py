"""
Paper Trading Engine for Scalping
Simulates live trading with real-time data (delayed) to validate
strategy before going live. Only unlocked after backtest passes validation gate.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import time

from scalping.config import CONSERVATIVE, AGGRESSIVE, VALIDATION_GATE, COSTS_INDIA, COSTS_US
from scalping.backtester import ScalpTrade, ScalpBacktestResult

logger = logging.getLogger(__name__)

PAPER_TRADES_FILE = "data/scalping/paper_trades.json"
PAPER_STATS_FILE = "data/scalping/paper_stats.json"


@dataclass
class PaperTradingSession:
    """Tracks a paper trading session"""
    session_id: str
    symbol: str
    strategy_name: str
    mode: str
    start_date: str
    capital: float
    trades: List[Dict] = field(default_factory=list)
    daily_stats: List[Dict] = field(default_factory=list)
    is_live_unlocked: bool = False


class PaperTrader:
    """
    Paper trading engine that simulates live scalping.
    Validates strategy performance before allowing live trading.

    Flow:
    1. Backtest must pass validation gate first
    2. Paper trade for minimum 14 days
    3. Must maintain 60%+ win rate
    4. Then live trading is unlocked
    """

    def __init__(self, capital: float = 100000, market: str = "NSE"):
        self.capital = capital
        self.market = market
        self.costs = COSTS_INDIA if market == "NSE" else COSTS_US
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs("data/scalping", exist_ok=True)

    def can_start_paper_trading(self, backtest_result: ScalpBacktestResult) -> Dict[str, Any]:
        """Check if backtest passed validation gate to allow paper trading"""
        if backtest_result.validation_passed:
            return {
                "allowed": True,
                "message": "Backtest passed validation gate. Paper trading unlocked.",
                "details": backtest_result.validation_details
            }
        else:
            return {
                "allowed": False,
                "message": "Backtest did not pass validation gate. Optimize strategy first.",
                "details": backtest_result.validation_details,
                "what_failed": {
                    k: v for k, v in backtest_result.validation_details.items()
                    if "check" in k
                }
            }

    def run_paper_session(
        self,
        strategy,
        symbol: str,
        mode: str = "conservative",
        days: int = 1
    ) -> Dict[str, Any]:
        """
        Run a paper trading session using recent real data.
        Uses last N days of 1-min data to simulate live trading.

        Args:
            strategy: Scalping strategy instance
            symbol: Stock symbol
            mode: conservative or aggressive
            days: Number of days to simulate (max 7 for 1m data)

        Returns:
            Session results with performance metrics
        """
        config = CONSERVATIVE if mode == "conservative" else AGGRESSIVE

        # Fetch recent data (yfinance allows last 7 days for 1m)
        period = f"{min(days, 7)}d"
        logger.info(f"Starting paper trading session: {symbol} | {strategy.name} | {period}")

        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1m")

        if data.empty:
            return {"error": f"No data available for {symbol}"}

        # Generate signals
        signals_df = strategy.generate_signals(data)

        # Simulate paper trades
        session_id = f"{symbol}_{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = PaperTradingSession(
            session_id=session_id,
            symbol=symbol,
            strategy_name=strategy.name,
            mode=mode,
            start_date=datetime.now().isoformat(),
            capital=self.capital
        )

        trades = self._simulate_paper_trades(signals_df, symbol, strategy.name, config)
        session.trades = [self._trade_to_dict(t) for t in trades]

        # Calculate session stats
        stats = self._calculate_session_stats(trades, session_id)
        session.daily_stats = stats['daily_breakdown']

        # Check if live trading should be unlocked
        session.is_live_unlocked = self._check_live_unlock(session_id)

        # Save session
        self._save_session(session)

        return {
            "session_id": session_id,
            "symbol": symbol,
            "strategy": strategy.name,
            "mode": mode,
            "period": period,
            "stats": stats,
            "live_trading_status": self._get_live_trading_status(session_id),
            "trades": session.trades
        }

    def _simulate_paper_trades(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_name: str,
        config: Dict
    ) -> List[ScalpTrade]:
        """Simulate paper trades from signals with realistic execution."""
        trades = []
        trade_id = 0
        capital = self.capital
        open_trades: List[ScalpTrade] = []   # support up to 5 simultaneous positions
        daily_count = 0
        current_day = None

        MAX_OPEN_POSITIONS = 5

        for timestamp, row in df.iterrows():
            trade_day = timestamp.date() if hasattr(timestamp, 'date') else timestamp
            if trade_day != current_day:
                current_day = trade_day
                daily_count = 0

            # Check exits for all open trades
            still_open = []
            for open_trade in open_trades:
                exit_price = None
                exit_reason = ""

                if open_trade.side == 'BUY':
                    if row['Low'] <= open_trade.stop_loss:
                        exit_price = open_trade.stop_loss
                        exit_reason = "STOP_LOSS"
                    elif row['High'] >= open_trade.take_profit:
                        exit_price = open_trade.take_profit
                        exit_reason = "TAKE_PROFIT"
                else:
                    if row['High'] >= open_trade.stop_loss:
                        exit_price = open_trade.stop_loss
                        exit_reason = "STOP_LOSS"
                    elif row['Low'] <= open_trade.take_profit:
                        exit_price = open_trade.take_profit
                        exit_reason = "TAKE_PROFIT"

                if exit_price is not None:
                    open_trade = self._close_paper_trade(open_trade, exit_price, timestamp, exit_reason)
                    capital += open_trade.net_pnl
                    trades.append(open_trade)
                    self._persist_stats(trades)
                else:
                    still_open.append(open_trade)

            open_trades = still_open

            # New entry — enforce max 5 simultaneous positions
            if (len(open_trades) < MAX_OPEN_POSITIONS
                    and row['signal'] in ['BUY', 'SELL']
                    and daily_count < config['max_trades_per_day']
                    and capital > 0
                    and not pd.isna(row.get('entry_price', float('nan')))
                    and not pd.isna(row.get('stop_loss', float('nan')))
                    and not pd.isna(row.get('take_profit', float('nan')))):

                raw_entry = float(row['entry_price'])

                # Random slippage 0.01% – 0.05%
                slippage_pct = np.random.uniform(0.0001, 0.0005)
                if row['signal'] == 'BUY':
                    entry_price = raw_entry * (1 + slippage_pct)
                else:
                    entry_price = raw_entry * (1 - slippage_pct)

                # Partial fill: if order > 1% of avg volume, fill at 70%
                avg_vol = df['Volume'].rolling(20).mean().loc[timestamp] if 'Volume' in df.columns else None
                base_qty = max(1, int(capital * config['max_risk_per_trade_pct'] / entry_price))
                if avg_vol and not np.isnan(avg_vol) and avg_vol > 0:
                    order_value = base_qty * entry_price
                    if order_value > 0.01 * avg_vol * entry_price:
                        base_qty = max(1, int(base_qty * 0.70))
                quantity = base_qty

                cost = self._calc_cost(entry_price, quantity) * 2

                open_trade = ScalpTrade(
                    id=trade_id,
                    symbol=symbol,
                    entry_time=timestamp,
                    exit_time=None,
                    entry_price=entry_price,
                    exit_price=None,
                    stop_loss=float(row['stop_loss']),
                    take_profit=float(row['take_profit']),
                    side=row['signal'],
                    quantity=quantity,
                    signal_score=float(row['signal_score']),
                    strategy_name=strategy_name,
                    status='OPEN',
                    transaction_cost=cost
                )
                open_trades.append(open_trade)
                trade_id += 1
                daily_count += 1

        # Close all remaining open trades at end of data
        last_price = float(df['Close'].iloc[-1])
        for open_trade in open_trades:
            open_trade = self._close_paper_trade(open_trade, last_price, df.index[-1], "END_OF_DAY")
            trades.append(open_trade)
        if open_trades:
            self._persist_stats(trades)

        return trades

    def _close_paper_trade(self, trade, exit_price, exit_time, exit_reason):
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason

        if trade.side == 'BUY':
            trade.gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.gross_pnl = (trade.entry_price - exit_price) * trade.quantity

        trade.net_pnl = trade.gross_pnl - trade.transaction_cost
        trade.pnl_pct = trade.net_pnl / (trade.entry_price * trade.quantity)
        trade.status = 'WIN' if trade.net_pnl > 0 else ('STOPPED' if exit_reason == 'STOP_LOSS' else 'LOSS')
        return trade

    def _calc_cost(self, price: float, quantity: int) -> float:
        trade_value = price * quantity
        if self.market == "NSE":
            return (self.costs['brokerage_per_trade']
                    + trade_value * self.costs['stt_pct']
                    + trade_value * self.costs['slippage_pct'])
        return trade_value * self.costs['slippage_pct']

    def _persist_stats(self, trades: List[ScalpTrade]) -> None:
        """Write a lightweight stats snapshot to paper_stats.json after every trade close."""
        closed = [t for t in trades if t.status != 'OPEN']
        wins = [t for t in closed if t.status == 'WIN']
        net_pnl = sum(t.net_pnl for t in closed)
        win_rate = len(wins) / len(closed) if closed else 0.0

        stats = {
            "updated_at": datetime.now().isoformat(),
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "win_rate": round(win_rate, 4),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(net_pnl / self.capital, 4),
        }
        os.makedirs("data/scalping", exist_ok=True)
        with open(PAPER_STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)

    def _calculate_session_stats(self, trades: List[ScalpTrade], session_id: str) -> Dict:
        closed = [t for t in trades if t.status != 'OPEN']
        if not closed:
            return {"total_trades": 0, "win_rate": 0, "net_pnl": 0, "daily_breakdown": []}

        wins = [t for t in closed if t.status == 'WIN']
        win_rate = len(wins) / len(closed)
        net_pnl = sum(t.net_pnl for t in closed)
        total_costs = sum(t.transaction_cost for t in closed)

        # Daily breakdown
        daily: Dict[str, Dict] = {}
        for t in closed:
            day = str(t.exit_time.date()) if t.exit_time else "unknown"
            if day not in daily:
                daily[day] = {"trades": 0, "wins": 0, "net_pnl": 0}
            daily[day]["trades"] += 1
            daily[day]["wins"] += 1 if t.status == 'WIN' else 0
            daily[day]["net_pnl"] += t.net_pnl

        daily_list = [
            {
                "date": d,
                "trades": v["trades"],
                "win_rate": v["wins"] / v["trades"] if v["trades"] > 0 else 0,
                "net_pnl": v["net_pnl"],
                "return_pct": v["net_pnl"] / self.capital
            }
            for d, v in daily.items()
        ]

        avg_daily_return = np.mean([d["return_pct"] for d in daily_list]) if daily_list else 0

        return {
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / self.capital,
            "avg_daily_return_pct": avg_daily_return,
            "total_costs": total_costs,
            "meets_conservative": win_rate >= CONSERVATIVE['win_rate_target'],
            "meets_aggressive": win_rate >= AGGRESSIVE['win_rate_target'],
            "daily_breakdown": daily_list
        }

    def _check_live_unlock(self, session_id: str) -> bool:
        """Check if enough paper trading history exists to unlock live trading"""
        all_sessions = self._load_all_sessions()
        if not all_sessions:
            return False

        all_trades = []
        session_days = set()
        for s in all_sessions:
            for t in s.get('trades', []):
                all_trades.append(t)
                if t.get('exit_time'):
                    try:
                        day = t['exit_time'][:10]
                        session_days.add(day)
                    except Exception:
                        pass

        if len(session_days) < VALIDATION_GATE['min_paper_trading_days']:
            return False

        if len(all_trades) < VALIDATION_GATE['min_trades_to_validate']:
            return False

        wins = sum(1 for t in all_trades if t.get('status') == 'WIN')
        win_rate = wins / len(all_trades) if all_trades else 0

        return win_rate >= VALIDATION_GATE['min_win_rate']

    def _get_live_trading_status(self, session_id: str) -> Dict[str, Any]:
        """Get current live trading unlock status"""
        all_sessions = self._load_all_sessions()
        all_trades = []
        session_days = set()

        for s in all_sessions:
            for t in s.get('trades', []):
                all_trades.append(t)
                if t.get('exit_time'):
                    try:
                        session_days.add(t['exit_time'][:10])
                    except Exception:
                        pass

        total_trades = len(all_trades)
        wins = sum(1 for t in all_trades if t.get('status') == 'WIN')
        win_rate = wins / total_trades if total_trades > 0 else 0
        days_traded = len(session_days)

        days_needed = max(0, VALIDATION_GATE['min_paper_trading_days'] - days_traded)
        trades_needed = max(0, VALIDATION_GATE['min_trades_to_validate'] - total_trades)

        unlocked = (
            days_traded >= VALIDATION_GATE['min_paper_trading_days']
            and total_trades >= VALIDATION_GATE['min_trades_to_validate']
            and win_rate >= VALIDATION_GATE['min_win_rate']
        )

        return {
            "live_trading_unlocked": unlocked,
            "days_traded": days_traded,
            "days_required": VALIDATION_GATE['min_paper_trading_days'],
            "days_remaining": days_needed,
            "total_paper_trades": total_trades,
            "trades_required": VALIDATION_GATE['min_trades_to_validate'],
            "trades_remaining": trades_needed,
            "current_win_rate": f"{win_rate:.1%}",
            "required_win_rate": f"{VALIDATION_GATE['min_win_rate']:.0%}",
            "message": (
                "Live trading UNLOCKED. Strategy validated!"
                if unlocked
                else f"Keep paper trading. {days_needed} more days and {trades_needed} more trades needed."
            )
        }

    def _trade_to_dict(self, trade: ScalpTrade) -> Dict:
        return {
            "id": trade.id,
            "symbol": trade.symbol,
            "entry_time": str(trade.entry_time),
            "exit_time": str(trade.exit_time) if trade.exit_time else None,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "side": trade.side,
            "quantity": trade.quantity,
            "status": trade.status,
            "gross_pnl": round(trade.gross_pnl, 2),
            "net_pnl": round(trade.net_pnl, 2),
            "pnl_pct": round(trade.pnl_pct, 5),
            "exit_reason": trade.exit_reason,
            "signal_score": trade.signal_score
        }

    def _save_session(self, session: PaperTradingSession):
        sessions = self._load_all_sessions()
        sessions.append({
            "session_id": session.session_id,
            "symbol": session.symbol,
            "strategy_name": session.strategy_name,
            "mode": session.mode,
            "start_date": session.start_date,
            "capital": session.capital,
            "trades": session.trades,
            "daily_stats": session.daily_stats
        })
        with open(PAPER_TRADES_FILE, 'w') as f:
            json.dump(sessions, f, indent=2)

    def _load_all_sessions(self) -> List[Dict]:
        if os.path.exists(PAPER_TRADES_FILE):
            try:
                with open(PAPER_TRADES_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def get_cumulative_stats(self) -> Dict[str, Any]:
        """Get cumulative paper trading performance across all sessions"""
        return self._get_live_trading_status("all")
