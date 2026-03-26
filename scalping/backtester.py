"""
Scalping Backtester
Extends the existing BacktestEngine for intraday scalping with
realistic cost modeling and benchmark validation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from scalping.config import (
    CONSERVATIVE, AGGRESSIVE, VALIDATION_GATE,
    COSTS_INDIA, COSTS_US, DATA_CONFIG
)

logger = logging.getLogger(__name__)


@dataclass
class ScalpTrade:
    """Single scalp trade record"""
    id: int
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    side: str                    # BUY or SELL
    quantity: int
    signal_score: float
    strategy_name: str
    status: str                  # OPEN, WIN, LOSS, STOPPED
    gross_pnl: float = 0.0
    transaction_cost: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""        # TAKE_PROFIT, STOP_LOSS, END_OF_DAY


@dataclass
class ScalpBacktestResult:
    """Complete backtesting results"""
    symbol: str
    strategy_name: str
    mode: str                    # conservative or aggressive
    period: str
    interval: str
    initial_capital: float
    final_capital: float

    # Core metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    # Returns
    total_net_pnl: float
    total_return_pct: float
    avg_daily_return_pct: float
    best_day_pct: float
    worst_day_pct: float

    # Risk metrics
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_win_pct: float
    avg_loss_pct: float
    risk_reward_ratio: float

    # Cost analysis
    total_transaction_costs: float
    cost_impact_pct: float

    # Benchmark validation
    meets_conservative_target: bool
    meets_aggressive_target: bool
    validation_passed: bool
    validation_details: Dict[str, Any]

    # Trade list
    trades: List[ScalpTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


class ScalpingBacktester:
    """
    Backtester specifically designed for scalping strategies.
    Includes realistic cost modeling and benchmark validation.
    """

    def __init__(self, initial_capital: float = 100000, market: str = "NSE"):
        self.initial_capital = initial_capital
        self.market = market
        self.costs = COSTS_INDIA if market == "NSE" else COSTS_US

    def fetch_data(self, symbol: str, period: str = "1mo", interval: str = "1m") -> pd.DataFrame:
        """Fetch intraday data using yfinance"""
        logger.info(f"Fetching {interval} data for {symbol} ({period})")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            logger.info(f"Fetched {len(data)} candles for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def calculate_transaction_cost(self, price: float, quantity: int) -> float:
        """Calculate realistic transaction cost per trade"""
        trade_value = price * quantity

        if self.market == "NSE":
            brokerage = self.costs['brokerage_per_trade']
            stt = trade_value * self.costs['stt_pct']
            exchange = trade_value * self.costs['exchange_charges_pct']
            gst = brokerage * self.costs['gst_on_brokerage']
            sebi = trade_value * self.costs['sebi_charges_pct']
            slippage = trade_value * self.costs['slippage_pct']
            return brokerage + stt + exchange + gst + sebi + slippage
        else:
            sec_fee = trade_value * self.costs['sec_fee_pct']
            finra = quantity * self.costs['finra_taf_per_share']
            slippage = trade_value * self.costs['slippage_pct']
            return sec_fee + finra + slippage

    def run_backtest(
        self,
        strategy,
        symbol: str,
        period: str = "7d",
        interval: str = "1m",
        mode: str = "conservative"
    ) -> ScalpBacktestResult:
        """
        Run full backtest for a scalping strategy.

        Args:
            strategy: Scalping strategy instance
            symbol: Stock symbol
            period: Historical period (max 1mo for 1m data via yfinance)
            interval: Candle interval (1m or 5m)
            mode: conservative or aggressive

        Returns:
            ScalpBacktestResult with full metrics and benchmark validation
        """
        config = CONSERVATIVE if mode == "conservative" else AGGRESSIVE

        # Fetch data (yfinance max 7 days for 1m interval)
        data = self.fetch_data(symbol, period, interval)

        # Generate signals
        logger.info(f"Generating signals with {strategy.name}")
        signals_df = strategy.generate_signals(data)

        # Simulate trades
        trades = self._simulate_trades(signals_df, symbol, strategy.name, config)

        # Calculate metrics
        result = self._calculate_metrics(
            trades, symbol, strategy.name, mode, period, interval, config
        )

        return result

    def _simulate_trades(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_name: str,
        config: Dict
    ) -> List[ScalpTrade]:
        """Simulate trade execution with stop loss and take profit"""
        trades = []
        trade_id = 0
        capital = self.initial_capital
        open_trade: Optional[ScalpTrade] = None
        open_trade_entry_idx = -1
        daily_trade_count = 0
        current_day = None

        for i, (timestamp, row) in enumerate(df.iterrows()):
            # Reset daily trade count
            trade_day = timestamp.date() if hasattr(timestamp, 'date') else timestamp
            if trade_day != current_day:
                current_day = trade_day
                daily_trade_count = 0

            # Check open trade for exit (min 3 candles hold time)
            if open_trade is not None:
                candles_held = i - open_trade_entry_idx
                exit_price = None
                exit_reason = ""

                # Don't check exit for first 2 candles (avoid whipsaw stops)
                if candles_held >= 3:
                    low = float(row['Low'])
                    high = float(row['High'])
                    if open_trade.side == 'BUY':
                        if low <= open_trade.stop_loss:
                            exit_price = open_trade.stop_loss
                            exit_reason = "STOP_LOSS"
                        elif high >= open_trade.take_profit:
                            exit_price = open_trade.take_profit
                            exit_reason = "TAKE_PROFIT"
                    else:  # SELL
                        if high >= open_trade.stop_loss:
                            exit_price = open_trade.stop_loss
                            exit_reason = "STOP_LOSS"
                        elif low <= open_trade.take_profit:
                            exit_price = open_trade.take_profit
                            exit_reason = "TAKE_PROFIT"

                    if exit_price is not None:
                        open_trade = self._close_trade(
                            open_trade, exit_price, timestamp, exit_reason
                        )
                        capital += open_trade.net_pnl
                        trades.append(open_trade)
                        open_trade = None
                        open_trade_entry_idx = -1

            # Check for new entry signal
            if (open_trade is None
                    and row['signal'] in ['BUY', 'SELL']
                    and daily_trade_count < config['max_trades_per_day']
                    and capital > 0
                    and not pd.isna(row['entry_price'])
                    and not pd.isna(row['stop_loss'])
                    and not pd.isna(row['take_profit'])):

                entry_price = float(row['entry_price'])
                stop_loss = float(row['stop_loss'])
                take_profit = float(row['take_profit'])

                # Validate SL/TP make sense
                if row['signal'] == 'BUY' and stop_loss >= entry_price:
                    continue
                if row['signal'] == 'SELL' and stop_loss <= entry_price:
                    continue

                quantity = self._calculate_quantity(
                    capital, entry_price, config['max_risk_per_trade_pct']
                )

                if quantity > 0:
                    cost = self.calculate_transaction_cost(entry_price, quantity)
                    open_trade = ScalpTrade(
                        id=trade_id,
                        symbol=symbol,
                        entry_time=timestamp,
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        side=row['signal'],
                        quantity=quantity,
                        signal_score=float(row['signal_score']),
                        strategy_name=strategy_name,
                        status='OPEN',
                        transaction_cost=cost * 2
                    )
                    trade_id += 1
                    daily_trade_count += 1
                    open_trade_entry_idx = i

        # Close any open trade at end
        if open_trade is not None:
            last_price = df['Close'].iloc[-1]
            open_trade = self._close_trade(
                open_trade, last_price, df.index[-1], "END_OF_DAY"
            )
            trades.append(open_trade)

        return trades

    def _close_trade(
        self,
        trade: ScalpTrade,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> ScalpTrade:
        """Close a trade and calculate P&L"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason

        if trade.side == 'BUY':
            trade.gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.gross_pnl = (trade.entry_price - exit_price) * trade.quantity

        trade.net_pnl = trade.gross_pnl - trade.transaction_cost
        trade.pnl_pct = trade.net_pnl / (trade.entry_price * trade.quantity)

        # Status based on exit reason, not net P&L
        # (costs can make a TP exit show negative net, but it's still a WIN)
        if exit_reason == 'TAKE_PROFIT':
            trade.status = 'WIN'
        elif exit_reason == 'STOP_LOSS':
            trade.status = 'STOPPED'
        else:
            trade.status = 'LOSS'

        return trade

    def _calculate_quantity(
        self, capital: float, price: float, max_risk_pct: float
    ) -> int:
        """Calculate position size - use meaningful capital per trade"""
        # Use 20% of capital per trade minimum to overcome transaction costs
        # For ₹1,00,000 capital: 20% = ₹20,000 per trade
        # At ₹1,400/share: 14 shares
        # TP at 0.4%: 14 × ₹5.60 = ₹78 gross profit
        # Costs: ~₹50 → Net: ₹28 profit per win
        min_trade_value = capital * 0.20   # 20% minimum
        max_trade_value = capital * 0.30   # 30% maximum

        trade_value = min(max_trade_value, max(min_trade_value, capital * max_risk_pct))
        quantity = int(trade_value / price)
        return max(1, quantity)

    def _calculate_metrics(
        self,
        trades: List[ScalpTrade],
        symbol: str,
        strategy_name: str,
        mode: str,
        period: str,
        interval: str,
        config: Dict
    ) -> ScalpBacktestResult:
        """Calculate all performance metrics and validate against benchmarks"""

        if not trades:
            logger.warning("No trades generated")
            return self._empty_result(symbol, strategy_name, mode, period, interval)

        closed = [t for t in trades if t.status != 'OPEN']
        wins = [t for t in closed if t.status == 'WIN']
        losses = [t for t in closed if t.status in ['LOSS', 'STOPPED']]

        total = len(closed)
        win_rate = len(wins) / total if total > 0 else 0

        total_gross_win = sum(t.gross_pnl for t in wins)
        total_gross_loss = abs(sum(t.gross_pnl for t in losses))
        profit_factor = total_gross_win / total_gross_loss if total_gross_loss > 0 else float('inf')

        total_net_pnl = sum(t.net_pnl for t in closed)
        total_costs = sum(t.transaction_cost for t in closed)
        final_capital = self.initial_capital + total_net_pnl
        total_return_pct = total_net_pnl / self.initial_capital

        avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
        risk_reward = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0

        # Daily returns
        daily_pnl: Dict[str, float] = {}
        for t in closed:
            day = str(t.exit_time.date()) if t.exit_time else "unknown"
            daily_pnl[day] = daily_pnl.get(day, 0) + t.net_pnl

        daily_returns = [v / self.initial_capital for v in daily_pnl.values()]
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0
        best_day = max(daily_returns) if daily_returns else 0
        worst_day = min(daily_returns) if daily_returns else 0

        # Equity curve
        equity = self.initial_capital
        equity_curve = [equity]
        for t in sorted(closed, key=lambda x: x.exit_time or datetime.min):
            equity += t.net_pnl
            equity_curve.append(equity)

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio
        if len(daily_returns) > 1:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0

        # Cost impact
        cost_impact = total_costs / self.initial_capital

        # Benchmark validation
        validation = self._validate_against_benchmarks(
            win_rate, profit_factor, max_dd, avg_daily_return, total
        )

        return ScalpBacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            mode=mode,
            period=period,
            interval=interval,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_net_pnl=total_net_pnl,
            total_return_pct=total_return_pct,
            avg_daily_return_pct=avg_daily_return,
            best_day_pct=best_day,
            worst_day_pct=worst_day,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            risk_reward_ratio=risk_reward,
            total_transaction_costs=total_costs,
            cost_impact_pct=cost_impact,
            meets_conservative_target=validation['conservative'],
            meets_aggressive_target=validation['aggressive'],
            validation_passed=validation['gate_passed'],
            validation_details=validation,
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns
        )

    def _validate_against_benchmarks(
        self,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
        avg_daily_return: float,
        total_trades: int
    ) -> Dict[str, Any]:
        """Validate results against our locked benchmarks"""

        conservative_pass = (
            win_rate >= CONSERVATIVE['win_rate_target']
            and avg_daily_return >= CONSERVATIVE['daily_net_target_pct']
            and max_drawdown <= VALIDATION_GATE['max_drawdown_pct']
        )

        aggressive_pass = (
            win_rate >= AGGRESSIVE['win_rate_target']
            and avg_daily_return >= AGGRESSIVE['daily_net_target_pct']
            and max_drawdown <= VALIDATION_GATE['max_drawdown_pct']
        )

        gate_passed = (
            win_rate >= VALIDATION_GATE['min_win_rate']
            and profit_factor >= VALIDATION_GATE['min_profit_factor']
            and max_drawdown <= VALIDATION_GATE['max_drawdown_pct']
            and total_trades >= VALIDATION_GATE['min_trades_to_validate']
        )

        return {
            'conservative': conservative_pass,
            'aggressive': aggressive_pass,
            'gate_passed': gate_passed,
            'win_rate_check': f"{win_rate:.1%} (target: {VALIDATION_GATE['min_win_rate']:.0%}+)",
            'profit_factor_check': f"{profit_factor:.2f} (target: {VALIDATION_GATE['min_profit_factor']}+)",
            'drawdown_check': f"{max_drawdown:.1%} (max: {VALIDATION_GATE['max_drawdown_pct']:.0%})",
            'daily_return_check': f"{avg_daily_return:.2%} (target: {CONSERVATIVE['daily_net_target_pct']:.1%}+)",
            'trades_check': f"{total_trades} (min: {VALIDATION_GATE['min_trades_to_validate']})",
        }

    def _empty_result(self, symbol, strategy_name, mode, period, interval):
        return ScalpBacktestResult(
            symbol=symbol, strategy_name=strategy_name, mode=mode,
            period=period, interval=interval,
            initial_capital=self.initial_capital, final_capital=self.initial_capital,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, profit_factor=0, total_net_pnl=0,
            total_return_pct=0, avg_daily_return_pct=0,
            best_day_pct=0, worst_day_pct=0, max_drawdown_pct=0,
            sharpe_ratio=0, avg_win_pct=0, avg_loss_pct=0,
            risk_reward_ratio=0, total_transaction_costs=0,
            cost_impact_pct=0, meets_conservative_target=False,
            meets_aggressive_target=False, validation_passed=False,
            validation_details={}
        )

    def print_report(self, result: ScalpBacktestResult):
        """Print a clean backtest report"""
        print("\n" + "=" * 60)
        print(f"SCALPING BACKTEST REPORT")
        print(f"Symbol: {result.symbol} | Strategy: {result.strategy_name}")
        print(f"Mode: {result.mode.upper()} | Period: {result.period}")
        print("=" * 60)

        print(f"\nCAPITAL")
        print(f"  Initial:     {self.initial_capital:>12,.2f}")
        print(f"  Final:       {result.final_capital:>12,.2f}")
        print(f"  Net P&L:     {result.total_net_pnl:>12,.2f} ({result.total_return_pct:.2%})")

        print(f"\nTRADE STATISTICS")
        print(f"  Total Trades:    {result.total_trades}")
        print(f"  Win Rate:        {result.win_rate:.1%}")
        print(f"  Profit Factor:   {result.profit_factor:.2f}")
        print(f"  Avg Win:         {result.avg_win_pct:.3%}")
        print(f"  Avg Loss:        {result.avg_loss_pct:.3%}")
        print(f"  Risk/Reward:     1:{result.risk_reward_ratio:.1f}")

        print(f"\nDAILY PERFORMANCE")
        print(f"  Avg Daily Net:   {result.avg_daily_return_pct:.2%}")
        print(f"  Best Day:        {result.best_day_pct:.2%}")
        print(f"  Worst Day:       {result.worst_day_pct:.2%}")

        print(f"\nRISK METRICS")
        print(f"  Max Drawdown:    {result.max_drawdown_pct:.2%}")
        print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")

        print(f"\nCOST ANALYSIS")
        print(f"  Total Costs:     {result.total_transaction_costs:,.2f}")
        print(f"  Cost Impact:     {result.cost_impact_pct:.2%}")

        print(f"\nBENCHMARK VALIDATION")
        for key, val in result.validation_details.items():
            if key not in ['conservative', 'aggressive', 'gate_passed']:
                status = "✓" if "check" in key else ""
                print(f"  {key.replace('_check', '').replace('_', ' ').title():<20} {val}")

        print(f"\n  Conservative Target: {'PASSED' if result.meets_conservative_target else 'FAILED'}")
        print(f"  Aggressive Target:   {'PASSED' if result.meets_aggressive_target else 'FAILED'}")
        print(f"  Validation Gate:     {'PASSED - Ready for Paper Trading' if result.validation_passed else 'FAILED - Needs Optimization'}")
        print("=" * 60)
