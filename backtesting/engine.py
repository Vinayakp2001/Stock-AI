"""
Backtesting Engine
Simulates trading strategies on historical data and calculates performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str  # 'BUY' or 'SELL'
    status: str  # 'OPEN', 'CLOSED', 'CANCELLED'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

@dataclass
class BacktestResult:
    """Results from backtesting a strategy"""
    symbol: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series

class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from data"""
        pass

class MovingAverageStrategy(Strategy):
    """Simple moving average crossover strategy"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(f"MA_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on moving average crossover"""
        signals = pd.Series(index=data.index, data=0)
        
        # Calculate moving averages
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals[short_ma > long_ma] = 1  # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        return signals

class RSIStrategy(Strategy):
    """RSI-based trading strategy"""
    
    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"RSI_{rsi_period}_{oversold}_{overbought}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI"""
        signals = pd.Series(index=data.index, data=0)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals[rsi < self.oversold] = 1  # Buy signal
        signals[rsi > self.overbought] = -1  # Sell signal
        
        return signals

class MACDStrategy(Strategy):
    """MACD-based trading strategy"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MACD"""
        signals = pd.Series(index=data.index, data=0)
        
        # Calculate MACD
        ema_fast = data['Close'].ewm(span=self.fast_period).mean()
        ema_slow = data['Close'].ewm(span=self.slow_period).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_period).mean()
        
        # Generate signals
        signals[macd > signal] = 1  # Buy signal
        signals[macd < signal] = -1  # Sell signal
        
        return signals

class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run_backtest(self, strategy: Strategy, data: pd.DataFrame, 
                    symbol: str, start_date: str = None, end_date: str = None) -> BacktestResult:
        """
        Run backtest for a strategy
        
        Args:
            strategy: Trading strategy to test
            data: Historical price data
            symbol: Stock symbol
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
        
        Returns:
            BacktestResult object with detailed performance metrics
        """
        logger.info(f"Running backtest for {strategy.name} on {symbol}")
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if data.empty:
            raise ValueError("No data available for the specified date range")
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        # Simulate trading
        for i, (date, row) in enumerate(data.iterrows()):
            signal = signals.iloc[i]
            price = row['Close']
            
            # Execute trades based on signals
            if signal == 1 and position <= 0:  # Buy signal
                if position < 0:  # Close short position
                    exit_trade = trades[-1]
                    exit_trade.exit_date = date
                    exit_trade.exit_price = price
                    exit_trade.status = 'CLOSED'
                    exit_trade.pnl = (exit_trade.entry_price - price) * abs(exit_trade.quantity)
                    exit_trade.pnl_pct = (exit_trade.entry_price - price) / exit_trade.entry_price
                    capital += exit_trade.pnl
                
                # Open long position
                quantity = int(capital * 0.95 / price)  # Use 95% of capital
                if quantity > 0:
                    trade = Trade(
                        symbol=symbol,
                        entry_date=date,
                        exit_date=None,
                        entry_price=price,
                        exit_price=None,
                        quantity=quantity,
                        side='BUY',
                        status='OPEN'
                    )
                    trades.append(trade)
                    position = quantity
                    capital -= quantity * price * (1 + self.commission)
            
            elif signal == -1 and position >= 0:  # Sell signal
                if position > 0:  # Close long position
                    exit_trade = trades[-1]
                    exit_trade.exit_date = date
                    exit_trade.exit_price = price
                    exit_trade.status = 'CLOSED'
                    exit_trade.pnl = (price - exit_trade.entry_price) * exit_trade.quantity
                    exit_trade.pnl_pct = (price - exit_trade.entry_price) / exit_trade.entry_price
                    capital += exit_trade.pnl
                    position = 0
            
            # Calculate current equity
            current_equity = capital + (position * price)
            equity_curve.append(current_equity)
        
        # Close any open positions at the end
        if position > 0:
            final_price = data['Close'].iloc[-1]
            exit_trade = trades[-1]
            exit_trade.exit_date = data.index[-1]
            exit_trade.exit_price = final_price
            exit_trade.status = 'CLOSED'
            exit_trade.pnl = (final_price - exit_trade.entry_price) * exit_trade.quantity
            exit_trade.pnl_pct = (final_price - exit_trade.entry_price) / exit_trade.entry_price
            capital += exit_trade.pnl
        
        # Calculate performance metrics
        final_capital = capital
        total_return = final_capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Calculate equity curve and returns
        equity_series = pd.Series(equity_curve, index=data.index)
        daily_returns = equity_series.pct_change().dropna()
        
        # Calculate additional metrics
        annualized_return = self._calculate_annualized_return(total_return_pct, data.index[0], data.index[-1])
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(equity_series)
        win_rate, winning_trades, losing_trades, avg_win, avg_loss, profit_factor = self._calculate_trade_metrics(trades)
        
        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            total_trades=len([t for t in trades if t.status == 'CLOSED']),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_series,
            daily_returns=daily_returns
        )
    
    def _calculate_annualized_return(self, total_return_pct: float, start_date: datetime, end_date: datetime) -> float:
        """Calculate annualized return"""
        days = (end_date - start_date).days
        if days == 0:
            return 0
        return ((1 + total_return_pct) ** (365 / days)) - 1
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty:
            return 0
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, float]:
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown_pct = drawdown.min()
        max_drawdown = max_drawdown_pct * equity_curve.iloc[0]
        return max_drawdown, max_drawdown_pct
    
    def _calculate_trade_metrics(self, trades: List[Trade]) -> Tuple[float, int, int, float, float, float]:
        """Calculate trade-based metrics"""
        closed_trades = [t for t in trades if t.status == 'CLOSED']
        
        if not closed_trades:
            return 0, 0, 0, 0, 0, 0
        
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(closed_trades)
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return win_rate, len(winning_trades), len(losing_trades), avg_win, avg_loss, profit_factor
    
    def compare_strategies(self, strategies: List[Strategy], data: pd.DataFrame, 
                          symbol: str) -> Dict[str, BacktestResult]:
        """Compare multiple strategies"""
        results = {}
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, data, symbol)
                results[strategy.name] = result
            except Exception as e:
                logger.error(f"Error testing strategy {strategy.name}: {str(e)}")
                continue
        return results

# Example usage
if __name__ == "__main__":
    from agents.data_agent import DataAgent
    
    # Initialize agents
    data_agent = DataAgent()
    
    # Get data
    apple_data = data_agent.get_stock_data('AAPL', period='1y')
    data = apple_data.data
    
    # Create strategies
    ma_strategy = MovingAverageStrategy(20, 50)
    rsi_strategy = RSIStrategy(14, 30, 70)
    macd_strategy = MACDStrategy(12, 26, 9)
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=100000)
    
    # Run backtests
    strategies = [ma_strategy, rsi_strategy, macd_strategy]
    results = engine.compare_strategies(strategies, data, 'AAPL')
    
    # Print results
    for strategy_name, result in results.items():
        print(f"\n=== {strategy_name} ===")
        print(f"Total Return: {result.total_return_pct:.2%}")
        print(f"Annualized Return: {result.annualized_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown_pct:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Profit Factor: {result.profit_factor:.2f}") 