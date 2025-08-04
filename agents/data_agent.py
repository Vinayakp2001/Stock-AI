"""
Data Collection Agent
Handles fetching stock data, calculating technical indicators, and data preprocessing
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Container for stock data and indicators"""
    symbol: str
    data: pd.DataFrame
    indicators: Dict[str, pd.Series]
    metadata: Dict[str, any]

class DataAgent:
    """Agent responsible for collecting and processing stock data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> StockData:
        """
        Fetch stock data and calculate technical indicators
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            StockData object containing price data and indicators
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < self.cache_duration:
                    logger.info(f"Using cached data for {symbol}")
                    return cached_data
            
            # Fetch data from Yahoo Finance
            logger.info(f"Fetching data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(data)
            
            # Get metadata
            metadata = self._get_metadata(ticker)
            
            # Create StockData object
            stock_data = StockData(
                symbol=symbol,
                data=data,
                indicators=indicators,
                metadata=metadata
            )
            
            # Cache the result
            self.cache[cache_key] = (stock_data, datetime.now())
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            # Moving averages
            indicators['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            indicators['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            indicators['sma_200'] = ta.trend.sma_indicator(data['Close'], window=200)
            indicators['ema_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            indicators['ema_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            
            # MACD
            macd = ta.trend.MACD(data['Close'])
            indicators['macd'] = macd.macd()
            indicators['macd_signal'] = macd.macd_signal()
            indicators['macd_histogram'] = macd.macd_diff()
            
            # RSI
            indicators['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            indicators['bb_upper'] = bb.bollinger_hband()
            indicators['bb_middle'] = bb.bollinger_mavg()
            indicators['bb_lower'] = bb.bollinger_lband()
            indicators['bb_width'] = bb.bollinger_wband()
            indicators['bb_percent'] = bb.bollinger_pband()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
            indicators['stoch_k'] = stoch.stoch()
            indicators['stoch_d'] = stoch.stoch_signal()
            
            # Volume indicators
            # indicators['volume_sma'] = ta.volume.volume_sma(data['Close'], data['Volume'])
            indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
            
            # ATR (Average True Range) for volatility
            indicators['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
            
            # Williams %R
            indicators['williams_r'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
            
            # Commodity Channel Index
            indicators['cci'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
            
            # Price patterns
            indicators['price_change'] = data['Close'].pct_change()
            indicators['price_change_5d'] = data['Close'].pct_change(periods=5)
            indicators['price_change_20d'] = data['Close'].pct_change(periods=20)
            
            # Volatility
            indicators['volatility'] = data['Close'].rolling(window=20).std()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
        
        return indicators
    
    def _get_metadata(self, ticker: yf.Ticker) -> Dict[str, any]:
        """Get stock metadata"""
        try:
            info = ticker.info
            return {
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'volume_avg': info.get('averageVolume', 0),
                'price_high_52w': info.get('fiftyTwoWeekHigh', 0),
                'price_low_52w': info.get('fiftyTwoWeekLow', 0),
            }
        except Exception as e:
            logger.warning(f"Error fetching metadata: {str(e)}")
            return {}
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, StockData]:
        """Fetch data for multiple stocks"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_stock_data(symbol, period)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                continue
        return results
    
    def get_market_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """Get market-wide data (S&P 500, NASDAQ, etc.)"""
        if symbols is None:
            symbols = ['^GSPC', '^IXIC', '^DJI']  # S&P 500, NASDAQ, Dow Jones
        
        market_data = {}
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period="1y")
                market_data[symbol] = data.data['Close']
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {str(e)}")
        
        return pd.DataFrame(market_data)
    
    def prepare_ml_data(self, stock_data: StockData, target_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning models with improved features
        
        Args:
            stock_data: StockData object
            target_days: Number of days to predict ahead
        
        Returns:
            Tuple of (features, target)
        """
        data = stock_data.data.copy()
        indicators = stock_data.indicators
        
        # Calculate price changes and returns
        data['price_change'] = data['Close'].pct_change()
        data['price_change_2d'] = data['Close'].pct_change(2)
        data['price_change_5d'] = data['Close'].pct_change(5)
        data['price_change_10d'] = data['Close'].pct_change(10)
        data['price_change_20d'] = data['Close'].pct_change(20)
        
        # Calculate volume features
        data['volume_change'] = data['Volume'].pct_change()
        data['volume_sma_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['volume_std_ratio'] = data['Volume'] / data['Volume'].rolling(20).std()
        
        # Calculate volatility
        data['volatility_5d'] = data['price_change'].rolling(5).std()
        data['volatility_10d'] = data['price_change'].rolling(10).std()
        data['volatility_20d'] = data['price_change'].rolling(20).std()
        
        # Calculate momentum features
        data['momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
        data['momentum_10d'] = data['Close'] / data['Close'].shift(10) - 1
        data['momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
        
        # Add key technical indicators
        key_indicators = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'stoch_k', 'stoch_d'
        ]
        
        for indicator in key_indicators:
            if indicator in indicators:
                data[f'indicator_{indicator}'] = indicators[indicator]
        
        # Calculate price relative to moving averages
        if 'sma_20' in indicators and 'sma_50' in indicators:
            data['price_vs_sma20'] = data['Close'] / indicators['sma_20'] - 1
            data['price_vs_sma50'] = data['Close'] / indicators['sma_50'] - 1
            data['sma_cross'] = (indicators['sma_20'] > indicators['sma_50']).astype(int)
            data['sma_distance'] = (indicators['sma_20'] - indicators['sma_50']) / indicators['sma_50']
        
        # Calculate RSI-based features
        if 'rsi' in indicators:
            data['rsi_oversold'] = (indicators['rsi'] < 30).astype(int)
            data['rsi_overbought'] = (indicators['rsi'] > 70).astype(int)
            data['rsi_trend'] = indicators['rsi'].rolling(5).mean() - indicators['rsi']
            data['rsi_momentum'] = indicators['rsi'] - indicators['rsi'].shift(5)
        
        # Calculate MACD-based features
        if 'macd' in indicators and 'macd_signal' in indicators:
            data['macd_cross'] = (indicators['macd'] > indicators['macd_signal']).astype(int)
            data['macd_strength'] = indicators['macd'] - indicators['macd_signal']
            data['macd_momentum'] = indicators['macd'] - indicators['macd'].shift(5)
        
        # Calculate Bollinger Bands features
        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            data['bb_position'] = (data['Close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            data['bb_squeeze'] = (indicators['bb_upper'] - indicators['bb_lower']) / data['Close']
            data['bb_breakout'] = ((data['Close'] > indicators['bb_upper']) | (data['Close'] < indicators['bb_lower'])).astype(int)
        
        # Create target variable (future price change)
        if target_days == 1:
            data['target'] = data['Close'].shift(-1) / data['Close'] - 1  # Next day return
        else:
            data['target'] = data['Close'].shift(-target_days) / data['Close'] - 1  # Future return
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Select features (exclude target and date columns)
        exclude_columns = ['target', 'Dividends', 'Stock Splits', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Ensure we have enough data
        if len(data) < 100:
            logger.warning(f"Not enough data for ML training: {len(data)} rows")
            return pd.DataFrame(), pd.Series()
        
        X = data[feature_columns]
        y = data['target']
        
        # Log feature information
        logger.info(f"Prepared ML data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target range: {y.min():.4f} to {y.max():.4f}")
        
        return X, y
    
    def get_support_resistance(self, stock_data: StockData, window: int = 20) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        data = stock_data.data
        
        # Find local minima and maxima
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        # Get recent levels
        recent_highs = highs.tail(50).dropna()
        recent_lows = lows.tail(50).dropna()
        
        # Calculate support and resistance
        resistance = recent_highs.quantile(0.8)  # 80th percentile of highs
        support = recent_lows.quantile(0.2)      # 20th percentile of lows
        
        return {
            'support': support,
            'resistance': resistance,
            'current_price': data['Close'].iloc[-1]
        }

# Example usage
if __name__ == "__main__":
    agent = DataAgent()
    
    # Get data for Apple
    apple_data = agent.get_stock_data('AAPL', period='6mo')
    print(f"Apple data shape: {apple_data.data.shape}")
    print(f"Indicators calculated: {len(apple_data.indicators)}")
    print(f"Current price: ${apple_data.data['Close'].iloc[-1]:.2f}")
    
    # Get support/resistance levels
    levels = agent.get_support_resistance(apple_data)
    print(f"Support: ${levels['support']:.2f}")
    print(f"Resistance: ${levels['resistance']:.2f}") 