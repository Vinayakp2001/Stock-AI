"""
Base Strategy ABC and shared IndicatorUtils.
All scalping strategies inherit from BaseStrategy.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all scalping strategies."""

    def __init__(self, params: dict = None):
        self.params: dict = params or {}
        self.name: str = ""  # subclass must set

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to a copy of data and return it."""
        ...

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame with columns:
          signal (BUY/SELL/HOLD), signal_score (0-100),
          entry_price, stop_loss, take_profit
        """
        ...

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "",
            "timeframe": "1m",
            "params": self.params,
        }


class IndicatorUtils:
    """Static helpers for common indicator calculations."""

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Wilder RSI. Returns NaN-filled Series if input too short."""
        if series.empty or len(series) < period:
            return pd.Series(np.nan, index=series.index, dtype=float)
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """Daily-reset VWAP. Requires Close, High, Low, Volume columns."""
        if df.empty:
            return pd.Series(dtype=float)
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        # Group by calendar date for daily reset
        dates = df.index.date if hasattr(df.index, "date") else pd.Series(df.index).dt.date.values
        date_series = pd.Series(dates, index=df.index)
        cum_tp_vol = (typical * df["Volume"]).groupby(date_series).cumsum()
        cum_vol = df["Volume"].groupby(date_series).cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        if df.empty or len(df) < period:
            return pd.Series(np.nan, index=df.index, dtype=float)
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift()).abs()
        lc = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average."""
        if series.empty:
            return pd.Series(dtype=float)
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Volume / rolling mean volume."""
        if df.empty or "Volume" not in df.columns:
            return pd.Series(dtype=float)
        avg = df["Volume"].rolling(window).mean().replace(0, np.nan)
        return df["Volume"] / avg
