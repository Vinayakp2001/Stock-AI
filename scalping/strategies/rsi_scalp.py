"""
RSI Divergence Scalping Strategy
Uses RSI divergence on 5-min chart for high-probability entries
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scalping.config import SIGNAL_CONFIG
from scalping.strategies.base import BaseStrategy, IndicatorUtils


class RSIScalpStrategy(BaseStrategy):
    """
    RSI-based scalping strategy using divergence and extreme levels.

    Entry conditions (BUY):
    - RSI crosses above 35 from oversold territory
    - Bullish divergence: price makes lower low, RSI makes higher low
    - Volume confirmation
    - EMA 9 > EMA 21 (trend filter)

    Entry conditions (SELL):
    - RSI crosses below 65 from overbought territory
    - Bearish divergence: price makes higher high, RSI makes lower high
    - Volume confirmation
    - EMA 9 < EMA 21 (trend filter)
    """

    def __init__(self, rsi_period: int = 14, oversold: int = 35, overbought: int = 65, params: dict = None):
        super().__init__(params=params)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI_Scalp_{oversold}_{overbought}"

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # RSI
        df['rsi'] = IndicatorUtils.rsi(df['Close'], self.rsi_period)

        # RSI crossover signals
        df['rsi_cross_up'] = (df['rsi'] > self.oversold) & (df['rsi'].shift(1) <= self.oversold)
        df['rsi_cross_down'] = (df['rsi'] < self.overbought) & (df['rsi'].shift(1) >= self.overbought)

        # EMAs for trend filter
        df['ema_9'] = IndicatorUtils.ema(df['Close'], 9)
        df['ema_21'] = IndicatorUtils.ema(df['Close'], 21)
        df['trend_up'] = df['ema_9'] > df['ema_21']

        # VWAP
        df['vwap'] = IndicatorUtils.vwap(df)

        # Volume ratio
        df['volume_ratio'] = IndicatorUtils.volume_ratio(df, 20)

        # ATR
        df['atr'] = IndicatorUtils.atr(df, 14)

        # RSI divergence detection (simplified)
        df['price_lower_low'] = (df['Close'] < df['Close'].shift(5)) & (df['Close'].shift(5) < df['Close'].shift(10))
        df['rsi_higher_low'] = (df['rsi'] > df['rsi'].shift(5)) & (df['rsi'].shift(5) > df['rsi'].shift(10))
        df['bullish_divergence'] = df['price_lower_low'] & df['rsi_higher_low']

        df['price_higher_high'] = (df['Close'] > df['Close'].shift(5)) & (df['Close'].shift(5) > df['Close'].shift(10))
        df['rsi_lower_high'] = (df['rsi'] < df['rsi'].shift(5)) & (df['rsi'].shift(5) < df['rsi'].shift(10))
        df['bearish_divergence'] = df['price_higher_high'] & df['rsi_lower_high']

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        df['signal'] = 'HOLD'
        df['signal_score'] = 0
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan

        for i in range(10, len(df)):
            row = df.iloc[i]
            score = 0

            # ── BUY: RSI crosses up from oversold ───────────────────────────
            if row['rsi_cross_up']:
                score += 35

                if row['trend_up']:
                    score += 20  # Trend confirmation

                if row['volume_ratio'] >= SIGNAL_CONFIG['min_volume_ratio']:
                    score += 20

                if row['bullish_divergence']:
                    score += 15  # Divergence bonus

                if row['Close'] > row['vwap']:
                    score += 10

                if score >= SIGNAL_CONFIG['min_signal_score']:
                    atr = row['atr'] if row['atr'] > 0 else row['Close'] * 0.002
                    sl_distance = max(atr * 1.5, row['Close'] * 0.0015)
                    tp_distance = sl_distance * 2.0
                    df.iloc[i, df.columns.get_loc('signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('signal_score')] = score
                    df.iloc[i, df.columns.get_loc('entry_price')] = row['Close']
                    df.iloc[i, df.columns.get_loc('stop_loss')] = row['Close'] - sl_distance
                    df.iloc[i, df.columns.get_loc('take_profit')] = row['Close'] + tp_distance

            # ── SELL: RSI crosses down from overbought ───────────────────────
            elif row['rsi_cross_down']:
                score += 35

                if not row['trend_up']:
                    score += 20

                if row['volume_ratio'] >= SIGNAL_CONFIG['min_volume_ratio']:
                    score += 20

                if row['bearish_divergence']:
                    score += 15

                if row['Close'] < row['vwap']:
                    score += 10

                if score >= SIGNAL_CONFIG['min_signal_score']:
                    atr = row['atr'] if row['atr'] > 0 else row['Close'] * 0.002
                    sl_distance = max(atr * 1.5, row['Close'] * 0.0015)
                    tp_distance = sl_distance * 2.0
                    df.iloc[i, df.columns.get_loc('signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('signal_score')] = score
                    df.iloc[i, df.columns.get_loc('entry_price')] = row['Close']
                    df.iloc[i, df.columns.get_loc('stop_loss')] = row['Close'] + sl_distance
                    df.iloc[i, df.columns.get_loc('take_profit')] = row['Close'] - tp_distance

        return df

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "RSI Scalp",
            "timeframe": "5m",
            "confirmation": "EMA trend + Volume + VWAP + Divergence",
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "params": self.params,
        }
