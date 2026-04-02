"""
VWAP Bounce Scalping Strategy
Price bounces off VWAP with volume confirmation
One of the most reliable intraday scalping strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scalping.config import SIGNAL_CONFIG
from scalping.strategies.base import BaseStrategy, IndicatorUtils


class VWAPStrategy(BaseStrategy):
    """
    VWAP Bounce strategy.

    Entry conditions (BUY - bounce from below VWAP):
    - Price touches VWAP from below (within 0.1%)
    - Price starts moving back up
    - Volume spike on the bounce candle
    - RSI not overbought (<65)

    Entry conditions (SELL - rejection from above VWAP):
    - Price touches VWAP from above (within 0.1%)
    - Price starts moving back down
    - Volume spike on rejection candle
    - RSI not oversold (>35)
    """

    def __init__(self, deviation_pct: float = 0.001, params: dict = None):
        super().__init__(params=params)
        self.deviation_pct = deviation_pct
        self.name = "VWAP_Bounce"

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # VWAP - resets each day
        df['vwap'] = IndicatorUtils.vwap(df)

        # VWAP bands (standard deviation bands)
        df['vwap_std'] = df['Close'].rolling(20).std()
        df['vwap_upper'] = df['vwap'] + df['vwap_std']
        df['vwap_lower'] = df['vwap'] - df['vwap_std']

        # Distance from VWAP
        df['vwap_distance'] = (df['Close'] - df['vwap']) / df['vwap']

        # Price momentum (rate of change)
        df['roc'] = df['Close'].pct_change(3)

        # RSI
        df['rsi'] = IndicatorUtils.rsi(df['Close'], 14)

        # Volume ratio
        df['volume_ratio'] = IndicatorUtils.volume_ratio(df, 20)

        # ATR
        df['atr'] = IndicatorUtils.atr(df, 14)

        # Near VWAP flag
        df['near_vwap'] = df['vwap_distance'].abs() <= self.deviation_pct

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        df['signal'] = 'HOLD'
        df['signal_score'] = 0
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan

        for i in range(2, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            score = 0

            # ── BUY: Bounce from VWAP support ───────────────────────────────
            # Price was below or at VWAP, now moving up
            if (row['near_vwap'] or prev['near_vwap']) and row['roc'] > 0:
                if prev['Close'] <= prev['vwap']:  # Was at or below VWAP
                    score += 40

                    if row['volume_ratio'] >= SIGNAL_CONFIG['min_volume_ratio']:
                        score += 25

                    if row['rsi'] < 65:
                        score += 20

                    if row['Close'] > row['vwap']:  # Crossed above VWAP
                        score += 15

                    if score >= SIGNAL_CONFIG['min_signal_score']:
                        atr = row['atr'] if row['atr'] > 0 else row['Close'] * 0.002
                        sl_distance = max(atr * 1.5, row['Close'] * 0.0015)
                        tp_distance = sl_distance * 2.0
                        df.iloc[i, df.columns.get_loc('signal')] = 'BUY'
                        df.iloc[i, df.columns.get_loc('signal_score')] = score
                        df.iloc[i, df.columns.get_loc('entry_price')] = row['Close']
                        df.iloc[i, df.columns.get_loc('stop_loss')] = row['vwap_lower']
                        df.iloc[i, df.columns.get_loc('take_profit')] = row['Close'] + tp_distance

            # ── SELL: Rejection from VWAP resistance ────────────────────────
            elif (row['near_vwap'] or prev['near_vwap']) and row['roc'] < 0:
                if prev['Close'] >= prev['vwap']:  # Was at or above VWAP
                    score += 40

                    if row['volume_ratio'] >= SIGNAL_CONFIG['min_volume_ratio']:
                        score += 25

                    if row['rsi'] > 35:
                        score += 20

                    if row['Close'] < row['vwap']:  # Crossed below VWAP
                        score += 15

                    if score >= SIGNAL_CONFIG['min_signal_score']:
                        atr = row['atr'] if row['atr'] > 0 else row['Close'] * 0.002
                        sl_distance = max(atr * 1.5, row['Close'] * 0.0015)
                        tp_distance = sl_distance * 2.0
                        df.iloc[i, df.columns.get_loc('signal')] = 'SELL'
                        df.iloc[i, df.columns.get_loc('signal_score')] = score
                        df.iloc[i, df.columns.get_loc('entry_price')] = row['Close']
                        df.iloc[i, df.columns.get_loc('stop_loss')] = row['vwap_upper']
                        df.iloc[i, df.columns.get_loc('take_profit')] = row['Close'] - tp_distance

        return df

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "VWAP Bounce",
            "timeframe": "1m",
            "confirmation": "Volume + RSI + Price momentum",
            "deviation_pct": self.deviation_pct,
            "params": self.params,
        }
