"""
EMA Crossover Scalping Strategy
Fast EMA (9) crosses above/below Slow EMA (21) on 1-min chart
Confirmed by 5-min trend, volume spike, RSI and VWAP
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scalping.config import SIGNAL_CONFIG


class EMACrossoverStrategy:
    """
    EMA 9/21 crossover strategy with strong whipsaw filters.

    Entry conditions (BUY):
    - EMA 9 crosses above EMA 21 on 1-min
    - EMA 9 > EMA 21 on 5-min (trend confirmation)
    - Volume >= 1.5x average
    - RSI between 40-65
    - Price above VWAP
    - EMA diff increasing (momentum building, not fading)

    Entry conditions (SELL):
    - EMA 9 crosses below EMA 21 on 1-min
    - EMA 9 < EMA 21 on 5-min (trend confirmation)
    - Volume >= 1.5x average
    - RSI between 35-60
    - Price below VWAP
    """

    def __init__(self, fast: int = 9, slow: int = 21):
        self.fast = fast
        self.slow = slow
        self.name = f"EMA_{fast}_{slow}_Crossover"

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = data.copy()

        # 1-min EMAs
        df['ema_fast'] = df['Close'].ewm(span=self.fast, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=self.slow, adjust=False).mean()

        # EMA crossover
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_cross_up'] = (df['ema_diff'] > 0) & (df['ema_diff'].shift(1) <= 0)
        df['ema_cross_down'] = (df['ema_diff'] < 0) & (df['ema_diff'].shift(1) >= 0)

        # Momentum: EMA diff growing (not shrinking after crossover)
        df['ema_diff_growing'] = df['ema_diff'].abs() > df['ema_diff'].abs().shift(1)

        # 5-min trend using 5-period rolling EMA on 1-min data
        df['ema_fast_5m'] = df['Close'].ewm(span=self.fast * 5, adjust=False).mean()
        df['ema_slow_5m'] = df['Close'].ewm(span=self.slow * 5, adjust=False).mean()
        df['trend_5m_up'] = df['ema_fast_5m'] > df['ema_slow_5m']

        # RSI (14 period)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # VWAP - reset per day
        df['date'] = df.index.date
        df['vwap'] = (
            (df['Close'] * df['Volume'])
            .groupby(df['date'])
            .cumsum()
            / df['Volume'].groupby(df['date']).cumsum()
        )

        # Volume ratio
        df['volume_avg'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_avg'].replace(0, np.nan)

        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        df['atr'] = (
            pd.concat([high_low, high_close, low_close], axis=1)
            .max(axis=1)
            .rolling(14)
            .mean()
        )

        # Price momentum (last 3 candles direction)
        df['price_momentum'] = df['Close'] - df['Close'].shift(3)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate BUY/SELL/HOLD signals with score"""
        df = self.calculate_indicators(data)
        df['signal'] = 'HOLD'
        df['signal_score'] = 0
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan

        # Overall trend: use 50-period EMA slope
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['market_trend'] = df['ema_50'] - df['ema_50'].shift(10)  # positive = uptrend

        for i in range(25, len(df)):
            row = df.iloc[i]
            score = 0
            market_up = row['market_trend'] > 0

            # ── BUY conditions (only in uptrend) ────────────────────────────
            if row['ema_cross_up'] and market_up:
                score += 30

                if row['trend_5m_up']:
                    score += 25

                if row['volume_ratio'] >= SIGNAL_CONFIG['min_volume_ratio']:
                    score += 15

                if 40 <= row['rsi'] <= 62:
                    score += 15

                if row['Close'] > row['vwap']:
                    score += 10

                if row['ema_diff_growing'] and row['price_momentum'] > 0:
                    score += 5

                if score >= SIGNAL_CONFIG['min_signal_score']:
                    atr = row['atr'] if row['atr'] > 0 else row['Close'] * 0.002
                    sl_distance = max(atr * 2.0, row['Close'] * 0.002)  # 0.2% min SL
                    tp_distance = sl_distance * 3.0                      # 1:3 RR
                    df.iloc[i, df.columns.get_loc('signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('signal_score')] = score
                    df.iloc[i, df.columns.get_loc('entry_price')] = row['Close']
                    df.iloc[i, df.columns.get_loc('stop_loss')] = row['Close'] - sl_distance
                    df.iloc[i, df.columns.get_loc('take_profit')] = row['Close'] + tp_distance

            # ── SELL conditions (only in downtrend) ─────────────────────────
            elif row['ema_cross_down'] and not market_up:
                score += 30

                if not row['trend_5m_up']:
                    score += 25

                if row['volume_ratio'] >= SIGNAL_CONFIG['min_volume_ratio']:
                    score += 15

                if 38 <= row['rsi'] <= 60:
                    score += 15

                if row['Close'] < row['vwap']:
                    score += 10

                if row['ema_diff_growing'] and row['price_momentum'] < 0:
                    score += 5

                if score >= SIGNAL_CONFIG['min_signal_score']:
                    atr = row['atr'] if row['atr'] > 0 else row['Close'] * 0.002
                    sl_distance = max(atr * 2.0, row['Close'] * 0.002)
                    tp_distance = sl_distance * 3.0                      # 1:3 RR
                    df.iloc[i, df.columns.get_loc('signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('signal_score')] = score
                    df.iloc[i, df.columns.get_loc('entry_price')] = row['Close']
                    df.iloc[i, df.columns.get_loc('stop_loss')] = row['Close'] + sl_distance
                    df.iloc[i, df.columns.get_loc('take_profit')] = row['Close'] - tp_distance

        return df

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "EMA Crossover",
            "timeframe": "1m",
            "confirmation": "5m trend + Volume + RSI + VWAP + Momentum",
            "fast_ema": self.fast,
            "slow_ema": self.slow,
        }
