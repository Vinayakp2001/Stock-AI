"""
Volatility Regime Detector
Classifies market as choppy / trending / strong_trend using ADX.
Choppy markets (ADX < 20) are rejected to avoid false signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# ADX thresholds (Req 2.2, 2.3, 2.4)
ADX_CHOPPY    = 20
ADX_STRONG    = 40

# ATR ratio threshold for position size reduction (Req 2.5)
LOW_VOL_ATR_RATIO = 1.0   # below avg ATR → reduce size


class VolatilityRegimeDetector:
    """
    Uses ADX to classify market regime and decide whether to trade.

    Regimes:
      choppy      → ADX < 20  → reject all signals
      trending    → 20 ≤ ADX < 40 → allow signals
      strong_trend → ADX ≥ 40 → allow with higher confidence
    """

    def __init__(self, adx_period: int = 14, atr_period: int = 14, atr_avg_period: int = 20):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.atr_avg_period = atr_avg_period

    # ── Public API ────────────────────────────────────────────────────────

    def detect_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute ADX + ATR and return regime dict.

        Returns:
            {
                'regime': 'trending' | 'choppy' | 'strong_trend',
                'adx': float,
                'atr': float,
                'atr_ratio': float,   # current ATR / avg ATR
                'tradeable': bool
            }
        """
        if len(data) < self.adx_period + 5:
            logger.warning("Not enough data for regime detection, defaulting to unknown")
            return self._unknown_regime()

        try:
            adx = self._calculate_adx(data)
            atr, atr_avg = self._calculate_atr(data)
            atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0

            regime = self._classify(adx)
            tradeable = regime != "choppy"

            result = {
                "regime": regime,
                "adx": round(adx, 2),
                "atr": round(atr, 4),
                "atr_ratio": round(atr_ratio, 3),
                "tradeable": tradeable,
            }

            logger.info(
                "Regime detected | regime=%s | adx=%.1f | atr_ratio=%.2f | tradeable=%s",
                regime, adx, atr_ratio, tradeable
            )
            return result

        except Exception as e:
            logger.warning("Regime detection failed: %s — defaulting to unknown", e)
            return self._unknown_regime()

    def get_position_size_multiplier(self, regime: Dict[str, Any]) -> float:
        """
        Returns position size multiplier based on regime.
          - Low volatility (ATR below avg): 0.5  (Req 2.5)
          - Normal trending:                1.0
          - Strong trend:                   1.2
        """
        if not regime.get("tradeable", True):
            return 0.0   # choppy → no position

        atr_ratio = regime.get("atr_ratio", 1.0)
        if atr_ratio < LOW_VOL_ATR_RATIO:
            return 0.5   # Req 2.5: reduce size in low volatility

        if regime.get("regime") == "strong_trend":
            return 1.2

        return 1.0

    # ── Private helpers ───────────────────────────────────────────────────

    def _classify(self, adx: float) -> str:
        if adx < ADX_CHOPPY:
            return "choppy"
        if adx >= ADX_STRONG:
            return "strong_trend"
        return "trending"

    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """Wilder's ADX calculation (last value)."""
        high  = data["High"].values.astype(float)
        low   = data["Low"].values.astype(float)
        close = data["Close"].values.astype(float)
        n = self.adx_period

        if len(high) < n * 2 + 5:
            return 25.0  # not enough data, return neutral

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:]  - close[:-1])
            )
        )

        # Directional movement
        up_move   = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder smoothing: first value = sum of first n, then rolling
        def wilder_smooth(arr, period):
            if len(arr) < period:
                return np.array([0.0])
            result = np.zeros(len(arr))
            result[period - 1] = arr[:period].sum()
            for i in range(period, len(arr)):
                result[i] = result[i - 1] - (result[i - 1] / period) + arr[i]
            return result

        atr_s   = wilder_smooth(tr, n)
        plus_s  = wilder_smooth(plus_dm, n)
        minus_s = wilder_smooth(minus_dm, n)

        # DI values — only valid from index n-1 onwards
        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di  = np.where(atr_s > 0, 100.0 * plus_s  / atr_s, 0.0)
            minus_di = np.where(atr_s > 0, 100.0 * minus_s / atr_s, 0.0)
            denom    = plus_di + minus_di
            dx       = np.where(denom > 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)

        # ADX = Wilder smooth of DX, starting from index n-1
        dx_valid = dx[n - 1:]  # skip the zero-padded warmup
        adx_arr  = wilder_smooth(dx_valid, n)

        # Return last valid ADX value, clamped to [0, 100]
        last_adx = float(adx_arr[-1])
        return max(0.0, min(100.0, last_adx))

    def _calculate_atr(self, data: pd.DataFrame):
        """Returns (current_atr, avg_atr_over_atr_avg_period)."""
        high  = data["High"]
        low   = data["Low"]
        close = data["Close"]

        hl  = high - low
        hc  = (high - close.shift()).abs()
        lc  = (low  - close.shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        atr_series = tr.rolling(self.atr_period).mean()
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        avg_atr     = float(atr_series.rolling(self.atr_avg_period).mean().iloc[-1]) if len(atr_series) >= self.atr_avg_period else current_atr

        return current_atr, avg_atr

    def _unknown_regime(self) -> Dict[str, Any]:
        """Fallback when calculation fails — allow trading (Req 2.6 error handling)."""
        return {
            "regime": "unknown",
            "adx": 25.0,
            "atr": 0.0,
            "atr_ratio": 1.0,
            "tradeable": True,
        }

    def _compute_adx_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute ADX for every row and return as a Series.
        Used by ImprovedScalpingStrategy for O(n) pre-computation.
        """
        high = data["High"].values.astype(float)
        low = data["Low"].values.astype(float)
        close = data["Close"].values.astype(float)
        n = self.adx_period
        size = len(data)
        adx_vals = np.full(size, 25.0)

        if size < n * 2 + 5:
            return pd.Series(adx_vals, index=data.index)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        up = high[1:] - high[:-1]
        down = low[:-1] - low[1:]
        pdm = np.where((up > down) & (up > 0), up, 0.0)
        ndm = np.where((down > up) & (down > 0), down, 0.0)

        def smooth(arr):
            s = np.zeros(len(arr))
            s[n - 1] = arr[:n].sum()
            for i in range(n, len(arr)):
                s[i] = s[i - 1] - s[i - 1] / n + arr[i]
            return s

        atr_s = smooth(tr)
        pdi_s = smooth(pdm)
        ndi_s = smooth(ndm)

        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = 100 * np.where(atr_s > 0, pdi_s / atr_s, 0)
            ndi = 100 * np.where(atr_s > 0, ndi_s / atr_s, 0)
            dx = 100 * np.abs(pdi - ndi) / np.where((pdi + ndi) > 0, pdi + ndi, 1)

        adx_raw = smooth(dx[n - 1:])
        start = n * 2 - 1
        end = start + len(adx_raw)
        adx_vals[start:end] = adx_raw[: size - start]
        return pd.Series(adx_vals, index=data.index)
