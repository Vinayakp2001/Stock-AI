"""
Market Regime Detection System — trading/market_regime.py

Detects broad market regime (BULL / BEAR / SIDEWAYS / VOLATILE) using:
- Price vs 50/200 SMA (trend direction)
- ADX (trend strength) — reuses VolatilityRegimeDetector
- Volatility (ATR relative to price)
- Volume trend

Produces a RegimeSignal used by TradingDecisionEngine to adjust
strategy aggressiveness and position sizing.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from scalping.filters.regime_filter import VolatilityRegimeDetector  # noqa: F401 (kept for re-export)

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "BULL"           # uptrend, low volatility
    BEAR = "BEAR"           # downtrend
    SIDEWAYS = "SIDEWAYS"   # range-bound, low ADX
    VOLATILE = "VOLATILE"   # high volatility, unclear direction


@dataclass
class RegimeSignal:
    regime: MarketRegime
    confidence: float           # 0-1
    adx: float
    trend_strength: str         # STRONG / MODERATE / WEAK
    price_vs_sma50: float       # % above/below 50 SMA
    price_vs_sma200: float      # % above/below 200 SMA
    volatility_pct: float       # ATR as % of price
    recommendation: str         # TRADE_NORMAL / TRADE_CAUTIOUS / AVOID
    details: Dict[str, Any]


class MarketRegimeDetector:
    """
    Detects market regime from OHLCV data.

    Can work with:
    - A pandas DataFrame (already fetched)
    - A symbol string (fetches daily data via yfinance)
    """

    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        adx_period: int = 14,
        volatility_threshold: float = 2.0,  # ATR% above this = volatile
    ) -> None:
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.adx_period = adx_period
        self.volatility_threshold = volatility_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, data: pd.DataFrame) -> RegimeSignal:
        """Detect regime from OHLCV DataFrame."""
        if len(data) < self.sma_slow + 10:
            logger.warning("MarketRegimeDetector: insufficient data (%d rows)", len(data))
            return self._neutral_signal()

        try:
            close = data["Close"]
            # Handle multi-level columns from yfinance (squeeze to 1D Series)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.squeeze()

            high = data["High"]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            high = high.squeeze()

            low = data["Low"]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]
            low = low.squeeze()

            # SMAs — use .item() to safely extract scalar from any Series shape
            sma50 = close.rolling(self.sma_fast).mean().iloc[-1]
            sma50 = float(sma50.item() if hasattr(sma50, "item") else sma50)
            sma200 = close.rolling(self.sma_slow).mean().iloc[-1]
            sma200 = float(sma200.item() if hasattr(sma200, "item") else sma200)
            last_close = close.iloc[-1]
            last_close = float(last_close.item() if hasattr(last_close, "item") else last_close)

            price_vs_sma50 = (last_close - sma50) / sma50 * 100
            price_vs_sma200 = (last_close - sma200) / sma200 * 100

            # ADX — calculate directly (daily timeframe)
            adx, atr = self._calculate_adx_atr(close, high, low)
            volatility_pct = atr / last_close * 100 if last_close > 0 else 0.0

            # Classify regime
            regime, confidence = self._classify(
                price_vs_sma50, price_vs_sma200, adx, volatility_pct
            )

            trend_strength = (
                "STRONG" if adx >= 40 else
                "MODERATE" if adx >= 25 else
                "WEAK"
            )

            recommendation = self._get_recommendation(regime, confidence)

            signal = RegimeSignal(
                regime=regime,
                confidence=round(confidence, 3),
                adx=adx,
                trend_strength=trend_strength,
                price_vs_sma50=round(price_vs_sma50, 2),
                price_vs_sma200=round(price_vs_sma200, 2),
                volatility_pct=round(volatility_pct, 3),
                recommendation=recommendation,
                details={
                    "sma50": round(sma50, 2),
                    "sma200": round(sma200, 2),
                    "last_close": round(last_close, 2),
                    "atr": round(atr, 4),
                },
            )
            logger.info(
                "MarketRegimeDetector: %s (conf=%.2f) ADX=%.1f vol=%.2f%% rec=%s",
                regime.value, confidence, adx, volatility_pct, recommendation,
            )
            return signal

        except Exception as exc:
            logger.warning("MarketRegimeDetector: detection failed: %s", exc)
            return self._neutral_signal()

    def detect_from_symbol(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> RegimeSignal:
        """Fetch data from yfinance and detect regime."""
        try:
            data = yf.download(symbol, period=period, interval=interval,
                               progress=False, auto_adjust=True)
            if data.empty:
                logger.warning("MarketRegimeDetector: no data for %s", symbol)
                return self._neutral_signal()
            return self.detect(data)
        except Exception as exc:
            logger.warning("MarketRegimeDetector: fetch failed for %s: %s", symbol, exc)
            return self._neutral_signal()

    def get_position_size_multiplier(self, signal: RegimeSignal) -> float:
        """
        Returns position size multiplier based on regime.
        BULL strong → 1.2, BULL moderate → 1.0,
        SIDEWAYS → 0.7, VOLATILE → 0.5, BEAR → 0.3
        """
        if signal.regime == MarketRegime.BEAR:
            return 0.3
        if signal.regime == MarketRegime.VOLATILE:
            return 0.5
        if signal.regime == MarketRegime.SIDEWAYS:
            return 0.7
        # BULL
        return 1.2 if signal.trend_strength == "STRONG" else 1.0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _calculate_adx_atr(self, close: pd.Series, high: pd.Series, low: pd.Series):
        """Simple pandas ADX + ATR for daily data."""
        n = self.adx_period
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = float(tr.rolling(n).mean().iloc[-1])

        up = high.diff()
        down = -low.diff()
        pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
        ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)

        atr_s = tr.ewm(alpha=1/n, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / atr_s.replace(0, np.nan)
        ndi = 100 * ndm.ewm(alpha=1/n, adjust=False).mean() / atr_s.replace(0, np.nan)
        dx = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
        adx = float(dx.ewm(alpha=1/n, adjust=False).mean().iloc[-1])
        adx = max(0.0, min(100.0, adx))
        return adx, atr

    def _classify(
        self,
        price_vs_sma50: float,
        price_vs_sma200: float,
        adx: float,
        volatility_pct: float,
    ):
        """Return (MarketRegime, confidence)."""

        # High volatility overrides everything
        if volatility_pct > self.volatility_threshold:
            confidence = min(1.0, volatility_pct / (self.volatility_threshold * 2))
            return MarketRegime.VOLATILE, confidence

        # Both SMAs bullish
        if price_vs_sma50 > 0 and price_vs_sma200 > 0:
            confidence = min(1.0, (price_vs_sma50 + price_vs_sma200) / 10 + adx / 100)
            return MarketRegime.BULL, round(confidence, 3)

        # Both SMAs bearish
        if price_vs_sma50 < 0 and price_vs_sma200 < 0:
            confidence = min(1.0, (abs(price_vs_sma50) + abs(price_vs_sma200)) / 10 + adx / 100)
            return MarketRegime.BEAR, round(confidence, 3)

        # Mixed / low ADX = sideways
        confidence = max(0.3, 1.0 - adx / 50)
        return MarketRegime.SIDEWAYS, round(confidence, 3)

    def _get_recommendation(self, regime: MarketRegime, confidence: float) -> str:
        if regime == MarketRegime.BEAR:
            return "AVOID"
        if regime == MarketRegime.VOLATILE:
            return "AVOID" if confidence > 0.7 else "TRADE_CAUTIOUS"
        if regime == MarketRegime.SIDEWAYS:
            return "TRADE_CAUTIOUS"
        return "TRADE_NORMAL"  # BULL

    def _neutral_signal(self) -> RegimeSignal:
        return RegimeSignal(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.5,
            adx=25.0,
            trend_strength="WEAK",
            price_vs_sma50=0.0,
            price_vs_sma200=0.0,
            volatility_pct=1.0,
            recommendation="TRADE_CAUTIOUS",
            details={},
        )
