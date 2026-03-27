"""
Adaptive Stop Loss
ATR-based dynamic SL/TP that adjusts to current volatility.

Rules (from requirements):
  - High vol (ATR > 1.5x avg): multiplier = 2.5  (Req 4.2)
  - Normal vol:                 multiplier = 2.0  (Req 4.3)
  - SL distance: min 0.15%, max 0.5% of entry    (Req 4.4, 4.5)
  - TP always = 3x SL distance (1:3 RR)          (Req 4.6)
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Constraints
MIN_SL_PCT = 0.0010   # 0.10%  (tighter SL)
MAX_SL_PCT = 0.0030   # 0.30%  (tighter max)
RR_RATIO   = 4.0      # 1:4 RR to overcome transaction costs

# ATR multipliers
HIGH_VOL_THRESHOLD  = 1.5
MULTIPLIER_HIGH_VOL = 1.5   # was 2.5 — tighter in high vol
MULTIPLIER_NORMAL   = 1.0   # was 2.0 — tighter in normal vol


class AdaptiveStopLoss:
    """
    Calculates ATR-based stop loss and take profit levels.
    """

    def calculate(
        self,
        entry_price: float,
        atr: float,
        avg_atr: float,
        side: str,          # 'BUY' or 'SELL'
    ) -> Dict[str, float]:
        """
        Returns:
            {
                'stop_loss':        float,
                'take_profit':      float,
                'sl_distance_pct':  float,
                'tp_distance_pct':  float,
                'atr_multiplier':   float,
            }
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        # Choose multiplier based on volatility regime (Req 4.2, 4.3)
        atr_ratio   = atr / avg_atr if avg_atr > 0 else 1.0
        multiplier  = MULTIPLIER_HIGH_VOL if atr_ratio > HIGH_VOL_THRESHOLD else MULTIPLIER_NORMAL

        # Raw SL distance from ATR
        raw_sl = atr * multiplier if atr > 0 else entry_price * MIN_SL_PCT

        # Clamp to [MIN_SL_PCT, MAX_SL_PCT] of entry price (Req 4.4, 4.5)
        min_dist = entry_price * MIN_SL_PCT
        max_dist = entry_price * MAX_SL_PCT
        sl_distance = max(min_dist, min(max_dist, raw_sl))

        tp_distance = sl_distance * RR_RATIO   # Req 4.6

        if side.upper() == "BUY":
            stop_loss   = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SELL
            stop_loss   = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        result = {
            "stop_loss":       round(stop_loss,   4),
            "take_profit":     round(take_profit, 4),
            "sl_distance_pct": round(sl_distance / entry_price, 6),
            "tp_distance_pct": round(tp_distance / entry_price, 6),
            "atr_multiplier":  multiplier,
        }

        logger.debug(
            "AdaptiveSL | side=%s | entry=%.2f | sl=%.2f | tp=%.2f | mult=%.1f | atr_ratio=%.2f",
            side, entry_price, stop_loss, take_profit, multiplier, atr_ratio
        )
        return result
