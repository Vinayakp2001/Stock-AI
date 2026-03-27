"""
Ensemble Signal Scorer
Combines EMA, VWAP, and RSI strategy signals with weighted scoring.
Only generates a trade signal when ensemble score >= 70.

Design: run all three strategies ONCE on the full dataset upfront,
then score each candle by index — O(n) not O(n^2).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from scalping.strategies.ema_crossover import EMACrossoverStrategy
from scalping.strategies.vwap_strategy import VWAPStrategy
from scalping.strategies.rsi_scalp import RSIScalpStrategy

logger = logging.getLogger(__name__)

# Weights must sum to 1.0  (Req 3.2)
WEIGHTS = {"ema": 0.40, "vwap": 0.35, "rsi": 0.25}

# Agreement bonuses (Req 3.5, 3.6)
BONUS_2_OF_3 = 10
BONUS_3_OF_3 = 20

# Minimum score to generate a signal (Req 3.3, 3.4)
MIN_SCORE = 70


class EnsembleScorer:
    """
    Runs all three strategies on the full dataset once, then scores
    each candle by combining the per-strategy signals at that index.
    """

    def __init__(self):
        self._ema = EMACrossoverStrategy()
        self._vwap = VWAPStrategy()
        self._rsi = RSIScalpStrategy()

    def score_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all three strategies on `data` once and return a DataFrame
        with ensemble columns added:
          ensemble_signal, ensemble_score, ensemble_agreement,
          entry_price, stop_loss, take_profit
        """
        ema_df = self._ema.generate_signals(data)
        vwap_df = self._vwap.generate_signals(data)
        rsi_df = self._rsi.generate_signals(data)

        result = data.copy()
        result["ensemble_signal"] = "HOLD"
        result["ensemble_score"] = 0.0
        result["ensemble_agreement"] = 0
        result["entry_price"] = np.nan
        result["stop_loss"] = np.nan
        result["take_profit"] = np.nan

        for i in range(len(data)):
            row = self._score_at(i, ema_df, vwap_df, rsi_df)
            result.iloc[i, result.columns.get_loc("ensemble_signal")] = row["direction"]
            result.iloc[i, result.columns.get_loc("ensemble_score")] = row["score"]
            result.iloc[i, result.columns.get_loc("ensemble_agreement")] = row["agreement"]
            if row["direction"] != "HOLD":
                result.iloc[i, result.columns.get_loc("entry_price")] = row["entry_price"]
                result.iloc[i, result.columns.get_loc("stop_loss")] = row["stop_loss"]
                result.iloc[i, result.columns.get_loc("take_profit")] = row["take_profit"]

        return result

    def _score_at(self, i: int, ema_df, vwap_df, rsi_df) -> Dict[str, Any]:
        """Score a single candle index across the three pre-computed signal DataFrames."""
        ema_sig = str(ema_df.iloc[i].get("signal", "HOLD"))
        vwap_sig = str(vwap_df.iloc[i].get("signal", "HOLD"))
        rsi_sig = str(rsi_df.iloc[i].get("signal", "HOLD"))

        ema_score = float(ema_df.iloc[i].get("signal_score", 0))
        vwap_score = float(vwap_df.iloc[i].get("signal_score", 0))
        rsi_score = float(rsi_df.iloc[i].get("signal_score", 0))

        # Each strategy contributes its full score weighted by its weight.
        # A single strong strategy (score=80) can reach: 80*0.40=32 (EMA alone)
        # which is too low. Instead: normalise so a single strategy at 100 = 100.
        # We scale the weighted score by 1/max_possible_weight so one strong
        # strategy can still pass the threshold.
        signals_firing = [(ema_sig, ema_score, WEIGHTS["ema"]),
                          (vwap_sig, vwap_score, WEIGHTS["vwap"]),
                          (rsi_sig, rsi_score, WEIGHTS["rsi"])]

        directions = [s for s, sc, w in signals_firing if s != "HOLD"]
        buy_count = directions.count("BUY")
        sell_count = directions.count("SELL")
        agreement = max(buy_count, sell_count)

        if buy_count > sell_count:
            dominant = "BUY"
        elif sell_count > buy_count:
            dominant = "SELL"
        else:
            dominant = "HOLD"

        if dominant == "HOLD":
            return {"score": 0.0, "direction": "HOLD", "agreement": 0,
                    "entry_price": None, "stop_loss": None, "take_profit": None}

        # Weighted score — only count strategies that agree with dominant direction
        weight_sum = 0.0
        weighted_score = 0.0
        for sig, score, weight in signals_firing:
            if sig == dominant:
                weighted_score += score * weight
                weight_sum += weight

        # Normalise: divide by the sum of agreeing weights so a single
        # strategy at score=80 still produces 80, not 32.
        base = weighted_score / weight_sum if weight_sum > 0 else 0.0

        # Agreement bonus on top
        if agreement == 3:
            base += BONUS_3_OF_3
        elif agreement == 2:
            base += BONUS_2_OF_3

        final_score = min(100.0, base)
        direction = dominant if final_score >= MIN_SCORE else "HOLD"

        # Pick SL/TP from highest-weighted agreeing strategy
        entry_price = stop_loss = take_profit = None
        for df, sig in [(ema_df, ema_sig), (vwap_df, vwap_sig), (rsi_df, rsi_sig)]:
            if sig == direction:
                r = df.iloc[i]
                ep = r.get("entry_price")
                if ep is not None and not (isinstance(ep, float) and np.isnan(ep)):
                    entry_price = ep
                    stop_loss = r.get("stop_loss")
                    take_profit = r.get("take_profit")
                    break

        return {
            "score": round(final_score, 2),
            "direction": direction,
            "agreement": agreement,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
