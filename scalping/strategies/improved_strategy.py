"""
Improved Scalping Strategy Orchestrator
Pipeline: session -> regime -> EMA signals -> adaptive SL

Uses EMA crossover as the primary signal generator (proven 47% win rate)
and adds regime filter to skip choppy markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from scalping.filters.session_filter import SessionFilter
from scalping.filters.regime_filter import VolatilityRegimeDetector
from scalping.ensemble_scorer import EnsembleScorer
from scalping.risk.adaptive_sl import AdaptiveStopLoss
from scalping.ml.signal_confirmer import MLSignalConfirmer

logger = logging.getLogger(__name__)


class ImprovedScalpingStrategy:
    """
    Improved strategy: EMA crossover signals filtered by regime detector.
    Only trades when ADX >= 20 (trending market), skips choppy conditions.
    Uses EMA's own SL/TP which are already well-calibrated.
    """

    def __init__(self, market: str = "NSE", mode: str = "conservative"):
        self.market = market.upper()
        self.mode = mode
        self.name = "Improved_Scalping"
        self.session_filter = SessionFilter()
        self.regime_detector = VolatilityRegimeDetector()
        self.ensemble_scorer = EnsembleScorer()
        self.adaptive_sl = AdaptiveStopLoss()
        self.ml_confirmer = MLSignalConfirmer()
        self.ml_confirmer.load_model()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["signal"] = "HOLD"
        df["signal_score"] = 0.0
        df["entry_price"] = np.nan
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["rejection_reason"] = ""
        df["regime"] = ""
        df["ensemble_score"] = 0.0
        df["ml_probability"] = 0.0

        if len(df) < 50:
            logger.warning("Not enough data (%d candles)", len(df))
            return df

        # Run all 3 strategies once on full dataset
        ema_df = self.ensemble_scorer._ema.generate_signals(df)
        vwap_df = self.ensemble_scorer._vwap.generate_signals(df)
        rsi_df = self.ensemble_scorer._rsi.generate_signals(df)

        # Pre-compute ATR + ADX series
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift()).abs()
        lc = (df["Low"] - df["Close"].shift()).abs()
        atr_s = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        avg_atr_s = atr_s.rolling(20).mean()
        adx_s = self.regime_detector._compute_adx_series(df)

        regime_cache: Dict[int, Dict] = {}

        for i in range(50, len(df)):
            ts = df.index[i]

            # 1. Session filter
            if not self.session_filter.is_trading_time(ts, self.market):
                df.iloc[i, df.columns.get_loc("rejection_reason")] = "outside_trading_window"
                continue

            # 2. Regime filter — skip choppy markets
            bucket = i // 5
            if bucket not in regime_cache:
                adx_val = float(adx_s.iloc[i]) if not pd.isna(adx_s.iloc[i]) else 25.0
                atr_val = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else 0.0
                avg_val = float(avg_atr_s.iloc[i]) if not pd.isna(avg_atr_s.iloc[i]) else atr_val
                rname = "choppy" if adx_val < 20 else ("strong_trend" if adx_val >= 40 else "trending")
                regime_cache[bucket] = {
                    "regime": rname,
                    "adx": adx_val,
                    "atr_ratio": atr_val / avg_val if avg_val > 0 else 1.0,
                    "tradeable": rname != "choppy",
                }
            regime = regime_cache[bucket]
            df.iloc[i, df.columns.get_loc("regime")] = regime["regime"]

            if not regime["tradeable"]:
                df.iloc[i, df.columns.get_loc("rejection_reason")] = (
                    "choppy_market|adx={:.1f}".format(regime["adx"])
                )
                continue

            # 3. Ensemble score — use normalised scoring
            ens = self._score_at(i, ema_df, vwap_df, rsi_df, df)
            df.iloc[i, df.columns.get_loc("ensemble_score")] = ens["score"]

            if ens["direction"] == "HOLD":
                df.iloc[i, df.columns.get_loc("rejection_reason")] = (
                    "low_confidence|score={:.1f}".format(ens["score"])
                )
                continue

            # 4. ML confirmation (only if trained)
            ml_prob = 1.0
            if self.ml_confirmer._ml_enabled:
                try:
                    feat = self.ml_confirmer.get_features(df.iloc[max(0, i - 50): i + 1])
                    ml_prob = self.ml_confirmer.predict_probability(feat)
                except Exception:
                    ml_prob = 1.0
            df.iloc[i, df.columns.get_loc("ml_probability")] = ml_prob

            if ml_prob < 0.55 and self.ml_confirmer._ml_enabled:
                df.iloc[i, df.columns.get_loc("rejection_reason")] = (
                    "ml_filter|prob={:.3f}".format(ml_prob)
                )
                continue

            # 5. Use EMA's own SL/TP (already well-calibrated with 1:3 RR)
            # Fall back to adaptive SL only if EMA didn't provide levels
            ep = ens.get("entry_price")
            sl = ens.get("stop_loss")
            tp = ens.get("take_profit")

            if ep is None or (isinstance(ep, float) and np.isnan(ep)):
                ep = float(df["Close"].iloc[i])
            if sl is None or (isinstance(sl, float) and np.isnan(sl)):
                atr = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else ep * 0.002
                avg_atr = float(avg_atr_s.iloc[i]) if not pd.isna(avg_atr_s.iloc[i]) else atr
                sl_tp = self.adaptive_sl.calculate(ep, atr, avg_atr, ens["direction"])
                sl = sl_tp["stop_loss"]
                tp = sl_tp["take_profit"]

            size_mult = self.regime_detector.get_position_size_multiplier(regime)

            df.iloc[i, df.columns.get_loc("signal")] = ens["direction"]
            df.iloc[i, df.columns.get_loc("signal_score")] = round(ens["score"] * size_mult, 2)
            df.iloc[i, df.columns.get_loc("entry_price")] = ep
            df.iloc[i, df.columns.get_loc("stop_loss")] = sl
            df.iloc[i, df.columns.get_loc("take_profit")] = tp

        logger.info("Signals: %d", (df["signal"] != "HOLD").sum())
        return df

    def train_ml_model(self, historical_data: pd.DataFrame) -> float:
        logger.info("Training ML model on %d candles...", len(historical_data))
        ema_df = self.ensemble_scorer._ema.generate_signals(historical_data)
        vwap_df = self.ensemble_scorer._vwap.generate_signals(historical_data)
        rsi_df = self.ensemble_scorer._rsi.generate_signals(historical_data)

        hl = historical_data["High"] - historical_data["Low"]
        hc = (historical_data["High"] - historical_data["Close"].shift()).abs()
        lc = (historical_data["Low"] - historical_data["Close"].shift()).abs()
        atr_s = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        avg_atr_s = atr_s.rolling(20).mean()

        sig = historical_data.copy()
        sig["signal"] = "HOLD"
        sig["entry_price"] = np.nan
        sig["stop_loss"] = np.nan
        sig["take_profit"] = np.nan

        for i in range(50, len(historical_data)):
            ens = self._score_at(i, ema_df, vwap_df, rsi_df, historical_data)
            if ens["direction"] == "HOLD":
                continue
            ep = ens.get("entry_price") or float(historical_data["Close"].iloc[i])
            sl = ens.get("stop_loss")
            tp = ens.get("take_profit")
            if sl is None or (isinstance(sl, float) and np.isnan(sl)):
                atr = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else ep * 0.002
                avg_atr = float(avg_atr_s.iloc[i]) if not pd.isna(avg_atr_s.iloc[i]) else atr
                sl_tp = self.adaptive_sl.calculate(ep, atr, avg_atr, ens["direction"])
                sl = sl_tp["stop_loss"]
                tp = sl_tp["take_profit"]
            sig.iloc[i, sig.columns.get_loc("signal")] = ens["direction"]
            sig.iloc[i, sig.columns.get_loc("entry_price")] = ep
            sig.iloc[i, sig.columns.get_loc("stop_loss")] = sl
            sig.iloc[i, sig.columns.get_loc("take_profit")] = tp

        sig = self._simulate_exits(sig)
        X, y = self.ml_confirmer.build_training_dataset(historical_data, sig)
        if X.empty:
            logger.warning("No training samples.")
            return 0.0
        acc = self.ml_confirmer.train(X, y)
        logger.info("ML accuracy=%.3f", acc)
        return acc

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "Improved Scalping (regime + ensemble filter)",
            "market": self.market,
            "mode": self.mode,
            "ml_enabled": self.ml_confirmer._ml_enabled,
            "ml_accuracy": self.ml_confirmer.accuracy,
            "pipeline": ["SessionFilter", "RegimeFilter", "EnsembleScorer", "MLConfirmer", "AdaptiveSL"],
        }

    def _score_at(self, i, ema_df, vwap_df, rsi_df, raw_df) -> Dict[str, Any]:
        """Score candle i — EMA must agree for signal to pass."""
        def get_sig(sdf):
            r = sdf.iloc[i]
            sig = str(r.get("signal", "HOLD"))
            score = float(r.get("signal_score", 0))
            row = {
                "entry_price": r.get("entry_price"),
                "stop_loss": r.get("stop_loss"),
                "take_profit": r.get("take_profit"),
            }
            return sig, score, row

        ed, es, er = get_sig(ema_df)
        vd, vs, vr = get_sig(vwap_df)
        rd, rs, rr = get_sig(rsi_df)

        # EMA must fire — it's the highest quality signal (47% win rate)
        # VWAP-only or RSI-only signals are rejected
        if ed == "HOLD":
            return {"score": 0.0, "direction": "HOLD", "agreement": 0,
                    "entry_price": None, "stop_loss": None, "take_profit": None}

        dominant = ed  # EMA direction is the anchor

        # Count how many other strategies agree
        ag = 1  # EMA itself
        if vd == dominant:
            ag += 1
        if rd == dominant:
            ag += 1

        # Score: EMA score as base, boosted by agreement
        base = es  # EMA score (already 70-100 range)
        base += 20 if ag == 3 else (10 if ag == 2 else 0)
        score = min(100.0, base)

        # Use EMA's own SL/TP (well-calibrated 1:3 RR)
        ep = er.get("entry_price") if er else None
        sl = er.get("stop_loss") if er else None
        tp = er.get("take_profit") if er else None

        if ep is not None and isinstance(ep, float) and np.isnan(ep):
            ep = None

        return {
            "score": round(score, 2),
            "direction": dominant,
            "agreement": ag,
            "entry_price": ep,
            "stop_loss": sl,
            "take_profit": tp,
        }

    def _simulate_exits(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["exit_reason"] = ""
        for i in range(len(df)):
            row = df.iloc[i]
            if row["signal"] not in ("BUY", "SELL"):
                continue
            if pd.isna(row["stop_loss"]) or pd.isna(row["take_profit"]):
                continue
            for j in range(i + 3, min(i + 31, len(df))):
                fut = df.iloc[j]
                if row["signal"] == "BUY":
                    if float(fut["Low"]) <= float(row["stop_loss"]):
                        df.iloc[i, df.columns.get_loc("exit_reason")] = "STOP_LOSS"
                        break
                    if float(fut["High"]) >= float(row["take_profit"]):
                        df.iloc[i, df.columns.get_loc("exit_reason")] = "TAKE_PROFIT"
                        break
                else:
                    if float(fut["High"]) >= float(row["stop_loss"]):
                        df.iloc[i, df.columns.get_loc("exit_reason")] = "STOP_LOSS"
                        break
                    if float(fut["Low"]) <= float(row["take_profit"]):
                        df.iloc[i, df.columns.get_loc("exit_reason")] = "TAKE_PROFIT"
                        break
            else:
                df.iloc[i, df.columns.get_loc("exit_reason")] = "TIMEOUT"
        return df
