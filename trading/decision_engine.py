"""
Trading Decision Engine (Issue #3)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "technical": 30.0,
    "fundamental": 25.0,
    "sentiment": 15.0,
    "ml_prediction": 20.0,
    "risk_metrics": 10.0,
}

DECISION_THRESHOLDS = [
    (80.0, "STRONG_BUY"),
    (60.0, "BUY"),
    (40.0, "HOLD"),
    (20.0, "SELL"),
    (0.0, "STRONG_SELL"),
]


class TradingDecisionEngine:
    def __init__(self, weights=None, initial_capital=100_000.0):
        self._weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        total = sum(self._weights.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Weights must sum to 100, got {total}")
        self._capital = initial_capital
        self._ensemble_scorer = None
        self._ml_confirmer = None
        self._risk_manager = None

    def calculate_trade_score(self, symbol, data, capital=None):
        cap = capital or self._capital
        scores = {}
        available = []

        scores["technical"] = self._score_technical(data)
        if scores["technical"] is not None:
            available.append("technical")

        scores["ml_prediction"] = self._score_ml(data)
        if scores["ml_prediction"] is not None:
            available.append("ml_prediction")

        scores["risk_metrics"] = self._score_risk(cap)
        if scores["risk_metrics"] is not None:
            available.append("risk_metrics")

        scores["fundamental"] = None
        scores["sentiment"] = None

        effective_weights = self._redistribute_weights(available)
        total_score = sum(scores[k] * effective_weights[k] for k in available)

        breakdown = {
            "component_scores": scores,
            "weights_used": effective_weights,
            "components_available": available,
            "timestamp": datetime.now().isoformat(),
        }
        return round(total_score, 2), breakdown

    def make_decision(self, symbol, data, capital=None):
        score, breakdown = self.calculate_trade_score(symbol, data, capital)
        decision = self._score_to_decision(score)
        return decision, score, breakdown

    def rank_symbols(self, symbols, data_map, capital=None):
        results = []
        for symbol in symbols:
            try:
                data = data_map.get(symbol)
                if data is None or data.empty:
                    continue
                decision, score, breakdown = self.make_decision(symbol, data, capital)
                results.append({"symbol": symbol, "score": score, "decision": decision, "breakdown": breakdown})
            except Exception as exc:
                logger.warning("Failed to score %s: %s", symbol, exc)
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results

    def print_report(self, results):
        if not results:
            print("No results to display.")
            return
        header = f"{'Rank':<5} {'Symbol':<15} {'Score':<8} {'Decision':<12} {'Technical':<11} {'ML':<8} {'Risk':<8}"
        sep = "=" * len(header)
        print(f"\n{sep}\n  MULTI-FACTOR STOCK SCORING REPORT\n{sep}")
        print(header)
        print("-" * len(header))
        for r in results:
            cs = r["breakdown"]["component_scores"]
            tech = f"{cs['technical']:.1f}" if cs["technical"] is not None else "N/A"
            ml = f"{cs['ml_prediction']:.1f}" if cs["ml_prediction"] is not None else "N/A"
            risk = f"{cs['risk_metrics']:.1f}" if cs["risk_metrics"] is not None else "N/A"
            label = {"STRONG_BUY": "STRONG", "BUY": "OK", "HOLD": "WEAK", "SELL": "SELL", "STRONG_SELL": "AVOID"}.get(r["decision"], "")
            print(f"{r['rank']:<5} {r['symbol']:<15} {r['score']:<8.1f} {r['decision']:<12} {tech:<11} {ml:<8} {risk:<8}  {label}")
        print(sep)
        best = results[0]
        print(f"\nBest candidate: {best['symbol']} ({best['decision']}, score={best['score']:.1f})\n")

    def _score_technical(self, data):
        try:
            if self._ensemble_scorer is None:
                from scalping.ensemble_scorer import EnsembleScorer
                self._ensemble_scorer = EnsembleScorer()
            scored = self._ensemble_scorer.score_all(data)
            last = scored.iloc[-1]
            raw_score = float(last.get("ensemble_score", 0))
            signal = str(last.get("ensemble_signal", "HOLD"))
            if "BUY" in signal:
                return min(100.0, raw_score)
            if "SELL" in signal:
                return max(0.0, 100.0 - raw_score)
            return 50.0
        except Exception as exc:
            logger.warning("Technical score failed: %s", exc)
            return 50.0

    def _score_ml(self, data):
        try:
            if self._ml_confirmer is None:
                from scalping.ml.signal_confirmer import MLSignalConfirmer
                self._ml_confirmer = MLSignalConfirmer()
                self._ml_confirmer.load_model()
            if not self._ml_confirmer._ml_enabled:
                return None
            features = self._ml_confirmer.get_features(data)
            prob = self._ml_confirmer.predict_probability(features)
            return round(prob * 100.0, 2)
        except Exception as exc:
            logger.warning("ML score failed: %s", exc)
            return None

    def _score_risk(self, capital):
        try:
            if self._risk_manager is None:
                from scalping.risk.risk_manager import RiskManager
                self._risk_manager = RiskManager(initial_capital=capital)
            status = self._risk_manager.get_status()
            if status["trading_halted"]:
                return 0.0
            score = max(0.0, 100.0 - status["drawdown_pct"] * 1000.0)
            if status["consecutive_losses"] >= 3:
                score = max(0.0, score - 30.0)
            return round(score, 2)
        except Exception as exc:
            logger.warning("Risk score failed: %s", exc)
            return 50.0

    def _redistribute_weights(self, available):
        avail_raw = {k: self._weights[k] for k in available}
        total = sum(avail_raw.values())
        if total == 0:
            equal = 1.0 / len(available) if available else 1.0
            return {k: equal for k in available}
        return {k: v / total for k, v in avail_raw.items()}

    def _score_to_decision(self, score):
        for threshold, decision in DECISION_THRESHOLDS:
            if score >= threshold:
                return decision
        return "STRONG_SELL"
