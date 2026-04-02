"""
Trading Decision Engine (Issue #3)
Combines technical, fundamental, sentiment, ML, and risk scores into
a final 0-100 trade score and a STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL decision.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from typing_extensions import TypedDict

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


class ScoreBreakdown(TypedDict):
    component_scores: Dict[str, Optional[float]]
    weights_used: Dict[str, float]
    components_available: List[str]
    timestamp: str


class TradingDecisionEngine:
    """
    Combines multiple scoring components into a final trade score and decision.
    Unavailable components have their weights redistributed proportionally.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        initial_capital: float = 100_000.0,
    ):
        self._weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        total = sum(self._weights.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Weights must sum to 100, got {total}")
        self._capital = initial_capital
        self._ensemble_scorer = None
        self._ml_confirmer = None
        self._risk_manager = None
        self._fundamental_analyzer = None  # lazy-init (Requirement 5.1)

    # ── Public API ────────────────────────────────────────────────────────

    def calculate_trade_score(
        self,
        symbol: str,
        data: pd.DataFrame,
        capital: Optional[float] = None,
    ) -> Tuple[float, ScoreBreakdown]:
        cap = capital or self._capital
        scores: Dict[str, Optional[float]] = {}
        available: List[str] = []

        scores["technical"] = self._score_technical(data)
        if scores["technical"] is not None:
            available.append("technical")

        scores["ml_prediction"] = self._score_ml(data)
        if scores["ml_prediction"] is not None:
            available.append("ml_prediction")

        scores["risk_metrics"] = self._score_risk(cap)
        if scores["risk_metrics"] is not None:
            available.append("risk_metrics")

        scores["fundamental"] = self._score_fundamental(symbol)
        if scores["fundamental"] is not None:
            available.append("fundamental")

        scores["sentiment"] = None
        logger.warning("Sentiment component unavailable — requires Issue #2")

        effective_weights = self._redistribute_weights(available)
        total_score = sum(scores[k] * effective_weights[k] for k in available)

        breakdown = ScoreBreakdown(
            component_scores=scores,
            weights_used=effective_weights,
            components_available=available,
            timestamp=datetime.now().isoformat(),
        )
        return round(total_score, 2), breakdown

    def make_decision(
        self,
        symbol: str,
        data: pd.DataFrame,
        capital: Optional[float] = None,
    ) -> Tuple[str, float, ScoreBreakdown]:
        score, breakdown = self.calculate_trade_score(symbol, data, capital)
        decision = self._score_to_decision(score)
        return decision, score, breakdown

    def rank_symbols(
        self,
        symbols: List[str],
        data_map: Dict[str, pd.DataFrame],
        capital: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for symbol in symbols:
            try:
                data = data_map.get(symbol)
                if data is None or data.empty:
                    logger.warning("No data for %s — skipping", symbol)
                    continue
                decision, score, breakdown = self.make_decision(symbol, data, capital)
                results.append(
                    {"symbol": symbol, "score": score, "decision": decision, "breakdown": breakdown}
                )
            except Exception as exc:
                logger.warning("Failed to score %s: %s — skipping", symbol, exc)

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results

    def print_report(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            print("No results to display.")
            return

        header = (
            f"{'Rank':<5} {'Symbol':<15} {'Score':<8} {'Decision':<12}"
            f" {'Technical':<11} {'ML':<8} {'Risk':<8}"
        )
        sep = "=" * len(header)
        print(f"\n{sep}")
        print("  MULTI-FACTOR STOCK SCORING REPORT")
        print(sep)
        print(header)
        print("-" * len(header))

        for r in results:
            cs = r["breakdown"]["component_scores"]
            tech = f"{cs['technical']:.1f}" if cs["technical"] is not None else "N/A"
            ml = f"{cs['ml_prediction']:.1f}" if cs["ml_prediction"] is not None else "N/A"
            risk = f"{cs['risk_metrics']:.1f}" if cs["risk_metrics"] is not None else "N/A"
            label = self._decision_label(r["decision"])
            print(
                f"{r['rank']:<5} {r['symbol']:<15} {r['score']:<8.1f}"
                f" {r['decision']:<12} {tech:<11} {ml:<8} {risk:<8}  {label}"
            )

        print(sep)
        best = results[0]
        print(f"\nBest candidate: {best['symbol']} ({best['decision']}, score={best['score']:.1f})\n")


    # ── Component Scorers ─────────────────────────────────────────────────

    def _score_fundamental(self, symbol: str) -> Optional[float]:
        """Lazy-init FundamentalAnalyzer and return 0-100 score (Requirement 5.1)."""
        try:
            if self._fundamental_analyzer is None:
                from agents.fundamental_agent import FundamentalAnalyzer
                self._fundamental_analyzer = FundamentalAnalyzer()
            score = self._fundamental_analyzer.calculate_fundamental_score(symbol)
            return score
        except Exception as exc:
            logger.warning("Fundamental score failed: %s — weight redistributed", exc)
            return None

    def _score_technical(self, data: pd.DataFrame) -> Optional[float]:
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
            logger.warning("Technical score failed: %s — using neutral 50", exc)
            return 50.0

    def _score_ml(self, data: pd.DataFrame) -> Optional[float]:
        try:
            if self._ml_confirmer is None:
                from scalping.ml.signal_confirmer import MLSignalConfirmer
                self._ml_confirmer = MLSignalConfirmer()
                self._ml_confirmer.load_model()
            if not self._ml_confirmer._ml_enabled:
                logger.warning("ML component disabled — weight will be redistributed")
                return None
            features = self._ml_confirmer.get_features(data)
            prob = self._ml_confirmer.predict_probability(features)
            return round(prob * 100.0, 2)
        except Exception as exc:
            logger.warning("ML score failed: %s — weight redistributed", exc)
            return None

    def _score_risk(self, capital: float) -> Optional[float]:
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
            logger.warning("Risk score failed: %s — using neutral 50", exc)
            return 50.0

    # ── Helpers ───────────────────────────────────────────────────────────

    def _redistribute_weights(self, available: List[str]) -> Dict[str, float]:
        avail_raw = {k: self._weights[k] for k in available}
        total = sum(avail_raw.values())
        if total == 0:
            equal = 1.0 / len(available) if available else 1.0
            return {k: equal for k in available}
        return {k: v / total for k, v in avail_raw.items()}

    def _score_to_decision(self, score: float) -> str:
        for threshold, decision in DECISION_THRESHOLDS:
            if score >= threshold:
                return decision
        return "STRONG_SELL"

    def _decision_label(self, decision: str) -> str:
        labels = {
            "STRONG_BUY": "STRONG",
            "BUY": "OK",
            "HOLD": "WEAK",
            "SELL": "SELL",
            "STRONG_SELL": "AVOID",
        }
        return labels.get(decision, "")
