"""
Continuous Learning System (Issue #42)
Automatically retrains the ML signal confirmer as new trade data accumulates,
tracks model performance over time, and degrades gracefully when accuracy drops.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paths
TRADE_LOG_PATH   = os.path.join("data", "learning", "trade_log.jsonl")
PERF_LOG_PATH    = os.path.join("data", "learning", "model_performance.json")
MODEL_DIR        = os.path.join("models", "scalping")

# Thresholds
RETRAIN_INTERVAL = 50        # retrain every N new trades
MIN_RETRAIN_SAMPLES = 100    # minimum trades before first retrain
ACCURACY_DECAY_THRESHOLD = 0.05   # retrain if accuracy drops by this much vs best


class ContinuousLearner:
    """
    Wraps MLSignalConfirmer with:
    - Persistent trade logging (JSONL)
    - Automatic retraining every RETRAIN_INTERVAL new trades
    - Performance history tracking
    - Accuracy-decay-triggered retraining
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        retrain_interval: int = RETRAIN_INTERVAL,
        min_samples: int = MIN_RETRAIN_SAMPLES,
    ):
        from scalping.ml.signal_confirmer import MLSignalConfirmer, MODEL_PATH_DEFAULT
        self._confirmer = MLSignalConfirmer(model_path or MODEL_PATH_DEFAULT)
        self._confirmer.load_model()

        self.retrain_interval = retrain_interval
        self.min_samples = min_samples

        self._trades_since_retrain = 0
        self._best_accuracy = self._confirmer.accuracy
        self._performance_history: List[Dict[str, Any]] = []

        os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        self._load_performance_log()

    # ── Public API ────────────────────────────────────────────────────────

    def log_trade(
        self,
        features: pd.DataFrame,
        exit_reason: str,
        symbol: str = "",
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Record a completed trade. Call this after every trade closes.

        Args:
            features:    Single-row DataFrame from MLSignalConfirmer.get_features()
            exit_reason: 'TAKE_PROFIT' → label=1, anything else → label=0
            symbol:      Instrument symbol (for logging)
            timestamp:   ISO timestamp string (defaults to now)
        """
        label = 1 if exit_reason == "TAKE_PROFIT" else 0
        record = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "symbol": symbol,
            "label": label,
            "exit_reason": exit_reason,
            "features": features.iloc[0].to_dict() if not features.empty else {},
        }

        with open(TRADE_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        self._trades_since_retrain += 1
        logger.debug(
            "Trade logged | label=%d | trades_since_retrain=%d",
            label, self._trades_since_retrain,
        )

        # Auto-retrain check
        if self._should_retrain():
            self.retrain()

    def retrain(self, force: bool = False) -> Optional[float]:
        """
        Retrain the model on all logged trades.

        Args:
            force: Skip minimum-sample check if True

        Returns:
            New accuracy, or None if skipped
        """
        X, y = self._load_training_data()
        if X.empty:
            logger.warning("No training data available — skipping retrain")
            return None

        n = len(X)
        if not force and n < self.min_samples:
            logger.info("Only %d samples (need %d) — skipping retrain", n, self.min_samples)
            return None

        logger.info("Retraining on %d samples...", n)
        new_accuracy = self._confirmer.train(X, y)

        self._trades_since_retrain = 0
        self._record_performance(new_accuracy, n)

        if new_accuracy > self._best_accuracy:
            self._best_accuracy = new_accuracy

        logger.info("Retrain complete | accuracy=%.3f | best=%.3f", new_accuracy, self._best_accuracy)
        return new_accuracy

    def get_status(self) -> Dict[str, Any]:
        """Return current learner status."""
        return {
            "model_enabled": self._confirmer._ml_enabled,
            "current_accuracy": self._confirmer.accuracy,
            "best_accuracy": self._best_accuracy,
            "trades_since_retrain": self._trades_since_retrain,
            "total_trades_logged": self._count_logged_trades(),
            "retrain_interval": self.retrain_interval,
            "next_retrain_in": max(0, self.retrain_interval - self._trades_since_retrain),
            "performance_history": self._performance_history[-10:],  # last 10
        }

    def predict(self, features: pd.DataFrame) -> float:
        """Pass-through to MLSignalConfirmer.predict_probability()."""
        return self._confirmer.predict_probability(features)

    def should_take_signal(self, features: pd.DataFrame) -> bool:
        """Pass-through to MLSignalConfirmer.should_take_signal()."""
        return self._confirmer.should_take_signal(features)

    # ── Private ───────────────────────────────────────────────────────────

    def _should_retrain(self) -> bool:
        """Check if retraining is due."""
        total = self._count_logged_trades()
        if total < self.min_samples:
            return False
        if self._trades_since_retrain >= self.retrain_interval:
            return True
        # Accuracy decay check
        if (
            self._best_accuracy > 0
            and self._confirmer.accuracy > 0
            and (self._best_accuracy - self._confirmer.accuracy) >= ACCURACY_DECAY_THRESHOLD
        ):
            logger.info(
                "Accuracy decay detected (%.3f → %.3f) — triggering retrain",
                self._best_accuracy, self._confirmer.accuracy,
            )
            return True
        return False

    def _load_training_data(self) -> tuple:
        """Load all logged trades into (X, y) DataFrames."""
        if not os.path.exists(TRADE_LOG_PATH):
            return pd.DataFrame(), pd.Series(dtype=int)

        rows, labels = [], []
        with open(TRADE_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("features"):
                        rows.append(rec["features"])
                        labels.append(rec["label"])
                except json.JSONDecodeError:
                    continue

        if not rows:
            return pd.DataFrame(), pd.Series(dtype=int)

        X = pd.DataFrame(rows).fillna(0)
        y = pd.Series(labels, dtype=int)
        return X, y

    def _count_logged_trades(self) -> int:
        if not os.path.exists(TRADE_LOG_PATH):
            return 0
        with open(TRADE_LOG_PATH) as f:
            return sum(1 for line in f if line.strip())

    def _record_performance(self, accuracy: float, n_samples: int) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": round(accuracy, 4),
            "n_samples": n_samples,
            "model_enabled": self._confirmer._ml_enabled,
        }
        self._performance_history.append(entry)
        self._save_performance_log()

    def _load_performance_log(self) -> None:
        if os.path.exists(PERF_LOG_PATH):
            try:
                with open(PERF_LOG_PATH) as f:
                    self._performance_history = json.load(f)
                if self._performance_history:
                    best = max(e["accuracy"] for e in self._performance_history)
                    self._best_accuracy = max(self._best_accuracy, best)
            except Exception:
                self._performance_history = []

    def _save_performance_log(self) -> None:
        with open(PERF_LOG_PATH, "w") as f:
            json.dump(self._performance_history, f, indent=2)
