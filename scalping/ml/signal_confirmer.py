"""
ML Signal Confirmer
RandomForestClassifier trained on 8 technical features to filter
low-probability signals before entry.

Features: RSI, MACD, EMA diff, volume ratio, ATR, ADX, VWAP distance, momentum
Labels:   1 = profitable trade (TAKE_PROFIT exit), 0 = losing trade

Req 5.1-5.7
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

MODEL_PATH_DEFAULT = os.path.join("models", "scalping", "signal_classifier.joblib")
MIN_ACCURACY       = 0.55   # Req 5.7: disable ML if below this
MIN_SAMPLES        = 500    # Req 5.6: minimum training samples
PROBABILITY_THRESHOLD = 0.55  # Req 5.3, 5.4


class MLSignalConfirmer:
    """
    Trains a RandomForestClassifier on historical trade data and
    uses it to filter signals at inference time.
    """

    def __init__(self, model_path: str = MODEL_PATH_DEFAULT):
        self.model_path  = model_path
        self.model       = None
        self.scaler      = None
        self.is_trained  = False
        self.accuracy    = 0.0
        self._ml_enabled = False   # disabled until trained with sufficient accuracy

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, data: pd.DataFrame, labels: pd.Series) -> float:
        """
        Train classifier on feature data.

        Args:
            data:   DataFrame with feature columns (output of get_features())
            labels: Series of 1 (win) / 0 (loss)

        Returns:
            test accuracy (float)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score

        if len(data) < MIN_SAMPLES:
            logger.warning(
                "Only %d samples available (need %d). ML filter disabled.",
                len(data), MIN_SAMPLES
            )
            self._ml_enabled = False
            return 0.0

        # Temporal 80/20 split — no shuffling to avoid look-ahead (Req 5.5)
        split = int(len(data) * 0.80)
        X_train, X_test = data.iloc[:split].values, data.iloc[split:].values
        y_train, y_test = labels.iloc[:split].values, labels.iloc[split:].values

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train_s, y_train)

        preds = self.model.predict(X_test_s)
        self.accuracy = float(accuracy_score(y_test, preds))

        if self.accuracy < MIN_ACCURACY:
            logger.warning(
                "ML model accuracy %.2f < %.2f threshold. ML filter DISABLED.",
                self.accuracy, MIN_ACCURACY
            )
            self._ml_enabled = False
        else:
            self.is_trained  = True
            self._ml_enabled = True
            self._save_model()
            logger.info("ML model trained. Accuracy=%.3f. Saved to %s", self.accuracy, self.model_path)

        return self.accuracy

    def load_model(self) -> bool:
        """Load a previously saved model. Returns True on success."""
        import joblib
        if not os.path.exists(self.model_path):
            return False
        try:
            bundle = joblib.load(self.model_path)
            self.model      = bundle["model"]
            self.scaler     = bundle["scaler"]
            self.accuracy   = bundle.get("accuracy", 0.0)
            self.is_trained = True
            self._ml_enabled = self.accuracy >= MIN_ACCURACY
            logger.info("ML model loaded from %s (accuracy=%.3f)", self.model_path, self.accuracy)
            return True
        except Exception as e:
            logger.warning("Failed to load ML model: %s", e)
            return False

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_probability(self, features: pd.DataFrame) -> float:
        """
        Returns probability (0-1) that the signal is a winning trade.
        Returns 1.0 (pass-through) if ML is disabled.
        """
        if not self._ml_enabled or self.model is None:
            return 1.0   # ML disabled → don't filter

        try:
            X = self.scaler.transform(features.values)
            prob = float(self.model.predict_proba(X)[0][1])
            return prob
        except Exception as e:
            logger.warning("ML prediction failed: %s — passing signal through", e)
            return 1.0

    def should_take_signal(self, features: pd.DataFrame) -> bool:
        """Returns True if ML probability >= threshold (Req 5.3, 5.4)."""
        prob = self.predict_probability(features)
        result = prob >= PROBABILITY_THRESHOLD
        logger.debug("ML filter | prob=%.3f | threshold=%.2f | pass=%s", prob, PROBABILITY_THRESHOLD, result)
        return result

    # ── Feature Engineering ───────────────────────────────────────────────

    def get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract 8 ML features from OHLCV data. (Req 5.5)
        Returns a single-row DataFrame for the last candle.
        """
        df = data.copy()

        # RSI
        delta = df["Close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD (12-26 EMA diff)
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26

        # EMA diff (9-21)
        ema9  = df["Close"].ewm(span=9,  adjust=False).mean()
        ema21 = df["Close"].ewm(span=21, adjust=False).mean()
        df["ema_diff"] = (ema9 - ema21) / df["Close"]

        # Volume ratio
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)

        # ATR (normalised by price)
        hl  = df["High"] - df["Low"]
        hc  = (df["High"] - df["Close"].shift()).abs()
        lc  = (df["Low"]  - df["Close"].shift()).abs()
        atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        df["atr_pct"] = atr / df["Close"]

        # ADX (simplified via ta library if available, else manual)
        df["adx"] = self._compute_adx(df)

        # VWAP distance
        vwap = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
        df["vwap_dist"] = (df["Close"] - vwap) / df["Close"]

        # Price momentum (3-candle)
        df["momentum"] = df["Close"].pct_change(3)

        feature_cols = ["rsi", "macd", "ema_diff", "volume_ratio", "atr_pct", "adx", "vwap_dist", "momentum"]
        last = df[feature_cols].iloc[[-1]].fillna(0)
        return last

    def build_training_dataset(self, data: pd.DataFrame, signals_df: pd.DataFrame) -> tuple:
        """
        Build (X, y) training set from historical signals.
        Labels: 1 if exit_reason == TAKE_PROFIT, else 0.
        """
        rows, labels = [], []
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            if row.get("signal") not in ("BUY", "SELL"):
                continue
            if "exit_reason" not in signals_df.columns:
                continue
            label = 1 if row.get("exit_reason") == "TAKE_PROFIT" else 0
            features = self.get_features(data.iloc[max(0, i - 50): i + 1])
            if not features.empty:
                rows.append(features.iloc[0])
                labels.append(label)

        if not rows:
            return pd.DataFrame(), pd.Series(dtype=int)

        return pd.DataFrame(rows).reset_index(drop=True), pd.Series(labels)

    # ── Private ───────────────────────────────────────────────────────────

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Lightweight ADX for feature extraction."""
        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)
        close = df["Close"].values.astype(float)
        n     = len(df)
        adx_vals = np.full(n, 25.0)

        if n < period + 5:
            return pd.Series(adx_vals, index=df.index)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        up   = high[1:] - high[:-1]
        down = low[:-1] - low[1:]
        pdm  = np.where((up > down) & (up > 0), up, 0.0)
        ndm  = np.where((down > up) & (down > 0), down, 0.0)

        def smooth(arr):
            s = np.zeros(len(arr))
            s[period - 1] = arr[:period].sum()
            for i in range(period, len(arr)):
                s[i] = s[i - 1] - s[i - 1] / period + arr[i]
            return s

        atr_s = smooth(tr);  pdi_s = smooth(pdm);  ndi_s = smooth(ndm)
        with np.errstate(divide='ignore', invalid='ignore'):
            pdi = 100 * np.where(atr_s > 0, pdi_s / atr_s, 0)
            ndi = 100 * np.where(atr_s > 0, ndi_s / atr_s, 0)
            dx  = 100 * np.abs(pdi - ndi) / np.where((pdi + ndi) > 0, pdi + ndi, 1)

        adx_raw = smooth(dx[period - 1:])
        start   = period * 2 - 1
        adx_vals[start: start + len(adx_raw)] = adx_raw
        return pd.Series(adx_vals, index=df.index)

    def _save_model(self):
        import joblib
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "accuracy": self.accuracy}, self.model_path)
