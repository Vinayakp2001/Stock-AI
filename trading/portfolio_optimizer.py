"""
Portfolio Theory Optimization (Issue #43)
Modern Portfolio Theory — efficient frontier, max Sharpe, min volatility,
and equal-weight fallback. Uses only numpy/scipy (no extra deps).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065   # ~6.5% India risk-free (10yr G-sec)


class PortfolioOptimizer:
    """
    Given a list of symbols, fetches historical returns and computes:
    - Max Sharpe Ratio portfolio
    - Minimum Volatility portfolio
    - Equal Weight portfolio (fallback)
    - Efficient Frontier points
    """

    def __init__(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        risk_free_rate: float = RISK_FREE_RATE,
    ):
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.risk_free_rate = risk_free_rate
        self._returns: Optional[pd.DataFrame] = None
        self._mean_returns: Optional[np.ndarray] = None
        self._cov_matrix: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────────

    def fetch_data(self) -> pd.DataFrame:
        """Download price data and compute daily returns."""
        logger.info("Fetching %s data for %d symbols", self.period, len(self.symbols))
        prices = yf.download(
            self.symbols, period=self.period, interval=self.interval,
            auto_adjust=True, progress=False
        )["Close"]

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(self.symbols[0])

        prices = prices.dropna(how="all").ffill().dropna()
        self._returns = prices.pct_change().dropna()
        self._mean_returns = self._returns.mean().values * TRADING_DAYS
        self._cov_matrix = self._returns.cov().values * TRADING_DAYS
        logger.info("Returns computed: %d days x %d symbols", *self._returns.shape)
        return self._returns

    def max_sharpe(self) -> Dict:
        """Return weights that maximise the Sharpe ratio."""
        self._ensure_data()
        n = len(self.symbols)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(
            lambda w: -self._sharpe(w),
            x0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        weights = self._clean_weights(result.x)
        ret, vol, sharpe = self._portfolio_stats(weights)
        return self._build_result("max_sharpe", weights, ret, vol, sharpe)

    def min_volatility(self) -> Dict:
        """Return weights that minimise portfolio volatility."""
        self._ensure_data()
        n = len(self.symbols)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(
            lambda w: self._volatility(w),
            x0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        weights = self._clean_weights(result.x)
        ret, vol, sharpe = self._portfolio_stats(weights)
        return self._build_result("min_volatility", weights, ret, vol, sharpe)

    def equal_weight(self) -> Dict:
        """Equal-weight portfolio — always available as fallback."""
        self._ensure_data()
        n = len(self.symbols)
        weights = np.ones(n) / n
        ret, vol, sharpe = self._portfolio_stats(weights)
        return self._build_result("equal_weight", weights, ret, vol, sharpe)

    def efficient_frontier(self, n_points: int = 50) -> List[Dict]:
        """
        Compute n_points along the efficient frontier by sweeping
        target returns from min to max.
        """
        self._ensure_data()
        n = len(self.symbols)
        min_ret = float(self._mean_returns.min())
        max_ret = float(self._mean_returns.max())
        targets = np.linspace(min_ret, max_ret, n_points)
        frontier = []

        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: self._return(w) - t},
            ]
            result = minimize(
                lambda w: self._volatility(w),
                np.ones(n) / n,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )
            if result.success:
                w = self._clean_weights(result.x)
                ret, vol, sharpe = self._portfolio_stats(w)
                frontier.append({"return": round(ret, 4), "volatility": round(vol, 4), "sharpe": round(sharpe, 4)})

        return frontier

    def optimize(self) -> Dict:
        """
        Run all three optimizations and return a combined report.
        Falls back to equal weight if optimization fails.
        """
        if self._returns is None:
            self.fetch_data()

        try:
            ms = self.max_sharpe()
        except Exception as e:
            logger.warning("Max Sharpe failed: %s — using equal weight", e)
            ms = self.equal_weight()

        try:
            mv = self.min_volatility()
        except Exception as e:
            logger.warning("Min Vol failed: %s — using equal weight", e)
            mv = self.equal_weight()

        ew = self.equal_weight()

        return {
            "symbols": self.symbols,
            "period": self.period,
            "max_sharpe": ms,
            "min_volatility": mv,
            "equal_weight": ew,
            "recommended": ms,   # max Sharpe is the default recommendation
        }

    def print_report(self, result: Optional[Dict] = None) -> None:
        """Print a clean optimization report."""
        if result is None:
            result = self.optimize()

        print("\n" + "=" * 55)
        print("  PORTFOLIO OPTIMIZATION REPORT")
        print("=" * 55)
        print(f"Symbols : {', '.join(result['symbols'])}")
        print(f"Period  : {result['period']}")
        print("-" * 55)

        for key in ("max_sharpe", "min_volatility", "equal_weight"):
            p = result[key]
            print(f"\n{p['strategy'].upper().replace('_', ' ')}")
            print(f"  Expected Return : {p['annual_return']:.2%}")
            print(f"  Volatility      : {p['annual_volatility']:.2%}")
            print(f"  Sharpe Ratio    : {p['sharpe_ratio']:.3f}")
            print("  Weights:")
            for sym, w in p["weights"].items():
                bar = "█" * int(w * 20)
                print(f"    {sym:<15} {w:.1%}  {bar}")

        print("\n" + "=" * 55)
        rec = result["recommended"]
        print(f"Recommended: {rec['strategy']} | Sharpe={rec['sharpe_ratio']:.3f}")
        print("=" * 55 + "\n")

    # ── Private ───────────────────────────────────────────────────────────

    def _ensure_data(self) -> None:
        if self._returns is None:
            self.fetch_data()

    def _return(self, weights: np.ndarray) -> float:
        return float(np.dot(weights, self._mean_returns))

    def _volatility(self, weights: np.ndarray) -> float:
        return float(np.sqrt(weights @ self._cov_matrix @ weights))

    def _sharpe(self, weights: np.ndarray) -> float:
        ret = self._return(weights)
        vol = self._volatility(weights)
        return (ret - self.risk_free_rate) / vol if vol > 0 else 0.0

    def _portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        ret = self._return(weights)
        vol = self._volatility(weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def _clean_weights(self, weights: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        """Zero out tiny weights and renormalise."""
        w = np.where(weights < threshold, 0.0, weights)
        total = w.sum()
        return w / total if total > 0 else np.ones(len(weights)) / len(weights)

    def _build_result(
        self, strategy: str, weights: np.ndarray,
        ret: float, vol: float, sharpe: float
    ) -> Dict:
        return {
            "strategy": strategy,
            "weights": {sym: round(float(w), 4) for sym, w in zip(self.symbols, weights)},
            "annual_return": round(ret, 4),
            "annual_volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 4),
        }
