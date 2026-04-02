"""
Fundamental Analysis Engine — agents/fundamental_agent.py

Fetches stock fundamentals via yfinance, calculates financial ratios,
and produces a 0-100 fundamental score for use in TradingDecisionEngine.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # Python < 3.8

import yfinance as yf

logger = logging.getLogger(__name__)


class FundamentalRatios(TypedDict, total=False):
    """All 12 fundamental ratio fields. Values are None when unavailable."""
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    roe: Optional[float]           # percentage (e.g. 22.5 means 22.5%)
    roa: Optional[float]           # percentage
    debt_to_equity: Optional[float]
    eps: Optional[float]
    revenue_growth: Optional[float]    # percentage
    profit_margin: Optional[float]     # percentage
    peg_ratio: Optional[float]
    dividend_yield: Optional[float]    # percentage
    free_cash_flow: Optional[float]
    current_ratio: Optional[float]


class FundamentalAnalyzer:
    """
    Fetches and scores stock fundamentals using yfinance.

    Results are cached in-memory for `cache_ttl_hours` hours to minimise
    API calls (Requirement 6.2).
    """

    def __init__(self, cache_ttl_hours: int = 24) -> None:
        self.cache_ttl_hours: int = cache_ttl_hours
        # In-memory cache: symbol -> {"timestamp": datetime, "info": dict}
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_financial_ratios(self, symbol: str) -> FundamentalRatios:
        """Return the 12 fundamental ratios for *symbol* (Requirement 1.2)."""
        info = self._fetch_from_cache_or_api(symbol)

        def _get(key: str, scale: float = 1.0) -> Optional[float]:
            val = info.get(key)
            if val is None:
                return None
            try:
                return float(val) * scale
            except (TypeError, ValueError):
                return None

        return FundamentalRatios(
            pe_ratio=_get("trailingPE"),
            pb_ratio=_get("priceToBook"),
            roe=_get("returnOnEquity", 100.0),       # yfinance gives 0-1 fraction
            roa=_get("returnOnAssets", 100.0),
            debt_to_equity=_get("debtToEquity"),
            eps=_get("trailingEps"),
            revenue_growth=_get("revenueGrowth", 100.0),
            profit_margin=_get("profitMargins", 100.0),
            peg_ratio=_get("pegRatio"),
            dividend_yield=_get("dividendYield"),       # yfinance already returns as percentage
            free_cash_flow=_get("freeCashflow"),
            current_ratio=_get("currentRatio"),
        )

    def calculate_fundamental_score(self, symbol: str) -> float:
        """Return a 0-100 fundamental score for *symbol* (Requirement 1.4)."""
        ratios = self.get_financial_ratios(symbol)

        val_score, val_available = self._valuation_score(ratios)
        prof_score, prof_available = self._profitability_score(ratios)
        health_score, health_available = self._health_score(ratios)

        # If nothing is available at all, return neutral 50 (Requirement 3.6)
        if not val_available and not prof_available and not health_available:
            logger.warning(
                "FundamentalAnalyzer: all ratios unavailable for %s — returning neutral 50",
                symbol,
            )
            return 50.0

        # Redistribute weights when a sub-score has no data (Requirement 3.5)
        raw_weights = {
            "valuation": 40.0 if val_available else 0.0,
            "profitability": 35.0 if prof_available else 0.0,
            "health": 25.0 if health_available else 0.0,
        }
        total_w = sum(raw_weights.values())
        weights = {k: v / total_w for k, v in raw_weights.items()}

        score = (
            val_score * weights["valuation"]
            + prof_score * weights["profitability"]
            + health_score * weights["health"]
        )
        return round(min(100.0, max(0.0, score)), 2)

    def get_sector_comparison(self, symbol: str) -> Dict[str, Any]:
        """Return sector comparison data for *symbol* (Requirement 4.1)."""
        info = self._fetch_from_cache_or_api(symbol)

        sector = info.get("sector")
        industry = info.get("industry")
        sector_pe = _safe_float(info.get("sectorPE") or info.get("forwardPE"))
        stock_pe = _safe_float(info.get("trailingPE"))
        sector_roe = None  # yfinance doesn't expose sector ROE directly
        stock_roe = _safe_float(info.get("returnOnEquity"))
        if stock_roe is not None:
            stock_roe = round(stock_roe * 100.0, 2)

        pe_vs_sector = None
        if stock_pe is not None and sector_pe is not None and sector_pe != 0:
            pe_vs_sector = round(stock_pe / sector_pe, 4)

        roe_vs_sector = None
        if stock_roe is not None and sector_roe is not None and sector_roe != 0:
            roe_vs_sector = round(stock_roe / sector_roe, 4)

        return {
            "sector": sector,
            "industry": industry,
            "sector_pe": sector_pe,
            "stock_pe": stock_pe,
            "pe_vs_sector": pe_vs_sector,
            "sector_roe": sector_roe,
            "stock_roe": stock_roe,
            "roe_vs_sector": roe_vs_sector,
        }

    def clear_cache(self) -> None:
        """Evict all cached entries (Requirement 6.2)."""
        self._cache.clear()
        logger.debug("FundamentalAnalyzer cache cleared")

    # ------------------------------------------------------------------
    # Private: cache + fetch
    # ------------------------------------------------------------------

    def _fetch_from_cache_or_api(self, symbol: str) -> Dict[str, Any]:
        """
        Return yfinance info dict for *symbol*.
        Serves from cache if within TTL, otherwise fetches fresh data.
        (Requirements 1.5, 6.2, 6.3)
        """
        now = datetime.now()
        entry = self._cache.get(symbol)
        if entry is not None:
            age = now - entry["timestamp"]
            if age < timedelta(hours=self.cache_ttl_hours):
                logger.debug("FundamentalAnalyzer: cache hit for %s", symbol)
                return entry["info"]

        # Cache miss or expired — fetch from yfinance (Requirement 6.1)
        try:
            logger.debug("FundamentalAnalyzer: fetching yfinance data for %s", symbol)
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
        except Exception as exc:
            logger.warning(
                "FundamentalAnalyzer: yfinance fetch failed for %s: %s", symbol, exc
            )
            info = {}

        self._cache[symbol] = {"timestamp": now, "info": info}
        return info

    # ------------------------------------------------------------------
    # Private: sub-score calculators
    # ------------------------------------------------------------------

    def _valuation_score(self, ratios: FundamentalRatios):
        """
        Score valuation metrics: P/E, P/B, PEG, Dividend Yield.
        Returns (score, has_data).  (Requirements 3.1, 3.2, 3.5)
        """
        scores = []

        pe = ratios.get("pe_ratio")
        if pe is not None:
            if pe <= 0 or pe > 50:
                scores.append(0.0)
            elif pe <= 25:
                scores.append(100.0)
            else:  # 25-50 linear decay
                scores.append(100.0 * (50.0 - pe) / 25.0)

        pb = ratios.get("pb_ratio")
        if pb is not None:
            if pb <= 0:
                scores.append(50.0)
            elif pb <= 1.5:
                scores.append(100.0)
            elif pb <= 5.0:
                scores.append(100.0 * (5.0 - pb) / 3.5)
            else:
                scores.append(0.0)

        peg = ratios.get("peg_ratio")
        if peg is not None:
            if peg <= 0:
                scores.append(50.0)
            elif peg <= 1.0:
                scores.append(100.0)
            elif peg <= 2.0:
                scores.append(100.0 * (2.0 - peg))
            else:
                scores.append(0.0)

        dy = ratios.get("dividend_yield")
        if dy is not None:
            # Cap at 20% — anything higher is a data artifact
            dy = min(dy, 20.0)
            if dy >= 3.0:
                scores.append(100.0)
            elif dy >= 1.0:
                scores.append(50.0 + 50.0 * (dy - 1.0) / 2.0)
            else:
                scores.append(50.0)

        if not scores:
            return 50.0, False
        return sum(scores) / len(scores), True

    def _profitability_score(self, ratios: FundamentalRatios):
        """
        Score profitability metrics: ROE, ROA, Profit Margin, Revenue Growth.
        Returns (score, has_data).  (Requirements 3.1, 3.3, 3.5)
        """
        scores = []

        roe = ratios.get("roe")
        if roe is not None:
            if roe < 0:
                scores.append(0.0)
            elif roe >= 20.0:
                scores.append(100.0)
            elif roe >= 10.0:
                scores.append(50.0 + 50.0 * (roe - 10.0) / 10.0)
            else:
                scores.append(50.0)

        roa = ratios.get("roa")
        if roa is not None:
            if roa < 0:
                scores.append(0.0)
            elif roa >= 10.0:
                scores.append(100.0)
            elif roa >= 5.0:
                scores.append(50.0 + 50.0 * (roa - 5.0) / 5.0)
            else:
                scores.append(50.0)

        pm = ratios.get("profit_margin")
        if pm is not None:
            if pm < 0:
                scores.append(0.0)
            elif pm >= 20.0:
                scores.append(100.0)
            elif pm >= 10.0:
                scores.append(50.0 + 50.0 * (pm - 10.0) / 10.0)
            else:
                scores.append(50.0)

        rg = ratios.get("revenue_growth")
        if rg is not None:
            if rg < 0:
                scores.append(0.0)
            elif rg >= 15.0:
                scores.append(100.0)
            elif rg >= 5.0:
                scores.append(50.0 + 50.0 * (rg - 5.0) / 10.0)
            else:
                scores.append(50.0)

        if not scores:
            return 50.0, False
        return sum(scores) / len(scores), True

    def _health_score(self, ratios: FundamentalRatios):
        """
        Score financial health metrics: D/E, Current Ratio, Free Cash Flow.
        Returns (score, has_data).  (Requirements 3.1, 3.4, 3.5)
        """
        scores = []

        de = ratios.get("debt_to_equity")
        if de is not None:
            # yfinance returns D/E as a percentage (e.g. 35.6 means 0.356 ratio)
            # Normalise: values > 5 are almost certainly in percentage form
            if de > 5.0:
                de = de / 100.0
            if de < 0:
                scores.append(50.0)
            elif de <= 0.5:
                scores.append(100.0)
            elif de <= 1.5:
                scores.append(100.0 - 50.0 * (de - 0.5) / 1.0)
            elif de <= 3.0:
                scores.append(50.0 - 50.0 * (de - 1.5) / 1.5)
            else:
                scores.append(0.0)

        cr = ratios.get("current_ratio")
        if cr is not None:
            if cr >= 2.0:
                scores.append(100.0)
            elif cr >= 1.0:
                scores.append(50.0 + 50.0 * (cr - 1.0))
            else:
                scores.append(50.0)

        fcf = ratios.get("free_cash_flow")
        if fcf is not None:
            scores.append(100.0 if fcf >= 0 else 0.0)

        if not scores:
            return 50.0, False
        return sum(scores) / len(scores), True


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
