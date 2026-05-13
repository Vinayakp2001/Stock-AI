"""
Sentiment Analysis Engine — agents/sentiment_agent.py

Fetches recent news headlines via yfinance and scores sentiment using
TextBlob, producing a 0-100 sentiment score for TradingDecisionEngine.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from textblob import TextBlob
import yfinance as yf

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Fetches stock news headlines via yfinance and scores sentiment
    using TextBlob polarity analysis.

    Score range: 0 (very negative) to 100 (very positive), 50 = neutral.
    Results cached in-memory for `cache_ttl_hours` hours.
    """

    def __init__(self, cache_ttl_hours: int = 1, max_headlines: int = 20) -> None:
        self.cache_ttl_hours = cache_ttl_hours
        self.max_headlines = max_headlines
        # cache: symbol -> {"timestamp": datetime, "score": float, "headlines": list}
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sentiment_score(self, symbol: str) -> float:
        """
        Return a 0-100 sentiment score for *symbol*.
        50 = neutral, >50 = positive, <50 = negative.
        Serves from cache if within TTL.
        """
        cached = self._get_from_cache(symbol)
        if cached is not None:
            return cached["score"]

        headlines = self._fetch_headlines(symbol)
        score = self._score_headlines(headlines)

        self._cache[symbol] = {
            "timestamp": datetime.now(),
            "score": score,
            "headlines": headlines,
        }
        logger.debug(
            "SentimentAnalyzer: %s — %d headlines, score=%.1f", symbol, len(headlines), score
        )
        return score

    def get_sentiment_detail(self, symbol: str) -> Dict[str, Any]:
        """
        Return score plus headline-level breakdown for inspection.
        """
        # Ensure cache is populated
        self.get_sentiment_score(symbol)
        entry = self._cache.get(symbol, {})
        headlines = entry.get("headlines", [])

        scored = []
        for h in headlines:
            polarity = TextBlob(h).sentiment.polarity
            scored.append({"headline": h, "polarity": round(polarity, 4)})

        return {
            "symbol": symbol,
            "score": entry.get("score", 50.0),
            "headline_count": len(headlines),
            "headlines": scored,
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.debug("SentimentAnalyzer cache cleared")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_from_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(symbol)
        if entry is None:
            return None
        age = datetime.now() - entry["timestamp"]
        if age < timedelta(hours=self.cache_ttl_hours):
            logger.debug("SentimentAnalyzer: cache hit for %s", symbol)
            return entry
        return None

    def _fetch_headlines(self, symbol: str) -> List[str]:
        """Fetch recent news headlines from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            headlines = []
            for item in news[: self.max_headlines]:
                # New yfinance API: title is nested under item['content']['title']
                content = item.get("content", {})
                title = content.get("title") or item.get("title", "")
                summary = content.get("summary", "")
                text = f"{title}. {summary}".strip(". ") if summary else title
                if text:
                    headlines.append(text)
            return headlines
        except Exception as exc:
            logger.warning("SentimentAnalyzer: news fetch failed for %s: %s", symbol, exc)
            return []

    def _score_headlines(self, headlines: List[str]) -> float:
        """
        Average TextBlob polarity across headlines, map to 0-100.
        TextBlob polarity: -1.0 (negative) to +1.0 (positive).
        Mapping: polarity * 50 + 50 → 0-100.
        Returns 50.0 (neutral) if no headlines.
        """
        if not headlines:
            logger.warning("SentimentAnalyzer: no headlines found — returning neutral 50")
            return 50.0

        polarities = [TextBlob(h).sentiment.polarity for h in headlines]
        avg_polarity = sum(polarities) / len(polarities)

        # Map [-1, 1] → [0, 100]
        score = avg_polarity * 50.0 + 50.0
        return round(min(100.0, max(0.0, score)), 2)
