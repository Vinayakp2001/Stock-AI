"""
Multiple Asset Classes (Issue #53)
Detects asset class from symbol and provides class-specific configuration:
- NSE/BSE Indian equities
- US equities (NYSE/NASDAQ)
- Crypto (via yfinance -USD/-USDT suffix)
- Indices (^NSEI, ^GSPC, etc.)
- Futures (BNF, NIFTY)

Used by backtester, benchmark, and decision engine to apply
correct costs, session hours, currency, and lot sizes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class AssetClass(Enum):
    NSE_EQUITY  = "NSE_EQUITY"
    BSE_EQUITY  = "BSE_EQUITY"
    US_EQUITY   = "US_EQUITY"
    CRYPTO      = "CRYPTO"
    INDEX       = "INDEX"
    FUTURES     = "FUTURES"
    UNKNOWN     = "UNKNOWN"


@dataclass
class AssetConfig:
    symbol:         str
    asset_class:    AssetClass
    currency:       str          # INR | USD
    exchange:       str          # NSE | BSE | NYSE | NASDAQ | CRYPTO | INDEX
    currency_symbol: str         # ₹ | $
    lot_size:       int          # 1 for equities, varies for futures
    tick_size:      float        # minimum price movement
    # Transaction costs
    brokerage:      float        # flat fee per trade
    stt_pct:        float        # securities transaction tax %
    slippage_pct:   float        # estimated slippage %
    # Session (IST hours for India, ET for US, 24h for crypto)
    session_start:  str          # "HH:MM"
    session_end:    str          # "HH:MM"
    session_tz:     str          # timezone label
    # Data
    max_history_days: int        # yfinance 1m data limit
    supports_short:  bool


# ── Pre-defined configs ───────────────────────────────────────────────────────

_NSE = dict(
    currency="INR", exchange="NSE", currency_symbol="₹",
    lot_size=1, tick_size=0.05,
    brokerage=20.0, stt_pct=0.00025, slippage_pct=0.0003,
    session_start="09:15", session_end="15:30", session_tz="IST",
    max_history_days=7, supports_short=False,
)

_BSE = dict(
    currency="INR", exchange="BSE", currency_symbol="₹",
    lot_size=1, tick_size=0.01,
    brokerage=20.0, stt_pct=0.00025, slippage_pct=0.0003,
    session_start="09:15", session_end="15:30", session_tz="IST",
    max_history_days=7, supports_short=False,
)

_US = dict(
    currency="USD", exchange="NYSE/NASDAQ", currency_symbol="$",
    lot_size=1, tick_size=0.01,
    brokerage=0.0, stt_pct=0.0, slippage_pct=0.0002,
    session_start="09:30", session_end="16:00", session_tz="ET",
    max_history_days=7, supports_short=True,
)

_CRYPTO = dict(
    currency="USD", exchange="CRYPTO", currency_symbol="$",
    lot_size=1, tick_size=0.01,
    brokerage=0.0, stt_pct=0.0, slippage_pct=0.001,
    session_start="00:00", session_end="23:59", session_tz="UTC",
    max_history_days=7, supports_short=True,
)

_INDEX = dict(
    currency="INR", exchange="INDEX", currency_symbol="₹",
    lot_size=1, tick_size=0.05,
    brokerage=0.0, stt_pct=0.0, slippage_pct=0.0,
    session_start="09:15", session_end="15:30", session_tz="IST",
    max_history_days=7, supports_short=False,
)

_FUTURES = dict(
    currency="INR", exchange="NSE", currency_symbol="₹",
    lot_size=25, tick_size=0.05,
    brokerage=20.0, stt_pct=0.0001, slippage_pct=0.0003,
    session_start="09:15", session_end="15:30", session_tz="IST",
    max_history_days=7, supports_short=True,
)


class AssetUniverse:
    """
    Detects asset class from symbol string and returns AssetConfig.

    Detection rules (in order):
    1. Starts with ^ → INDEX
    2. Ends with -USD or -USDT → CRYPTO
    3. Ends with .NS → NSE_EQUITY
    4. Ends with .BO → BSE_EQUITY
    5. Known futures keywords (BNF, NIFTY, BANKNIFTY) → FUTURES
    6. All uppercase, 1-5 chars, no dot → US_EQUITY
    7. Fallback → UNKNOWN
    """

    # Curated watchlists per asset class
    WATCHLISTS: Dict[str, List[str]] = {
        "NSE_LARGE_CAP": [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
            "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS",
            "BHARTIARTL.NS", "AXISBANK.NS",
        ],
        "US_LARGE_CAP": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "BRK-B", "JPM", "V",
        ],
        "CRYPTO": [
            "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
        ],
        "INDICES": [
            "^NSEI", "^BSESN", "^GSPC", "^IXIC", "^DJI",
        ],
    }

    @classmethod
    def detect(cls, symbol: str) -> AssetConfig:
        """Detect asset class and return config for the given symbol."""
        ac = cls._classify(symbol)
        template = cls._template(ac)
        return AssetConfig(symbol=symbol, asset_class=ac, **template)

    @classmethod
    def _classify(cls, symbol: str) -> AssetClass:
        s = symbol.upper()
        if s.startswith("^"):
            return AssetClass.INDEX
        if s.endswith("-USD") or s.endswith("-USDT"):
            return AssetClass.CRYPTO
        if symbol.endswith(".NS"):
            return AssetClass.NSE_EQUITY
        if symbol.endswith(".BO"):
            return AssetClass.BSE_EQUITY
        if any(k in s for k in ("BANKNIFTY", "NIFTY", "BNF")):
            return AssetClass.FUTURES
        # US equity: uppercase letters only, 1-5 chars, optional hyphen
        if s.replace("-", "").isalpha() and 1 <= len(s.replace("-", "")) <= 5:
            return AssetClass.US_EQUITY
        return AssetClass.UNKNOWN

    @classmethod
    def _template(cls, ac: AssetClass) -> dict:
        mapping = {
            AssetClass.NSE_EQUITY: _NSE,
            AssetClass.BSE_EQUITY: _BSE,
            AssetClass.US_EQUITY:  _US,
            AssetClass.CRYPTO:     _CRYPTO,
            AssetClass.INDEX:      _INDEX,
            AssetClass.FUTURES:    _FUTURES,
            AssetClass.UNKNOWN:    _NSE,   # safe default
        }
        return dict(mapping[ac])

    @classmethod
    def get_watchlist(cls, category: str) -> List[str]:
        return cls.WATCHLISTS.get(category, [])

    @classmethod
    def transaction_cost(cls, symbol: str, price: float, quantity: int) -> float:
        """Calculate total transaction cost for one side of a trade."""
        cfg = cls.detect(symbol)
        trade_value = price * quantity
        return cfg.brokerage + trade_value * (cfg.stt_pct + cfg.slippage_pct)

    @classmethod
    def is_market_open(cls, symbol: str) -> bool:
        """Simple check — returns True during session hours (naive, no holidays)."""
        from datetime import datetime
        import pytz
        cfg = cls.detect(symbol)
        tz_map = {"IST": "Asia/Kolkata", "ET": "America/New_York", "UTC": "UTC"}
        tz = pytz.timezone(tz_map.get(cfg.session_tz, "UTC"))
        now = datetime.now(tz)
        start = datetime.strptime(cfg.session_start, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day, tzinfo=tz)
        end = datetime.strptime(cfg.session_end, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day, tzinfo=tz)
        return start <= now <= end
