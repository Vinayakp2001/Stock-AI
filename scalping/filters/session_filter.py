"""
Session Filter - Only trade during highest-liquidity windows.

NSE:  09:15 - 11:15 IST  (optimal window)
NYSE: 09:30 - 11:30 EST  (optimal window)

Positions are closed 15 minutes before market close.
"""

from datetime import time, datetime
from typing import Tuple
import pytz
import logging

logger = logging.getLogger(__name__)

# Market close times (used for end-of-day position close trigger)
_MARKET_CLOSE = {
    "NSE":  time(15, 30),   # 3:30 PM IST
    "NYSE": time(16, 0),    # 4:00 PM EST
}

# Optimal trading windows (high-liquidity open)
_TRADING_WINDOWS = {
    "NSE":  (time(9, 15),  time(15, 15)),   # Full day minus last 15 min (Req 1.1)
    "NYSE": (time(9, 30),  time(15, 45)),   # Full day minus last 15 min (Req 1.4)
}

# Timezones
_TIMEZONES = {
    "NSE":  pytz.timezone("Asia/Kolkata"),
    "NYSE": pytz.timezone("America/New_York"),
}

CLOSE_BUFFER_MINUTES = 15   # Req 1.3


class SessionFilter:
    """
    Checks whether a given timestamp falls within the optimal
    trading window for a market, and whether open positions
    should be closed ahead of market close.
    """

    def get_trading_window(self, market: str = "NSE") -> Tuple[time, time]:
        """Return (start_time, end_time) for the market's optimal window."""
        market = market.upper()
        if market not in _TRADING_WINDOWS:
            raise ValueError(f"Unknown market '{market}'. Supported: {list(_TRADING_WINDOWS)}")
        return _TRADING_WINDOWS[market]

    def _local_time(self, timestamp, market: str) -> time:
        """Convert a timestamp to the market's local time."""
        tz = _TIMEZONES[market]
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                # Assume UTC if naive
                timestamp = pytz.utc.localize(timestamp)
            local_dt = timestamp.astimezone(tz)
        else:
            # pandas Timestamp
            try:
                local_dt = timestamp.tz_convert(tz)
            except Exception:
                local_dt = pytz.utc.localize(timestamp.to_pydatetime()).astimezone(tz)
        return local_dt.time()

    def is_trading_time(self, timestamp, market: str = "NSE") -> bool:
        """
        Returns True if timestamp is within the optimal trading window.
        Rejects signal and logs reason when outside window. (Req 1.1, 1.2, 1.4, 1.5)
        """
        market = market.upper()
        local_t = self._local_time(timestamp, market)
        start, end = _TRADING_WINDOWS[market]

        if start <= local_t <= end:
            return True

        logger.info(
            "Signal rejected | reason=outside_trading_window | market=%s | time=%s | window=%s-%s",
            market, local_t, start, end
        )
        return False

    def should_close_positions(self, timestamp, market: str = "NSE") -> bool:
        """
        Returns True if within CLOSE_BUFFER_MINUTES of market close.
        Triggers end-of-day position close. (Req 1.3)
        """
        market = market.upper()
        local_t = self._local_time(timestamp, market)
        close_t = _MARKET_CLOSE[market]

        # Convert both to minutes-since-midnight for easy comparison
        local_mins = local_t.hour * 60 + local_t.minute
        close_mins = close_t.hour * 60 + close_t.minute

        return close_mins - CLOSE_BUFFER_MINUTES <= local_mins < close_mins
