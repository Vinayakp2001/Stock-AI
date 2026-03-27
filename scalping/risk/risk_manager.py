"""
Risk Management Module (Issue #4)
Portfolio-level capital protection: position sizing, daily loss circuit-breaker,
drawdown guard, and consecutive loss cooldown.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, Optional
from typing_extensions import TypedDict

from scalping.config import CONSERVATIVE, AGGRESSIVE

logger = logging.getLogger(__name__)

# Default base risk pct (overridden by mode config)
_DEFAULT_RISK_PCT = 0.25   # 25% of capital — within the 20-30% band

# Rule thresholds
CAPITAL_MIN          = 10_000.0   # ₹10,000
DAILY_LOSS_LIMIT_PCT = 0.02       # -2% of daily start capital
DRAWDOWN_REDUCE_PCT  = 0.05       # 5%  → halve risk_pct
DRAWDOWN_HALT_PCT    = 0.10       # 10% → halt
COOLDOWN_LOSSES      = 3          # consecutive losses before cooldown
COOLDOWN_MINUTES     = 15         # cooldown duration


class RiskDecision(TypedDict):
    allowed: bool
    quantity: int       # 0 if blocked
    reason: str         # "" if allowed, else rule name constant
    risk_pct: float     # effective risk % used for sizing


class RiskManager:
    """
    Single entry-point for all portfolio-level risk rules.

    Usage:
        rm = RiskManager(initial_capital=100_000, mode="conservative")
        decision = rm.check_entry("BUY", capital, entry_price, timestamp)
        if decision["allowed"]:
            # open trade …
            rm.record_trade_result(trade)
    """

    def __init__(self, initial_capital: float, mode: str = "conservative"):
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if mode not in ("conservative", "aggressive"):
            raise ValueError(f"mode must be 'conservative' or 'aggressive', got {mode}")

        self._mode = mode
        cfg = CONSERVATIVE if mode == "conservative" else AGGRESSIVE
        self._base_risk_pct: float = cfg.get("max_risk_per_trade_pct", _DEFAULT_RISK_PCT)

        # Internal state
        self._peak_capital: float        = initial_capital
        self._daily_start_capital: float = initial_capital
        self._daily_pnl: float           = 0.0
        self._trading_halted: bool       = False
        self._halt_reason: str           = ""
        self._consecutive_losses: int    = 0
        self._cooldown_until: Optional[datetime] = None
        self._current_day: Optional[date]        = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def check_entry(
        self,
        signal: str,
        capital: float,
        entry_price: float,
        timestamp: datetime,
    ) -> RiskDecision:
        """
        Evaluate all risk rules and return a RiskDecision.

        Rules are evaluated in priority order; first block wins.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if capital <= 0:
            raise ValueError(f"capital must be positive, got {capital}")
        if signal not in ("BUY", "SELL"):
            raise ValueError(f"signal must be 'BUY' or 'SELL', got {signal}")

        # Auto-reset on day boundary
        self._maybe_reset_day(timestamp, capital)

        # ── Rule 1: Capital too low ───────────────────────────────────────────
        if capital < CAPITAL_MIN:
            return self._block("CAPITAL_TOO_LOW", 0.0)

        # ── Rule 2: Daily loss circuit-breaker ───────────────────────────────
        daily_loss_threshold = -DAILY_LOSS_LIMIT_PCT * self._daily_start_capital
        if self._daily_pnl <= daily_loss_threshold:
            if not self._trading_halted:
                self._trading_halted = True
                self._halt_reason = "DAILY_LOSS_LIMIT"
                logger.warning(
                    "DAILY_LOSS_LIMIT hit at %s | daily_pnl=%.2f | threshold=%.2f",
                    timestamp, self._daily_pnl, daily_loss_threshold,
                )
            return self._block("DAILY_LOSS_LIMIT", 0.0)

        # ── Rule 3: Max drawdown halt ─────────────────────────────────────────
        drawdown = self._drawdown(capital)
        if drawdown > DRAWDOWN_HALT_PCT:
            if not self._trading_halted:
                self._trading_halted = True
                self._halt_reason = "MAX_DRAWDOWN"
                logger.warning(
                    "MAX_DRAWDOWN hit at %s | drawdown=%.2f%% | capital=%.2f",
                    timestamp, drawdown * 100, capital,
                )
            return self._block("MAX_DRAWDOWN", 0.0)

        # ── Rule 4: Consecutive loss cooldown ─────────────────────────────────
        if self._cooldown_until is not None and timestamp < self._cooldown_until:
            return self._block("CONSECUTIVE_LOSS_COOLDOWN", 0.0)

        # ── All clear: compute position size ─────────────────────────────────
        effective_risk_pct = self._base_risk_pct
        if drawdown > DRAWDOWN_REDUCE_PCT:
            effective_risk_pct = self._base_risk_pct * 0.5

        trade_value = capital * effective_risk_pct
        trade_value = max(capital * 0.20, min(capital * 0.30, trade_value))
        quantity = max(1, int(trade_value / entry_price))

        return RiskDecision(
            allowed=True,
            quantity=quantity,
            reason="",
            risk_pct=effective_risk_pct,
        )

    def record_trade_result(self, trade: Any) -> None:
        """
        Update internal state after a trade closes.
        `trade` must have: net_pnl (float), status (str: WIN/LOSS/STOPPED).
        """
        net_pnl = float(trade.net_pnl)
        status  = str(trade.status)

        self._daily_pnl += net_pnl

        # Update peak capital (use daily_start + daily_pnl as proxy for current capital)
        current_capital = self._daily_start_capital + self._daily_pnl
        if current_capital > self._peak_capital:
            self._peak_capital = current_capital

        # Consecutive loss tracking
        if status in ("LOSS", "STOPPED"):
            self._consecutive_losses += 1
            if self._consecutive_losses >= COOLDOWN_LOSSES:
                # Only set cooldown if not already active
                if self._cooldown_until is None or datetime.now() >= self._cooldown_until:
                    self._cooldown_until = datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
                    logger.warning(
                        "CONSECUTIVE_LOSS_COOLDOWN activated | losses=%d | until=%s",
                        self._consecutive_losses, self._cooldown_until,
                    )
        elif status == "WIN":
            self._consecutive_losses = 0

        # Restore full risk_pct if drawdown recovered below 5%
        # (handled dynamically in check_entry — no extra state needed)

    def reset_day(self, new_capital: Optional[float] = None) -> None:
        """Reset all daily counters. Call at the start of each trading day."""
        if new_capital is not None and new_capital > 0:
            self._daily_start_capital = new_capital
        self._daily_pnl      = 0.0
        self._trading_halted = False
        self._halt_reason    = ""
        self._current_day    = None   # will be set on next check_entry call

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of current risk state."""
        current_capital = self._daily_start_capital + self._daily_pnl
        drawdown = self._drawdown(current_capital)
        daily_pnl_pct = (
            self._daily_pnl / self._daily_start_capital
            if self._daily_start_capital > 0 else 0.0
        )
        cooldown_active = (
            self._cooldown_until is not None
            and datetime.now() < self._cooldown_until
        )
        return {
            "capital":           current_capital,
            "peak_capital":      self._peak_capital,
            "drawdown_pct":      round(drawdown, 6),
            "daily_pnl":         round(self._daily_pnl, 2),
            "daily_pnl_pct":     round(daily_pnl_pct, 6),
            "trading_halted":    self._trading_halted,
            "halt_reason":       self._halt_reason,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_active":   cooldown_active,
            "cooldown_until":    self._cooldown_until,
            "mode":              self._mode,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _block(self, reason: str, risk_pct: float) -> RiskDecision:
        return RiskDecision(allowed=False, quantity=0, reason=reason, risk_pct=risk_pct)

    def _drawdown(self, capital: float) -> float:
        if self._peak_capital <= 0:
            return 0.0
        return max(0.0, (self._peak_capital - capital) / self._peak_capital)

    def _maybe_reset_day(self, timestamp: datetime, capital: float) -> None:
        """Auto-reset daily state when the trading day changes."""
        today = timestamp.date() if hasattr(timestamp, "date") else date.today()
        if self._current_day is None:
            self._current_day = today
            self._daily_start_capital = capital
        elif today != self._current_day:
            logger.info("New trading day %s — resetting daily risk counters", today)
            self._current_day        = today
            self._daily_start_capital = capital
            self._daily_pnl          = 0.0
            self._trading_halted     = False
            self._halt_reason        = ""
