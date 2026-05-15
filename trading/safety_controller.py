"""
Safety Control System (Issue #46)
Emergency kill switch + circuit breakers for the trading bot.

Checks:
- Daily loss limit (hard stop)
- Max drawdown limit
- Consecutive loss streak
- Max trades per day
- Position size limit
- Manual kill switch
- Unusual activity (trade frequency spike)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_PATH = os.path.join("data", "safety", "safety_state.json")


@dataclass
class SafetyConfig:
    max_daily_loss_pct:     float = 3.0    # halt if daily loss > 3%
    max_drawdown_pct:       float = 10.0   # halt if drawdown > 10%
    max_consecutive_losses: int   = 5      # halt after 5 losses in a row
    max_trades_per_day:     int   = 20     # hard cap on daily trades
    max_position_size_pct:  float = 30.0   # max % of capital per trade
    max_trades_per_minute:  float = 10.0   # unusual activity threshold


@dataclass
class SafetyStatus:
    trading_allowed:      bool
    kill_switch_active:   bool
    halt_reason:          str
    daily_loss_pct:       float
    drawdown_pct:         float
    consecutive_losses:   int
    trades_today:         int
    checks_passed:        Dict[str, bool]
    timestamp:            str = field(default_factory=lambda: datetime.now().isoformat())


class SafetyController:
    """
    Central safety gate. Call check_trade() before every order.
    All trading halts are logged and persisted across restarts.
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        initial_capital: float = 100_000.0,
        alert_system=None,
    ):
        self.config = config or SafetyConfig()
        self.initial_capital = initial_capital
        self._alert = alert_system

        # State
        self._kill_switch: bool = False
        self._halt_reason: str = ""
        self._peak_capital: float = initial_capital
        self._current_capital: float = initial_capital
        self._daily_loss: float = 0.0
        self._consecutive_losses: int = 0
        self._trades_today: int = 0
        self._trade_day: date = date.today()
        self._recent_trade_times: List[datetime] = []

        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        self._load_state()

    # ── Public API ────────────────────────────────────────────────────────

    def check_trade(
        self,
        capital: float,
        position_size: float,
        symbol: str = "",
    ) -> bool:
        """
        Gate check before placing any order.
        Returns True if trading is allowed, False if blocked.
        """
        self._refresh_day(capital)

        if self._kill_switch:
            logger.warning("SafetyController: KILL SWITCH active — trade blocked (%s)", symbol)
            return False

        checks = self._run_checks(capital, position_size)
        all_passed = all(checks.values())

        if not all_passed:
            failed = [k for k, v in checks.items() if not v]
            self._halt_reason = ", ".join(failed)
            logger.warning("SafetyController: trade BLOCKED — %s", self._halt_reason)
            if self._alert:
                self._alert.risk_breach(self._halt_reason, capital, self.initial_capital, symbol)
        return all_passed

    def record_trade_result(self, pnl: float, capital: float) -> None:
        """Call after every trade closes."""
        self._refresh_day(capital)
        self._current_capital = capital
        self._peak_capital = max(self._peak_capital, capital)
        self._trades_today += 1
        self._recent_trade_times.append(datetime.now())

        if pnl < 0:
            self._daily_loss += abs(pnl)
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._save_state()

        # Auto-halt checks after recording
        drawdown = (self._peak_capital - capital) / self._peak_capital * 100
        daily_loss_pct = self._daily_loss / self.initial_capital * 100

        if daily_loss_pct >= self.config.max_daily_loss_pct:
            self._trigger_halt(f"Daily loss {daily_loss_pct:.1f}% >= limit {self.config.max_daily_loss_pct}%")
        elif drawdown >= self.config.max_drawdown_pct:
            self._trigger_halt(f"Drawdown {drawdown:.1f}% >= limit {self.config.max_drawdown_pct}%")
        elif self._consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger_halt(f"{self._consecutive_losses} consecutive losses")

    def activate_kill_switch(self, reason: str = "Manual") -> None:
        """Immediately halt all trading."""
        self._kill_switch = True
        self._halt_reason = f"KILL SWITCH: {reason}"
        logger.critical("SafetyController: KILL SWITCH ACTIVATED — %s", reason)
        if self._alert:
            self._alert.risk_breach("KILL_SWITCH", 0, 0)
        self._save_state()

    def reset_kill_switch(self) -> None:
        """Re-enable trading after manual review."""
        self._kill_switch = False
        self._halt_reason = ""
        self._consecutive_losses = 0
        self._daily_loss = 0.0
        self._trades_today = 0
        self._recent_trade_times = []
        logger.info("SafetyController: kill switch reset — trading re-enabled")
        self._save_state()

    def reset_day(self, capital: float) -> None:
        """Call at start of each trading day."""
        self._trade_day = date.today()
        self._daily_loss = 0.0
        self._trades_today = 0
        self._consecutive_losses = 0
        self._recent_trade_times = []
        self._current_capital = capital
        # Don't reset kill switch — requires manual reset
        logger.info("SafetyController: daily reset | capital=%.2f", capital)
        self._save_state()

    def get_status(self) -> SafetyStatus:
        drawdown = (self._peak_capital - self._current_capital) / self._peak_capital * 100
        daily_loss_pct = self._daily_loss / self.initial_capital * 100
        checks = self._run_checks(self._current_capital, 0)
        return SafetyStatus(
            trading_allowed=not self._kill_switch and all(checks.values()),
            kill_switch_active=self._kill_switch,
            halt_reason=self._halt_reason,
            daily_loss_pct=round(daily_loss_pct, 3),
            drawdown_pct=round(drawdown, 3),
            consecutive_losses=self._consecutive_losses,
            trades_today=self._trades_today,
            checks_passed=checks,
        )

    # ── Private ───────────────────────────────────────────────────────────

    def _run_checks(self, capital: float, position_size: float) -> Dict[str, bool]:
        drawdown = (self._peak_capital - capital) / self._peak_capital * 100
        daily_loss_pct = self._daily_loss / self.initial_capital * 100
        pos_pct = position_size / capital * 100 if capital > 0 else 0

        # Unusual activity: trades per minute in last 60s
        now = datetime.now()
        recent = [t for t in self._recent_trade_times
                  if (now - t).total_seconds() <= 60]
        trade_freq = len(recent)

        return {
            "daily_loss_ok":       daily_loss_pct < self.config.max_daily_loss_pct,
            "drawdown_ok":         drawdown < self.config.max_drawdown_pct,
            "consecutive_loss_ok": self._consecutive_losses < self.config.max_consecutive_losses,
            "trades_per_day_ok":   self._trades_today < self.config.max_trades_per_day,
            "position_size_ok":    pos_pct <= self.config.max_position_size_pct or position_size == 0,
            "activity_ok":         trade_freq <= self.config.max_trades_per_minute,
        }

    def _trigger_halt(self, reason: str) -> None:
        if not self._kill_switch:
            self._kill_switch = True
            self._halt_reason = reason
            logger.critical("SafetyController: AUTO-HALT — %s", reason)
            if self._alert:
                self._alert.risk_breach("AUTO_HALT", 0, 0)
            self._save_state()

    def _refresh_day(self, capital: float) -> None:
        today = date.today()
        if today != self._trade_day:
            self.reset_day(capital)

    def _save_state(self) -> None:
        try:
            state = {
                "kill_switch":         self._kill_switch,
                "halt_reason":         self._halt_reason,
                "peak_capital":        self._peak_capital,
                "current_capital":     self._current_capital,
                "daily_loss":          self._daily_loss,
                "consecutive_losses":  self._consecutive_losses,
                "trades_today":        self._trades_today,
                "trade_day":           str(self._trade_day),
                "saved_at":            datetime.now().isoformat(),
            }
            with open(STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning("SafetyController: failed to save state: %s", e)

    def _load_state(self) -> None:
        if not os.path.exists(STATE_PATH):
            return
        try:
            with open(STATE_PATH) as f:
                s = json.load(f)
            saved_day = date.fromisoformat(s.get("trade_day", str(date.today())))
            if saved_day == date.today():
                self._kill_switch        = s.get("kill_switch", False)
                self._halt_reason        = s.get("halt_reason", "")
                self._daily_loss         = s.get("daily_loss", 0.0)
                self._consecutive_losses = s.get("consecutive_losses", 0)
                self._trades_today       = s.get("trades_today", 0)
            self._peak_capital    = s.get("peak_capital", self.initial_capital)
            self._current_capital = s.get("current_capital", self.initial_capital)
            logger.info("SafetyController: state loaded (kill_switch=%s)", self._kill_switch)
        except Exception as e:
            logger.warning("SafetyController: failed to load state: %s", e)
