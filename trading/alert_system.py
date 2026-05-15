"""
Alert & Notification System (Issue #45)
Sends alerts via console logging and optional email for:
- Trade signals (BUY/SELL)
- Risk limit breaches (drawdown, daily loss, consecutive losses)
- Market regime changes
- Price target hits
- System errors
"""

import logging
import os
import smtplib
import json
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ALERT_LOG_PATH = os.path.join("data", "alerts", "alert_history.jsonl")


class AlertLevel(Enum):
    INFO    = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    TRADE_SIGNAL     = "TRADE_SIGNAL"
    RISK_BREACH      = "RISK_BREACH"
    REGIME_CHANGE    = "REGIME_CHANGE"
    PRICE_TARGET     = "PRICE_TARGET"
    SYSTEM_ERROR     = "SYSTEM_ERROR"
    DAILY_SUMMARY    = "DAILY_SUMMARY"


@dataclass
class Alert:
    alert_type:  AlertType
    level:       AlertLevel
    title:       str
    message:     str
    symbol:      str = ""
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())
    data:        Dict = field(default_factory=dict)


class AlertSystem:
    """
    Central alert dispatcher. Supports:
    - Console (always on)
    - File log (always on)
    - Email (optional, configure via env vars or constructor)
    - Custom callbacks (register any function)
    """

    def __init__(
        self,
        email_to:   Optional[str] = None,
        email_from: Optional[str] = None,
        smtp_host:  str = "smtp.gmail.com",
        smtp_port:  int = 587,
        smtp_user:  Optional[str] = None,
        smtp_pass:  Optional[str] = None,
        min_level:  AlertLevel = AlertLevel.INFO,
    ):
        # Email config — fall back to env vars
        self._email_to   = email_to   or os.getenv("ALERT_EMAIL_TO")
        self._email_from = email_from or os.getenv("ALERT_EMAIL_FROM")
        self._smtp_host  = smtp_host
        self._smtp_port  = smtp_port
        self._smtp_user  = smtp_user  or os.getenv("ALERT_SMTP_USER")
        self._smtp_pass  = smtp_pass  or os.getenv("ALERT_SMTP_PASS")
        self._min_level  = min_level
        self._callbacks: List[Callable[[Alert], None]] = []
        self._history:   List[Alert] = []
        self._last_regime: Optional[str] = None

        os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def register_callback(self, fn: Callable[[Alert], None]) -> None:
        """Register a custom alert handler (e.g. Telegram, Slack)."""
        self._callbacks.append(fn)

    def send(self, alert: Alert) -> None:
        """Dispatch an alert through all enabled channels."""
        if not self._should_send(alert):
            return

        self._history.append(alert)
        self._log_to_console(alert)
        self._log_to_file(alert)

        if self._email_to:
            self._send_email(alert)

        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.warning("Alert callback failed: %s", e)

    # ── Convenience methods ───────────────────────────────────────────────

    def trade_signal(self, symbol: str, signal: str, score: float,
                     entry: float, sl: float, tp: float) -> None:
        level = AlertLevel.INFO if signal == "HOLD" else AlertLevel.WARNING
        self.send(Alert(
            alert_type=AlertType.TRADE_SIGNAL,
            level=level,
            title=f"Trade Signal: {signal} {symbol}",
            message=(f"{signal} signal for {symbol} | Score={score:.1f} | "
                     f"Entry=₹{entry:.2f} SL=₹{sl:.2f} TP=₹{tp:.2f}"),
            symbol=symbol,
            data={"signal": signal, "score": score, "entry": entry, "sl": sl, "tp": tp},
        ))

    def risk_breach(self, breach_type: str, value: float, limit: float,
                    symbol: str = "") -> None:
        self.send(Alert(
            alert_type=AlertType.RISK_BREACH,
            level=AlertLevel.CRITICAL,
            title=f"Risk Breach: {breach_type}",
            message=f"{breach_type} breached: {value:.2f} (limit={limit:.2f})",
            symbol=symbol,
            data={"breach_type": breach_type, "value": value, "limit": limit},
        ))

    def regime_change(self, old_regime: str, new_regime: str,
                      confidence: float, recommendation: str) -> None:
        if old_regime == new_regime:
            return
        self.send(Alert(
            alert_type=AlertType.REGIME_CHANGE,
            level=AlertLevel.WARNING,
            title=f"Regime Change: {old_regime} → {new_regime}",
            message=(f"Market regime changed from {old_regime} to {new_regime} "
                     f"(confidence={confidence:.1%}) | Action: {recommendation}"),
            data={"old": old_regime, "new": new_regime,
                  "confidence": confidence, "recommendation": recommendation},
        ))
        self._last_regime = new_regime

    def price_target(self, symbol: str, target_type: str,
                     price: float, target: float) -> None:
        self.send(Alert(
            alert_type=AlertType.PRICE_TARGET,
            level=AlertLevel.WARNING,
            title=f"Price Target Hit: {symbol} {target_type}",
            message=f"{symbol} hit {target_type} at ₹{price:.2f} (target=₹{target:.2f})",
            symbol=symbol,
            data={"target_type": target_type, "price": price, "target": target},
        ))

    def daily_summary(self, trades: int, wins: int, pnl: float,
                      win_rate: float) -> None:
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING
        self.send(Alert(
            alert_type=AlertType.DAILY_SUMMARY,
            level=level,
            title="Daily Trading Summary",
            message=(f"Trades={trades} | Wins={wins} | Win Rate={win_rate:.1%} | "
                     f"Net P&L=₹{pnl:+,.2f}"),
            data={"trades": trades, "wins": wins, "pnl": pnl, "win_rate": win_rate},
        ))

    def error(self, component: str, error_msg: str) -> None:
        self.send(Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.CRITICAL,
            title=f"System Error: {component}",
            message=f"Error in {component}: {error_msg}",
            data={"component": component, "error": error_msg},
        ))

    def get_history(self, alert_type: Optional[AlertType] = None,
                    limit: int = 50) -> List[Alert]:
        history = self._history
        if alert_type:
            history = [a for a in history if a.alert_type == alert_type]
        return history[-limit:]

    def check_regime(self, detector, data) -> None:
        """
        Convenience: detect current regime and fire alert if it changed.
        Pass a MarketRegimeDetector instance and OHLCV DataFrame.
        """
        try:
            signal = detector.detect(data)
            new_regime = signal.regime.value
            old_regime = self._last_regime or new_regime
            self.regime_change(old_regime, new_regime,
                               signal.confidence, signal.recommendation)
            self._last_regime = new_regime
        except Exception as e:
            logger.warning("check_regime failed: %s", e)

    # ── Private ───────────────────────────────────────────────────────────

    def _should_send(self, alert: Alert) -> bool:
        levels = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL]
        return levels.index(alert.level) >= levels.index(self._min_level)

    def _log_to_console(self, alert: Alert) -> None:
        icons = {
            AlertLevel.INFO:     "ℹ️ ",
            AlertLevel.WARNING:  "⚠️ ",
            AlertLevel.CRITICAL: "🚨",
        }
        icon = icons.get(alert.level, "")
        log_fn = (logger.critical if alert.level == AlertLevel.CRITICAL
                  else logger.warning if alert.level == AlertLevel.WARNING
                  else logger.info)
        log_fn("%s [%s] %s — %s", icon, alert.alert_type.value,
               alert.title, alert.message)

    def _log_to_file(self, alert: Alert) -> None:
        try:
            record = {
                "timestamp":  alert.timestamp,
                "type":       alert.alert_type.value,
                "level":      alert.level.value,
                "title":      alert.title,
                "message":    alert.message,
                "symbol":     alert.symbol,
                "data":       alert.data,
            }
            with open(ALERT_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("Failed to write alert to file: %s", e)

    def _send_email(self, alert: Alert) -> None:
        if not all([self._smtp_user, self._smtp_pass, self._email_from, self._email_to]):
            return
        try:
            subject = f"[Stock AI] {alert.level.value}: {alert.title}"
            body = (f"Time: {alert.timestamp}\n"
                    f"Type: {alert.alert_type.value}\n"
                    f"Level: {alert.level.value}\n\n"
                    f"{alert.message}\n\n"
                    f"Data: {json.dumps(alert.data, indent=2)}")
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = self._email_from
            msg["To"]      = self._email_to

            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                server.login(self._smtp_user, self._smtp_pass)
                server.send_message(msg)
            logger.info("Alert email sent to %s", self._email_to)
        except Exception as e:
            logger.warning("Failed to send alert email: %s", e)
