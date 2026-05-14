"""
Automated Position Management — trading/position_manager.py

Monitors open positions and automatically manages:
- Trailing stop-loss updates
- Time-based exits (max hold time)
- Max loss per position (hard stop)
- Take-profit targets
- Daily P&L limits
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from brokers.base_broker import BrokerBase, OrderSide, Position
from trading.order_executor import OrderExecutor

logger = logging.getLogger(__name__)


@dataclass
class PositionConfig:
    """Per-position management rules."""
    symbol: str
    entry_price: float
    quantity: float
    side: OrderSide
    stop_loss_pct: float = 1.0        # % below entry for initial stop
    take_profit_pct: float = 2.0      # % above entry for take profit
    trailing_stop_pct: float = 0.5    # % trailing stop once in profit
    max_hold_minutes: int = 60        # force exit after N minutes
    entered_at: datetime = field(default_factory=datetime.now)
    highest_price: float = 0.0        # for trailing stop tracking
    lowest_price: float = float("inf")
    is_active: bool = True
    exit_reason: str = ""


@dataclass
class PositionStatus:
    symbol: str
    current_price: float
    entry_price: float
    pnl_pct: float
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: float
    hold_minutes: float
    action: str  # HOLD / EXIT_SL / EXIT_TP / EXIT_TRAILING / EXIT_TIME / EXIT_MAX_LOSS


class PositionManager:
    """
    Monitors and auto-manages open positions.

    Usage:
        pm = PositionManager(broker, executor)
        pm.add_position(PositionConfig("RELIANCE.NS", entry=1360, qty=2, side=BUY))
        # Call pm.monitor() periodically (e.g. every 30 seconds)
        exits = pm.monitor()
    """

    def __init__(
        self,
        broker: BrokerBase,
        executor: OrderExecutor,
        max_daily_loss_pct: float = 3.0,   # halt all trading if daily loss > 3%
        initial_capital: float = 100_000.0,
    ) -> None:
        self.broker = broker
        self.executor = executor
        self.max_daily_loss_pct = max_daily_loss_pct
        self.initial_capital = initial_capital
        self._positions: Dict[str, PositionConfig] = {}
        self._daily_pnl: float = 0.0
        self._trading_halted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_position(self, config: PositionConfig) -> None:
        """Register a new position for monitoring."""
        config.highest_price = config.entry_price
        config.lowest_price = config.entry_price
        self._positions[config.symbol] = config
        logger.info(
            "PositionManager: tracking %s qty=%s entry=%.2f SL=%.1f%% TP=%.1f%%",
            config.symbol, config.quantity, config.entry_price,
            config.stop_loss_pct, config.take_profit_pct,
        )

    def remove_position(self, symbol: str) -> None:
        self._positions.pop(symbol, None)

    def monitor(self) -> List[PositionStatus]:
        """
        Check all tracked positions against current prices.
        Automatically exits positions that hit SL/TP/time/trailing stop.
        Returns list of PositionStatus for all active positions.
        """
        if self._trading_halted:
            logger.warning("PositionManager: trading halted — daily loss limit reached")
            return []

        statuses = []
        for symbol, config in list(self._positions.items()):
            if not config.is_active:
                continue
            try:
                status = self._check_position(config)
                statuses.append(status)
                if status.action != "HOLD":
                    self._execute_exit(config, status)
            except Exception as exc:
                logger.warning("PositionManager: error monitoring %s: %s", symbol, exc)

        return statuses

    def get_summary(self) -> Dict:
        """Return summary of all tracked positions."""
        active = [c for c in self._positions.values() if c.is_active]
        return {
            "active_positions": len(active),
            "symbols": [c.symbol for c in active],
            "daily_pnl": round(self._daily_pnl, 2),
            "trading_halted": self._trading_halted,
        }

    def reset_daily_pnl(self) -> None:
        """Call at start of each trading day."""
        self._daily_pnl = 0.0
        self._trading_halted = False
        logger.info("PositionManager: daily P&L reset")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_position(self, config: PositionConfig) -> PositionStatus:
        """Evaluate a single position and determine action."""
        quote = self.broker.get_quote(config.symbol)
        price = quote.last_price

        # Update high/low watermarks
        config.highest_price = max(config.highest_price, price)
        config.lowest_price = min(config.lowest_price, price)

        # Calculate prices
        if config.side == OrderSide.BUY:
            sl_price = config.entry_price * (1 - config.stop_loss_pct / 100)
            tp_price = config.entry_price * (1 + config.take_profit_pct / 100)
            trailing_sl = config.highest_price * (1 - config.trailing_stop_pct / 100)
            pnl_pct = (price - config.entry_price) / config.entry_price * 100
        else:  # SELL/short
            sl_price = config.entry_price * (1 + config.stop_loss_pct / 100)
            tp_price = config.entry_price * (1 - config.take_profit_pct / 100)
            trailing_sl = config.lowest_price * (1 + config.trailing_stop_pct / 100)
            pnl_pct = (config.entry_price - price) / config.entry_price * 100

        hold_minutes = (datetime.now() - config.entered_at).total_seconds() / 60

        # Determine action
        action = "HOLD"

        if config.side == OrderSide.BUY:
            if price <= sl_price:
                action = "EXIT_SL"
            elif price >= tp_price:
                action = "EXIT_TP"
            elif pnl_pct > 0 and price <= trailing_sl:
                action = "EXIT_TRAILING"
        else:
            if price >= sl_price:
                action = "EXIT_SL"
            elif price <= tp_price:
                action = "EXIT_TP"
            elif pnl_pct > 0 and price >= trailing_sl:
                action = "EXIT_TRAILING"

        if hold_minutes >= config.max_hold_minutes:
            action = "EXIT_TIME"

        if pnl_pct < -config.stop_loss_pct * 1.5:
            action = "EXIT_MAX_LOSS"

        return PositionStatus(
            symbol=config.symbol,
            current_price=price,
            entry_price=config.entry_price,
            pnl_pct=round(pnl_pct, 4),
            stop_loss_price=round(sl_price, 2),
            take_profit_price=round(tp_price, 2),
            trailing_stop_price=round(trailing_sl, 2),
            hold_minutes=round(hold_minutes, 1),
            action=action,
        )

    def _execute_exit(self, config: PositionConfig, status: PositionStatus) -> None:
        """Execute exit order and update daily P&L."""
        exit_side = OrderSide.SELL if config.side == OrderSide.BUY else OrderSide.BUY
        result = self.executor.execute_market(config.symbol, exit_side, config.quantity)

        if result.success:
            pnl = status.pnl_pct / 100 * config.entry_price * config.quantity
            self._daily_pnl += pnl
            config.is_active = False
            config.exit_reason = status.action
            self._positions.pop(config.symbol, None)

            logger.info(
                "PositionManager: exited %s reason=%s pnl=%.2f (%.2f%%) daily_pnl=%.2f",
                config.symbol, status.action, pnl, status.pnl_pct, self._daily_pnl,
            )

            # Check daily loss limit
            daily_loss_pct = abs(self._daily_pnl) / self.initial_capital * 100
            if self._daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
                self._trading_halted = True
                logger.warning(
                    "PositionManager: TRADING HALTED — daily loss %.2f%% >= limit %.2f%%",
                    daily_loss_pct, self.max_daily_loss_pct,
                )
        else:
            logger.error("PositionManager: exit order failed for %s", config.symbol)
