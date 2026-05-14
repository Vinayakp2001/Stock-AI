"""
Intelligent Order Execution System — trading/order_executor.py

Sits on top of any BrokerBase implementation and provides:
- Slippage-aware market orders
- Retry logic on transient failures
- Order splitting for large quantities
- Bracket orders (entry + stop-loss + take-profit)
- Order status polling until fill
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from brokers.base_broker import (
    BrokerBase, Order, OrderSide, OrderStatus, OrderType, Position, ProductType,
)

logger = logging.getLogger(__name__)


@dataclass
class BracketOrder:
    """Entry + stop-loss + take-profit as a single logical unit."""
    entry_order: Order
    stop_loss_price: float
    take_profit_price: float
    sl_order: Optional[Order] = None
    tp_order: Optional[Order] = None
    is_active: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Result of an order execution attempt."""
    success: bool
    order: Optional[Order]
    message: str = ""
    attempts: int = 1


class OrderExecutor:
    """
    Intelligent order execution layer over any BrokerBase broker.

    Features:
    - Retry on REJECTED/timeout (configurable max_retries)
    - Chunk large orders into smaller lots
    - Bracket order management (entry + SL + TP)
    - Poll until filled with timeout
    """

    def __init__(
        self,
        broker: BrokerBase,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
        max_lot_size: int = 500,
        poll_interval_sec: float = 0.5,
        fill_timeout_sec: float = 30.0,
    ) -> None:
        self.broker = broker
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.max_lot_size = max_lot_size
        self.poll_interval_sec = poll_interval_sec
        self.fill_timeout_sec = fill_timeout_sec
        self._active_brackets: List[BracketOrder] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_market(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        product_type: ProductType = ProductType.INTRADAY,
    ) -> ExecutionResult:
        """
        Execute a market order with retry logic.
        Splits into chunks if quantity > max_lot_size.
        """
        if quantity <= 0:
            return ExecutionResult(False, None, "Quantity must be positive")

        chunks = self._split_quantity(quantity)
        filled_orders = []

        for chunk_qty in chunks:
            result = self._execute_with_retry(
                Order(
                    symbol=symbol,
                    side=side,
                    quantity=chunk_qty,
                    order_type=OrderType.MARKET,
                    product_type=product_type,
                )
            )
            if not result.success:
                logger.warning(
                    "OrderExecutor: chunk failed for %s qty=%s — %s",
                    symbol, chunk_qty, result.message
                )
                break
            filled_orders.append(result.order)

        if not filled_orders:
            return ExecutionResult(False, None, "All chunks failed")

        # Merge fills into a summary order
        total_qty = sum(o.filled_quantity for o in filled_orders)
        avg_price = (
            sum(o.filled_quantity * o.average_price for o in filled_orders) / total_qty
            if total_qty > 0 else 0.0
        )
        summary = filled_orders[0]
        summary.filled_quantity = total_qty
        summary.average_price = round(avg_price, 4)
        return ExecutionResult(True, summary, f"Filled {total_qty} @ {avg_price:.2f}")

    def execute_limit(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        product_type: ProductType = ProductType.INTRADAY,
        wait_for_fill: bool = False,
    ) -> ExecutionResult:
        """Place a limit order, optionally polling until filled."""
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            product_type=product_type,
        )
        result = self._execute_with_retry(order)
        if result.success and wait_for_fill:
            result.order = self._poll_until_filled(result.order)
        return result

    def execute_bracket(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_loss_price: float,
        take_profit_price: float,
        product_type: ProductType = ProductType.INTRADAY,
    ) -> Tuple[ExecutionResult, BracketOrder]:
        """
        Place entry order then immediately place SL and TP orders.
        Returns (entry_result, bracket) so caller can track/cancel.
        """
        entry_result = self.execute_market(symbol, side, quantity, product_type)
        if not entry_result.success:
            bracket = BracketOrder(
                entry_order=Order(symbol=symbol, side=side, quantity=quantity,
                                  order_type=OrderType.MARKET),
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
            return entry_result, bracket

        # Opposite side for SL and TP
        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        # Stop-loss order
        sl_order = Order(
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_loss_price,
            product_type=product_type,
        )
        sl_result = self._execute_with_retry(sl_order)

        # Take-profit limit order
        tp_order = Order(
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=take_profit_price,
            product_type=product_type,
        )
        tp_result = self._execute_with_retry(tp_order)

        bracket = BracketOrder(
            entry_order=entry_result.order,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            sl_order=sl_result.order if sl_result.success else None,
            tp_order=tp_result.order if tp_result.success else None,
            is_active=True,
        )
        self._active_brackets.append(bracket)

        logger.info(
            "OrderExecutor: bracket placed for %s — entry=%s SL=%s TP=%s",
            symbol,
            entry_result.order.order_id if entry_result.order else "?",
            bracket.sl_order.order_id if bracket.sl_order else "failed",
            bracket.tp_order.order_id if bracket.tp_order else "failed",
        )
        return entry_result, bracket

    def cancel_bracket(self, bracket: BracketOrder) -> None:
        """Cancel both legs of an active bracket order."""
        for leg_order in [bracket.sl_order, bracket.tp_order]:
            if leg_order and leg_order.order_id:
                try:
                    self.broker.cancel_order(leg_order.order_id)
                except Exception as exc:
                    logger.warning("OrderExecutor: cancel failed for %s: %s",
                                   leg_order.order_id, exc)
        bracket.is_active = False

    def get_active_brackets(self) -> List[BracketOrder]:
        return [b for b in self._active_brackets if b.is_active]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_with_retry(self, order: Order) -> ExecutionResult:
        """Submit order with up to max_retries attempts."""
        for attempt in range(1, self.max_retries + 1):
            try:
                placed = self.broker.place_order(order)
                if placed.status not in (OrderStatus.REJECTED,):
                    return ExecutionResult(True, placed, attempts=attempt)
                logger.warning(
                    "OrderExecutor: order rejected (attempt %d/%d) — %s",
                    attempt, self.max_retries,
                    placed.broker_metadata.get("error", "unknown")
                )
            except Exception as exc:
                logger.warning(
                    "OrderExecutor: exception on attempt %d/%d: %s",
                    attempt, self.max_retries, exc
                )
            if attempt < self.max_retries:
                time.sleep(self.retry_delay_sec)

        return ExecutionResult(False, order, f"Failed after {self.max_retries} attempts")

    def _poll_until_filled(self, order: Order) -> Order:
        """Poll order status until FILLED or timeout."""
        if order.order_id is None:
            return order
        deadline = time.time() + self.fill_timeout_sec
        while time.time() < deadline:
            try:
                updated = self.broker.get_order_status(order.order_id)
                if updated.status == OrderStatus.FILLED:
                    return updated
                if updated.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                    logger.warning("OrderExecutor: order %s ended with %s",
                                   order.order_id, updated.status)
                    return updated
            except Exception as exc:
                logger.warning("OrderExecutor: poll error: %s", exc)
            time.sleep(self.poll_interval_sec)
        logger.warning("OrderExecutor: fill timeout for order %s", order.order_id)
        return order

    def _split_quantity(self, quantity: float) -> List[float]:
        """Split quantity into chunks of max_lot_size."""
        if quantity <= self.max_lot_size:
            return [quantity]
        chunks = []
        remaining = quantity
        while remaining > 0:
            chunk = min(remaining, self.max_lot_size)
            chunks.append(chunk)
            remaining -= chunk
        return chunks
