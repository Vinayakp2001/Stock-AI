"""
Paper Trading Broker — brokers/paper_broker.py

In-memory broker implementation for paper trading and testing.
No API keys required. Uses yfinance for live quotes.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import yfinance as yf

from brokers.base_broker import (
    AccountInfo, BrokerBase, Order, OrderSide, OrderStatus,
    OrderType, Position, ProductType, Quote,
)

logger = logging.getLogger(__name__)


class PaperBroker(BrokerBase):
    """
    Simulated broker for paper trading.
    Executes orders instantly at last market price.
    """

    def __init__(self, initial_capital: float = 100_000.0, currency: str = "INR") -> None:
        super().__init__("PaperBroker")
        self._cash = initial_capital
        self._initial_capital = initial_capital
        self._currency = currency
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        self._connected = True
        logger.info("PaperBroker: connected (capital=%.2f %s)", self._cash, self._currency)
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("PaperBroker: disconnected")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_info(self) -> AccountInfo:
        position_value = sum(
            p.quantity * p.current_price for p in self._positions.values()
        )
        total = self._cash + position_value
        return AccountInfo(
            broker_name=self.name,
            account_id="PAPER-001",
            cash_balance=round(self._cash, 2),
            total_value=round(total, 2),
            used_margin=round(position_value, 2),
            available_margin=round(self._cash, 2),
            currency=self._currency,
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> Order:
        order.order_id = str(uuid.uuid4())[:8]
        order.created_at = datetime.now()

        # Get fill price
        try:
            quote = self.get_quote(order.symbol)
            fill_price = quote.last_price
        except Exception as exc:
            logger.warning("PaperBroker: quote failed for %s: %s", order.symbol, exc)
            order.status = OrderStatus.REJECTED
            self._orders[order.order_id] = order
            return order

        if order.order_type == OrderType.LIMIT and order.limit_price:
            fill_price = order.limit_price

        cost = fill_price * order.quantity

        if order.side == OrderSide.BUY:
            if cost > self._cash:
                logger.warning("PaperBroker: insufficient funds for %s", order.symbol)
                order.status = OrderStatus.REJECTED
                self._orders[order.order_id] = order
                return order
            self._cash -= cost
            self._update_position(order.symbol, order.quantity, fill_price, order.product_type)

        elif order.side == OrderSide.SELL:
            pos = self._positions.get(order.symbol)
            if pos is None or pos.quantity < order.quantity:
                logger.warning("PaperBroker: insufficient position for %s", order.symbol)
                order.status = OrderStatus.REJECTED
                self._orders[order.order_id] = order
                return order
            self._cash += cost
            self._update_position(order.symbol, -order.quantity, fill_price, order.product_type)

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.updated_at = datetime.now()
        self._orders[order.order_id] = order

        logger.info(
            "PaperBroker: %s %s x%s @ %.2f [%s]",
            order.side.value, order.symbol, order.quantity, fill_price, order.order_id
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.OPEN:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def get_order_status(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")
        return order

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.status == OrderStatus.OPEN]

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Position]:
        self._refresh_position_prices()
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        self._refresh_position_prices()
        return self._positions.get(symbol)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if price is None:
            raise ValueError(f"Could not get price for {symbol}")
        return Quote(
            symbol=symbol,
            last_price=float(price),
            bid=info.get("bid"),
            ask=info.get("ask"),
            volume=info.get("volume"),
            timestamp=datetime.now(),
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        quotes = {}
        for sym in symbols:
            try:
                quotes[sym] = self.get_quote(sym)
            except Exception as exc:
                logger.warning("PaperBroker: quote failed for %s: %s", sym, exc)
        return quotes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_position(self, symbol: str, qty_delta: float,
                         price: float, product_type: ProductType) -> None:
        pos = self._positions.get(symbol)
        if pos is None:
            if qty_delta > 0:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=qty_delta,
                    average_price=price,
                    current_price=price,
                    pnl=0.0,
                    pnl_pct=0.0,
                    product_type=product_type,
                )
        else:
            new_qty = pos.quantity + qty_delta
            if new_qty <= 0:
                del self._positions[symbol]
            else:
                # Weighted average price on adds
                if qty_delta > 0:
                    total_cost = pos.average_price * pos.quantity + price * qty_delta
                    pos.average_price = total_cost / new_qty
                pos.quantity = new_qty
                pos.current_price = price
                pos.pnl = (price - pos.average_price) * new_qty
                pos.pnl_pct = (price - pos.average_price) / pos.average_price * 100

    def _refresh_position_prices(self) -> None:
        for symbol, pos in list(self._positions.items()):
            try:
                quote = self.get_quote(symbol)
                pos.current_price = quote.last_price
                pos.pnl = (pos.current_price - pos.average_price) * pos.quantity
                pos.pnl_pct = (pos.current_price - pos.average_price) / pos.average_price * 100
            except Exception:
                pass
