"""
Zerodha Kite Connect Broker — brokers/zerodha_broker.py

Implements BrokerBase using the kiteconnect library.
Requires: pip install kiteconnect

Set credentials via environment variables:
    ZERODHA_API_KEY
    ZERODHA_ACCESS_TOKEN   (generated after daily login flow)
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from brokers.base_broker import (
    AccountInfo, BrokerBase, Order, OrderSide, OrderStatus,
    OrderType, Position, ProductType, Quote,
)

logger = logging.getLogger(__name__)

# Zerodha product type mapping
_PRODUCT_MAP = {
    ProductType.INTRADAY: "MIS",
    ProductType.DELIVERY: "CNC",
}

# Zerodha order type mapping
_ORDER_TYPE_MAP = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP: "SL-M",
    OrderType.STOP_LIMIT: "SL",
}

_SIDE_MAP = {
    OrderSide.BUY: "BUY",
    OrderSide.SELL: "SELL",
}

_STATUS_MAP = {
    "COMPLETE": OrderStatus.FILLED,
    "OPEN": OrderStatus.OPEN,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "PENDING": OrderStatus.PENDING,
}


class ZerodhaBroker(BrokerBase):
    """
    Zerodha Kite Connect broker integration.

    Usage:
        broker = ZerodhaBroker()          # reads env vars
        broker.connect()
        order = broker.buy_market("RELIANCE", 1)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        super().__init__("Zerodha")
        self._api_key = api_key or os.environ.get("ZERODHA_API_KEY", "")
        self._access_token = access_token or os.environ.get("ZERODHA_ACCESS_TOKEN", "")
        self._kite = None  # kiteconnect.KiteConnect instance

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            from kiteconnect import KiteConnect  # type: ignore
            self._kite = KiteConnect(api_key=self._api_key)
            self._kite.set_access_token(self._access_token)
            # Verify by fetching profile
            profile = self._kite.profile()
            self._connected = True
            logger.info("ZerodhaBroker: connected as %s", profile.get("user_name", "unknown"))
            return True
        except ImportError:
            logger.error("ZerodhaBroker: kiteconnect not installed. Run: pip install kiteconnect")
            return False
        except Exception as exc:
            logger.error("ZerodhaBroker: connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._kite = None
        self._connected = False
        logger.info("ZerodhaBroker: disconnected")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_info(self) -> AccountInfo:
        self._require_connection()
        margins = self._kite.margins()
        equity = margins.get("equity", {})
        net = equity.get("net", 0.0)
        used = equity.get("utilised", {}).get("debits", 0.0)
        return AccountInfo(
            broker_name=self.name,
            account_id=self._kite.profile().get("user_id", ""),
            cash_balance=float(net),
            total_value=float(net),
            used_margin=float(used),
            available_margin=float(net - used),
            currency="INR",
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> Order:
        self._require_connection()
        try:
            params = {
                "tradingsymbol": order.symbol,
                "exchange": "NSE",
                "transaction_type": _SIDE_MAP[order.side],
                "quantity": int(order.quantity),
                "order_type": _ORDER_TYPE_MAP[order.order_type],
                "product": _PRODUCT_MAP[order.product_type],
                "validity": "DAY",
            }
            if order.limit_price:
                params["price"] = order.limit_price
            if order.stop_price:
                params["trigger_price"] = order.stop_price

            order_id = self._kite.place_order(variety="regular", **params)
            order.order_id = str(order_id)
            order.status = OrderStatus.OPEN
            order.created_at = datetime.now()
            logger.info("ZerodhaBroker: placed order %s", order.order_id)
        except Exception as exc:
            logger.error("ZerodhaBroker: place_order failed: %s", exc)
            order.status = OrderStatus.REJECTED
            order.broker_metadata["error"] = str(exc)
        return order

    def cancel_order(self, order_id: str) -> bool:
        self._require_connection()
        try:
            self._kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except Exception as exc:
            logger.error("ZerodhaBroker: cancel_order failed: %s", exc)
            return False

    def get_order_status(self, order_id: str) -> Order:
        self._require_connection()
        orders = self._kite.orders()
        for o in orders:
            if str(o["order_id"]) == str(order_id):
                return self._parse_order(o)
        raise ValueError(f"Order {order_id} not found")

    def get_open_orders(self) -> List[Order]:
        self._require_connection()
        return [
            self._parse_order(o)
            for o in self._kite.orders()
            if o.get("status") in ("OPEN", "PENDING", "TRIGGER PENDING")
        ]

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Position]:
        self._require_connection()
        data = self._kite.positions()
        result = []
        for p in data.get("net", []):
            if p.get("quantity", 0) != 0:
                result.append(self._parse_position(p))
        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        self._require_connection()
        instrument = f"NSE:{symbol}"
        data = self._kite.quote([instrument])
        q = data.get(instrument, {})
        return Quote(
            symbol=symbol,
            last_price=float(q.get("last_price", 0)),
            bid=q.get("depth", {}).get("buy", [{}])[0].get("price"),
            ask=q.get("depth", {}).get("sell", [{}])[0].get("price"),
            volume=q.get("volume"),
            timestamp=datetime.now(),
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        self._require_connection()
        instruments = [f"NSE:{s}" for s in symbols]
        data = self._kite.quote(instruments)
        quotes = {}
        for sym in symbols:
            key = f"NSE:{sym}"
            q = data.get(key, {})
            if q:
                quotes[sym] = Quote(
                    symbol=sym,
                    last_price=float(q.get("last_price", 0)),
                    volume=q.get("volume"),
                    timestamp=datetime.now(),
                )
        return quotes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_connection(self) -> None:
        if not self._connected or self._kite is None:
            raise RuntimeError("ZerodhaBroker: not connected. Call connect() first.")

    def _parse_order(self, o: dict) -> Order:
        side = OrderSide.BUY if o.get("transaction_type") == "BUY" else OrderSide.SELL
        status = _STATUS_MAP.get(o.get("status", ""), OrderStatus.PENDING)
        return Order(
            symbol=o.get("tradingsymbol", ""),
            side=side,
            quantity=float(o.get("quantity", 0)),
            order_type=OrderType.MARKET,
            order_id=str(o.get("order_id", "")),
            status=status,
            filled_quantity=float(o.get("filled_quantity", 0)),
            average_price=float(o.get("average_price", 0)),
            broker_metadata=o,
        )

    def _parse_position(self, p: dict) -> Position:
        qty = float(p.get("quantity", 0))
        avg = float(p.get("average_price", 0))
        ltp = float(p.get("last_price", avg))
        pnl = float(p.get("pnl", 0))
        pnl_pct = (ltp - avg) / avg * 100 if avg else 0.0
        return Position(
            symbol=p.get("tradingsymbol", ""),
            quantity=qty,
            average_price=avg,
            current_price=ltp,
            pnl=pnl,
            pnl_pct=round(pnl_pct, 4),
            broker_metadata=p,
        )
