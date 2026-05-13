"""
Alpaca API Broker — brokers/alpaca_broker.py

Implements BrokerBase using the alpaca-trade-api library.
Requires: pip install alpaca-trade-api

Set credentials via environment variables:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    ALPACA_BASE_URL   (optional, defaults to paper trading URL)
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

_PAPER_URL = "https://paper-api.alpaca.markets"
_LIVE_URL = "https://api.alpaca.markets"

_ORDER_TYPE_MAP = {
    OrderType.MARKET: "market",
    OrderType.LIMIT: "limit",
    OrderType.STOP: "stop",
    OrderType.STOP_LIMIT: "stop_limit",
}

_STATUS_MAP = {
    "filled": OrderStatus.FILLED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "new": OrderStatus.OPEN,
    "accepted": OrderStatus.OPEN,
    "pending_new": OrderStatus.PENDING,
    "canceled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "expired": OrderStatus.CANCELLED,
}


class AlpacaBroker(BrokerBase):
    """
    Alpaca broker integration for US stocks.

    Usage:
        broker = AlpacaBroker()           # reads env vars, uses paper URL
        broker = AlpacaBroker(live=True)  # live trading
        broker.connect()
        order = broker.buy_market("AAPL", 1)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        live: bool = False,
    ) -> None:
        super().__init__("Alpaca")
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._base_url = _LIVE_URL if live else os.environ.get("ALPACA_BASE_URL", _PAPER_URL)
        self._api = None  # alpaca_trade_api.REST instance

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            import alpaca_trade_api as tradeapi  # type: ignore
            self._api = tradeapi.REST(
                self._api_key,
                self._secret_key,
                self._base_url,
                api_version="v2",
            )
            account = self._api.get_account()
            self._connected = True
            logger.info(
                "AlpacaBroker: connected — status=%s equity=$%s",
                account.status, account.equity
            )
            return True
        except ImportError:
            logger.error("AlpacaBroker: alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            return False
        except Exception as exc:
            logger.error("AlpacaBroker: connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._api = None
        self._connected = False
        logger.info("AlpacaBroker: disconnected")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_info(self) -> AccountInfo:
        self._require_connection()
        acc = self._api.get_account()
        return AccountInfo(
            broker_name=self.name,
            account_id=acc.id,
            cash_balance=float(acc.cash),
            total_value=float(acc.equity),
            used_margin=float(acc.initial_margin),
            available_margin=float(acc.buying_power),
            currency="USD",
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> Order:
        self._require_connection()
        try:
            side = "buy" if order.side == OrderSide.BUY else "sell"
            time_in_force = "day" if order.product_type == ProductType.INTRADAY else "gtc"

            kwargs = {
                "symbol": order.symbol,
                "qty": order.quantity,
                "side": side,
                "type": _ORDER_TYPE_MAP[order.order_type],
                "time_in_force": time_in_force,
            }
            if order.limit_price:
                kwargs["limit_price"] = str(order.limit_price)
            if order.stop_price:
                kwargs["stop_price"] = str(order.stop_price)

            resp = self._api.submit_order(**kwargs)
            order.order_id = resp.id
            order.status = _STATUS_MAP.get(resp.status, OrderStatus.PENDING)
            order.created_at = datetime.now()
            logger.info("AlpacaBroker: placed order %s", order.order_id)
        except Exception as exc:
            logger.error("AlpacaBroker: place_order failed: %s", exc)
            order.status = OrderStatus.REJECTED
            order.broker_metadata["error"] = str(exc)
        return order

    def cancel_order(self, order_id: str) -> bool:
        self._require_connection()
        try:
            self._api.cancel_order(order_id)
            return True
        except Exception as exc:
            logger.error("AlpacaBroker: cancel_order failed: %s", exc)
            return False

    def get_order_status(self, order_id: str) -> Order:
        self._require_connection()
        o = self._api.get_order(order_id)
        return self._parse_order(o)

    def get_open_orders(self) -> List[Order]:
        self._require_connection()
        return [self._parse_order(o) for o in self._api.list_orders(status="open")]

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Position]:
        self._require_connection()
        return [self._parse_position(p) for p in self._api.list_positions()]

    def get_position(self, symbol: str) -> Optional[Position]:
        self._require_connection()
        try:
            return self._parse_position(self._api.get_position(symbol))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        self._require_connection()
        try:
            bars = self._api.get_latest_bar(symbol)
            return Quote(
                symbol=symbol,
                last_price=float(bars.c),
                volume=float(bars.v),
                timestamp=datetime.now(),
            )
        except Exception:
            # Fallback to last trade
            trade = self._api.get_latest_trade(symbol)
            return Quote(
                symbol=symbol,
                last_price=float(trade.price),
                timestamp=datetime.now(),
            )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        quotes = {}
        for sym in symbols:
            try:
                quotes[sym] = self.get_quote(sym)
            except Exception as exc:
                logger.warning("AlpacaBroker: quote failed for %s: %s", sym, exc)
        return quotes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_connection(self) -> None:
        if not self._connected or self._api is None:
            raise RuntimeError("AlpacaBroker: not connected. Call connect() first.")

    def _parse_order(self, o) -> Order:
        side = OrderSide.BUY if o.side == "buy" else OrderSide.SELL
        status = _STATUS_MAP.get(o.status, OrderStatus.PENDING)
        return Order(
            symbol=o.symbol,
            side=side,
            quantity=float(o.qty),
            order_type=OrderType.MARKET,
            order_id=o.id,
            status=status,
            filled_quantity=float(o.filled_qty or 0),
            average_price=float(o.filled_avg_price or 0),
            broker_metadata={"raw": str(o)},
        )

    def _parse_position(self, p) -> Position:
        qty = float(p.qty)
        avg = float(p.avg_entry_price)
        ltp = float(p.current_price)
        pnl = float(p.unrealized_pl)
        pnl_pct = float(p.unrealized_plpc) * 100
        return Position(
            symbol=p.symbol,
            quantity=qty,
            average_price=avg,
            current_price=ltp,
            pnl=pnl,
            pnl_pct=round(pnl_pct, 4),
            broker_metadata={"raw": str(p)},
        )
