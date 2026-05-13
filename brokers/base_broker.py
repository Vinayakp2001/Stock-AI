"""
Universal Broker API Interface — brokers/base_broker.py

Abstract base class that all broker implementations must follow.
Zerodha, Alpaca, and any future broker plug in by subclassing BrokerBase.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ProductType(Enum):
    INTRADAY = "INTRADAY"   # MIS in Zerodha, day order in Alpaca
    DELIVERY = "DELIVERY"   # CNC in Zerodha, GTC in Alpaca


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """Standardised order representation across all brokers."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    product_type: ProductType = ProductType.INTRADAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    broker_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Standardised position representation."""
    symbol: str
    quantity: float          # positive = long, negative = short
    average_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    product_type: ProductType = ProductType.INTRADAY
    broker_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountInfo:
    """Standardised account/margin information."""
    broker_name: str
    account_id: str
    cash_balance: float
    total_value: float
    used_margin: float
    available_margin: float
    currency: str = "INR"
    broker_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Quote:
    """Standardised market quote."""
    symbol: str
    last_price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Abstract base broker
# ---------------------------------------------------------------------------

class BrokerBase(ABC):
    """
    Abstract base class for all broker integrations.

    All broker implementations (Zerodha, Alpaca, paper trader) must
    implement every abstract method so the rest of the system can
    work with any broker interchangeably.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._connected = False
        logger.info("BrokerBase: initialised broker '%s'", name)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """
        Authenticate and establish connection to the broker.
        Returns True on success, False on failure.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection cleanly."""

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Return current account balance and margin details."""

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """
        Submit an order to the broker.
        Returns the Order with broker-assigned order_id and status updated.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully cancelled."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Fetch current status of an order by its broker order_id."""

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Return all currently open/pending orders."""

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Return all current open positions."""

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Return position for a specific symbol, or None if not held."""

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return latest market quote for a symbol."""

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Return latest quotes for multiple symbols."""

    # ------------------------------------------------------------------
    # Convenience helpers (concrete — brokers can override if needed)
    # ------------------------------------------------------------------

    def buy_market(self, symbol: str, quantity: float,
                   product_type: ProductType = ProductType.INTRADAY) -> Order:
        """Place a market buy order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            product_type=product_type,
        )
        return self.place_order(order)

    def sell_market(self, symbol: str, quantity: float,
                    product_type: ProductType = ProductType.INTRADAY) -> Order:
        """Place a market sell order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            product_type=product_type,
        )
        return self.place_order(order)

    def buy_limit(self, symbol: str, quantity: float, limit_price: float,
                  product_type: ProductType = ProductType.INTRADAY) -> Order:
        """Place a limit buy order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            product_type=product_type,
        )
        return self.place_order(order)

    def sell_limit(self, symbol: str, quantity: float, limit_price: float,
                   product_type: ProductType = ProductType.INTRADAY) -> Order:
        """Place a limit sell order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            product_type=product_type,
        )
        return self.place_order(order)

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"<{self.__class__.__name__} broker='{self.name}' {status}>"
