"""Tests for RiskManager"""
import pytest
from scalping.risk.risk_manager import RiskManager


@pytest.fixture
def rm():
    return RiskManager(initial_capital=100_000)


def test_initial_state(rm):
    status = rm.get_status()
    assert status["trading_halted"] is False
    assert status["consecutive_losses"] == 0
    assert status["drawdown_pct"] == 0.0


def test_check_entry_allowed(rm):
    result = rm.check_entry("BUY", 100_000, 1000.0, None)
    assert result["allowed"] is True
    assert result["quantity"] > 0


def test_consecutive_losses_halt(rm):
    from unittest.mock import MagicMock
    from datetime import datetime
    trade = MagicMock()
    trade.net_pnl = -500
    trade.status = "STOPPED"
    for _ in range(3):
        rm.record_trade_result(trade)
    # RiskManager activates cooldown after 3 losses — check entry should be blocked
    result = rm.check_entry("BUY", 100_000, 1000.0, datetime.now())
    assert result["allowed"] is False


def test_reset_day(rm):
    from unittest.mock import MagicMock
    trade = MagicMock()
    trade.net_pnl = -500
    trade.status = "STOPPED"
    for _ in range(5):
        rm.record_trade_result(trade)
    rm.reset_day(new_capital=100_000)
    assert rm.get_status()["trading_halted"] is False
