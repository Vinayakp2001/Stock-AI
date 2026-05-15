"""Tests for SafetyController"""
import os
import pytest
from trading.safety_controller import SafetyController, SafetyConfig, STATE_PATH


@pytest.fixture(autouse=True)
def clean_state():
    """Remove persisted state before each test."""
    if os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)
    yield
    if os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)


@pytest.fixture
def sc():
    cfg = SafetyConfig(max_consecutive_losses=3, max_trades_per_day=50)
    return SafetyController(config=cfg, initial_capital=100_000)


def test_trade_allowed_initially(sc):
    assert sc.check_trade(100_000, 20_000) is True


def test_auto_halt_on_consecutive_losses(sc):
    for _ in range(3):
        sc.record_trade_result(-500, 98_500)
    assert sc.get_status().trading_allowed is False


def test_kill_switch(sc):
    sc.activate_kill_switch("test")
    assert sc.check_trade(100_000, 20_000) is False


def test_reset_clears_halt(sc):
    for _ in range(3):
        sc.record_trade_result(-500, 98_500)
    sc.reset_kill_switch()
    assert sc.check_trade(100_000, 20_000) is True


def test_position_size_limit(sc):
    # 40% of capital exceeds 30% limit
    assert sc.check_trade(100_000, 40_000) is False
