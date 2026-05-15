"""Tests for AlertSystem"""
import pytest
from trading.alert_system import AlertSystem, AlertLevel, AlertType


@pytest.fixture
def system():
    return AlertSystem(min_level=AlertLevel.INFO)


def test_trade_signal_alert(system):
    captured = []
    system.register_callback(lambda a: captured.append(a))
    system.trade_signal("RELIANCE.NS", "BUY", 72.5, 1360.0, 1347.0, 1387.0)
    assert len(captured) == 1
    assert captured[0].alert_type == AlertType.TRADE_SIGNAL


def test_risk_breach_is_critical(system):
    captured = []
    system.register_callback(lambda a: captured.append(a))
    system.risk_breach("DRAWDOWN", 4.5, 3.0)
    assert captured[0].level == AlertLevel.CRITICAL


def test_regime_change_no_duplicate(system):
    captured = []
    system.register_callback(lambda a: captured.append(a))
    system.regime_change("BULL", "BULL", 0.8, "TRADE_NORMAL")  # same regime
    assert len(captured) == 0


def test_daily_summary_positive_pnl(system):
    captured = []
    system.register_callback(lambda a: captured.append(a))
    system.daily_summary(8, 5, 1200.0, 0.625)
    assert captured[0].level == AlertLevel.INFO


def test_history_tracking(system):
    system.trade_signal("TCS.NS", "SELL", 55.0, 3500.0, 3520.0, 3460.0)
    system.risk_breach("LOSS_STREAK", 5, 3)
    assert len(system.get_history()) == 2
    assert len(system.get_history(AlertType.RISK_BREACH)) == 1


def test_min_level_filter(system):
    system._min_level = AlertLevel.CRITICAL
    captured = []
    system.register_callback(lambda a: captured.append(a))
    system.daily_summary(5, 3, 500.0, 0.6)  # INFO level — should be filtered
    assert len(captured) == 0
