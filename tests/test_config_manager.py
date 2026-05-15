"""Tests for ConfigManager"""
import os
import pytest


def test_load_yaml_values():
    # Reset singleton
    import trading.config_manager as cm
    cm._cfg = None
    cm.ConfigManager._instance = None

    cfg = cm.get_config()
    assert cfg.get("broker.default") == "paper"
    assert cfg.get("risk.max_daily_loss_pct") == 3.0
    assert cfg.get("scalping.default_strategy") == "ema"


def test_default_fallback():
    import trading.config_manager as cm
    cfg = cm.get_config()
    assert cfg.get("nonexistent.key", default=99) == 99


def test_env_var_override(monkeypatch):
    import trading.config_manager as cm
    cm._cfg = None
    cm.ConfigManager._instance = None

    monkeypatch.setenv("STOCKAI_TRADING_INITIAL_CAPITAL", "999999")
    cfg = cm.ConfigManager()
    assert cfg.get("trading.initial_capital") == 999999


def test_section_returns_dict():
    import trading.config_manager as cm
    cfg = cm.get_config()
    section = cfg.get_section("risk")
    assert isinstance(section, dict)
    assert "max_drawdown_pct" in section


def test_secrets_masked():
    import trading.config_manager as cm
    cfg = cm.get_config()
    safe = cfg.all()
    assert safe["broker"]["alpaca_api_key"] == "***"
