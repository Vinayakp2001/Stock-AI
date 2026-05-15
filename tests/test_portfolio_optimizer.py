"""Tests for PortfolioOptimizer"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from trading.portfolio_optimizer import PortfolioOptimizer


def _mock_returns():
    """Generate synthetic daily returns for 4 symbols."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    data = pd.DataFrame(
        np.random.randn(300, 4) * 0.01 + 0.0003,
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    return data


@pytest.fixture
def opt():
    symbols = ["A", "B", "C", "D"]
    o = PortfolioOptimizer(symbols, period="1y")
    returns = _mock_returns()
    o._returns = returns
    o._mean_returns = returns.mean().values * 252
    o._cov_matrix = returns.cov().values * 252
    return o


def test_max_sharpe_weights_sum_to_one(opt):
    result = opt.max_sharpe()
    assert abs(sum(result["weights"].values()) - 1.0) < 0.01


def test_min_volatility_weights_sum_to_one(opt):
    result = opt.min_volatility()
    assert abs(sum(result["weights"].values()) - 1.0) < 0.01


def test_equal_weight(opt):
    result = opt.equal_weight()
    for w in result["weights"].values():
        assert abs(w - 0.25) < 0.01


def test_optimize_returns_all_strategies(opt):
    result = opt.optimize()
    assert "max_sharpe" in result
    assert "min_volatility" in result
    assert "equal_weight" in result
    assert "recommended" in result


def test_efficient_frontier_points(opt):
    frontier = opt.efficient_frontier(n_points=10)
    assert len(frontier) > 0
    for point in frontier:
        assert "return" in point
        assert "volatility" in point
