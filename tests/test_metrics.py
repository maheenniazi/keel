import numpy as np
import pandas as pd
import pytest

from keel import metrics as M


def daily_index(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_max_drawdown_known_value():
    equity = pd.Series([1, 2, 1, 2], index=daily_index(4))
    # peak hits 2, then halves to 1 -> -50% drawdown
    assert M.max_drawdown(equity) == pytest.approx(-0.5)


def test_no_drawdown_when_monotonic():
    equity = pd.Series([1, 1.1, 1.3, 1.5], index=daily_index(4))
    assert M.max_drawdown(equity) == pytest.approx(0.0)


def test_cagr_doubling_in_one_year():
    idx = pd.to_datetime(["2020-01-01", "2020-12-31"])
    equity = pd.Series([1.0, 2.0], index=idx)
    assert M.cagr(equity) == pytest.approx(1.0, abs=0.01)   # ~+100%/yr


def test_zero_variance_returns_give_zero_sharpe_and_vol():
    r = pd.Series([0.001] * 100, index=daily_index(100))
    assert M.ann_vol(r) == pytest.approx(0.0)
    assert M.sharpe(r) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 2000), index=daily_index(2000))
    assert M.sharpe(r) > 0


def test_sortino_at_least_sharpe_for_symmetric_noise():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 3000), index=daily_index(3000))
    # downside deviation <= total stdev, so Sortino >= Sharpe here
    assert M.sortino(r) >= M.sharpe(r) - 1e-9
