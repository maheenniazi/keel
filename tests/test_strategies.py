import numpy as np
import pandas as pd

from keel.strategies import EqualWeight, MeanReversion, Momentum, RiskParity


def idx(n):
    return pd.date_range("2016-01-01", periods=n, freq="B")


def test_equal_weight_sums_to_one_and_equal():
    px = pd.DataFrame(np.ones((300, 4)), index=idx(300), columns=list("ABCD"))
    w = EqualWeight().weights(px)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w.values, 0.25)


def test_momentum_picks_the_trending_name():
    n = 400
    data = {}
    data["FLAT1"] = 100 * np.ones(n)
    data["FLAT2"] = 100 * np.ones(n)
    data["UP"] = 100 * np.linspace(1.0, 3.0, n)      # steadily rises
    px = pd.DataFrame(data, index=idx(n))
    w = Momentum(lookback=252, skip=21, top_frac=0.3).weights(px)  # top 1 of 3
    assert w.get("UP", 0) > 0
    assert w.index.tolist() == ["UP"]


def test_risk_parity_favours_calmer_asset():
    n = 300
    rng = np.random.default_rng(3)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))   # low vol
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.03, n))    # high vol
    px = pd.DataFrame({"CALM": calm, "WILD": wild}, index=idx(n))
    w = RiskParity(window=60).weights(px)
    assert w["CALM"] > w["WILD"]
    assert np.isclose(w.sum(), 1.0)


def test_mean_reversion_is_dollar_neutral():
    n = 120
    rng = np.random.default_rng(4)
    px = pd.DataFrame(100 + rng.normal(0, 5, (n, 8)), index=idx(n),
                      columns=[f"S{i}" for i in range(8)])
    w = MeanReversion(window=20).weights(px)
    assert abs(w.sum()) < 1e-9             # longs and shorts cancel
    assert np.isclose(w.abs().sum(), 1.0)  # gross exposure = 1
