import numpy as np
import pandas as pd

from keel import Backtest
from keel.strategies import EqualWeight


def synth_prices(n_days=800, n_assets=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B")
    rets = rng.normal(0.0004, 0.01, size=(n_days, n_assets))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=idx, columns=cols)


def test_equity_starts_near_capital():
    px = synth_prices()
    r = Backtest(px, EqualWeight(), warmup=60, capital=1_000_000).run()
    assert len(r.equity) > 0
    assert 0.9e6 < r.equity.iloc[0] < 1.1e6


def test_costs_reduce_final_equity():
    px = synth_prices()
    free = Backtest(px, EqualWeight(), warmup=60, cost_bps=0, slippage_bps=0).run()
    costly = Backtest(px, EqualWeight(), warmup=60, cost_bps=50, slippage_bps=20).run()
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_long_only_weights_sum_to_one_on_rebalance():
    px = synth_prices()
    r = Backtest(px, EqualWeight(), warmup=60).run()
    d = r.turnover.index[1]                     # a rebalance date
    assert np.isclose(r.weights.loc[d].sum(), 1.0)


def test_no_same_day_trading():
    # positions set on a rebalance date take effect the *next* day, so the
    # portfolio return on the first rebalance date itself must be ~0 (still cash).
    px = synth_prices()
    r = Backtest(px, EqualWeight(), warmup=60, cost_bps=0, slippage_bps=0).run()
    first_rebal = r.turnover.index[0]
    assert abs(r.returns.get(first_rebal, 0.0)) < 1e-12


def test_deterministic():
    px = synth_prices()
    a = Backtest(px, EqualWeight(), warmup=60).run()
    b = Backtest(px, EqualWeight(), warmup=60).run()
    assert a.equity.iloc[-1] == b.equity.iloc[-1]
