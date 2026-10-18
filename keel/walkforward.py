"""Walk-forward (out-of-sample) evaluation.

Picking the strategy parameter that looked best on the whole history is how
backtests lie. Walk-forward instead chooses the parameter on a *training*
window, then measures it on the *next, unseen* window, rolls forward, and
stitches those out-of-sample pieces together. The gap between the best
in-sample Sharpe and the realized out-of-sample Sharpe is the overfitting tax.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import metrics
from .engine import Backtest


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series          # stitched out-of-sample daily returns
    oos_equity: pd.Series
    choices: pd.DataFrame           # which param won each window + its IS/OOS Sharpe
    oos_sharpe: float


def walk_forward(prices: pd.DataFrame, make_strategy, grid: list, *,
                 train_years: int = 2, test_years: int = 1, rebalance: str = "ME",
                 cost_bps: float = 5.0, slippage_bps: float = 2.0, warmup: int = 252):
    """`make_strategy(param)` builds a Strategy; `grid` is the list of params to try."""
    idx = prices.index
    start = idx[0]
    rows, oos_parts = [], []
    train = pd.DateOffset(years=train_years)
    test = pd.DateOffset(years=test_years)

    t0 = start
    while True:
        tr_end = t0 + train
        te_end = tr_end + test
        if tr_end > idx[-1]:
            break
        train_px = prices.loc[t0:tr_end]
        test_px = prices.loc[tr_end:te_end]
        if len(test_px) < 2 or len(train_px) < warmup + 2:
            t0 = t0 + test
            continue

        # choose the parameter with the best in-sample Sharpe
        best = None
        for p in grid:
            r = Backtest(train_px, make_strategy(p), rebalance=rebalance, warmup=warmup,
                         cost_bps=cost_bps, slippage_bps=slippage_bps).run()
            s = r.stats["Sharpe"]
            if best is None or s > best[1]:
                best = (p, s)

        # evaluate that choice out-of-sample on the next window
        # (warm the strategy with the tail of training history so signals are defined)
        oos_px = prices.loc[t0:te_end]
        oos = Backtest(oos_px, make_strategy(best[0]), rebalance=rebalance, warmup=warmup,
                       cost_bps=cost_bps, slippage_bps=slippage_bps).run()
        oos_window = oos.returns.loc[tr_end:te_end]
        rows.append({"test_start": tr_end.date(), "param": best[0],
                     "IS Sharpe": round(best[1], 2),
                     "OOS Sharpe": round(metrics.sharpe(oos_window), 2)})
        oos_parts.append(oos_window)
        t0 = t0 + test

    oos_returns = pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype=float)
    oos_equity = (1 + oos_returns).cumprod()
    return WalkForwardResult(
        oos_returns=oos_returns, oos_equity=oos_equity,
        choices=pd.DataFrame(rows), oos_sharpe=metrics.sharpe(oos_returns))
