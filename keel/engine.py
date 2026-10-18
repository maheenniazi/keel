"""The backtest engine.

Design decisions that matter for honesty:

* **No look-ahead.** At each rebalance date the strategy only sees prices up
  to and including that date. The weights it returns are then applied to the
  *next* day's returns (we `shift(1)` positions), so a signal can never trade
  on information from its own bar.
* **Real frictions.** Every rebalance pays transaction cost + slippage in
  proportion to the turnover it generates, charged the day the trade takes
  effect. Zero-cost backtests flatter every strategy; this one doesn't.
* **Explicit accounting.** We track target weights, held weights, per-rebalance
  turnover, gross and net returns, and the equity curve separately, so each
  number can be inspected.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import metrics


@dataclass
class BacktestResult:
    equity: pd.Series          # portfolio value over the active period
    returns: pd.Series         # net daily returns
    weights: pd.DataFrame      # target weights through time (forward-filled)
    turnover: pd.Series        # one-way turnover on each rebalance date
    ann_turnover: float        # average one-way turnover per year
    stats: dict
    name: str = "strategy"


# resample-style aliases -> period aliases used by to_period
_PERIOD = {"ME": "M", "M": "M", "W": "W", "QE": "Q", "Q": "Q", "YE": "Y", "Y": "Y"}


class Backtest:
    def __init__(self, prices: pd.DataFrame, strategy, *, rebalance: str = "M",
                 cost_bps: float = 5.0, slippage_bps: float = 2.0,
                 capital: float = 1_000_000.0, rf: float = 0.0, warmup: int = 252):
        self.prices = prices.sort_index()
        self.strategy = strategy
        self.rebalance = _PERIOD.get(rebalance, rebalance)
        self.cost_rate = (cost_bps + slippage_bps) / 1e4
        self.capital = capital
        self.rf = rf
        self.warmup = warmup

    def _rebalance_dates(self) -> list[pd.Timestamp]:
        # last actual trading day in each rebalance period, after the warmup
        idx = self.prices.index
        grp = pd.Series(idx, index=idx).groupby(idx.to_period(self.rebalance)).last()
        return [d for d in grp.values if idx.get_loc(d) >= self.warmup]

    def run(self) -> BacktestResult:
        prices = self.prices
        rets = prices.pct_change()
        tickers = prices.columns

        target = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
        turnover = {}
        prev_w = pd.Series(0.0, index=tickers)

        for d in self._rebalance_dates():
            hist = prices.loc[:d]                         # info only up to d
            w = self.strategy.weights(hist).reindex(tickers).fillna(0.0)
            target.loc[d] = w.values
            turnover[d] = float((w - prev_w).abs().sum())
            prev_w = w

        target = target.ffill().fillna(0.0)
        held = target.shift(1).fillna(0.0)                # trade with 1-day delay

        gross = (held * rets).sum(axis=1)
        # cost charged the day a new target takes effect (one day after it's set)
        rebal_turnover = target.diff().abs().sum(axis=1)
        cost = rebal_turnover.shift(1).fillna(0.0) * self.cost_rate
        net = (gross - cost).fillna(0.0)

        # trim to the active period (first day we actually hold something)
        active = held.abs().sum(axis=1) > 0
        if active.any():
            start = active.idxmax()
            net = net.loc[start:]
        equity = self.capital * (1 + net).cumprod()

        turn = pd.Series(turnover).sort_index()
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
        ann_turn = float(turn.sum() / years)

        stats = metrics.summary(equity, net, rf=self.rf, ann_turnover=ann_turn)
        return BacktestResult(equity=equity, returns=net, weights=target,
                              turnover=turn, ann_turnover=ann_turn, stats=stats,
                              name=getattr(self.strategy, "name", "strategy"))
