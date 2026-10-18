"""Cross-sectional momentum.

The classic "12-1" signal: rank stocks by their return from ~12 months ago to
~1 month ago (skipping the most recent month, which tends to mean-revert), then
hold an equal-weighted basket of the top names. One of the most robust,
well-documented anomalies in the academic literature.
"""
from __future__ import annotations

import math

import pandas as pd

from ..strategy import Strategy


class Momentum(Strategy):
    def __init__(self, lookback: int = 252, skip: int = 21, top_frac: float = 0.2):
        self.lookback, self.skip, self.top_frac = lookback, skip, top_frac
        self.name = f"Momentum ({lookback//21}-{skip//21}m, top {int(top_frac*100)}%)"

    def weights(self, prices: pd.DataFrame) -> pd.Series:
        if len(prices) < self.lookback + 1:
            return pd.Series(dtype=float)
        past = prices.iloc[-1 - self.lookback]
        recent = prices.iloc[-1 - self.skip]
        signal = (recent / past - 1).dropna()
        if signal.empty:
            return pd.Series(dtype=float)
        k = max(1, math.ceil(self.top_frac * len(signal)))
        winners = signal.nlargest(k).index
        return pd.Series(1.0 / k, index=winners)
