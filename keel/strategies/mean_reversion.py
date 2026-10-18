"""Cross-sectional mean reversion.

Score each stock by how far its price sits above/below its own recent average
(a z-score). Then go long the names that are unusually cheap and short the ones
that are unusually rich, dollar-neutral (longs and shorts sized equally). The
bet: short-term dislocations snap back.
"""
from __future__ import annotations

import pandas as pd

from ..strategy import Strategy


class MeanReversion(Strategy):
    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"Mean reversion ({window}d, long/short)"

    def weights(self, prices: pd.DataFrame) -> pd.Series:
        if len(prices) < self.window + 1:
            return pd.Series(dtype=float)
        recent = prices.iloc[-self.window:]
        mean, std = recent.mean(), recent.std(ddof=1)
        z = ((prices.iloc[-1] - mean) / std).replace([float("inf"), float("-inf")], float("nan")).dropna()
        if z.empty:
            return pd.Series(dtype=float)
        signal = -(z - z.mean())              # cheap (low z) -> long; rich -> short
        gross = signal.abs().sum()
        if gross == 0:
            return pd.Series(dtype=float)
        return signal / gross                 # dollar-neutral, gross exposure = 1
