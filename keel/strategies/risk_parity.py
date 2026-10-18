"""Inverse-volatility ("risk parity"-style) weighting.

Give calmer stocks more weight and jumpier stocks less, so each name
contributes roughly the same amount of risk. Long-only, weights sum to 1.
A simple, well-known way to build a steadier portfolio than cap/equal weight.
"""
from __future__ import annotations

import pandas as pd

from ..strategy import Strategy


class RiskParity(Strategy):
    def __init__(self, window: int = 60):
        self.window = window
        self.name = f"Risk parity (inverse-vol, {window}d)"

    def weights(self, prices: pd.DataFrame) -> pd.Series:
        if len(prices) < self.window + 1:
            return pd.Series(dtype=float)
        vol = prices.pct_change().iloc[-self.window:].std(ddof=1)
        inv = (1.0 / vol.replace(0, float("nan"))).dropna()
        if inv.empty:
            return pd.Series(dtype=float)
        return inv / inv.sum()
