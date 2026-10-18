"""Equal-weight benchmark: hold every name in the universe, rebalanced.

The honest yardstick every strategy should be compared against — if you can't
beat equal-weight after costs, your signal isn't adding anything.
"""
from __future__ import annotations

import pandas as pd

from ..strategy import Strategy


class EqualWeight(Strategy):
    name = "Equal-weight"

    def weights(self, prices: pd.DataFrame) -> pd.Series:
        live = prices.iloc[-1].dropna()
        if live.empty:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(live), index=live.index)
