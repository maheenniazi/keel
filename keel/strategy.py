"""Strategy interface.

A strategy maps a window of price history (everything up to a rebalance date)
to a set of target weights across the universe. Long-only strategies return
non-negative weights summing to 1; long/short strategies may return negatives
(the engine handles both). Returning all-zeros means "hold cash".
"""
from __future__ import annotations

import pandas as pd


class Strategy:
    name = "strategy"

    def weights(self, prices: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
