"""Load the bundled price panel (wide: dates x tickers of daily closes)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "prices.csv"


def load_prices(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or _DEFAULT, parse_dates=["date"], index_col="date")
    return df.sort_index()
