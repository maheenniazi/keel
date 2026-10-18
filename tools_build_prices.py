#!/usr/bin/env python3
"""Build keel's bundled price panel from the public S&P 500 daily dataset.

Keeps the 100 most liquid names (by average dollar volume) that have a complete
daily history over the common window, and writes a wide date x ticker CSV of
closes. Source: plotly/datasets all_stocks_5yr.csv (Kaggle S&P 500, 2013-2018).

Usage: python3 tools_build_prices.py all_stocks_5yr.csv data/prices.csv
"""
import sys
import pandas as pd

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "all_stocks_5yr.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/prices.csv"

    df = pd.read_csv(src, usecols=["date", "close", "volume", "Name"], parse_dates=["date"])
    close = df.pivot(index="date", columns="Name", values="close").sort_index()
    vol = df.pivot(index="date", columns="Name", values="volume").sort_index()

    # keep names with a complete history over the full window
    full = close.columns[close.notna().all()]
    close, vol = close[full], vol[full]

    # 100 most liquid by average dollar volume
    dollar_vol = (close * vol).mean().sort_values(ascending=False)
    universe = dollar_vol.head(100).index.tolist()

    panel = close[sorted(universe)].round(4)
    panel.index.name = "date"
    panel.to_csv(out)
    print(f"wrote {out}: {panel.shape[1]} tickers x {panel.shape[0]} days "
          f"({panel.index[0].date()} .. {panel.index[-1].date()})")

if __name__ == "__main__":
    main()
