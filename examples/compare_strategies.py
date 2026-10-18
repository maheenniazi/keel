"""Run every strategy against the benchmark and print a tearsheet.

    python examples/compare_strategies.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from keel import Backtest, load_prices, tearsheet
from keel.strategies import EqualWeight, MeanReversion, Momentum, RiskParity


def main():
    prices = load_prices()
    strategies = [EqualWeight(), Momentum(), MeanReversion(), RiskParity()]

    results = {}
    for s in strategies:
        r = Backtest(prices, s, rebalance="ME", cost_bps=5, slippage_bps=2).run()
        results[r.name] = r

    print(f"Universe: {prices.shape[1]} names, {prices.index[0].date()} .. {prices.index[-1].date()}")
    print("Monthly rebalance, 7 bps round-trip cost + slippage.\n")
    print(tearsheet.format_table(results))
    out = tearsheet.plot_tearsheet(results)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
