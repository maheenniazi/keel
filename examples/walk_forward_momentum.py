"""Walk-forward test of momentum's lookback — honest out-of-sample performance.

    python examples/walk_forward_momentum.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from keel import load_prices, walk_forward
from keel.strategies import Momentum


def main():
    prices = load_prices()
    grid = [63, 126, 189, 252]      # ~3, 6, 9, 12-month lookbacks

    wf = walk_forward(prices, lambda lb: Momentum(lookback=lb), grid,
                      train_years=2, test_years=1)

    print("Walk-forward: pick the best lookback on 2y of history, trade it the next 1y.\n")
    print(wf.choices.to_string(index=False))
    print(f"\nStitched out-of-sample Sharpe: {wf.oos_sharpe:.2f}")
    print("(Compare the OOS column to the IS column — that gap is the overfitting tax.)")


if __name__ == "__main__":
    main()
