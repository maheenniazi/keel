"""Keel — an honest backtesting engine + strategy research toolkit."""
from . import metrics, strategies, tearsheet
from .data import load_prices
from .engine import Backtest, BacktestResult
from .strategy import Strategy
from .walkforward import WalkForwardResult, walk_forward

__version__ = "1.0.0"
__all__ = [
    "Backtest",
    "BacktestResult",
    "Strategy",
    "WalkForwardResult",
    "load_prices",
    "metrics",
    "strategies",
    "tearsheet",
    "walk_forward",
]
