"""Performance metrics for a strategy's daily returns / equity curve.

All annualized figures assume ~252 trading days. Returns are simple daily
returns unless noted. These are deliberately small, tested, and dependency-light.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ann_return(returns: pd.Series) -> float:
    """Geometric annualized return from a series of daily returns."""
    if len(returns) == 0:
        return 0.0
    growth = float((1 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS / len(returns)) - 1


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio (rf is an annual rate)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / sd) if sd > 1e-12 else 0.0


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    """Like Sharpe but penalizes only downside volatility."""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / dd) if dd > 1e-12 else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak at each point (<= 0)."""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min()) if len(equity) else 0.0


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def calmar(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    return cagr(equity) / mdd if mdd > 0 else 0.0


def hit_rate(returns: pd.Series) -> float:
    return float((returns > 0).mean()) if len(returns) else 0.0


def summary(equity: pd.Series, returns: pd.Series | None = None,
            rf: float = 0.0, ann_turnover: float | None = None) -> dict:
    """A compact dict of headline stats for a tearsheet row."""
    if returns is None:
        returns = equity.pct_change().dropna()
    out = {
        "CAGR": cagr(equity),
        "Ann. return": ann_return(returns),
        "Ann. vol": ann_vol(returns),
        "Sharpe": sharpe(returns, rf),
        "Sortino": sortino(returns, rf),
        "Max drawdown": max_drawdown(equity),
        "Calmar": calmar(equity),
        "Hit rate": hit_rate(returns),
    }
    if ann_turnover is not None:
        out["Turnover/yr"] = ann_turnover
    return out
