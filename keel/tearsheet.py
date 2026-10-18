"""Turn backtest results into a metrics table and a tearsheet figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics as M
from .engine import BacktestResult

_PCT = {"CAGR", "Ann. return", "Ann. vol", "Max drawdown", "Hit rate"}
_COLORS = ["#2a7ae2", "#e07a55", "#3f8f6b", "#b4689e", "#d9a43a", "#6a8ec9"]


def metrics_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """A tidy DataFrame of headline stats, one column per strategy."""
    df = pd.DataFrame({name: r.stats for name, r in results.items()})
    return df


def format_table(results: dict[str, BacktestResult]) -> str:
    df = metrics_table(results)
    out = df.astype(object)
    for row in out.index:
        if row in _PCT:
            out.loc[row] = [f"{x*100:6.1f}%" for x in df.loc[row]]
        elif row == "Turnover/yr":
            out.loc[row] = [f"{x:6.2f}x" for x in df.loc[row]]
        else:
            out.loc[row] = [f"{x:6.2f}" for x in df.loc[row]]
    return out.to_string()


def plot_tearsheet(results: dict[str, BacktestResult], path: str = "tearsheet.png",
                   title: str = "Keel — strategy comparison") -> str:
    fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 1]})
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for i, (name, r) in enumerate(results.items()):
        c = _COLORS[i % len(_COLORS)]
        eq = r.equity / r.equity.iloc[0]
        ax[0].plot(eq.index, eq.values, label=f"{name}  (Sharpe {r.stats['Sharpe']:.2f})", lw=1.5, color=c)
        dd = M.drawdown_series(r.equity)
        ax[1].plot(dd.index, dd.values * 100, lw=1.1, color=c)
        roll = r.returns.rolling(252).apply(lambda x: M.sharpe(pd.Series(x)), raw=False)
        ax[2].plot(roll.index, roll.values, lw=1.1, color=c)

    ax[0].set_ylabel("growth of $1"); ax[0].legend(loc="upper left", fontsize=9)
    ax[0].axhline(1, color="#999", lw=0.6)
    ax[1].set_ylabel("drawdown %"); ax[1].axhline(0, color="#999", lw=0.6)
    ax[2].set_ylabel("rolling 1y Sharpe"); ax[2].axhline(0, color="#999", lw=0.6)
    ax[2].set_xlabel("date")
    for a in ax:
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
