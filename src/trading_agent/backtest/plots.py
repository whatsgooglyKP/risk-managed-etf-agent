"""Matplotlib charts saved as PNG (headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"Agent (no RL)": "#1f77b4", "Agent + RL": "#2ca02c", "Buy & hold SPY": "#7f7f7f",
          "5-ETF (v1)": "#1f77b4", "Wide-50, top 5 (v2)": "#ff7f0e"}


def plot_equity(curves: dict[str, pd.Series], path: Path, title: str, floor: float | None = None):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, s in curves.items():
        ax.plot(s.index, s.values, label=name, lw=1.6, color=COLORS.get(name))
    if floor:
        ax.axhline(floor, color="#d62728", ls="--", lw=1, label=f"Hard floor ${floor:.0f}")
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_title(title)
    ax.set_ylabel("Equity ($, log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_drawdown(curves: dict[str, pd.Series], path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, s in curves.items():
        dd = s / s.cummax() - 1
        ax.fill_between(dd.index, dd.values * 100, 0, alpha=0.35, label=name, color=COLORS.get(name))
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_exposure(expo: dict[str, pd.Series], path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for name, s in expo.items():
        ax.plot(s.index, s.rolling(21).mean().values * 100, label=name, lw=1.2, color=COLORS.get(name))
    ax.set_title(title)
    ax.set_ylabel("Invested (%), 1-mo avg")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
