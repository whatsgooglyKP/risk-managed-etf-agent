"""Standard performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_trade_pnl(trades: pd.DataFrame) -> list[float]:
    """Average-cost realized P&L per sell (a 'trade' = any reduction of a position)."""
    pnl, qty, cost = [], {}, {}
    if trades is None or trades.empty:
        return pnl
    for t in trades.itertuples():
        s = t.symbol
        if t.side == "buy":
            cost[s] = cost.get(s, 0.0) + t.qty * t.price
            qty[s] = qty.get(s, 0.0) + t.qty
        elif qty.get(s, 0) > 0:
            avg = cost[s] / qty[s]
            pnl.append((t.price - avg) * t.qty)
            frac = min(1.0, t.qty / qty[s])
            cost[s] *= 1 - frac
            qty[s] -= t.qty
    return pnl


def compute_metrics(equity: pd.Series, trades: pd.DataFrame | None = None,
                    exposure: pd.Series | None = None, floor_triggers: int = 0) -> dict:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    sd = rets.std()
    downside = rets[rets < 0].std()
    sharpe = np.sqrt(252) * rets.mean() / sd if sd > 0 else 0.0
    sortino = np.sqrt(252) * rets.mean() / downside if downside and downside > 0 else 0.0
    dd = equity / equity.cummax() - 1
    pnl = realized_trade_pnl(trades) if trades is not None else []
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "start_equity": round(float(equity.iloc[0]), 2),
        "end_equity": round(float(equity.iloc[-1]), 2),
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": float(sd * np.sqrt(252)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd.min()),
        "win_rate": float(np.mean([p > 0 for p in pnl])) if pnl else float("nan"),
        "exposure": float(exposure.mean()) if exposure is not None else 1.0,
        "n_trades": int(len(trades)) if trades is not None else 1,
        "floor_triggers": int(floor_triggers),
        "min_equity": round(float(equity.min()), 2),
    }


def format_table(rows: dict[str, dict]) -> str:
    """Markdown table: one column per strategy."""
    keys = [("total_return", "Total return", "{:.1%}"), ("cagr", "CAGR", "{:.2%}"),
            ("ann_vol", "Volatility (ann.)", "{:.1%}"), ("sharpe", "Sharpe", "{:.2f}"),
            ("sortino", "Sortino", "{:.2f}"), ("max_drawdown", "Max drawdown", "{:.1%}"),
            ("win_rate", "Win rate (closed trades)", "{:.1%}"), ("exposure", "Avg. exposure", "{:.0%}"),
            ("n_trades", "# orders", "{:d}"), ("avg_positions", "Avg. # holdings", "{:.1f}"),
            ("turnover", "Turnover (x equity / yr)", "{:.1f}"),
            ("costs_paid", "Trading costs paid ($)", "{:.2f}"),
            ("floor_triggers", "Floor triggers", "{:d}"),
            ("min_equity", "Lowest equity ($; floor is $60)", "{:.2f}"),
            ("end_equity", "Ending equity ($)", "{:.2f}")]
    names = list(rows)
    out = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k, label, fmt in keys:
        cells = []
        for n in names:
            v = rows[n].get(k)
            try:
                cells.append("n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v))
            except (ValueError, TypeError):
                cells.append(str(v))
        if all(c == "n/a" for c in cells) and k in ("avg_positions", "turnover", "costs_paid"):
            continue
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)
