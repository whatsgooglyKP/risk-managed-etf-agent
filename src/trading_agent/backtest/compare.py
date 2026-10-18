"""Side-by-side walk-forward comparison of configs (e.g. 5-ETF v1 vs wide universe v2) vs SPY.

Same engine, same costs, same periods for every config:
  * full out-of-sample period (backtest.start -> end)
  * 2008-2016 and 2017-present, each started fresh with the same $77
Writes results/comparison.json, docs/comparison.md and docs/images/compare_*.png.
"""
from __future__ import annotations

import copy
import json
import logging

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, load_config, resolve
from ..pipeline import get_forecasts, get_prices
from .engine import Backtester, buy_and_hold
from .metrics import compute_metrics, format_table
from .plots import plot_drawdown, plot_equity, plot_exposure

log = logging.getLogger("trading_agent.compare")

SPLIT = "2017-01-01"     # same split as the RL held-out test in the original report


def _cost_bps(cfg) -> float:
    return cfg["costs"]["slippage_bps"] + cfg["costs"]["spread_bps"] / 2


def run_one(cfg, prices, fc, start, end) -> tuple[dict, object]:
    res = Backtester(prices, fc, cfg, start, end).run()
    m = compute_metrics(res.equity, res.trades, res.exposure, res.floor_triggers)
    years = max((res.equity.index[-1] - res.equity.index[0]).days / 365.25, 1e-9)
    notional = float(res.trades["notional"].sum()) if not res.trades.empty else 0.0
    m["costs_paid"] = notional * _cost_bps(cfg) / 1e4
    m["turnover"] = notional / float(res.equity.mean()) / years
    held = pd.Series(dtype=float)
    if not res.trades.empty:
        t = res.trades.copy()
        t["signed"] = np.where(t["side"] == "buy", t["qty"], -t["qty"])
        qty = t.pivot_table(index="date", columns="symbol", values="signed", aggfunc="sum").cumsum()
        qty = qty.reindex(res.equity.index).ffill().fillna(0.0)
        held = (qty > 1e-9).sum(axis=1)
    m["avg_positions"] = float(held.mean()) if len(held) else 0.0
    return m, res


def _variants(cfg) -> dict[str, dict]:
    """Every variant that was tried is reported (no silent cherry-picking)."""
    etfs = [s for s in cfg["universe"]
            if s in {"SPY", "QQQ", "IWM", "MDY", "EFA", "EEM", "EWJ", "EWZ", "FXI", "TLT", "IEF", "TIP",
                     "LQD", "AGG", "GLD", "SLV", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV",
                     "XLY", "IYR"}]
    return {
        "as designed (top 5)": {},
        "top_n=3": {"portfolio": {"top_n": 3}},
        "top_n=8": {"portfolio": {"top_n": 8}, "risk": {"max_open_positions": 9}},
        "no corr filter": {"portfolio": {"max_corr": None}},
        "no portfolio vol cap": {"risk": {"portfolio_vol_cap": None}},
        "hold until own exit": {"portfolio": {"hold_buffer": None}},
        "rank by mu/vol": {"portfolio": {"rank_by": "mu_over_vol"}},
        "ETFs only (26), hold until exit": {
            "universe": etfs, "sectors": {s: g for s, g in cfg["sectors"].items() if s in etfs},
            "portfolio": {"hold_buffer": None}},
    }


def run_sensitivity(cfg, start, end, prices=None) -> dict:
    """Variants of the wide config on the SAME price data (sub-universes are column slices)."""
    from ..data.loader import PriceData

    prices = prices if prices is not None else get_prices(cfg)
    out = {}
    for name, ov in _variants(cfg).items():
        c = copy.deepcopy(dict(cfg))
        for k, v in ov.items():
            if isinstance(v, dict) and k not in ("sectors",):
                c[k].update(v)
            else:
                c[k] = v
        log.info("Sensitivity: %s", name)
        u = list(c["universe"])
        p = prices if u == list(prices.symbols) else PriceData(prices.open[u], prices.close[u])
        out[name] = run_one(c, p, get_forecasts(c, p), start, end)[0]
    return out


def run_compare(config_paths: list[str], sensitivity: bool = False) -> dict:
    img = PROJECT_ROOT / "docs" / "images"
    img.mkdir(parents=True, exist_ok=True)
    cfgs = [load_config(p) for p in config_paths]
    base = cfgs[0]
    start, end = base["backtest"]["start"], base["backtest"]["end"]
    cap, floor = base["risk"]["initial_capital"], base["risk"]["hard_floor"]
    split_end = str((pd.Timestamp(SPLIT) - pd.Timedelta(days=1)).date())
    periods = {"full": (start, end), "2008-2016": (start, split_end), "2017-now": (SPLIT, end)}

    tables = {k: {} for k in periods}
    curves = {k: {} for k in periods}
    expo = {}
    data = []
    for cfg, path in zip(cfgs, config_paths):
        name = cfg.get("name") or path
        prices = get_prices(cfg)
        fc = get_forecasts(cfg, prices)
        data.append((name, cfg, prices, fc))
        for k, (a, b) in periods.items():
            log.info("Backtest %s, %s", name, k)
            m, res = run_one(cfg, prices, fc, a, b)
            tables[k][name] = m
            curves[k][name] = res.equity
            if k == "full":
                expo[name] = res.exposure
                res.trades.to_csv(resolve(cfg["backtest"]["results_dir"]) /
                                  f"trades_{name.split()[0].lower().strip(',')}.csv", index=False)

    spy_prices = data[0][2]
    for k, (a, b) in periods.items():
        first = str(next(iter(curves[k].values())).index[0].date())
        spy = buy_and_hold(spy_prices, base["benchmark"], first, b, cap, _cost_bps(base))
        tables[k]["Buy & hold SPY"] = compute_metrics(spy, None, None)
        curves[k]["Buy & hold SPY"] = spy

    first_full = next(iter(curves["full"].values())).index
    span = f"{first_full[0].date()} to {first_full[-1].date()}"
    plot_equity(curves["full"], img / "compare_equity.png",
                f"Out-of-sample equity, ${cap:.0f} start, costs included ({span})", floor)
    plot_drawdown(curves["full"], img / "compare_drawdown.png", "Drawdown (full period)")
    plot_exposure(expo, img / "compare_exposure.png", "Market exposure (full period)")
    plot_equity(curves["2017-now"], img / "compare_equity_2017.png",
                f"Fresh ${cap:.0f} start on {SPLIT}", floor)

    summary = {"periods": periods, "tables": tables}
    if sensitivity and len(data) > 1:
        summary["sensitivity"] = run_sensitivity(data[-1][1], start, end, data[-1][2])
        tables["sensitivity (wide, full period)"] = summary["sensitivity"]

    out = resolve(base["backtest"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(summary, indent=2, default=str))
    titles = {"full": f"Full out-of-sample period ({span})",
              "2008-2016": "2008-2016 (fresh $77 start)",
              "2017-now": f"{SPLIT[:4]}-present (fresh $77 start)"}
    md = ["# 5-ETF (v1) vs wide universe (v2)\n",
          "Generated by `trading-agent compare --sensitivity`. Walk-forward, out-of-sample, "
          f"${cap:.0f} start, {_cost_bps(base):.0f} bps cost per trade side, same engine and risk rules.\n"]
    for k, rows in tables.items():
        md.append(f"## {titles.get(k, k)}\n\n{format_table(rows)}\n")
    (PROJECT_ROOT / "docs" / "comparison.md").write_text("\n".join(md))
    for k, rows in tables.items():
        log.info("%s\n%s", k, format_table(rows))
    return summary
