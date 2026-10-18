"""Leak-free features for the ML model (everything at row t uses data <= t)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def symbol_features(close: pd.Series, market: pd.Series) -> pd.DataFrame:
    r = np.log(close).diff()
    mr = np.log(market).diff()
    f = pd.DataFrame(index=close.index)
    for lag in (1, 2, 3, 5):
        f[f"ret_lag{lag}"] = r.shift(lag - 1)
    for w in (5, 20, 60, 120, 250):
        f[f"mom_{w}"] = np.log(close / close.shift(w))
    f["vol_20"] = r.rolling(20).std() * np.sqrt(252)
    f["vol_60"] = r.rolling(60).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_20"] / f["vol_60"]
    f["dist_ma50"] = close / close.rolling(50).mean() - 1
    f["dist_ma200"] = close / close.rolling(200).mean() - 1
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = up / (up + dn)
    f["mkt_mom_20"] = np.log(market / market.shift(20))
    f["mkt_vol_20"] = mr.rolling(20).std() * np.sqrt(252)
    f["drawdown_250"] = close / close.rolling(250).max() - 1
    return f


def panel_features(close: pd.DataFrame, benchmark: str, horizon: int) -> pd.DataFrame:
    """Stacked (date, symbol) feature panel plus forward-return targets.

    `fwd_ret` is the log return from close t to close t+horizon; it is NaN for the
    last `horizon` rows, and callers must only train on rows whose target date <= the
    training cut-off (see ensemble._ml_walk_forward).
    """
    market = close[benchmark]
    frames = []
    for i, sym in enumerate(close.columns):
        f = symbol_features(close[sym], market)
        f["sym_id"] = i
        f["fwd_ret"] = np.log(close[sym].shift(-horizon) / close[sym])
        f["symbol"] = sym
        frames.append(f.reset_index().rename(columns={f.index.name or "index": "date"}))
    panel = pd.concat(frames, ignore_index=True)
    return panel.set_index(["date", "symbol"]).sort_index()


FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "mom_5", "mom_20", "mom_60", "mom_120",
    "mom_250", "vol_20", "vol_60", "vol_ratio", "dist_ma50", "dist_ma200", "rsi14",
    "mkt_mom_20", "mkt_vol_20", "drawdown_250", "sym_id",
]
