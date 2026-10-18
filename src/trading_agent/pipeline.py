"""Shared helpers: load data + cached walk-forward forecasts."""
from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from .config import resolve
from .data.loader import load_prices
from .forecasting.ensemble import combine, walk_forward_forecasts

log = logging.getLogger("trading_agent.pipeline")


def get_prices(cfg, refresh: bool = False):
    return load_prices(cfg["universe"], cfg["data"]["start"], resolve(cfg["data"]["cache_dir"]),
                       refresh=refresh)


def _fc_key(cfg) -> str:
    f = {k: v for k, v in cfg["forecasting"].items() if k not in ("weights", "n_jobs", "cache_dir")}
    blob = json.dumps([cfg["universe"], cfg["data"]["start"], cfg["backtest"]["start"], f], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def get_forecasts(cfg, prices, refresh: bool = False) -> pd.DataFrame:
    """Walk-forward OOS forecasts, cached on disk (they take a few minutes to compute)."""
    out_dir = resolve(cfg["backtest"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"forecasts_{_fc_key(cfg)}.pkl"
    if path.exists() and not refresh:
        fc = pd.read_pickle(path)
        last = prices.close.index[-1]
        if fc.index.get_level_values("date").max() >= last:
            return combine(fc, cfg["forecasting"]["weights"])
    fc = walk_forward_forecasts(prices.close, cfg["backtest"]["start"], cfg["forecasting"],
                                cfg["benchmark"], cfg["backtest"]["end"],
                                cache_dir=resolve(cfg["forecasting"].get("cache_dir", "results/fc_cache")))
    fc.to_pickle(path)
    return fc
