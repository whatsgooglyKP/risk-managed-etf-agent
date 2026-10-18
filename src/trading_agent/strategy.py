"""Turn ensemble forecasts into long/flat signals (the risk engine decides the size)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(close: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """Annualised EWMA (RiskMetrics) volatility, known at each close."""
    r = np.log(close).diff()
    var = (r ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=20).mean()
    return np.sqrt(var * 252)


def trend_ok(close: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    return close > close.rolling(window).mean()


def long_signals(fc: pd.DataFrame, currently_long: dict[str, bool], scfg,
                 trend: dict[str, bool] | None = None) -> dict[str, bool]:
    """Hysteresis rule per symbol.

    Enter when P(up) >= enter_prob, >= min_agreement models are positive and (optionally)
    price > 200-day MA. Stay in while P(up) >= exit_prob and the trend filter holds.
    `fc` is indexed by symbol with columns p_up, agreement.
    """
    out = {}
    for sym, row in fc.iterrows():
        t_ok = True if not scfg["trend_filter"] or trend is None else bool(trend.get(sym, False))
        if currently_long.get(sym, False):
            out[sym] = bool(row["p_up"] >= scfg["exit_prob"] and t_ok)
        else:
            out[sym] = bool(row["p_up"] >= scfg["enter_prob"]
                            and row["agreement"] >= scfg["min_agreement"] and t_ok)
    return out
