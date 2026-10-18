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


def rank_scores(fc: pd.DataFrame, rank_by: str, vols: dict[str, float] | None = None) -> pd.Series:
    """Higher = stronger long forecast. NaN forecasts (no data yet) rank last."""
    if rank_by == "confidence":
        s = fc["confidence"] * np.sign(fc["p_up"] - 0.5)
    elif rank_by == "mu_over_vol":
        v = pd.Series(vols or {}).reindex(fc.index).clip(lower=0.05) / np.sqrt(252)
        s = fc["mu"] / v
    else:
        s = fc["p_up"]
    return s.astype(float)


def select_portfolio(fc: pd.DataFrame, currently_long: dict[str, bool], scfg, pcfg,
                     trend: dict[str, bool] | None = None, vols: dict[str, float] | None = None,
                     returns: pd.DataFrame | None = None) -> dict[str, bool]:
    """Pick which symbols to hold.

    `portfolio.top_n` null -> the original rule: every symbol whose own signal is on.
    Otherwise:
      1. per-symbol eligibility uses the same hysteresis rule as `long_signals`
         (enter at p >= enter_prob with agreement + trend; stay while p >= exit_prob + trend);
      2. eligible symbols are ranked by forecast strength (`rank_by`);
      3. current holdings are kept first if they still rank within top_n + hold_buffer
         (avoids selling a name that slipped from #5 to #6 and buying it back tomorrow);
         hold_buffer null = keep a holding until its own exit rule fires;
      4. remaining slots go to the best new candidates;
      5. a candidate is skipped if its return correlation (over `returns`, the last
         `corr_window` days) with an already-picked symbol exceeds `max_corr`.
    """
    base = long_signals(fc, currently_long, scfg, trend)
    pcfg = pcfg or {}
    top_n = pcfg.get("top_n")
    if top_n is None:
        return base
    scores = rank_scores(fc, pcfg.get("rank_by", "p_up"), vols)
    eligible = [s for s, ok in base.items() if ok and np.isfinite(scores.get(s, np.nan))]
    ranked = sorted(eligible, key=lambda s: -scores[s])
    rank_of = {s: i for i, s in enumerate(ranked)}
    hb = pcfg.get("hold_buffer", 0)
    keep_limit = len(ranked) if hb is None else top_n + int(hb)   # null = hold until own exit rule
    holders = [s for s in ranked if currently_long.get(s, False) and rank_of[s] < keep_limit]
    newcomers = [s for s in ranked if s not in holders]

    corr = None
    max_corr = pcfg.get("max_corr")
    if max_corr is not None and returns is not None and len(ranked) > 1:
        sub = returns.reindex(columns=ranked)
        corr = sub.corr(min_periods=max(20, len(sub) // 2))

    picked: list[str] = []
    for s in holders + newcomers:
        if len(picked) >= top_n:
            break
        if corr is not None and picked:
            c = corr.loc[s, picked]
            if (c > max_corr).any():
                continue
        picked.append(s)
    return {s: s in picked for s in fc.index}


def recent_returns(close: pd.DataFrame, end, window: int) -> pd.DataFrame:
    """Daily log returns for the `window` sessions up to and including `end` (no look-ahead)."""
    r = np.log(close.loc[:end]).diff()
    return r.iloc[-window:]
