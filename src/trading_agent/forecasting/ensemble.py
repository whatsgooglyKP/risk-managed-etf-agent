"""Walk-forward ensemble of ETS + ARIMA + gradient boosting with a confidence estimate.

Output (one row per date x symbol), all known at that date's close:
    mu_ets, sd_ets, mu_arima, sd_arima, mu_ml, p_ml,
    p_up       ensemble probability that the next return is positive
    mu         ensemble expected next-day log return
    agreement  number of models (0-3) with a positive expected return
    dispersion std of the three model means (model disagreement)
    confidence 2*|p_up - 0.5| in [0, 1]
"""
from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLS, panel_features
from .models import arima_walk_forward, ets_walk_forward, prob_up

log = logging.getLogger("trading_agent.forecasting")


STAT_COLS = ["mu_ets", "sd_ets", "mu_arima", "sd_arima"]


def eligible_dates(close: pd.Series, dates: pd.DatetimeIndex, min_history: int) -> pd.DatetimeIndex:
    """Dates (within `dates`) on which `close` has at least `min_history` bars up to and incl. d."""
    s = close.dropna()
    if len(s) < min_history:
        return pd.DatetimeIndex([])
    first_ok = s.index[min_history - 1]
    return dates[(dates >= first_ok) & dates.isin(s.index)]


def _stat_key(sym: str, close: pd.Series, dates: pd.DatetimeIndex, window: int, refit: int,
              orders) -> str:
    """Cache key: the model settings, the forecast dates and the exact price history used."""
    s = close.dropna().loc[: dates[-1]]
    h = hashlib.sha1()
    h.update(repr((sym, window, refit, [tuple(o) for o in orders], str(dates[0]), str(dates[-1]),
                   len(dates))).encode())
    h.update(pd.util.hash_pandas_object(s.round(6), index=True).values.tobytes())
    return h.hexdigest()[:16]


def _single_thread_worker():
    """One BLAS thread per worker process: N processes x M BLAS threads oversubscribes the CPU."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001 - optional optimisation
        pass


def _stat_job(args):
    sym, close, dates, window, refit_every, orders, cache_dir = args
    if len(dates) == 0:
        return pd.DataFrame(columns=STAT_COLS, index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), []], names=["date", "symbol"]))
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{sym}_{_stat_key(sym, close, dates, window, refit_every, orders)}.pkl"
        if path.exists():
            try:
                return pd.read_pickle(path)
            except Exception:  # noqa: BLE001 - corrupt cache file: recompute
                pass
    series = close.dropna()
    ets = ets_walk_forward(series, dates, window, refit_every)
    ari = arima_walk_forward(series, dates, window, refit_every, orders)
    df = ets.join(ari)
    df["symbol"] = sym
    df = df.reset_index().set_index(["date", "symbol"])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        df.to_pickle(tmp)
        tmp.replace(path)
    return df


def _make_models(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    common = dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
                  l2_regularization=1.0, random_state=seed)
    return HistGradientBoostingClassifier(**common), HistGradientBoostingRegressor(**common)


def _ml_walk_forward(close: pd.DataFrame, dates: pd.DatetimeIndex, fcfg, benchmark: str) -> pd.DataFrame:
    """Pooled (all symbols) gradient boosting, retrained on an expanding window.

    Leakage guard: a row at date d has target fwd_ret over (d, d+h]; when training at
    cut-off T we keep only rows with d <= T - h trading days, so no future data is used.
    """
    h = int(fcfg["ml_horizon"])
    panel = panel_features(close, benchmark, h).dropna(subset=FEATURE_COLS)
    all_dates = close.index
    pos = {d: i for i, d in enumerate(all_dates)}
    out = []
    clf = reg = None
    last_train = None
    for d in dates:
        if clf is None or pos[d] - pos[last_train] >= fcfg["ml_retrain_every"]:
            cutoff = all_dates[pos[d] - h] if pos[d] - h >= 0 else None
            train = panel.loc[panel.index.get_level_values("date") <= cutoff] if cutoff is not None else panel.iloc[:0]
            train = train.dropna(subset=["fwd_ret"])
            if len(train) >= fcfg["ml_min_train"]:
                clf, reg = _make_models()
                clf.fit(train[FEATURE_COLS].values, (train["fwd_ret"] > 0).astype(int).values)
                reg.fit(train[FEATURE_COLS].values, train["fwd_ret"].values / h)
                last_train = d
        try:
            rows = panel.xs(d, level="date", drop_level=False)
        except KeyError:
            continue
        if clf is None:
            p = np.full(len(rows), 0.5)
            mu = np.zeros(len(rows))
        else:
            X = rows[FEATURE_COLS].values
            p = clf.predict_proba(X)[:, 1]
            mu = reg.predict(X)
        out.append(pd.DataFrame({"p_ml": p, "mu_ml": mu}, index=rows.index))
    return pd.concat(out).sort_index() if out else pd.DataFrame(columns=["p_ml", "mu_ml"])


def combine(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Blend model outputs into ensemble probability, mean, agreement and confidence."""
    df = df.copy()
    w_e, w_a, w_m = weights["ets"], weights["arima"], weights["ml"]
    tot = w_e + w_a + w_m
    p_e = prob_up(df["mu_ets"], df["sd_ets"])
    p_a = prob_up(df["mu_arima"], df["sd_arima"])
    df["p_up"] = (w_e * p_e + w_a * p_a + w_m * df["p_ml"]) / tot
    df["mu"] = (w_e * df["mu_ets"] + w_a * df["mu_arima"] + w_m * df["mu_ml"]) / tot
    means = df[["mu_ets", "mu_arima", "mu_ml"]]
    df["agreement"] = (means > 0).sum(axis=1)
    df["dispersion"] = means.std(axis=1)
    df["confidence"] = (2 * (df["p_up"] - 0.5).abs()).clip(0, 1)
    return df


def walk_forward_forecasts(close: pd.DataFrame, start: str, fcfg, benchmark: str,
                           end: str | None = None, n_jobs: int | None = None,
                           cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Out-of-sample forecasts for every trading day in [start, end].

    Symbols with a shorter history (listed later) get forecasts only once they have
    `min_history` bars. ETS/ARIMA run one process per symbol and, if `cache_dir` is given,
    each symbol's walk-forward result is cached on disk keyed by its exact price history,
    so the same symbol is never recomputed across configs or reruns.
    """
    dates = close.loc[start:end].index
    window, refit = int(fcfg["train_window"]), int(fcfg["refit_every"])
    min_hist = int(fcfg.get("min_history", 252))
    orders = [tuple(o) for o in fcfg["arima_orders"]]
    if n_jobs is None:
        n_jobs = int(fcfg.get("n_jobs", 1) or 1)
    n_jobs = max(1, min(n_jobs, os.cpu_count() or 1))
    jobs = [(s, close[s], eligible_dates(close[s], dates, min_hist), window, refit, orders, cache_dir)
            for s in close.columns]
    log.info("Walk-forward ETS/ARIMA for %d symbols x %d days (%d workers)", len(jobs), len(dates), n_jobs)
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs)),
                                 initializer=_single_thread_worker) as ex:
            stat = pd.concat(list(ex.map(_stat_job, jobs)))
    else:
        stat = pd.concat([_stat_job(j) for j in jobs])
    log.info("Walk-forward gradient boosting")
    ml = _ml_walk_forward(close, dates, fcfg, benchmark)
    df = stat.join(ml, how="left")
    df["p_ml"] = df["p_ml"].fillna(0.5)
    df["mu_ml"] = df["mu_ml"].fillna(0.0)
    return combine(df.sort_index(), fcfg["weights"])


def latest_forecasts(close: pd.DataFrame, fcfg, benchmark: str) -> pd.DataFrame:
    """Forecasts for the next session using all data through the last close (live use)."""
    last = close.index[-1]
    fcfg = dict(fcfg)
    fcfg["ml_retrain_every"] = 1
    df = walk_forward_forecasts(close, str(last.date()), fcfg, benchmark)
    return df.xs(last, level="date")
