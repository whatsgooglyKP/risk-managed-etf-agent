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

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .features import FEATURE_COLS, panel_features
from .models import arima_walk_forward, ets_walk_forward, prob_up

log = logging.getLogger("trading_agent.forecasting")


def _stat_job(args):
    sym, close, dates, window, refit_every, orders = args
    ets = ets_walk_forward(close, dates, window, refit_every)
    ari = arima_walk_forward(close, dates, window, refit_every, orders)
    df = ets.join(ari)
    df["symbol"] = sym
    return df.reset_index().set_index(["date", "symbol"])


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
                           end: str | None = None, n_jobs: int = 5) -> pd.DataFrame:
    """Out-of-sample forecasts for every trading day in [start, end]."""
    dates = close.loc[start:end].index
    window, refit = int(fcfg["train_window"]), int(fcfg["refit_every"])
    orders = [tuple(o) for o in fcfg["arima_orders"]]
    jobs = [(s, close[s], dates, window, refit, orders) for s in close.columns]
    log.info("Walk-forward ETS/ARIMA for %d symbols x %d days", len(jobs), len(dates))
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
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
    df = walk_forward_forecasts(close, str(last.date()), fcfg, benchmark, n_jobs=1)
    return df.xs(last, level="date")
