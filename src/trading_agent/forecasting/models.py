"""Per-symbol statistical forecasters: ETS (Holt damped trend) and ARIMA.

Both are walk-forward: parameters are (re)fit every `refit_every` days on a trailing
window, and between refits the fitted model is updated with each new observation
without re-estimating parameters. Every forecast for day t+1 uses data <= t only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm


def _quiet():
    warnings.filterwarnings("ignore")
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        warnings.simplefilter("ignore", ConvergenceWarning)
    except Exception:  # pragma: no cover
        pass


# ----------------------------------------------------------------------------- ETS
class HoltDampedState:
    """Additive damped-trend exponential smoothing, ETS(A,Ad,N), on log prices."""

    def __init__(self, alpha: float, beta: float, phi: float, level: float, trend: float,
                 resid_sd: float):
        self.alpha, self.beta, self.phi = alpha, beta, phi
        self.level, self.trend, self.resid_sd = level, trend, resid_sd

    @classmethod
    def fit(cls, log_price: np.ndarray) -> "HoltDampedState":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        _quiet()
        m = ExponentialSmoothing(log_price, trend="add", damped_trend=True,
                                 initialization_method="estimated").fit()
        p = m.params
        resid = log_price - m.fittedvalues
        return cls(float(p["smoothing_level"]), float(p["smoothing_trend"]),
                   float(p["damping_trend"]), float(m.level[-1]), float(m.trend[-1]),
                   float(np.std(resid[20:], ddof=1)))

    def update(self, y: float) -> None:
        prev_level = self.level
        self.level = self.alpha * y + (1 - self.alpha) * (prev_level + self.phi * self.trend)
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.phi * self.trend

    def forecast_return(self, last_log_price: float) -> tuple[float, float]:
        """Expected next-day log return and its standard deviation."""
        return self.level + self.phi * self.trend - last_log_price, self.resid_sd


def ets_walk_forward(close: pd.Series, dates: pd.DatetimeIndex, window: int,
                     refit_every: int) -> pd.DataFrame:
    logp = np.log(close.values)
    idx = {d: i for i, d in enumerate(close.index)}
    out, state, since = [], None, refit_every
    last_i = None
    for d in dates:
        i = idx[d]
        if state is None or since >= refit_every:
            state = HoltDampedState.fit(logp[max(0, i - window + 1): i + 1])
            since = 0
        else:
            for j in range(last_i + 1, i + 1):
                state.update(logp[j])
            since += i - last_i
        last_i = i
        mu, sd = state.forecast_return(logp[i])
        out.append((d, mu, sd))
    return pd.DataFrame(out, columns=["date", "mu_ets", "sd_ets"]).set_index("date")


# --------------------------------------------------------------------------- ARIMA
def fit_arima(returns: np.ndarray, orders: list[tuple[int, int, int]]):
    from statsmodels.tsa.arima.model import ARIMA

    _quiet()
    best = None
    for order in orders:
        try:
            res = ARIMA(returns, order=tuple(order), trend="c").fit()
        except Exception:
            continue
        if best is None or res.aic < best.aic:
            best = res
    if best is None:
        raise RuntimeError("ARIMA failed to fit for every candidate order")
    return best


def arima_walk_forward(close: pd.Series, dates: pd.DatetimeIndex, window: int,
                       refit_every: int, orders) -> pd.DataFrame:
    rets = np.log(close).diff().values
    idx = {d: i for i, d in enumerate(close.index)}
    out, base, base_i, since = [], None, None, refit_every
    for d in dates:
        i = idx[d]
        if base is None or since >= refit_every:
            lo = max(1, i - window + 1)
            base, base_i, since = fit_arima(rets[lo: i + 1], orders), i, 0
            res = base
        else:
            since = i - base_i
            res = base.extend(rets[base_i + 1: i + 1])
        fc = res.get_forecast(1)
        mu = float(np.asarray(fc.predicted_mean)[0])
        sd = float(np.asarray(fc.se_mean)[0])
        out.append((d, mu, sd))
    return pd.DataFrame(out, columns=["date", "mu_arima", "sd_arima"]).set_index("date")


def prob_up(mu, sd):
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-9)
    return norm.cdf(np.asarray(mu, dtype=float) / sd)
