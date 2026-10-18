import numpy as np
import pandas as pd

from trading_agent.forecasting.ensemble import _ml_walk_forward, combine
from trading_agent.forecasting.models import HoltDampedState, arima_walk_forward, ets_walk_forward


def synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    rets = rng.normal(0.0003, 0.01, size=(n, 3))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=["SPY", "TLT", "GLD"])


def test_ets_recursion_matches_statsmodels():
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES

    lp = np.log(synthetic()["SPY"].values)
    s = HoltDampedState.fit(lp[:500])
    m = ES(lp[:500], trend="add", damped_trend=True, initialization_method="estimated").fit()
    for y in lp[500:510]:
        s.update(y)
    ref = ES(lp[:510], trend="add", damped_trend=True, initialization_method="known",
             initial_level=m.params["initial_level"], initial_trend=m.params["initial_trend"]).fit(
        smoothing_level=m.params["smoothing_level"], smoothing_trend=m.params["smoothing_trend"],
        damping_trend=m.params["damping_trend"], optimized=False)
    assert np.isclose(s.level, ref.level[-1]) and np.isclose(s.trend, ref.trend[-1])


def test_stat_forecasts_do_not_peek():
    """Changing prices AFTER date t must not change the forecast made at t."""
    close = synthetic()["SPY"]
    dates = close.index[600:620]
    a1 = arima_walk_forward(close, dates, 300, 10, [(1, 0, 0)])
    e1 = ets_walk_forward(close, dates, 300, 10)
    shocked = close.copy()
    shocked.loc[dates[10]:] *= 1.5
    a2 = arima_walk_forward(shocked, dates, 300, 10, [(1, 0, 0)])
    e2 = ets_walk_forward(shocked, dates, 300, 10)
    pd.testing.assert_frame_equal(a1.iloc[:10], a2.iloc[:10])
    pd.testing.assert_frame_equal(e1.iloc[:10], e2.iloc[:10])


def test_ml_does_not_peek():
    close = synthetic()
    dates = close.index[800:820]
    fcfg = {"ml_horizon": 5, "ml_retrain_every": 63, "ml_min_train": 300}
    m1 = _ml_walk_forward(close, dates, fcfg, "SPY")
    shocked = close.copy()
    shocked.loc[dates[5]:] *= 0.5
    m2 = _ml_walk_forward(shocked, dates, fcfg, "SPY")
    first = m1.index.get_level_values("date") < dates[5]
    pd.testing.assert_frame_equal(m1[first], m2[first])


def test_combine_outputs_probabilities():
    df = pd.DataFrame({"mu_ets": [0.001, -0.001], "sd_ets": [0.01, 0.01], "mu_arima": [0.002, -0.002],
                       "sd_arima": [0.01, 0.01], "mu_ml": [0.001, -0.001], "p_ml": [0.7, 0.3]})
    out = combine(df, {"ets": 0.25, "arima": 0.25, "ml": 0.5})
    assert out["p_up"].iloc[0] > 0.5 > out["p_up"].iloc[1]
    assert list(out["agreement"]) == [3, 0]
    assert out["confidence"].between(0, 1).all()
