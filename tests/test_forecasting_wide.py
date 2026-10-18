import numpy as np
import pandas as pd

from trading_agent.config import load_config
from trading_agent.forecasting import ensemble
from trading_agent.forecasting.ensemble import eligible_dates, walk_forward_forecasts


def panel(n=700, late=450, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, 3)), axis=0)), index=idx,
                     columns=["SPY", "TLT", "NEW"])
    c.iloc[:late, 2] = np.nan                 # NEW lists on day `late`
    return c


def test_eligible_dates_need_min_history():
    c = panel()
    dates = c.index[500:]
    el = eligible_dates(c["NEW"], dates, 252)
    assert len(el) == 0                       # only 250 bars by the end of the window... none eligible
    el2 = eligible_dates(c["NEW"], c.index[600:], 100)
    assert el2[0] == c.index[600] and len(el2) == 100


def test_late_symbol_gets_forecasts_only_after_min_history(tmp_path):
    cfg = load_config()
    f = dict(cfg["forecasting"], min_history=150, train_window=200, refit_every=50,
             ml_min_train=300, ml_retrain_every=50, n_jobs=1)
    c = panel()
    fc = walk_forward_forecasts(c, str(c.index[560].date()), f, "SPY", cache_dir=tmp_path)
    new = fc.xs("NEW", level="symbol")
    assert new.index[0] == c["NEW"].dropna().index[149]
    assert np.isfinite(new[["mu_ets", "mu_arima", "p_up"]].values).all()
    assert len(fc.xs("SPY", level="symbol")) == len(c.loc[c.index[560]:])


def test_stat_cache_is_reused_and_keyed_on_prices(tmp_path, monkeypatch):
    cfg = load_config()
    f = dict(cfg["forecasting"], min_history=150, train_window=200, refit_every=50,
             ml_min_train=300, ml_retrain_every=50, n_jobs=1)
    c = panel()[["SPY", "TLT"]]
    start = str(c.index[650].date())
    a = walk_forward_forecasts(c, start, f, "SPY", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == 2

    def boom(*a, **k):
        raise AssertionError("should have used the cache")

    monkeypatch.setattr(ensemble, "ets_walk_forward", boom)
    b = walk_forward_forecasts(c, start, f, "SPY", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(a, b)
    c2 = c.copy()
    c2.iloc[-1, 0] *= 1.01                    # different price history -> cache miss -> recompute
    try:
        walk_forward_forecasts(c2, start, f, "SPY", cache_dir=tmp_path)
        raise RuntimeError("expected a recompute")
    except AssertionError:
        pass
