import numpy as np
import pandas as pd

from trading_agent.backtest.engine import Backtester
from trading_agent.backtest.metrics import compute_metrics
from trading_agent.data.loader import PriceData

UNI = ["SPY", "QQQ", "TLT", "GLD", "IWM"]


def fake_market(n=400, drift=0.0004, crash_at=None, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    r = rng.normal(drift, 0.01, size=(n, len(UNI)))
    if crash_at:
        r[crash_at:crash_at + 30] = -0.03
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=UNI)
    opn = close.shift(1).fillna(close.iloc[0])
    fc = []
    for d in idx:
        for s in UNI:
            fc.append({"date": d, "symbol": s, "p_up": 0.7, "mu": 0.001, "agreement": 3, "confidence": 0.4})
    return PriceData(opn, close), pd.DataFrame(fc).set_index(["date", "symbol"])


def test_backtest_runs_and_respects_exposure(cfg):
    prices, fc = fake_market()
    cfg["signal"]["trend_filter"] = False
    res = Backtester(prices, fc, cfg, "2018-03-01").run()
    assert len(res.equity) > 100
    assert res.exposure.max() <= 1.0 + 1e-6
    m = compute_metrics(res.equity, res.trades, res.exposure)
    assert np.isfinite(m["sharpe"]) and m["n_trades"] > 0


def test_crash_never_breaches_floor_by_much_and_halts(cfg):
    prices, fc = fake_market(crash_at=150)
    cfg["signal"]["trend_filter"] = False
    cfg["risk"]["max_drawdown_hard"] = 0.99   # disable drawdown brake to force the floor
    cfg["risk"]["max_drawdown_soft"] = 0.99
    cfg["risk"]["daily_loss_cap"] = 0.99
    cfg["risk"]["max_position"] = 1.0
    cfg["risk"]["target_vol"] = 5.0
    res = Backtester(prices, fc, cfg, "2018-01-10").run()
    assert res.floor_triggers == 1 and res.halted_on is not None
    after = res.equity.loc[pd.Timestamp(res.halted_on) + pd.Timedelta(days=3):]
    assert (res.exposure.loc[after.index] == 0).all()          # flat forever after the halt
    assert after.nunique() == 1                                  # equity frozen in cash
