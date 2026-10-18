"""Risk engine limits added for the wide universe."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_agent.risk.engine import Order, PortfolioSnapshot, RiskEngine, RiskViolation

D = date(2026, 10, 20)
SECTORS = {"AAPL": "tech", "MSFT": "tech", "XLK": "tech", "XOM": "energy", "TLT": "bonds", "SPY": "broad"}
UNI = list(SECTORS)


def engine(cfg, tmp_path, **risk_over):
    r = dict(cfg["risk"], **risk_over)
    return RiskEngine(r, UNI, tmp_path / "rs.json", tmp_path / "HALT", sectors=SECTORS)


def snap(pos=None, eq=77.0):
    pos = pos or {}
    return PortfolioSnapshot(eq, eq - sum(pos.values()), pos, eq)


def test_sector_cap_scales_group(cfg, tmp_path):
    rk = engine(cfg, tmp_path, max_sector=0.40, max_position=0.30, target_vol=1.0)
    longs = {"AAPL": True, "MSFT": True, "XLK": True, "XOM": True}
    vols = {s: 0.2 for s in UNI}
    dec = rk.size_targets(longs, vols, snap(), D)
    tech = sum(dec.targets[s] for s in ("AAPL", "MSFT", "XLK"))
    assert tech == pytest.approx(0.40)
    assert dec.targets["XOM"] == pytest.approx(0.30)


def test_sector_cap_rejects_buy(cfg, tmp_path):
    rk = engine(cfg, tmp_path, max_sector=0.40, max_position=0.35)
    s = snap({"AAPL": 20.0, "MSFT": 10.0})
    with pytest.raises(RiskViolation, match="sector cap"):
        rk.validate_order(Order("XLK", "buy", 5.0), s, D)
    rk.validate_order(Order("XOM", "buy", 5.0), s, D)      # other sector is fine


def test_max_open_positions(cfg, tmp_path):
    rk = engine(cfg, tmp_path, max_open_positions=2, max_position=0.30)
    s = snap({"AAPL": 10.0, "XOM": 10.0})
    with pytest.raises(RiskViolation, match="max 2"):
        rk.validate_order(Order("TLT", "buy", 5.0), s, D)
    rk.validate_order(Order("AAPL", "buy", 5.0), s, D)     # adding to a holding is fine
    dec = rk.size_targets({"AAPL": True, "XOM": True, "TLT": True}, {u: 0.2 for u in UNI}, s, D)
    assert sum(1 for w in dec.targets.values() if w > 0) == 2
    assert dec.targets["TLT"] == 0.0                         # holdings kept first


def test_portfolio_vol_cap_scales_down_only(cfg, tmp_path):
    rk = engine(cfg, tmp_path, portfolio_vol_cap=0.10, max_position=0.5, target_vol=1.0)
    cov = pd.DataFrame(np.full((2, 2), 0.04), index=["AAPL", "XOM"], columns=["AAPL", "XOM"])  # vol 20%, corr 1
    dec = rk.size_targets({"AAPL": True, "XOM": True}, {"AAPL": 0.2, "XOM": 0.2}, snap(), D, cov=cov)
    w = np.array([dec.targets["AAPL"], dec.targets["XOM"]])
    assert np.sqrt(w @ cov.values @ w) == pytest.approx(0.10)
    low = cov * 0.01                                          # 2% vol: cap not binding, no scaling up
    dec2 = rk.size_targets({"AAPL": True, "XOM": True}, {"AAPL": 0.2, "XOM": 0.2}, snap(), D, cov=low)
    assert dec2.targets["AAPL"] == pytest.approx(0.49)        # max_position, then 98% gross cap


def test_portfolio_vol_cap_unknown_covariance_is_conservative(cfg, tmp_path):
    rk = engine(cfg, tmp_path, portfolio_vol_cap=0.10, max_position=0.5, target_vol=1.0)
    cov = pd.DataFrame([[0.04, np.nan], [np.nan, 0.04]], index=["AAPL", "XOM"], columns=["AAPL", "XOM"])
    dec = rk.size_targets({"AAPL": True, "XOM": True}, {"AAPL": 0.2, "XOM": 0.2}, snap(), D, cov=cov)
    assert sum(dec.targets.values()) == pytest.approx(0.5)    # treated as perfectly correlated


def test_intraday_block_and_cooldown_reject_buys(cfg, tmp_path):
    rk = engine(cfg, tmp_path)
    rk.block_buys_today(D, "test")
    with pytest.raises(RiskViolation, match="intraday monitor"):
        rk.validate_order(Order("SPY", "buy", 5.0), snap(), D)
    rk.validate_order(Order("SPY", "buy", 5.0), snap(), date(2026, 10, 21))   # next day ok
    dec = rk.size_targets({"SPY": True}, {"SPY": 0.2}, snap(), D)
    assert dec.buys_blocked and dec.targets["SPY"] == 0.0
    rk.start_cooldown(D, "test")
    with pytest.raises(RiskViolation, match="cooldown"):
        rk.validate_order(Order("SPY", "buy", 5.0), snap(), date(2026, 10, 22))
    # selling is always allowed
    rk.validate_order(Order("SPY", "sell", 5.0), snap({"SPY": 10.0}), D)
