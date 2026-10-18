from datetime import date

import pytest

from trading_agent.risk.engine import Order, PortfolioSnapshot, RiskViolation

D = date(2026, 10, 19)
ALL_LONG = {s: True for s in ["SPY", "QQQ", "TLT", "GLD", "IWM"]}


def snap(equity=77.0, positions=None, last=None, cash=None):
    positions = positions or {}
    return PortfolioSnapshot(equity, equity - sum(positions.values()) if cash is None else cash, positions, last)


def test_vol_targeting(risk, cfg):
    dec = risk.size_targets({"SPY": True}, {"SPY": 0.40}, snap(), D)
    assert dec.targets["SPY"] == pytest.approx(cfg["risk"]["target_vol"] / 0.40)


def test_higher_vol_gets_smaller_weight(risk):
    dec = risk.size_targets({"SPY": True, "QQQ": True}, {"SPY": 0.30, "QQQ": 0.60}, snap(), D)
    assert dec.targets["QQQ"] < dec.targets["SPY"]


def test_max_position_cap(risk, cfg):
    dec = risk.size_targets({"TLT": True}, {"TLT": 0.02}, snap(), D)   # tiny vol -> capped
    assert dec.targets["TLT"] == pytest.approx(cfg["risk"]["max_position"])


def test_gross_exposure_never_exceeds_cash(risk, cfg):
    dec = risk.size_targets(ALL_LONG, {s: 0.05 for s in ALL_LONG}, snap(), D)
    cap = cfg["risk"]["max_gross_exposure"] - cfg["risk"]["cash_buffer"]
    assert sum(dec.targets.values()) <= cap + 1e-9
    assert all(w >= 0 for w in dec.targets.values())   # no shorts


def test_flat_signal_means_zero(risk):
    dec = risk.size_targets({"SPY": False}, {"SPY": 0.15}, snap(), D)
    assert dec.targets["SPY"] == 0


def test_missing_vol_means_flat(risk):
    dec = risk.size_targets({"SPY": True}, {"SPY": float("nan")}, snap(), D)
    assert dec.targets["SPY"] == 0


def test_rl_scale_can_only_reduce(risk):
    full = risk.size_targets({"SPY": True}, {"SPY": 0.4}, snap(), D).targets["SPY"]
    half = risk.size_targets({"SPY": True}, {"SPY": 0.4}, snap(), D, rl_scale=0.5).targets["SPY"]
    over = risk.size_targets({"SPY": True}, {"SPY": 0.4}, snap(), D, rl_scale=5.0).targets["SPY"]
    assert half == pytest.approx(full / 2)
    assert over == pytest.approx(full)


def test_soft_drawdown_halves_size(risk):
    risk.size_targets({"SPY": True}, {"SPY": 0.4}, snap(100.0), D)        # peak 100
    dec = risk.size_targets({"SPY": True}, {"SPY": 0.4}, snap(91.0), D)   # 9% dd
    assert dec.exposure_mult == 0.5


def test_hard_drawdown_goes_flat_then_cools_down(risk):
    risk.size_targets({"SPY": True}, {"SPY": 0.4}, snap(100.0), D)
    dec = risk.size_targets({"SPY": True}, {"SPY": 0.4}, snap(87.0), D)   # 13% dd
    assert dec.targets["SPY"] == 0 and risk.state.cooldown_until
    later = risk.size_targets({"SPY": True}, {"SPY": 0.4}, snap(87.0), date(2026, 10, 22))
    assert later.targets["SPY"] == 0
    resumed = risk.size_targets({"SPY": True}, {"SPY": 0.4}, snap(87.0), date(2026, 11, 3))
    assert 0 < resumed.targets["SPY"] and resumed.exposure_mult == 0.5


def test_daily_loss_cap_blocks_new_buying(risk):
    s = snap(74.0, positions={"SPY": 10.0}, last=77.0)   # -3.9% today
    dec = risk.size_targets({"SPY": True, "QQQ": True}, {"SPY": 0.1, "QQQ": 0.1}, s, D)
    assert dec.buys_blocked
    assert dec.targets["QQQ"] == 0
    assert dec.targets["SPY"] <= 10.0 / 74.0 + 1e-9
    with pytest.raises(RiskViolation, match="daily loss"):
        risk.validate_order(Order("QQQ", "buy", 5.0), s, D)


def test_cash_only_buy_limit(risk):
    with pytest.raises(RiskViolation, match="cash"):
        risk.validate_order(Order("SPY", "buy", 20.0), snap(77.0, positions={"QQQ": 25.0}, cash=10.0), D)


def test_max_position_enforced_per_order(risk):
    with pytest.raises(RiskViolation, match="max position"):
        risk.validate_order(Order("SPY", "buy", 40.0), snap(), D)


def test_no_shorting(risk):
    with pytest.raises(RiskViolation, match="shorting"):
        risk.validate_order(Order("SPY", "sell", 5.0), snap(), D)


def test_cannot_oversell(risk):
    with pytest.raises(RiskViolation, match="exceeds holding"):
        risk.validate_order(Order("SPY", "sell", 30.0), snap(positions={"SPY": 10.0}), D)


def test_symbol_outside_universe_rejected(risk):
    with pytest.raises(RiskViolation, match="universe"):
        risk.validate_order(Order("AAPL", "buy", 5.0), snap(), D)


def test_leveraged_etf_rejected(risk):
    with pytest.raises(RiskViolation, match="leveraged"):
        risk.validate_order(Order("TQQQ", "buy", 5.0), snap(), D)


def test_leveraged_etf_not_allowed_in_config():
    from trading_agent.config import load_config

    with pytest.raises(ValueError):
        load_config(overrides={"universe": ["SPY", "TQQQ"]})


def test_margin_not_allowed_in_config():
    from trading_agent.config import load_config

    with pytest.raises(ValueError):
        load_config(overrides={"risk": {"max_gross_exposure": 1.5}})


def test_min_order_size(risk):
    with pytest.raises(RiskViolation, match="minimum"):
        risk.validate_order(Order("SPY", "buy", 0.5), snap(), D)


def test_state_persists(cfg, risk, tmp_path):
    from trading_agent.risk.engine import RiskEngine

    risk.size_targets({"SPY": True}, {"SPY": 0.2}, snap(90.0), D)
    again = RiskEngine(cfg["risk"], ["SPY"], tmp_path / "risk_state.json", tmp_path / "HALT")
    assert again.state.peak_equity == 90.0
