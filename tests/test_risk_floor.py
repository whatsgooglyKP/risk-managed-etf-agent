from datetime import date

import pytest

from trading_agent.risk.engine import Order, PortfolioSnapshot, RiskEngine, RiskViolation

D = date(2026, 10, 19)


def snap(equity, cash=None, positions=None, last=None):
    positions = positions or {}
    cash = equity - sum(positions.values()) if cash is None else cash
    return PortfolioSnapshot(equity, cash, positions, last)


def test_floor_triggers_liquidation_and_kill_file(risk, tmp_path):
    s = snap(59.99, positions={"SPY": 40.0})
    dec = risk.size_targets({"SPY": True}, {"SPY": 0.15}, s, D)
    assert dec.halted and dec.liquidate
    assert all(w == 0 for w in dec.targets.values())
    assert (tmp_path / "HALT").exists()
    assert risk.state.floor_triggers == 1


def test_floor_exactly_at_60_halts(risk):
    assert risk.check_floor(snap(60.0)) is True


def test_above_floor_does_not_halt(risk):
    assert risk.check_floor(snap(60.01)) is False
    assert not risk.is_halted()


def test_halt_is_permanent_across_restarts(cfg, risk, tmp_path):
    risk.check_floor(snap(55.0))
    # new process, equity recovered (e.g. deposit) -> still halted because of kill file + state
    again = RiskEngine(cfg["risk"], ["SPY"], tmp_path / "risk_state.json", tmp_path / "HALT")
    dec = again.size_targets({"SPY": True}, {"SPY": 0.1}, snap(100.0), D)
    assert dec.halted and dec.targets["SPY"] == 0


def test_kill_file_alone_halts(cfg, tmp_path):
    (tmp_path / "HALT").write_text("manual stop")
    eng = RiskEngine(cfg["risk"], ["SPY"], None, tmp_path / "HALT")
    assert eng.is_halted()


def test_halted_rejects_buys_but_allows_sells(risk):
    risk.check_floor(snap(58.0, positions={"SPY": 30.0}))
    s = snap(58.0, positions={"SPY": 30.0})
    with pytest.raises(RiskViolation):
        risk.validate_order(Order("QQQ", "buy", 5.0), s, D)
    risk.validate_order(Order("SPY", "sell", 30.0, close_position=True), s, D, emergency=True)


def test_buy_rejected_at_or_below_floor_even_without_halt_state(cfg):
    eng = RiskEngine(cfg["risk"], ["SPY"])
    with pytest.raises(RiskViolation):
        eng.validate_order(Order("SPY", "buy", 5.0), snap(60.0), D)


def test_floor_must_be_below_capital():
    from trading_agent.config import load_config

    with pytest.raises(ValueError):
        load_config(overrides={"risk": {"hard_floor": 80.0}})
