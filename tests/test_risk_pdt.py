from datetime import date

import pytest

from trading_agent.risk.engine import Order, PortfolioSnapshot, RiskViolation

MON, TUE, WED, THU, FRI = (date(2026, 10, d) for d in (19, 20, 21, 22, 23))
NEXT_MON, NEXT_TUE = date(2026, 10, 26), date(2026, 10, 27)


def s(positions=None, equity=77.0, broker_dt=None):
    positions = positions or {"SPY": 20.0}
    return PortfolioSnapshot(equity, equity - sum(positions.values()), positions, None, broker_dt)


def day_trade(risk, sym, day):
    risk.record_fill(sym, "buy", day)
    risk.record_fill(sym, "sell", day)


def test_round_trip_same_day_counts_as_day_trade(risk):
    day_trade(risk, "SPY", MON)
    assert risk.day_trades_in_window(MON) == 1


def test_overnight_hold_is_not_a_day_trade(risk):
    risk.record_fill("SPY", "buy", MON)
    risk.record_fill("SPY", "sell", TUE)
    assert risk.day_trades_in_window(TUE) == 0


def test_fourth_day_trade_blocked(risk):
    day_trade(risk, "SPY", MON)
    day_trade(risk, "QQQ", TUE)
    day_trade(risk, "GLD", WED)
    risk.record_fill("IWM", "buy", THU)
    with pytest.raises(RiskViolation, match="PDT"):
        risk.validate_order(Order("IWM", "sell", 10.0), s({"IWM": 20.0}), THU)


def test_selling_overnight_position_allowed_even_at_limit(risk):
    day_trade(risk, "SPY", MON)
    day_trade(risk, "QQQ", TUE)
    day_trade(risk, "GLD", WED)
    risk.validate_order(Order("TLT", "sell", 10.0), s({"TLT": 20.0}), THU)  # bought earlier: fine


def test_window_rolls_after_five_business_days(risk):
    day_trade(risk, "SPY", MON)
    day_trade(risk, "QQQ", TUE)
    day_trade(risk, "GLD", WED)
    assert risk.day_trades_in_window(FRI) == 3
    assert risk.day_trades_in_window(NEXT_MON) == 2   # Monday dropped out
    assert risk.day_trades_in_window(NEXT_TUE) == 1
    risk.record_fill("IWM", "buy", NEXT_TUE)
    risk.validate_order(Order("IWM", "sell", 10.0), s({"IWM": 20.0}), NEXT_TUE)


def test_broker_count_is_respected(risk):
    risk.record_fill("SPY", "buy", MON)
    with pytest.raises(RiskViolation, match="PDT"):
        risk.validate_order(Order("SPY", "sell", 10.0), s(broker_dt=3), MON)


def test_pdt_not_applied_above_25k(risk):
    for d, sym in ((MON, "SPY"), (TUE, "QQQ"), (WED, "GLD")):
        day_trade(risk, sym, d)
    risk.record_fill("IWM", "buy", THU)
    risk.validate_order(Order("IWM", "sell", 1000.0), s({"IWM": 5000.0}, equity=30000.0), THU)


def test_emergency_liquidation_overrides_pdt(risk):
    for d, sym in ((MON, "SPY"), (TUE, "QQQ"), (WED, "GLD")):
        day_trade(risk, sym, d)
    risk.record_fill("IWM", "buy", THU)
    risk.validate_order(Order("IWM", "sell", 20.0, close_position=True), s({"IWM": 20.0}), THU, emergency=True)
