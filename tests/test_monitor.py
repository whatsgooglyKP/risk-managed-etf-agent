"""Intraday risk monitor against a fake broker and a fake (mocked) Alpaca price feed."""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from trading_agent.config import load_config
from trading_agent.execution import monitor
from trading_agent.execution.ledger import Ledger
from trading_agent.execution.live import make_risk
from trading_agent.risk.engine import PortfolioSnapshot

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 20, 11, 30, tzinfo=NY).astimezone(timezone.utc)
TODAY = date(2026, 10, 20)


class FakeBroker:
    """Positions are held as share quantities at fixed broker marks (prices)."""

    def __init__(self, cash=17.0, qty=None, marks=None, last_equity=77.0, is_open=True, daytrades=0):
        self.cash = cash
        self.qty = dict(qty or {"SPY": 0.1, "QQQ": 0.1})
        self.marks = dict(marks or {"SPY": 300.0, "QQQ": 300.0})
        self.last_equity = last_equity
        self.is_open = is_open
        self.daytrades = daytrades
        self.submitted = []
        self.cancelled = 0
        self.session = TODAY

    def snapshot(self):
        pos = {s: q * self.marks[s] for s, q in self.qty.items() if q > 0}
        return PortfolioSnapshot(self.cash + sum(pos.values()), self.cash, pos, self.last_equity, self.daytrades)

    def position_quantities(self):
        return dict(self.qty)

    def cancel_open_orders(self):
        self.cancelled += 1
        return 0

    def clock(self):
        return SimpleNamespace(is_open=self.is_open, timestamp=NOW,
                               next_open=NOW + timedelta(hours=20), next_close=NOW + timedelta(hours=4))

    def submit(self, order):
        self.submitted.append(order)
        q = self.qty.pop(order.symbol)
        self.cash += q * self.marks[order.symbol]
        return {"status": "filled", "qty": q, "price": self.marks[order.symbol],
                "notional": q * self.marks[order.symbol], "client_order_id": f"x-{order.tag}"}


class FakeFeed:
    def __init__(self, prices=None, age_sec=5, fail=False):
        self.prices = prices or {}
        self.age = age_sec
        self.fail = fail
        self.calls = 0

    def latest(self, symbols):
        self.calls += 1
        if self.fail:
            raise ConnectionError("feed down")
        ts = NOW - timedelta(seconds=self.age)
        return {s: (self.prices[s], ts) for s in symbols if s in self.prices}


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"execution": {"state_dir": str(tmp_path / "state"),
                                                "log_dir": str(tmp_path / "logs")},
                                  "risk": {"kill_file": str(tmp_path / "state" / "HALT")}})


def run(cfg, broker, feed, **kw):
    risk = make_risk(cfg)
    if risk.state.peak_equity is None:
        risk.state.peak_equity = 77.0
    return monitor.check_once(cfg, broker, feed, risk, NOW, Ledger(cfg["execution"]["state_dir"]), **kw), risk


def test_ok_when_nothing_is_breached(cfg):
    b = FakeBroker()
    res, risk = run(cfg, b, FakeFeed({"SPY": 300.5, "QQQ": 299.0}))
    assert res.status == "ok" and not b.submitted
    assert res.equity == pytest.approx(76.95)                 # min(broker 77.00, live marks 76.95)
    assert not risk.buys_blocked_today(TODAY)


def test_live_prices_breach_floor_before_broker_marks_do(cfg, tmp_path):
    # broker still marks $77; live IEX trades say the holdings collapsed -> equity ~$47
    b = FakeBroker()
    res, risk = run(cfg, b, FakeFeed({"SPY": 150.0, "QQQ": 150.0}))
    assert res.status == "halted"
    assert res.equity == pytest.approx(47.0)
    assert {o.symbol for o in b.submitted} == {"SPY", "QQQ"} and all(o.side == "sell" for o in b.submitted)
    assert b.cancelled == 1                                   # open orders cancelled first
    assert (tmp_path / "state" / "HALT").exists() and risk.is_halted()
    assert (tmp_path / "state" / "monitor.csv").exists()


def test_floor_liquidation_overrides_pdt_when_budget_used(cfg):
    b = FakeBroker(daytrades=3)
    risk = make_risk(cfg)
    risk.state.peak_equity = 77.0
    risk.record_fill("SPY", "buy", TODAY)                     # bought this morning -> selling = day trade
    risk.save()
    res, risk = run(cfg, b, FakeFeed({"SPY": 150.0, "QQQ": 150.0}))
    assert res.status == "halted"
    assert {o.symbol for o in b.submitted} == {"SPY", "QQQ"}  # emergency: PDT guard overridden
    assert risk.day_trades_in_window(TODAY) == 1


def test_daily_loss_blocks_buys_but_does_not_sell_by_default(cfg):
    b = FakeBroker()                                          # last close $77
    res, risk = run(cfg, b, FakeFeed({"SPY": 285.0, "QQQ": 285.0}))   # -$3 = -3.9%
    assert res.status == "daily_loss" and not b.submitted
    assert risk.buys_blocked_today(TODAY)
    assert make_risk(cfg).buys_blocked_today(TODAY)          # persisted for the daily run


def test_daily_loss_flatten_respects_pdt(cfg):
    cfg["monitor"]["on_daily_loss"] = "flatten"
    risk = make_risk(cfg)
    risk.state.peak_equity = 77.0
    risk.record_fill("QQQ", "buy", TODAY)
    risk.save()
    b = FakeBroker()
    res, _ = run(cfg, b, FakeFeed({"SPY": 285.0, "QQQ": 285.0}))
    assert [o.symbol for o in b.submitted] == ["SPY"]         # QQQ bought today: kept (no day trade)
    assert any("PDT" in a for a in res.actions)


def test_drawdown_starts_cooldown_and_flattens(cfg):
    risk = make_risk(cfg)
    risk.state.peak_equity = 85.0                             # 77 / 85 = -9.4% ... push lower with feed
    risk.save()
    b = FakeBroker(last_equity=74.0)
    res, risk = run(cfg, b, FakeFeed({"SPY": 280.0, "QQQ": 280.0}))   # equity 73 -> dd 14%
    assert res.status == "drawdown"
    assert risk.state.cooldown_until is not None and risk.buys_blocked_today(TODAY)
    assert {o.symbol for o in b.submitted} == {"SPY", "QQQ"}


def test_stale_or_missing_quotes_fall_back_to_broker_marks(cfg):
    b = FakeBroker()
    res, _ = run(cfg, b, FakeFeed({"SPY": 1.0, "QQQ": 1.0}, age_sec=3600))   # hour-old quotes
    assert res.status == "ok" and set(res.stale_symbols) == {"SPY", "QQQ"}
    res2, _ = run(cfg, b, FakeFeed(fail=True))
    assert res2.status == "ok" and res2.equity == pytest.approx(77.0)


def test_conservative_equity_uses_the_lower_value(cfg):
    b = FakeBroker()
    res, _ = run(cfg, b, FakeFeed({"SPY": 400.0, "QQQ": 400.0}))   # feed higher than broker
    assert res.equity == pytest.approx(77.0) and res.feed_equity == pytest.approx(97.0)


def test_dry_run_places_no_orders(cfg):
    b = FakeBroker()
    res, _ = run(cfg, b, FakeFeed({"SPY": 150.0, "QQQ": 150.0}), dry_run=True)
    assert res.status == "halted" and not b.submitted
    assert any("DRY RUN" in a for a in res.actions)


def test_run_monitor_market_closed_does_nothing(cfg, monkeypatch):
    import trading_agent.execution.alpaca_broker as ab

    b = FakeBroker(is_open=False)
    feed = FakeFeed({"SPY": 1.0, "QQQ": 1.0})
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.setattr(ab, "AlpacaBroker", lambda *a, **k: b)
    monkeypatch.setattr(monitor, "AlpacaPriceFeed", lambda *a, **k: feed)
    assert monitor.run_monitor(cfg) == 0
    assert not b.submitted and feed.calls == 0


def test_run_monitor_loop_checks_repeatedly(cfg, monkeypatch):
    import trading_agent.execution.alpaca_broker as ab

    b = FakeBroker()
    feed = FakeFeed({"SPY": 300.0, "QQQ": 300.0})
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.setattr(ab, "AlpacaBroker", lambda *a, **k: b)
    monkeypatch.setattr(monitor, "AlpacaPriceFeed", lambda *a, **k: feed)
    sleeps = []
    assert monitor.run_monitor(cfg, loop=True, max_checks=3, sleep=sleeps.append) == 0
    assert feed.calls == 3 and len(sleeps) == 2 and not b.submitted


def test_monitor_orders_get_their_own_client_ids():
    from trading_agent.execution.alpaca_broker import AlpacaBroker
    from trading_agent.risk.engine import Order

    ab = AlpacaBroker.__new__(AlpacaBroker)
    ab.session = TODAY
    daily = ab.client_order_id(Order("SPY", "sell", 5.0))
    mon = ab.client_order_id(Order("SPY", "sell", 5.0, tag="mon1130"))
    assert daily != mon and mon.endswith("-mon1130")
