from datetime import date
from types import SimpleNamespace

import pytest

from trading_agent.execution.guarded import GuardedBroker
from trading_agent.execution.orders import plan_orders
from trading_agent.execution.sim import SimBroker
from trading_agent.risk.engine import Order, PortfolioSnapshot

D = date(2026, 10, 19)


def test_plan_sells_first_and_funds_buys_from_cash():
    snap = PortfolioSnapshot(77.0, 2.0, {"SPY": 40.0, "TLT": 35.0})
    orders = plan_orders({"SPY": 0.0, "TLT": 0.3, "GLD": 0.3}, snap, 0.05, 1.0)
    assert [o.side for o in orders][:1] == ["sell"]
    assert orders[0].symbol == "SPY" and orders[0].close_position
    buys = [o for o in orders if o.side == "buy"]
    assert sum(o.notional for o in buys) <= 2.0 + 40.0 + 35.0 - 0.3 * 77 + 1e-6


def test_rebalance_band_skips_tiny_changes():
    snap = PortfolioSnapshot(77.0, 47.0, {"SPY": 30.0})
    assert plan_orders({"SPY": 0.40}, snap, 0.05, 1.0) == []   # 30.8 vs 30 -> skip


def test_guarded_broker_rejects_and_never_submits():
    sim = SimBroker(77.0, 0.0)
    sim.set_prices({"SPY": 100.0, "QQQ": 100.0})
    from trading_agent.config import load_config
    from trading_agent.risk.engine import RiskEngine

    cfg = load_config()
    guard = GuardedBroker(sim, RiskEngine(cfg["risk"], ["SPY", "QQQ"]))
    res = guard.execute([Order("SPY", "buy", 70.0)], D)     # > 35% max position
    assert res[0]["status"] == "rejected"
    assert sim.cash == 77.0 and not sim.trades
    res = guard.execute([Order("SPY", "buy", 20.0)], D)
    assert res[0]["status"] == "filled" and sim.qty["SPY"] == pytest.approx(0.2)


class FakeOrder(SimpleNamespace):
    pass


class FakeClient:
    """Minimal stand-in for alpaca TradingClient to test idempotency + status polling."""

    def __init__(self, existing=None):
        self.submitted = []
        self.existing = existing

    def get_order_by_client_id(self, cid):
        if self.existing and self.existing.client_order_id == cid:
            return self.existing
        raise Exception("404 order not found")

    def get_orders(self, req):
        return []

    def submit_order(self, req):
        o = FakeOrder(id="abc", symbol=req.symbol, status="filled", filled_qty="0.05",
                      filled_avg_price="600.0", side="buy", client_order_id=req.client_order_id)
        self.submitted.append(req)
        return o

    def get_order_by_id(self, oid):
        return self.existing


def make_broker(client):
    from trading_agent.execution.alpaca_broker import AlpacaBroker

    b = AlpacaBroker.__new__(AlpacaBroker)
    b.client, b.session, b.timeout, b.paper = client, D, 5, True
    from trading_agent.execution.retry import with_retries

    b._retry = with_retries(0)
    return b


def test_alpaca_submit_fills_and_reports():
    client = FakeClient()
    res = make_broker(client).submit(Order("SPY", "buy", 30.0))
    assert res["status"] == "filled" and res["qty"] == pytest.approx(0.05)
    assert client.submitted[0].client_order_id == "ta-2026-10-19-SPY-buy"


def test_alpaca_rerun_same_day_does_not_resubmit():
    existing = FakeOrder(id="x1", symbol="SPY", status="filled", filled_qty="0.05",
                         filled_avg_price="600.0", side="buy", client_order_id="ta-2026-10-19-SPY-buy")
    client = FakeClient(existing)
    res = make_broker(client).submit(Order("SPY", "buy", 30.0))
    assert client.submitted == [] and res["status"] == "filled"


def test_unfilled_order_is_cancelled_after_timeout(monkeypatch):
    pending = FakeOrder(id="p1", symbol="SPY", status="new", filled_qty="0", filled_avg_price=None,
                        side="buy", client_order_id="ta-2026-10-19-SPY-buy")
    client = FakeClient(pending)
    cancelled = []

    def cancel(oid):
        cancelled.append(oid)
        pending.status = "canceled"

    client.cancel_order_by_id = cancel
    b = make_broker(client)
    b.timeout = 0
    monkeypatch.setattr("time.sleep", lambda s: None)
    res = b.submit(Order("SPY", "buy", 30.0))
    assert cancelled == ["p1"] and res["status"] == "canceled"


def test_retry_only_on_transient():
    from trading_agent.execution.retry import with_retries

    calls = {"n": 0}

    class E(Exception):
        status_code = 503

    @with_retries(2, base_delay=0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise E()
        return "ok"

    assert flaky() == "ok" and calls["n"] == 3

    @with_retries(3, base_delay=0)
    def bad():
        calls["n"] += 100
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        bad()


def test_live_requires_flag_and_config(cfg):
    from trading_agent.execution.live import resolve_mode

    assert resolve_mode(cfg, False) == "paper"
    with pytest.raises(SystemExit):
        resolve_mode(cfg, True)                       # config still says paper
    cfg["execution"]["mode"] = "live"
    assert resolve_mode(cfg, False) == "paper"        # flag missing -> paper
    assert resolve_mode(cfg, True) == "live"
