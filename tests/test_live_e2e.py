"""End-to-end daily run against a fake Alpaca broker (no network, no keys)."""
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from trading_agent.config import load_config
from trading_agent.data.loader import PriceData
from trading_agent.execution import live
from trading_agent.execution.ledger import Ledger
from trading_agent.risk.engine import PortfolioSnapshot

NY = ZoneInfo("America/New_York")
UNI = ["SPY", "QQQ", "TLT", "GLD", "IWM"]
SESSION = date(2026, 10, 20)   # a Tuesday
PREV = date(2026, 10, 19)


class FakeBroker:
    instances = []

    def __init__(self, key, secret, paper=True, session=None, order_timeout_sec=90, max_retries=4,
                 equity=77.0, is_open=True, positions=None):
        self.paper, self.session = paper, session
        self.cash = equity - sum((positions or {}).values())
        self.positions = dict(positions or {})
        self.last_equity = equity
        self.is_open = is_open
        self.submitted = []
        FakeBroker.instances.append(self)

    def clock(self):
        ts = datetime(2026, 10, 20, 10, 0, tzinfo=NY)
        return SimpleNamespace(is_open=self.is_open, timestamp=ts,
                               next_open=ts + timedelta(days=1), next_close=ts.replace(hour=16))

    def account(self):
        return SimpleNamespace(trading_blocked=False, account_blocked=False)

    def previous_session(self, today):
        return PREV

    def peak_equity(self):
        return None

    def snapshot(self):
        eq = self.cash + sum(self.positions.values())
        return PortfolioSnapshot(eq, self.cash, dict(self.positions), self.last_equity, 0)

    def submit(self, order):
        self.submitted.append(order)
        if order.side == "buy":
            self.cash -= order.notional
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.notional
        else:
            amt = self.positions.pop(order.symbol) if order.close_position else order.notional
            if not order.close_position:
                self.positions[order.symbol] -= amt
            self.cash += amt
        return {"status": "filled", "qty": 0.1, "price": 100.0, "notional": order.notional,
                "client_order_id": f"ta-{SESSION}-{order.symbol}-{order.side}"}


def prices(last=PREV, n=400):
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=n)
    rng = np.random.default_rng(0)
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, (n, 5)), axis=0)), index=idx, columns=UNI)
    return PriceData(c.shift(1).bfill(), c)


def fc_table(p_up=0.7):
    return pd.DataFrame({"p_up": p_up, "mu": 0.001, "agreement": 3, "confidence": 0.4}, index=UNI)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = load_config(overrides={"execution": {"state_dir": str(tmp_path / "state"),
                                               "log_dir": str(tmp_path / "logs")},
                                 "risk": {"kill_file": str(tmp_path / "state" / "HALT")}})
    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.setattr(live, "fetch_prices", lambda cfg, k, s: prices())
    monkeypatch.setattr(live, "latest_forecasts", lambda close, f, b: fc_table())
    monkeypatch.setattr(live, "check_freshness", lambda p, d, t: None)
    FakeBroker.instances = []
    return cfg, tmp_path


def use_broker(monkeypatch, **kw):
    import trading_agent.execution.alpaca_broker as ab

    monkeypatch.setattr(ab, "AlpacaBroker", lambda *a, **k: FakeBroker(*a, **{**k, **kw}))


def test_daily_run_places_orders_and_is_idempotent(env, monkeypatch):
    cfg, tmp = env
    use_broker(monkeypatch)
    assert live.run_daily(cfg) == 0
    b = FakeBroker.instances[-1]
    assert b.submitted and all(o.side == "buy" for o in b.submitted)
    assert sum(o.notional for o in b.submitted) <= 77.0
    led = Ledger(tmp / "state")
    assert led.run_status(str(SESSION), "paper") == "completed"
    assert (tmp / "state" / "orders.csv").exists()
    # second run the same day: nothing happens
    assert live.run_daily(cfg) == 0
    assert FakeBroker.instances[-1].submitted == []


def test_floor_breach_liquidates_and_halts(env, monkeypatch):
    cfg, tmp = env
    use_broker(monkeypatch, equity=59.0, positions={"SPY": 30.0, "QQQ": 20.0})
    assert live.run_daily(cfg) == 0
    b = FakeBroker.instances[-1]
    assert {o.symbol for o in b.submitted} == {"SPY", "QQQ"}
    assert all(o.side == "sell" for o in b.submitted)
    assert (tmp / "state" / "HALT").exists()
    # later run (even with money back): still halted, no buys
    use_broker(monkeypatch, equity=100.0)
    assert live.run_daily(cfg, force=True) == 0
    assert all(o.side == "sell" for o in FakeBroker.instances[-1].submitted)


def test_stale_data_fails_safe_without_orders(env, monkeypatch):
    cfg, _ = env
    use_broker(monkeypatch)
    monkeypatch.setattr(live, "fetch_prices", lambda cfg, k, s: prices(last=date(2026, 10, 15)))
    assert live.run_daily(cfg) == 1
    assert FakeBroker.instances[-1].submitted == []


def test_market_closed_does_nothing(env, monkeypatch):
    cfg, _ = env
    use_broker(monkeypatch, is_open=False)
    assert live.run_daily(cfg) == 0
    assert FakeBroker.instances[-1].submitted == []


def test_dry_run_places_no_orders(env, monkeypatch):
    cfg, _ = env
    use_broker(monkeypatch)
    assert live.run_daily(cfg, dry_run=True) == 0
    assert FakeBroker.instances[-1].submitted == []


def test_low_confidence_holds_cash(env, monkeypatch):
    cfg, _ = env
    use_broker(monkeypatch)
    monkeypatch.setattr(live, "latest_forecasts", lambda close, f, b: fc_table(p_up=0.51))
    assert live.run_daily(cfg) == 0
    assert FakeBroker.instances[-1].submitted == []


def test_missing_keys_fails_safe(env, monkeypatch):
    cfg, _ = env
    monkeypatch.delenv("ALPACA_API_KEY")
    monkeypatch.setattr("trading_agent.config.load_secrets", lambda: (None, None))
    monkeypatch.setattr(live, "load_secrets", lambda: (None, None))
    assert live.run_daily(cfg) == 1
