"""Intraday risk monitor: re-checks the account during market hours using live prices.

The daily run decides positions once a day. Between runs, this monitor (no forecasting,
a few seconds per check) marks every holding with Alpaca's free real-time IEX prices and
re-applies the risk limits:

  1. HARD FLOOR / kill file  -> cancel open orders, sell everything, permanent halt.
     This is an EMERGENCY liquidation: it is allowed even if it creates a day trade and
     the PDT budget is used up (losing the account is worse than a PDT flag; this is the
     only case where the PDT guard is overridden, and it is logged).
  2. DRAWDOWN >= max_drawdown_hard -> start the same cooldown the daily run uses and (with
     monitor.on_drawdown: flatten) sell positions. Not an emergency: positions bought
     today are NOT sold (that would be a day trade); the next daily run sells them.
  3. DAILY LOSS >= daily_loss_cap vs the previous close -> block new buying for the rest
     of the day (state survives into the daily run). With monitor.on_daily_loss: flatten
     it also sells positions, under the same PDT rule as (2).

Equity is computed conservatively: min(broker-reported equity, equity re-marked with the
latest IEX trades). A quote older than monitor.max_price_age_sec is ignored and the
broker's own mark is used instead. If neither broker nor feed can be read, the check
fails (exit code 1) without trading. Run it with `trading-agent monitor` (one check, e.g.
from cron or GitHub Actions every 15-30 min) or `trading-agent monitor --loop` (local).
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ..risk.engine import Order, PortfolioSnapshot, RiskEngine
from .guarded import GuardedBroker

log = logging.getLogger("trading_agent.monitor")
NY = ZoneInfo("America/New_York")


# ----------------------------------------------------------------------------- price feed
class AlpacaPriceFeed:
    """Latest trade per symbol from Alpaca's free IEX feed (REST polling, no websocket)."""

    def __init__(self, api_key: str, secret_key: str):
        from alpaca.data.historical import StockHistoricalDataClient

        self.client = StockHistoricalDataClient(api_key, secret_key)

    def latest(self, symbols: list[str]) -> dict[str, tuple[float, datetime]]:
        if not symbols:
            return {}
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestTradeRequest

        res = self.client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX))
        return {s: (float(t.price), t.timestamp) for s, t in res.items() if t is not None}


# ----------------------------------------------------------------------------- result
@dataclass
class MonitorResult:
    status: str                       # ok | halted | drawdown | daily_loss | closed | error
    ts: str = ""
    broker_equity: float | None = None
    feed_equity: float | None = None
    equity: float | None = None       # conservative value used for the checks
    last_equity: float | None = None
    peak_equity: float | None = None
    daily_change: float | None = None
    drawdown: float | None = None
    stale_symbols: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)


def mark_to_market(snap: PortfolioSnapshot, qty: dict[str, float],
                   quotes: dict[str, tuple[float, datetime]], now: datetime,
                   max_age_sec: float) -> tuple[float, dict[str, float], list[str]]:
    """Re-mark holdings with fresh quotes. Returns (equity, market values, stale symbols)."""
    values, stale = {}, []
    for sym, mv in snap.positions.items():
        q = qty.get(sym)
        quote = quotes.get(sym)
        fresh = False
        if q and quote is not None:
            px, ts = quote
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            fresh = px > 0 and (now - ts).total_seconds() <= max_age_sec
        if fresh:
            values[sym] = q * quote[0]
        else:
            values[sym] = mv
            stale.append(sym)
    equity = snap.equity - sum(snap.positions.values()) + sum(values.values())
    return equity, values, stale


def _append_csv(path: Path, res: MonitorResult) -> None:
    row = asdict(res)
    row["stale_symbols"] = ";".join(res.stale_symbols)
    row["actions"] = " | ".join(res.actions)
    row["orders"] = len(res.orders)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------- core check
def check_once(cfg, broker, feed, risk: RiskEngine, now: datetime | None = None,
               ledger=None, mode: str = "paper", dry_run: bool = False) -> MonitorResult:
    """One intraday risk check. Pure orchestration over injectable broker/feed/risk."""
    mcfg = cfg.get("monitor") or {}
    rcfg = cfg["risk"]
    now = now or datetime.now(timezone.utc)
    today: date = now.astimezone(NY).date()
    res = MonitorResult(status="ok", ts=now.astimezone(NY).isoformat(timespec="seconds"))
    guard = GuardedBroker(broker, risk)

    snap = broker.snapshot()
    res.broker_equity = round(snap.equity, 4)
    qty = broker.position_quantities() if snap.positions else {}
    quotes = {}
    if snap.positions:
        try:
            quotes = feed.latest(sorted(snap.positions))
        except Exception as exc:  # noqa: BLE001 - feed outage: fall back to broker marks
            log.warning("Price feed failed (%s); using broker marks only", exc)
    feed_eq, values, stale = mark_to_market(snap, qty, quotes, now,
                                            float(mcfg.get("max_price_age_sec", 900)))
    res.feed_equity = round(feed_eq, 4)
    res.stale_symbols = stale
    equity = min(snap.equity, feed_eq)
    msnap = PortfolioSnapshot(equity, snap.cash, values, snap.last_equity, snap.broker_daytrade_count)
    res.equity = round(equity, 4)
    res.last_equity = snap.last_equity
    peak = max(risk.state.peak_equity or equity, equity)
    res.peak_equity = round(peak, 4)
    res.drawdown = 1 - equity / peak if peak > 0 else 0.0
    res.daily_change = (equity / snap.last_equity - 1) if snap.last_equity else None

    def sell_all(emergency: bool, why: str) -> None:
        orders = []
        for sym in sorted(msnap.positions):
            if not emergency and risk.bought_today(sym, today):
                res.actions.append(f"kept {sym}: bought today, selling now would be a day trade (PDT)")
                continue
            orders.append(Order(sym, "sell", msnap.positions[sym], close_position=True,
                                reason=why, tag=f"mon{now.astimezone(NY):%H%M}"))
        if not orders:
            return
        if dry_run:
            res.actions.append(f"DRY RUN: would sell {[o.symbol for o in orders]}")
            return
        if emergency and hasattr(broker, "cancel_open_orders"):
            try:
                broker.cancel_open_orders()
            except Exception as exc:  # noqa: BLE001
                log.error("Could not cancel open orders: %s", exc)
        session = str(today)
        out = guard.execute(orders, today, emergency=emergency,
                            on_result=(lambda o, r: ledger.record_order(session, mode, o, r)) if ledger else None)
        res.orders.extend(out)
        res.actions.append(f"sold {[o.symbol for o, r in zip(orders, out) if r.get('status') == 'filled']}")

    # 1. hard floor / kill file (emergency, PDT override allowed)
    if risk.is_halted() or risk.check_floor(msnap):
        res.status = "halted"
        res.actions.append(f"HALT: {risk.state.halt_reason}")
        if msnap.positions:
            log.critical("Intraday monitor: %s -> emergency liquidation", risk.state.halt_reason)
            sell_all(emergency=True, why="floor liquidation (intraday monitor)")
        return _finish(cfg, res)

    # 2. drawdown hard limit
    if res.drawdown >= rcfg["max_drawdown_hard"]:
        res.status = "drawdown"
        risk.start_cooldown(today, f"intraday drawdown {res.drawdown:.1%}")
        risk.block_buys_today(today, "intraday drawdown limit")
        res.actions.append(f"drawdown {res.drawdown:.1%} >= {rcfg['max_drawdown_hard']:.0%}: cooldown "
                           f"until {risk.state.cooldown_until}, buys blocked")
        if mcfg.get("on_drawdown", "flatten") == "flatten" and msnap.positions:
            sell_all(emergency=False, why="drawdown limit (intraday monitor)")
        return _finish(cfg, res)

    # 3. daily loss cap
    if res.daily_change is not None and res.daily_change <= -rcfg["daily_loss_cap"]:
        res.status = "daily_loss"
        risk.block_buys_today(today, f"intraday loss {res.daily_change:.1%}")
        res.actions.append(f"down {res.daily_change:.1%} today >= cap {rcfg['daily_loss_cap']:.0%}: "
                           "no new buying today")
        if mcfg.get("on_daily_loss", "block_buys") == "flatten" and msnap.positions:
            sell_all(emergency=False, why="daily loss cap (intraday monitor)")
        return _finish(cfg, res)

    risk.save()
    return _finish(cfg, res)


def _finish(cfg, res: MonitorResult) -> MonitorResult:
    lvl = logging.INFO if res.status == "ok" else logging.WARNING
    log.log(lvl, "Monitor %s: equity $%.2f (broker $%.2f, live marks $%.2f), day %s, drawdown %.1f%%%s",
            res.status.upper(), res.equity, res.broker_equity, res.feed_equity,
            f"{res.daily_change:+.2%}" if res.daily_change is not None else "n/a",
            100 * (res.drawdown or 0), f", stale marks: {res.stale_symbols}" if res.stale_symbols else "")
    for a in res.actions:
        log.warning("Monitor action: %s", a)
    try:
        from ..config import resolve

        _append_csv(resolve(cfg["execution"]["state_dir"]) / "monitor.csv", res)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write monitor.csv: %s", exc)
    return res


# ----------------------------------------------------------------------------- entry points
def run_monitor(cfg, live: bool = False, loop: bool = False, dry_run: bool = False,
                max_checks: int | None = None, sleep=time.sleep) -> int:
    """CLI entry point. Exit code 0 = ok / nothing to do, 1 = a check failed (no trades made)."""
    from ..config import load_secrets, resolve
    from .alpaca_broker import AlpacaBroker
    from .ledger import Ledger
    from .live import make_risk, resolve_mode
    from .lock import state_lock

    mode = resolve_mode(cfg, live)
    key, secret = load_secrets()
    if not key or not secret:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY not set (see .env.example).")
        return 1
    interval = int((cfg.get("monitor") or {}).get("interval_sec", 60))
    ledger = Ledger(resolve(cfg["execution"]["state_dir"]))
    rc, n = 0, 0
    try:
        broker = AlpacaBroker(key, secret, paper=(mode == "paper"))
        feed = AlpacaPriceFeed(key, secret)
    except Exception as exc:  # noqa: BLE001
        log.error("Monitor could not connect: %s", exc)
        return 1
    while True:
        n += 1
        try:
            clock = broker.clock()
            if not clock.is_open:
                log.info("Market closed; monitor has nothing to do (next open %s ET).",
                         clock.next_open.astimezone(NY).strftime("%a %Y-%m-%d %H:%M"))
                if not loop:
                    return 0
                wait = (clock.next_open - clock.timestamp).total_seconds()
                if wait > 2 * 3600:          # local loop: exit instead of idling overnight
                    log.info("Next open is more than 2 hours away; exiting loop.")
                    return rc
                sleep(max(interval, min(wait, 3600)))
                continue
            broker.session = clock.timestamp.astimezone(NY).date()
            with state_lock(resolve(cfg["execution"]["state_dir"])):
                risk = make_risk(cfg)          # reload state each time (daily run may have changed it)
                res = check_once(cfg, broker, feed, risk, clock.timestamp, ledger, mode, dry_run)
            if res.status == "halted" and not broker.snapshot().positions:
                if loop:
                    log.critical("Halted and flat; stopping the monitor loop.")
                    return 0
        except Exception as exc:  # noqa: BLE001 - fail safe, try again next interval
            log.error("Monitor check failed (no trades made): %s", exc)
            rc = 1
            if not loop:
                return 1
        if not loop or (max_checks is not None and n >= max_checks):
            return rc
        sleep(interval)
