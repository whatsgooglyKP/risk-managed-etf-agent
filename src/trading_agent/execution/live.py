"""Daily live/paper run: one command, idempotent, fails safe (any error = no new trades).

Intended schedule: once per trading day, >= 10 minutes after the 9:30 ET open.
  1. Connect, read the market clock (do nothing if closed / too early / too late).
  2. Skip if today's session already completed (idempotent re-runs).
  3. Hard-floor / kill-file check -> emergency liquidation + permanent halt.
  4. Fresh daily data through the PREVIOUS close (yfinance, Alpaca IEX fallback),
     verified to be exactly the last completed session.
  5. Ensemble forecasts -> long/flat signals -> RiskEngine targets (-> optional RL scale).
  6. Orders (sells first, then buys) through GuardedBroker; each polled until final.
  7. Ledger (SQLite + CSV) records signals, orders, equity, run status.
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import load_secrets, resolve
from ..data.loader import PriceData, check_freshness, load_prices, load_prices_alpaca
from ..forecasting.ensemble import latest_forecasts
from ..risk.engine import RiskEngine
from ..strategy import ewma_vol, recent_returns, select_portfolio, trend_ok
from .guarded import GuardedBroker
from .ledger import Ledger
from .orders import plan_orders

log = logging.getLogger("trading_agent.live")
NY = ZoneInfo("America/New_York")


class RunAbort(Exception):
    """Expected, safe early exit (market closed, already ran, ...)."""


def resolve_mode(cfg, live_flag: bool) -> str:
    cfg_mode = cfg["execution"]["mode"]
    if live_flag and cfg_mode != "live":
        raise SystemExit("Refusing LIVE: pass --live AND set execution.mode: live in your config.")
    if cfg_mode == "live" and not live_flag:
        log.warning("Config says live but --live flag not given -> running in PAPER mode.")
    return "live" if live_flag else "paper"


def make_risk(cfg) -> RiskEngine:
    state_dir = resolve(cfg["execution"]["state_dir"])
    return RiskEngine(cfg["risk"], cfg["universe"], state_dir / "risk_state.json",
                      resolve(cfg["risk"]["kill_file"]), sectors=cfg.get("sectors"))


def fetch_prices(cfg, key, secret) -> PriceData:
    try:
        return load_prices(cfg["universe"], cfg["data"]["start"], resolve(cfg["data"]["cache_dir"]),
                           refresh=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance failed (%s); falling back to Alpaca IEX bars", exc)
        if not key:
            raise
        return load_prices_alpaca(cfg["universe"], cfg["data"]["start"], key, secret)


def compute_plan(cfg, prices: PriceData, snap, risk: RiskEngine, prev_longs: dict, today,
                 equity_hist: pd.Series | None = None):
    """Forecast -> signals -> risk targets -> orders. Pure function of its inputs."""
    universe = cfg["universe"]
    fc = latest_forecasts(prices.close[universe], cfg["forecasting"], cfg["benchmark"]).reindex(universe)
    if not prev_longs:
        prev_longs = {s: snap.positions.get(s, 0) > 0 for s in universe}
    trend = trend_ok(prices.close).iloc[-1].to_dict()
    vols = ewma_vol(prices.close).iloc[-1].to_dict()
    pcfg = cfg.get("portfolio") or {}
    rets = recent_returns(prices.close[universe], prices.close.index[-1],
                          int(pcfg.get("corr_window", 120) or 120))
    longs = select_portfolio(fc, prev_longs, cfg["signal"], pcfg, trend, vols, rets)
    picks = [s for s, v in longs.items() if v]
    cov = rets[picks].cov(min_periods=20) * 252 if picks else None
    scale = rl_scale(cfg, prices, fc, longs, snap, equity_hist)
    decision = risk.size_targets(longs, vols, snap, today, scale, cov=cov)
    orders = plan_orders(decision.targets, snap, cfg["risk"]["rebalance_band"],
                         cfg["risk"]["min_trade_notional"], 0.0, liquidate=decision.liquidate)
    return fc, longs, decision, orders


def rl_scale(cfg, prices, fc, longs, snap, equity_hist) -> float:
    if not cfg["rl"]["enabled"]:
        return 1.0
    path = resolve(cfg["rl"]["model_path"])
    if not path.exists():
        log.warning("rl.enabled but no model at %s -> scale 1.0 (train with `trading-agent train-rl`)", path)
        return 1.0
    from ..rl.env import build_obs
    from ..rl.train import load_policy

    spy = prices.close[cfg["benchmark"]]
    eq = np.append(equity_hist.values if equity_hist is not None else [], snap.equity)
    obs = build_obs(eq, snap.gross / snap.equity, float(ewma_vol(prices.close)[cfg["benchmark"]].iloc[-1]),
                    float(np.log(spy.iloc[-1] / spy.iloc[-21])), float(fc["p_up"].mean()),
                    float(np.mean(list(longs.values()))))
    s = load_policy(cfg, path).scale(obs)
    log.info("RL exposure scale: %.2f", s)
    return s


def run_daily(cfg, live: bool = False, dry_run: bool = False, force: bool = False) -> int:
    """Returns a process exit code: 0 = ok / nothing to do, 1 = failed safely (no new trades)."""
    from .lock import state_lock

    try:
        with state_lock(resolve(cfg["execution"]["state_dir"])):
            return _run_daily(cfg, live, dry_run, force)
    except TimeoutError as exc:
        log.error("RUN SKIPPED (no orders): %s", exc)
        return 1


def _run_daily(cfg, live: bool, dry_run: bool, force: bool) -> int:
    from .alpaca_broker import AlpacaBroker

    mode = resolve_mode(cfg, live)
    key, secret = load_secrets()
    if not key or not secret:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Copy .env.example to .env and add your "
                  "PAPER keys (or run `trading-agent preview` for an offline, no-keys preview).")
        return 1
    ledger = Ledger(resolve(cfg["execution"]["state_dir"]))
    session = datetime.now(NY).date()
    ex = cfg["execution"]
    try:
        broker = AlpacaBroker(key, secret, paper=(mode == "paper"), session=session,
                              order_timeout_sec=ex["order_timeout_sec"], max_retries=ex["max_retries"])
        clock = broker.clock()
        now = clock.timestamp.astimezone(NY)
        session = now.date()
        broker.session = session
        if not force:
            if not clock.is_open:
                raise RunAbort(f"Market closed (next open {clock.next_open.astimezone(NY):%a %Y-%m-%d %H:%M} ET).")
            opened = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if (now - opened).total_seconds() < ex["min_minutes_after_open"] * 60:
                raise RunAbort(f"Too soon after the open; run >= {ex['min_minutes_after_open']} min after 9:30 ET.")
            if (clock.next_close.astimezone(NY) - now).total_seconds() < 15 * 60:
                raise RunAbort("Less than 15 minutes to the close; skipping today.")
            if ledger.run_status(str(session), mode) == "completed":
                raise RunAbort(f"Already completed for {session} ({mode}); nothing to do.")

        acct = broker.account()
        if getattr(acct, "trading_blocked", False) or getattr(acct, "account_blocked", False):
            raise RuntimeError("Alpaca reports the account is blocked from trading")
        ledger.start_run(str(session), mode)
        risk = make_risk(cfg)
        guard = GuardedBroker(broker, risk)
        snap = guard.snapshot()
        log.info("Account equity $%.2f | cash $%.2f | positions %s | broker day trades %s",
                 snap.equity, snap.cash, {k: round(v, 2) for k, v in snap.positions.items()},
                 snap.broker_daytrade_count)

        if risk.state.peak_equity is None:            # e.g. fresh machine / lost cache
            risk.state.peak_equity = broker.peak_equity()

        # ---- 1. hard floor / kill switch (before anything else can fail)
        if risk.is_halted() or risk.check_floor(snap):
            return _liquidate(cfg, guard, risk, ledger, session, mode, snap, dry_run)

        # ---- 2. data, verified fresh
        prices = fetch_prices(cfg, key, secret)
        check_freshness(prices, cfg["data"]["max_staleness_days"], session)
        prev = broker.previous_session(session)
        if prices.close.index[-1].date() != prev:
            raise RuntimeError(f"Data ends {prices.close.index[-1].date()} but last session was {prev}")

        # ---- 3. plan
        fc, longs, decision, orders = compute_plan(cfg, prices, snap, risk, ledger.last_longs(mode),
                                                   session, ledger.equity_history(mode))
        _log_plan(fc, longs, decision, orders)
        ledger.record_signals(str(session), mode, fc, longs, decision.targets)
        if dry_run:
            ledger.finish_run(str(session), mode, "dry_run", "; ".join(decision.reasons))
            log.info("DRY RUN: no orders submitted.")
            return 0

        # ---- 4. execute (each order re-validated by the risk engine)
        guard.execute(orders, session, emergency=decision.liquidate,
                      on_result=lambda o, r: ledger.record_order(str(session), mode, o, r))
        after = guard.snapshot()
        risk.check_floor(after)
        ledger.record_equity(str(session), mode, after, decision.drawdown, risk.is_halted(),
                             risk.day_trades_in_window(session), "; ".join(decision.reasons))
        ledger.finish_run(str(session), mode, "completed", "; ".join(decision.reasons))
        log.info("Run complete. Equity $%.2f, cash $%.2f", after.equity, after.cash)
        return 0
    except RunAbort as msg:
        log.info("%s", msg)
        return 0
    except Exception as exc:  # noqa: BLE001 - fail safe: log, record, no further orders
        log.error("RUN FAILED SAFELY (no further orders): %s\n%s", exc, traceback.format_exc())
        try:
            ledger.finish_run(str(session), mode, "failed", str(exc))
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        try:
            ledger.export_csv()
        except Exception:  # noqa: BLE001
            pass


def _liquidate(cfg, guard, risk, ledger, session, mode, snap, dry_run) -> int:
    log.critical("HALTED (%s). Liquidating all positions.", risk.state.halt_reason)
    orders = plan_orders({}, snap, 0.0, 0.0, liquidate=True)
    if dry_run:
        log.info("DRY RUN: would liquidate %s", [o.symbol for o in orders])
        return 0
    guard.execute(orders, session, emergency=True,
                  on_result=lambda o, r: ledger.record_order(str(session), mode, o, r))
    after = guard.snapshot()
    ledger.record_equity(str(session), mode, after, 0.0, True, risk.day_trades_in_window(session),
                         risk.state.halt_reason)
    ledger.finish_run(str(session), mode, "halted", risk.state.halt_reason)
    return 0


def _log_plan(fc, longs, decision, orders):
    log.info("Forecasts:\n%s", fc[["p_up", "mu", "agreement", "confidence"]].round(4).to_string())
    log.info("Signals (long): %s", [s for s, v in longs.items() if v] or "none -> cash")
    log.info("Risk-approved targets: %s", {s: round(w, 3) for s, w in decision.targets.items() if w > 0})
    for r in decision.reasons:
        log.info("Risk note: %s", r)
    if not orders:
        log.info("No orders needed today.")
    for o in orders:
        log.info("Planned order: %s %s $%.2f (%s)", o.side.upper(), o.symbol, o.notional, o.reason)


def guard_check(cfg, live: bool = False) -> int:
    """Backward-compatible alias: one intraday monitor check (floor, drawdown, daily loss)."""
    from .monitor import run_monitor

    return run_monitor(cfg, live=live, loop=False)


def preview_offline(cfg) -> int:
    """No keys needed: today's signals and the orders a fresh $77 account would place."""
    from ..risk.engine import PortfolioSnapshot

    prices = load_prices(cfg["universe"], cfg["data"]["start"], resolve(cfg["data"]["cache_dir"]))
    cap = cfg["risk"]["initial_capital"]
    snap = PortfolioSnapshot(cap, cap, {}, cap)
    risk = RiskEngine(cfg["risk"], cfg["universe"], sectors=cfg.get("sectors"))  # in-memory only
    fc, longs, decision, orders = compute_plan(cfg, prices, snap, risk, {}, datetime.now(NY).date())
    log.info("Preview using data through %s (offline, no broker):", prices.close.index[-1].date())
    _log_plan(fc, longs, decision, orders)
    return 0
