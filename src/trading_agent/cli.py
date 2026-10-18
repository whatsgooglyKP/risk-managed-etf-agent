"""Command-line entry point: `trading-agent <command>`."""
from __future__ import annotations

import argparse
import sys

from .config import resolve
from .logging_setup import setup_logging


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trading-agent", description="Risk-managed ETF trading agent")
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="walk-forward backtest + charts + HTML report")
    b.add_argument("--no-rl", action="store_true", help="skip the RL comparison")
    b.add_argument("--retrain-rl", action="store_true", help="retrain the PPO model first")
    b.add_argument("--refresh", action="store_true", help="re-download data and recompute forecasts")

    sub.add_parser("train-rl", help="train the PPO exposure scaler")

    d = sub.add_parser("daily", help="the once-a-day run (paper by default)")
    d.add_argument("--live", action="store_true", help="REAL MONEY (also needs execution.mode: live)")
    d.add_argument("--dry-run", action="store_true", help="connect and plan, but place no orders")
    d.add_argument("--force", action="store_true", help="ignore clock/idempotency checks (testing)")

    m = sub.add_parser("monitor", help="intraday risk monitor: live prices vs floor / daily loss / drawdown")
    m.add_argument("--live", action="store_true")
    m.add_argument("--loop", action="store_true", help="keep checking every monitor.interval_sec until the close")
    m.add_argument("--dry-run", action="store_true", help="check and log, but place no orders")
    g = sub.add_parser("guard", help="alias for one `monitor` check (kept for old cron lines)")
    g.add_argument("--live", action="store_true")

    cmp_ = sub.add_parser("compare", help="walk-forward backtest of several configs side by side vs SPY")
    cmp_.add_argument("configs", nargs="*", default=["configs/default.yaml", "configs/wide.yaml"])
    cmp_.add_argument("--sensitivity", action="store_true", help="also run top_n 3/5/8 on the wide config")

    sub.add_parser("preview", help="offline: today's signals for a fresh $77 account (no keys)")
    c = sub.add_parser("check", help="verify Alpaca keys and show the account (places no orders)")
    c.add_argument("--live", action="store_true")
    sub.add_parser("status", help="show risk state, kill switch and recent ledger entries")
    sub.add_parser("dashboard", help="launch the Streamlit dashboard")

    args = p.parse_args(argv)
    from .config import load_config

    cfg = load_config(args.config)
    setup_logging(resolve(cfg["execution"]["log_dir"]))

    if args.cmd == "backtest":
        from .backtest.suite import run_suite

        run_suite(cfg, with_rl=not args.no_rl, retrain_rl=args.retrain_rl, refresh=args.refresh)
        return 0
    if args.cmd == "train-rl":
        from .pipeline import get_forecasts, get_prices
        from .rl.train import train_ppo

        prices = get_prices(cfg)
        train_ppo(prices, get_forecasts(cfg, prices), cfg, resolve(cfg["rl"]["model_path"]))
        return 0
    if args.cmd == "daily":
        from .execution.live import run_daily

        return run_daily(cfg, live=args.live, dry_run=args.dry_run, force=args.force)
    if args.cmd == "monitor":
        from .execution.monitor import run_monitor

        return run_monitor(cfg, live=args.live, loop=args.loop, dry_run=args.dry_run)
    if args.cmd == "compare":
        from .backtest.compare import run_compare

        run_compare(args.configs, sensitivity=args.sensitivity)
        return 0
    if args.cmd == "guard":
        from .execution.live import guard_check

        return guard_check(cfg, live=args.live)
    if args.cmd == "preview":
        from .execution.live import preview_offline

        return preview_offline(cfg)
    if args.cmd == "check":
        return _check(cfg, args.live)
    if args.cmd == "status":
        return _status(cfg)
    if args.cmd == "dashboard":
        import subprocess
        from pathlib import Path

        app = Path(__file__).parent / "dashboard" / "app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    return 1


def _check(cfg, live: bool) -> int:
    from .config import load_secrets
    from .execution.alpaca_broker import AlpacaBroker
    from .execution.live import resolve_mode

    mode = resolve_mode(cfg, live)
    key, secret = load_secrets()
    if not key or not secret:
        print("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. Copy .env.example to .env and fill them in.")
        return 1
    b = AlpacaBroker(key, secret, paper=(mode == "paper"))
    a, clock, snap = b.account(), b.clock(), b.snapshot()
    print(f"Mode: {mode.upper()}  status={a.status}  equity=${float(a.equity):.2f}  cash=${snap.cash:.2f}")
    print(f"Pattern day trader flag: {a.pattern_day_trader}  day trades (5d): {a.daytrade_count}")
    print(f"Trading blocked: {a.trading_blocked}  market open: {clock.is_open}")
    print(f"Positions: {snap.positions or 'none'}")
    print("Keys work. No orders were placed.")
    return 0


def _status(cfg) -> int:
    from .execution.ledger import Ledger
    from .execution.live import make_risk

    risk = make_risk(cfg)
    st = risk.state
    print(f"Halted: {risk.is_halted()}  reason: {st.halt_reason or '-'}")
    print(f"Kill file: {resolve(cfg['risk']['kill_file'])} exists={resolve(cfg['risk']['kill_file']).exists()}")
    print(f"Peak equity: {st.peak_equity}  cooldown until: {st.cooldown_until}  floor triggers: {st.floor_triggers}")
    led = Ledger(resolve(cfg["execution"]["state_dir"]))
    for t in ("runs", "orders", "equity"):
        df = led.table(t)
        print(f"\n== last {t} ==\n{df.tail(5).to_string(index=False) if not df.empty else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
