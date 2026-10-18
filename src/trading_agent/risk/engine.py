"""Risk engine: sizes every position and vets every single order.

The engine is the only component allowed to turn signals into target weights, and the
execution layer (``execution.guarded.GuardedBroker``) calls ``validate_order`` before
*every* order, in backtests and live alike. Nothing downstream can raise exposure above
what this module approves.

Rules (all configurable in configs/default.yaml -> risk):
  * Hard floor: equity <= hard_floor  -> liquidate everything, write the kill file, halt
    permanently. While halted only risk-reducing sells are accepted.
  * Volatility sizing: weight_i = min(max_position, target_vol / vol_i).
  * Cash only: gross exposure <= max_gross_exposure - cash_buffer (<= 100%), buys capped
    by available cash, never sell more than held (no shorting / margin).
  * Drawdown control: dd >= soft -> half size; dd >= hard -> flat for cooldown_days.
  * Daily loss cap: equity down >= cap vs last close -> no new buying today.
  * PDT guard: under $25k, at most N day trades per rolling 5 business days.
  * Wide-universe limits (optional): max weight per sector group (`max_sector`), max number
    of symbols held (`max_open_positions`) and an ex-ante portfolio volatility cap
    (`portfolio_vol_cap`, using the recent covariance of the chosen symbols).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FORBIDDEN_SYMBOLS

log = logging.getLogger("trading_agent.risk")

EPS = 1e-9


class RiskViolation(Exception):
    """Raised when an order is rejected by the risk engine."""


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    positions: dict[str, float]          # symbol -> market value ($)
    last_equity: float | None = None     # equity at the previous close (daily loss cap)
    broker_daytrade_count: int | None = None  # broker's own rolling count, if known

    def weight(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0) / max(self.equity, EPS)

    @property
    def gross(self) -> float:
        return sum(v for v in self.positions.values() if v > 0)


@dataclass
class Order:
    symbol: str
    side: str                 # "buy" | "sell"
    notional: float           # dollar amount (always positive)
    close_position: bool = False
    reason: str = ""
    tag: str = ""             # distinguishes intraday-monitor orders from the daily run's (order ids)


@dataclass
class RiskDecision:
    targets: dict[str, float]
    halted: bool = False
    liquidate: bool = False
    exposure_mult: float = 1.0
    drawdown: float = 0.0
    buys_blocked: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class RiskState:
    halted: bool = False
    halt_reason: str = ""
    peak_equity: float | None = None
    cooldown_until: str | None = None              # ISO date
    fills: list[list] = field(default_factory=list)       # [iso_date, symbol, side]
    day_trades: list[list] = field(default_factory=list)  # [iso_date, symbol]
    floor_triggers: int = 0
    buys_blocked_on: str | None = None              # ISO date: intraday monitor blocked buys for that day


class RiskEngine:
    def __init__(self, rcfg, universe: list[str], state_path: str | Path | None = None,
                 kill_file: str | Path | None = None, sectors: dict[str, str] | None = None):
        self.cfg = dict(rcfg)
        self.universe = [s.upper() for s in universe]
        self.sectors = {k.upper(): v for k, v in (sectors or {}).items()}
        self.state_path = Path(state_path) if state_path else None
        self.kill_file = Path(kill_file) if kill_file else None
        self.state = self._load_state()

    # ------------------------------------------------------------------ persistence
    def _load_state(self) -> RiskState:
        if self.state_path and self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            return RiskState(**data)
        return RiskState()

    def save(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self.state), indent=2))
            tmp.replace(self.state_path)

    # ------------------------------------------------------------------ halt / floor
    def is_halted(self) -> bool:
        if self.kill_file is not None and self.kill_file.exists():
            self.state.halted = True
            if not self.state.halt_reason:
                self.state.halt_reason = f"kill file present: {self.kill_file}"
        return self.state.halted

    def trigger_halt(self, reason: str) -> None:
        if not self.state.halted:
            log.critical("PERMANENT HALT: %s", reason)
        self.state.halted = True
        self.state.halt_reason = reason
        if self.kill_file is not None:
            self.kill_file.parent.mkdir(parents=True, exist_ok=True)
            self.kill_file.write_text(f"{datetime.now().isoformat()} {reason}\n"
                                      "Delete this file ONLY after a manual review.\n")
        self.save()

    def check_floor(self, snap: PortfolioSnapshot) -> bool:
        """True (and halts permanently) if equity is at or below the hard floor."""
        if snap.equity <= self.cfg["hard_floor"] + EPS:
            if not self.state.halted:
                self.state.floor_triggers += 1
            self.trigger_halt(f"hard floor hit: equity ${snap.equity:.2f} <= ${self.cfg['hard_floor']:.2f}")
            return True
        return False

    # ------------------------------------------------------------------ PDT
    @staticmethod
    def _bday_window(today: date, n: int = 5) -> set[str]:
        days = pd.bdate_range(end=pd.Timestamp(today), periods=n)
        return {d.date().isoformat() for d in days}

    def day_trades_in_window(self, today: date) -> int:
        window = self._bday_window(today)
        local = sum(1 for d, _ in self.state.day_trades if d in window)
        return local

    def pdt_applies(self, snap: PortfolioSnapshot) -> bool:
        return snap.equity < self.cfg["pdt_equity_threshold"]

    def bought_today(self, symbol: str, today: date) -> bool:
        t = today.isoformat()
        return any(d == t and s == symbol and side == "buy" for d, s, side in self.state.fills)

    def record_fill(self, symbol: str, side: str, today: date) -> None:
        t = today.isoformat()
        if side == "sell" and self.bought_today(symbol, today):
            self.state.day_trades.append([t, symbol])
            log.warning("Day trade recorded for %s on %s", symbol, t)
        self.state.fills.append([t, symbol, side])
        # keep state small: drop history older than ~10 business days
        keep = self._bday_window(today, 10)
        self.state.fills = [f for f in self.state.fills if f[0] in keep]
        self.state.day_trades = [d for d in self.state.day_trades if d[0] in keep]

    # ------------------------------------------------------------------ sizing
    def sector_of(self, symbol: str) -> str:
        return self.sectors.get(symbol.upper(), symbol.upper())   # ungrouped = its own group

    def sector_weight(self, snap: PortfolioSnapshot, sector: str) -> float:
        return sum(v for s, v in snap.positions.items() if self.sector_of(s) == sector) / max(snap.equity, EPS)

    def apply_sector_cap(self, targets: dict[str, float], reasons: list[str]) -> dict[str, float]:
        cap = self.cfg.get("max_sector")
        if not cap or not self.sectors:
            return targets
        out = dict(targets)
        groups: dict[str, list[str]] = {}
        for sym, w in out.items():
            if w > 0:
                groups.setdefault(self.sector_of(sym), []).append(sym)
        for g, syms in groups.items():
            tot = sum(out[s] for s in syms)
            if tot > cap + EPS:
                for s in syms:
                    out[s] *= cap / tot
                reasons.append(f"sector {g} {tot:.0%} scaled to cap {cap:.0%}")
        return out

    def apply_portfolio_vol_cap(self, targets: dict[str, float], cov: pd.DataFrame | None,
                                reasons: list[str]) -> dict[str, float]:
        """Scale all weights down (never up) so ex-ante annualised portfolio vol <= cap."""
        cap = self.cfg.get("portfolio_vol_cap")
        held = [s for s, w in targets.items() if w > 0]
        if not cap or not held:
            return targets
        if cov is None or not set(held) <= set(cov.columns):
            return targets
        c = cov.loc[held, held].to_numpy(dtype=float)
        if not np.isfinite(c).all():
            # unknown covariance for a new listing: assume perfect correlation (conservative)
            sd = np.sqrt(np.nan_to_num(np.diag(c), nan=0.0))
            c = np.outer(sd, sd)
        w = np.array([targets[s] for s in held])
        vol = float(np.sqrt(max(w @ c @ w, 0.0)))
        if vol > cap:
            k = cap / vol
            reasons.append(f"ex-ante portfolio vol {vol:.1%} scaled to cap {cap:.0%}")
            return {s: (wt * k if wt > 0 else 0.0) for s, wt in targets.items()}
        return targets

    def size_targets(self, longs: dict[str, bool], vols: dict[str, float],
                     snap: PortfolioSnapshot, today: date, rl_scale: float = 1.0,
                     cov: pd.DataFrame | None = None) -> RiskDecision:
        """Convert long/flat signals into risk-approved target weights.

        `cov` (optional) is the annualised covariance of recent returns, used only for the
        portfolio volatility cap.
        """
        reasons: list[str] = []
        if self.is_halted() or self.check_floor(snap):
            return RiskDecision({s: 0.0 for s in self.universe}, halted=True, liquidate=True,
                                exposure_mult=0.0, reasons=[self.state.halt_reason])

        # drawdown control (peak tracked in persistent state)
        peak = max(self.state.peak_equity or snap.equity, snap.equity)
        self.state.peak_equity = peak
        dd = 1.0 - snap.equity / peak
        mult = 1.0
        if self.state.cooldown_until:
            if today.isoformat() < self.state.cooldown_until:
                mult = 0.0
                reasons.append(f"drawdown cooldown until {self.state.cooldown_until}")
            else:
                reasons.append("cooldown over: drawdown peak reset, resuming at half size")
                self.state.cooldown_until = None
                self.state.peak_equity = peak = snap.equity
                dd = 0.0
                mult = 0.5
        if mult > 0 and dd >= self.cfg["max_drawdown_hard"]:
            end = pd.Timestamp(today) + pd.offsets.BDay(int(self.cfg["cooldown_days"]))
            self.state.cooldown_until = end.date().isoformat()
            mult = 0.0
            reasons.append(f"drawdown {dd:.1%} >= hard limit: flat until {self.state.cooldown_until}")
        elif mult > 0 and dd >= self.cfg["max_drawdown_soft"]:
            mult = min(mult, 0.5)
            reasons.append(f"drawdown {dd:.1%} >= soft limit: half size")

        scale = float(np.clip(rl_scale, 0.0, 1.0))
        targets = {}
        for sym in self.universe:
            if not longs.get(sym, False):
                targets[sym] = 0.0
                continue
            vol = max(float(vols.get(sym, np.nan)), 0.05) if np.isfinite(vols.get(sym, np.nan)) else None
            if vol is None:
                targets[sym] = 0.0
                reasons.append(f"{sym}: no volatility estimate -> flat")
                continue
            targets[sym] = min(self.cfg["max_position"], self.cfg["target_vol"] / vol) * mult * scale

        mop = self.cfg.get("max_open_positions")
        live_syms = [s for s, w in targets.items() if w > 0]
        if mop and len(live_syms) > mop:
            # keep current holdings first, then the largest targets
            order = sorted(live_syms, key=lambda s: (snap.positions.get(s, 0) <= 0, -targets[s]))
            for s in order[mop:]:
                targets[s] = 0.0
            reasons.append(f"max {mop} open positions: dropped {order[mop:]}")

        targets = self.apply_sector_cap(targets, reasons)
        targets = self.apply_portfolio_vol_cap(targets, cov, reasons)

        cap = self.cfg["max_gross_exposure"] - self.cfg["cash_buffer"]
        gross = sum(targets.values())
        if gross > cap:
            targets = {s: w * cap / gross for s, w in targets.items()}
            reasons.append(f"gross {gross:.0%} scaled to cap {cap:.0%}")

        buys_blocked = False
        if snap.last_equity and snap.equity / snap.last_equity - 1 <= -self.cfg["daily_loss_cap"]:
            buys_blocked = True
            targets = {s: min(w, snap.weight(s)) for s, w in targets.items()}
            reasons.append("daily loss cap hit: no new buying today")
        elif self.buys_blocked_today(today):
            buys_blocked = True
            targets = {s: min(w, snap.weight(s)) for s, w in targets.items()}
            reasons.append("intraday monitor blocked buying for today")

        self.save()
        return RiskDecision(targets, exposure_mult=mult * scale, drawdown=dd,
                            buys_blocked=buys_blocked, reasons=reasons)

    def block_buys_today(self, today: date, reason: str) -> None:
        if self.state.buys_blocked_on != today.isoformat():
            log.warning("Buying blocked for %s: %s", today, reason)
        self.state.buys_blocked_on = today.isoformat()
        self.save()

    def buys_blocked_today(self, today: date) -> bool:
        return self.state.buys_blocked_on == today.isoformat()

    def start_cooldown(self, today: date, reason: str) -> None:
        """Same rule as the daily drawdown brake: flat for cooldown_days, then half size."""
        end = pd.Timestamp(today) + pd.offsets.BDay(int(self.cfg["cooldown_days"]))
        if not self.state.cooldown_until or self.state.cooldown_until < end.date().isoformat():
            self.state.cooldown_until = end.date().isoformat()
        log.warning("Drawdown cooldown until %s: %s", self.state.cooldown_until, reason)
        self.save()

    # ------------------------------------------------------------------ per-order gate
    def validate_order(self, order: Order, snap: PortfolioSnapshot, today: date,
                       emergency: bool = False) -> None:
        """Raise RiskViolation unless the order is allowed. Called before EVERY order."""
        sym = order.symbol.upper()
        if order.side not in ("buy", "sell"):
            raise RiskViolation(f"unknown side {order.side}")
        if not np.isfinite(order.notional) or order.notional <= 0:
            raise RiskViolation(f"{sym}: notional must be positive")
        if sym in FORBIDDEN_SYMBOLS:
            raise RiskViolation(f"{sym}: leveraged/inverse products are forbidden")
        if self.cfg.get("allowed_symbols_only", True) and sym not in self.universe:
            raise RiskViolation(f"{sym}: not in the configured universe")

        held = snap.positions.get(sym, 0.0)
        if order.side == "sell":
            if held <= 0:
                raise RiskViolation(f"{sym}: nothing to sell (shorting is not allowed)")
            if order.notional > held * 1.02 + 0.01 and not order.close_position:
                raise RiskViolation(f"{sym}: sell ${order.notional:.2f} exceeds holding ${held:.2f}")
            if self.pdt_applies(snap) and self.bought_today(sym, today):
                used = max(self.day_trades_in_window(today), snap.broker_daytrade_count or 0)
                if used >= self.cfg["pdt_max_day_trades"]:
                    if not emergency:
                        raise RiskViolation(f"{sym}: PDT guard - {used} day trades in 5 days")
                    log.warning("PDT limit reached but emergency liquidation proceeds for %s", sym)
            return

        # ---- buys
        if self.is_halted():
            raise RiskViolation("trading is halted: buys are not allowed")
        if snap.equity <= self.cfg["hard_floor"]:
            raise RiskViolation("equity at/below hard floor: buys are not allowed")
        if order.notional < self.cfg["min_trade_notional"] - EPS:
            raise RiskViolation(f"{sym}: ${order.notional:.2f} below minimum order size")
        if order.notional > snap.cash + EPS:
            raise RiskViolation(f"{sym}: ${order.notional:.2f} exceeds cash ${snap.cash:.2f} (cash only)")
        if snap.last_equity and snap.equity / snap.last_equity - 1 <= -self.cfg["daily_loss_cap"]:
            raise RiskViolation("daily loss cap hit: no new buying today")
        if self.buys_blocked_today(today):
            raise RiskViolation("intraday monitor blocked buying for today")
        if self.state.cooldown_until and today.isoformat() < self.state.cooldown_until:
            raise RiskViolation(f"drawdown cooldown until {self.state.cooldown_until}: no buying")
        tol = 0.01
        if (held + order.notional) / snap.equity > self.cfg["max_position"] + tol:
            raise RiskViolation(f"{sym}: would exceed max position {self.cfg['max_position']:.0%}")
        if (snap.gross + order.notional) / snap.equity > self.cfg["max_gross_exposure"] + tol:
            raise RiskViolation("would exceed max gross exposure (no margin)")
        sec_cap = self.cfg.get("max_sector")
        if sec_cap and self.sectors:
            g = self.sector_of(sym)
            if self.sector_weight(snap, g) + order.notional / snap.equity > sec_cap + tol:
                raise RiskViolation(f"{sym}: would exceed sector cap {sec_cap:.0%} for {g}")
        mop = self.cfg.get("max_open_positions")
        if mop and held <= 0:
            n_held = sum(1 for v in snap.positions.values() if v > 0)
            if n_held >= mop:
                raise RiskViolation(f"{sym}: already holding {n_held} symbols (max {mop})")
