"""Event-driven daily backtester sharing the live code path.

Timeline for each trading day d:
  09:30  orders decided at the previous close are filled at d's OPEN (+ slippage/spread)
  16:00  portfolio marked at d's CLOSE; forecasts known at d's close -> signals ->
         RiskEngine.size_targets -> target weights for the next open.
Every order goes through GuardedBroker -> RiskEngine.validate_order, exactly as live.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.loader import PriceData
from ..execution.guarded import GuardedBroker
from ..execution.orders import plan_orders
from ..execution.sim import SimBroker
from ..risk.engine import RiskDecision, RiskEngine
from ..strategy import ewma_vol, select_portfolio, trend_ok


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    decisions: list = field(default_factory=list)
    floor_triggers: int = 0
    day_trades: int = 0
    halted_on: str | None = None


class Backtester:
    def __init__(self, prices: PriceData, forecasts: pd.DataFrame, cfg, start: str,
                 end: str | None = None):
        self.cfg = cfg
        self.prices = prices
        self.fc = forecasts
        self.universe = list(cfg["universe"])
        self.dates = prices.close.loc[start:end].index.intersection(
            forecasts.index.get_level_values("date").unique())
        self.fc_by_date = {d: g.droplevel("date").reindex(self.universe)
                           for d, g in forecasts.groupby(level="date")}
        self.vol = ewma_vol(prices.close)
        self.trend = trend_ok(prices.close)
        self.logret = np.log(prices.close).diff()
        self.pos_of = {d: i for i, d in enumerate(prices.close.index)}
        pcfg = cfg.get("portfolio") or {}
        self.pcfg = pcfg
        self.corr_window = int(pcfg.get("corr_window", 120) or 120)
        self.need_returns = (pcfg.get("top_n") is not None and pcfg.get("max_corr") is not None) \
            or bool(cfg["risk"].get("portfolio_vol_cap"))
        c = cfg["costs"]
        self.cost_bps = c["slippage_bps"] + c["spread_bps"] / 2
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self):
        r = self.cfg["risk"]
        self.broker = SimBroker(r["initial_capital"], self.cost_bps, self.cfg["costs"]["commission_per_order"])
        self.risk = RiskEngine(r, self.universe, sectors=self.cfg.get("sectors"))  # in-memory, no kill file
        self.guard = GuardedBroker(self.broker, self.risk)
        self.k = 0
        self.pending: RiskDecision | None = None
        self.sig_long = {s: False for s in self.universe}
        self.eq, self.expo, self.dec_log = [], [], []
        self.halted_on = None
        return self

    @property
    def done(self) -> bool:
        return self.k >= len(self.dates)

    def current_date(self):
        return self.dates[min(self.k, len(self.dates) - 1)]

    # ------------------------------------------------------------------ one day
    def open_and_close(self) -> float:
        """Execute pending orders at today's open, mark at the close. Returns equity."""
        d = self.dates[self.k]
        today = d.date()
        o = self.prices.open.loc[d, self.universe].to_dict()
        self.broker.set_prices(o, now=d)
        if self.pending is not None:
            snap = self.broker.snapshot()
            orders = plan_orders(self.pending.targets, snap, self.cfg["risk"]["rebalance_band"],
                                 self.cfg["risk"]["min_trade_notional"], 0.0,
                                 liquidate=self.pending.liquidate)
            self.guard.execute(orders, today, emergency=self.pending.liquidate)
        self.broker.set_prices(self.prices.close.loc[d, self.universe].to_dict(), now=d)
        eq = self.broker.equity()
        snap = self.broker.snapshot()
        self.eq.append((d, eq))
        self.expo.append((d, snap.gross / eq if eq > 0 else 0.0))
        return eq

    def decide(self, rl_scale: float = 1.0) -> RiskDecision:
        """At today's close: forecasts -> signals -> risk-approved targets for tomorrow."""
        d = self.dates[self.k]
        fc = self.fc_by_date[d]
        trend = self.trend.loc[d].to_dict()
        vols = self.vol.loc[d].to_dict()
        rets = cov = None
        if self.need_returns:
            i = self.pos_of[d]
            rets = self.logret.iloc[max(1, i - self.corr_window + 1): i + 1]
        self.sig_long = select_portfolio(fc, self.sig_long, self.cfg["signal"], self.pcfg, trend, vols, rets)
        if rets is not None and self.cfg["risk"].get("portfolio_vol_cap"):
            picks = [s for s, v in self.sig_long.items() if v]
            if picks:
                cov = rets[picks].cov(min_periods=20) * 252
        snap = self.broker.snapshot()
        dec = self.risk.size_targets(self.sig_long, vols, snap, d.date(), rl_scale, cov=cov)
        if dec.halted and self.halted_on is None:
            self.halted_on = str(d.date())
        self.pending = dec
        self.broker.last_equity = snap.equity
        self.dec_log.append((d, dec.exposure_mult, dec.drawdown, sum(dec.targets.values())))
        self.k += 1
        return dec

    def run(self, policy=None) -> BacktestResult:
        """policy(backtester) -> exposure scale in [0, 1]; None = always 1.0 (no RL)."""
        self.reset()
        while not self.done:
            self.open_and_close()
            scale = 1.0 if policy is None else float(policy(self))
            self.decide(scale)
        return self.result()

    def result(self) -> BacktestResult:
        eq = pd.Series(dict(self.eq), name="equity")
        return BacktestResult(eq, pd.DataFrame(self.broker.trades),
                              pd.Series(dict(self.expo), name="exposure"), self.dec_log,
                              self.risk.state.floor_triggers, len(self.risk.state.day_trades),
                              self.halted_on)


def buy_and_hold(prices: PriceData, symbol: str, start: str, end: str | None, capital: float,
                 cost_bps: float) -> pd.Series:
    close = prices.close[symbol].loc[start:end]
    first_open = prices.open[symbol].loc[close.index[0]]
    shares = capital / (first_open * (1 + cost_bps / 1e4))
    return (close * shares).rename(f"buy_hold_{symbol}")
