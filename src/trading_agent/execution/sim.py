"""Simulated broker for backtests: fractional fills at the open with costs."""
from __future__ import annotations

from ..risk.engine import Order, PortfolioSnapshot


class SimBroker:
    def __init__(self, cash: float, cost_bps: float, commission: float = 0.0):
        self.cash = float(cash)
        self.qty: dict[str, float] = {}
        self.prices: dict[str, float] = {}
        self.cost = cost_bps / 1e4
        self.commission = commission
        self.last_equity: float | None = None
        self.trades: list[dict] = []
        self.now = None

    def set_prices(self, prices: dict[str, float], now=None) -> None:
        self.prices = {k: float(v) for k, v in prices.items()}
        self.now = now

    def equity(self) -> float:
        return self.cash + sum(q * self.prices[s] for s, q in self.qty.items())

    def snapshot(self) -> PortfolioSnapshot:
        pos = {s: q * self.prices[s] for s, q in self.qty.items() if q > 1e-12}
        return PortfolioSnapshot(self.equity(), self.cash, pos, self.last_equity)

    def submit(self, order: Order) -> dict:
        px = self.prices[order.symbol]
        if order.side == "buy":
            fill = px * (1 + self.cost)
            spend = min(order.notional, self.cash - self.commission)
            if spend <= 0:
                return {"status": "rejected", "reason": "no cash"}
            q = spend / fill
            self.qty[order.symbol] = self.qty.get(order.symbol, 0.0) + q
            self.cash -= spend + self.commission
        else:
            fill = px * (1 - self.cost)
            held = self.qty.get(order.symbol, 0.0)
            q = held if order.close_position else min(held, order.notional / px)
            self.qty[order.symbol] = held - q
            if self.qty[order.symbol] <= 1e-12:
                self.qty.pop(order.symbol)
            self.cash += q * fill - self.commission
        rec = {"date": self.now, "symbol": order.symbol, "side": order.side, "qty": q,
               "price": fill, "notional": q * fill, "reason": order.reason, "status": "filled"}
        self.trades.append(rec)
        return rec
