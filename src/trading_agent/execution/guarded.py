"""GuardedBroker: the ONLY path to a broker. Every order passes RiskEngine.validate_order."""
from __future__ import annotations

import logging
import math
from datetime import date
from typing import Callable

from ..risk.engine import Order, PortfolioSnapshot, RiskEngine, RiskViolation

log = logging.getLogger("trading_agent.execution")


class GuardedBroker:
    def __init__(self, broker, risk: RiskEngine):
        self._broker = broker
        self.risk = risk

    def snapshot(self) -> PortfolioSnapshot:
        return self._broker.snapshot()

    def execute(self, orders: list[Order], today: date, emergency: bool = False,
                on_result: Callable[[Order, dict], None] | None = None) -> list[dict]:
        """Validate and submit orders one by one, refreshing the snapshot after each fill."""
        results = []
        for order in orders:
            snap = self._broker.snapshot()
            if order.side == "buy" and snap.cash < order.notional <= snap.cash * 1.01 + 0.01:
                # tiny shortfall from slippage on earlier fills: shrink to the cash we have
                order.notional = math.floor(snap.cash * 100) / 100
            try:
                self.risk.validate_order(order, snap, today, emergency=emergency)
            except RiskViolation as exc:
                log.warning("REJECTED by risk engine: %s %s $%.2f - %s",
                            order.side, order.symbol, order.notional, exc)
                res = {"status": "rejected", "reason": str(exc)}
            else:
                res = self._broker.submit(order)
                if res.get("status") == "filled":
                    self.risk.record_fill(order.symbol, order.side, today)
            results.append(res)
            if on_result:
                on_result(order, res)
        self.risk.save()
        return results
