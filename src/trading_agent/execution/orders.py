"""Translate target weights into a minimal list of orders (sells first, then buys)."""
from __future__ import annotations

import math

from ..risk.engine import Order, PortfolioSnapshot


def plan_orders(targets: dict[str, float], snap: PortfolioSnapshot, band: float,
                min_notional: float, cash_buffer: float = 0.0, liquidate: bool = False) -> list[Order]:
    sells, buys = [], []
    eq = snap.equity
    for sym in sorted(set(targets) | set(snap.positions)):
        cur = snap.positions.get(sym, 0.0)
        tgt = targets.get(sym, 0.0) * eq
        diff = tgt - cur
        if tgt <= 0 and cur > 0:
            sells.append(Order(sym, "sell", cur, close_position=True,
                               reason="liquidate" if liquidate else "exit"))
            continue
        if abs(diff) < band * eq or abs(diff) < min_notional:
            continue
        if diff < 0:
            sells.append(Order(sym, "sell", -diff, reason="trim"))
        else:
            buys.append(Order(sym, "buy", diff, reason="enter/add"))
    # buys are funded only by cash left after sells (cash account, no margin)
    cash = snap.cash + sum(o.notional for o in sells) - cash_buffer * eq
    funded = []
    for o in sorted(buys, key=lambda o: -o.notional):
        amt = math.floor(min(o.notional, cash) * 100) / 100     # whole cents, never above cash
        if amt >= min_notional:
            funded.append(Order(o.symbol, "buy", amt, reason=o.reason))
            cash -= amt
    return sells + funded
