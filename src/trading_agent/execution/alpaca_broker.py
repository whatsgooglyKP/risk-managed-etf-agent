"""Alpaca adapter (alpaca-py). PAPER by default; LIVE only when explicitly requested.

Reliability features:
  * deterministic client_order_id per (session, symbol, side) -> re-running the same day
    can never double-submit (Alpaca rejects duplicate ids; we then look the order up)
  * retries with backoff on transient errors (429 / 5xx / network)
  * every order is polled until filled / cancelled / rejected; unfilled orders are
    cancelled after `order_timeout_sec` so nothing is left working unattended
  * fractional, notional (dollar) market orders, time-in-force DAY
"""
from __future__ import annotations

import logging
import time
from datetime import date

from ..risk.engine import Order, PortfolioSnapshot
from .retry import with_retries

log = logging.getLogger("trading_agent.alpaca")

FINAL = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True, session: date | None = None,
                 order_timeout_sec: int = 90, max_retries: int = 4):
        from alpaca.trading.client import TradingClient

        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set (see .env.example)")
        self.paper = paper
        self.client = TradingClient(api_key, secret_key, paper=paper)
        self.session = session or date.today()
        self.timeout = order_timeout_sec
        self._retry = with_retries(max_retries)
        log.info("Connected to Alpaca (%s)", "PAPER" if paper else "LIVE - REAL MONEY")

    def _call(self, fn, *a, **kw):
        return self._retry(fn)(*a, **kw)

    # ------------------------------------------------------------------ account
    def account(self):
        return self._call(self.client.get_account)

    def clock(self):
        return self._call(self.client.get_clock)

    def previous_session(self, today: date) -> date:
        from datetime import timedelta

        from alpaca.trading.requests import GetCalendarRequest

        cal = self._call(self.client.get_calendar,
                         GetCalendarRequest(start=today - timedelta(days=10), end=today - timedelta(days=1)))
        return max(c.date for c in cal)

    def snapshot(self) -> PortfolioSnapshot:
        acct = self.account()
        positions = self._call(self.client.get_all_positions)
        cash = float(acct.cash)
        nmbp = getattr(acct, "non_marginable_buying_power", None)
        if nmbp is not None:
            cash = min(cash, float(nmbp))      # cash only: never use margin buying power
        return PortfolioSnapshot(
            equity=float(acct.equity),
            cash=max(cash, 0.0),
            positions={p.symbol: float(p.market_value) for p in positions if float(p.qty) > 0},
            last_equity=float(acct.last_equity) if acct.last_equity else None,
            broker_daytrade_count=int(acct.daytrade_count or 0),
        )

    def peak_equity(self) -> float | None:
        """Highest daily equity in the last year (used if local risk state was lost)."""
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest

            h = self._call(self.client.get_portfolio_history,
                           GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
            vals = [v for v in (h.equity or []) if v]
            return float(max(vals)) if vals else None
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read portfolio history: %s", exc)
            return None

    def open_orders(self):
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return self._call(self.client.get_orders, GetOrdersRequest(status=QueryOrderStatus.OPEN))

    # ------------------------------------------------------------------ orders
    def client_order_id(self, order: Order) -> str:
        return f"ta-{self.session.isoformat()}-{order.symbol}-{order.side}"

    def submit(self, order: Order) -> dict:
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        cid = self.client_order_id(order)
        existing = self._find_by_client_id(cid)
        if existing is not None:
            log.info("Order %s already exists (status %s); not resubmitting", cid, existing.status)
            return self._wait(existing, cid)

        if any(o.symbol == order.symbol for o in self.open_orders()):
            return {"status": "skipped", "reason": f"{order.symbol} already has an open order",
                    "client_order_id": cid}
        try:
            if order.close_position:
                sent = self._call(self.client.close_position, order.symbol)
            else:
                req = MarketOrderRequest(symbol=order.symbol, notional=round(order.notional, 2),
                                         side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                                         time_in_force=TimeInForce.DAY, client_order_id=cid)
                sent = self._call(self.client.submit_order, req)
        except APIError as exc:
            if "client_order_id" in str(exc) and "unique" in str(exc).lower():
                existing = self._find_by_client_id(cid)
                if existing is not None:
                    return self._wait(existing, cid)
            log.error("Order %s %s $%.2f failed: %s", order.side, order.symbol, order.notional, exc)
            return {"status": "error", "reason": str(exc), "client_order_id": cid}
        log.info("Submitted %s %s $%.2f (id %s)", order.side, order.symbol, order.notional, sent.id)
        return self._wait(sent, cid)

    def _find_by_client_id(self, cid: str):
        try:
            return self.client.get_order_by_client_id(cid)
        except Exception:  # noqa: BLE001 - 404 when it does not exist
            return None

    def _wait(self, order, cid: str) -> dict:
        """Poll until the order reaches a final state; cancel if it times out."""
        deadline = time.time() + self.timeout
        o = order
        while True:
            status = getattr(o.status, "value", str(o.status))
            if status in FINAL:
                break
            if time.time() > deadline:
                log.warning("Order %s not filled after %ss; cancelling", o.id, self.timeout)
                try:
                    self._call(self.client.cancel_order_by_id, o.id)
                except Exception as exc:  # noqa: BLE001
                    log.error("Cancel failed for %s: %s", o.id, exc)
                o = self._call(self.client.get_order_by_id, o.id)
                status = getattr(o.status, "value", str(o.status))
                break
            time.sleep(2)
            o = self._call(self.client.get_order_by_id, o.id)
        qty = float(o.filled_qty or 0)
        px = float(o.filled_avg_price) if o.filled_avg_price else None
        res = {"status": status, "symbol": o.symbol, "side": getattr(o.side, "value", str(o.side)),
               "qty": qty, "price": px, "notional": qty * px if px else 0.0,
               "broker_order_id": str(o.id), "client_order_id": cid}
        log.info("Order %s -> %s (qty %.6f @ %s)", cid, status, qty, px)
        if status == "partially_filled" or (status == "canceled" and qty > 0):
            res["status"] = "filled"
            res["partial"] = True
        return res
