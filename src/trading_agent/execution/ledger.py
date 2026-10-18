"""SQLite ledger of runs, signals, orders and equity (+ CSV exports for spreadsheets)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (session TEXT, mode TEXT, status TEXT, started_at TEXT,
    finished_at TEXT, message TEXT, PRIMARY KEY (session, mode));
CREATE TABLE IF NOT EXISTS signals (session TEXT, mode TEXT, symbol TEXT, p_up REAL, mu REAL,
    agreement INTEGER, confidence REAL, long INTEGER, target_weight REAL,
    PRIMARY KEY (session, mode, symbol));
CREATE TABLE IF NOT EXISTS orders (ts TEXT, session TEXT, mode TEXT, symbol TEXT, side TEXT,
    notional REAL, qty REAL, price REAL, status TEXT, reason TEXT, client_order_id TEXT,
    broker_order_id TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT, session TEXT, mode TEXT, equity REAL, cash REAL,
    exposure REAL, drawdown REAL, halted INTEGER, day_trades INTEGER, note TEXT);
"""


class Ledger:
    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.db"
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    # ------------------------------------------------------------------ runs
    def run_status(self, session: str, mode: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT status FROM runs WHERE session=? AND mode=?", (session, mode)).fetchone()
        return row[0] if row else None

    def start_run(self, session: str, mode: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
                      (session, mode, "running", datetime.now().isoformat(), None, ""))

    def finish_run(self, session: str, mode: str, status: str, message: str = "") -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET status=?, finished_at=?, message=? WHERE session=? AND mode=?",
                      (status, datetime.now().isoformat(), message[:2000], session, mode))

    # ------------------------------------------------------------------ records
    def record_signals(self, session: str, mode: str, fc: pd.DataFrame, longs: dict, targets: dict):
        with self._conn() as c:
            for sym, row in fc.iterrows():
                c.execute("INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?)",
                          (session, mode, sym, float(row["p_up"]), float(row["mu"]),
                           int(row["agreement"]), float(row["confidence"]), int(longs.get(sym, False)),
                           float(targets.get(sym, 0.0))))

    def record_order(self, session: str, mode: str, order, res: dict):
        with self._conn() as c:
            c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (datetime.now().isoformat(), session, mode, order.symbol, order.side,
                       float(res.get("notional") or order.notional), float(res.get("qty") or 0),
                       res.get("price"), res.get("status"), res.get("reason", order.reason),
                       res.get("client_order_id"), res.get("broker_order_id")))

    def record_equity(self, session: str, mode: str, snap, drawdown: float, halted: bool,
                      day_trades: int, note: str = ""):
        expo = snap.gross / snap.equity if snap.equity > 0 else 0.0
        with self._conn() as c:
            c.execute("INSERT INTO equity VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (datetime.now().isoformat(), session, mode, snap.equity, snap.cash, expo,
                       drawdown, int(halted), day_trades, note))

    # ------------------------------------------------------------------ reads
    def table(self, name: str, mode: str | None = None) -> pd.DataFrame:
        q = f"SELECT * FROM {name}" + (" WHERE mode=?" if mode else "")
        with self._conn() as c:
            return pd.read_sql_query(q, c, params=(mode,) if mode else None)

    def last_longs(self, mode: str) -> dict[str, bool]:
        df = self.table("signals", mode)
        if df.empty:
            return {}
        last = df[df["session"] == df["session"].max()]
        return {r.symbol: bool(r.long) for r in last.itertuples()}

    def equity_history(self, mode: str) -> pd.Series:
        df = self.table("equity", mode)
        if df.empty:
            return pd.Series(dtype=float)
        return df.groupby("session")["equity"].last()

    def export_csv(self) -> None:
        for t in ("runs", "signals", "orders", "equity"):
            self.table(t).to_csv(self.dir / f"{t}.csv", index=False)
