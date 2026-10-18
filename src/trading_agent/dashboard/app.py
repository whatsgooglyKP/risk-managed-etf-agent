"""Streamlit dashboard: `trading-agent dashboard` (or `streamlit run src/trading_agent/dashboard/app.py`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `streamlit run` from a checkout

from trading_agent.config import load_config, resolve  # noqa: E402
from trading_agent.execution.ledger import Ledger  # noqa: E402

st.set_page_config(page_title="Trading Agent", page_icon="📈", layout="wide")
cfg = load_config()
results = resolve(cfg["backtest"]["results_dir"])
state_dir = resolve(cfg["execution"]["state_dir"])
r = cfg["risk"]

st.title("📈 Risk-managed ETF trading agent")
view = st.sidebar.radio("View", ["Live / paper account", "Backtest"])
st.sidebar.markdown(f"**Universe:** {', '.join(cfg['universe'])}  \n**Hard floor:** ${r['hard_floor']:.0f} "
                    f"of ${r['initial_capital']:.0f}  \n**RL scaler:** {'on' if cfg['rl']['enabled'] else 'off'}")


def line(curves: dict[str, pd.Series], floor: float | None = None, log: bool = False):
    fig = go.Figure()
    for name, s in curves.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines"))
    if floor:
        fig.add_hline(y=floor, line_dash="dash", line_color="red", annotation_text="hard floor")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_type="log" if log else "linear", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")


def fmt_metrics(m: dict) -> pd.DataFrame:
    df = pd.DataFrame(m).T
    for c in ("total_return", "cagr", "ann_vol", "max_drawdown", "win_rate", "exposure"):
        if c in df:
            df[c] = df[c].astype(float).map(lambda v: f"{v:.1%}" if pd.notna(v) else "n/a")
    for c in ("sharpe", "sortino"):
        if c in df:
            df[c] = df[c].astype(float).map(lambda v: f"{v:.2f}")
    return df


if view == "Backtest":
    mpath = results / "metrics.json"
    if not mpath.exists():
        st.info("No backtest yet. Run `trading-agent backtest` first.")
        st.stop()
    m = json.loads(mpath.read_text())
    st.subheader("Full out-of-sample period")
    st.dataframe(fmt_metrics(m["full"]), width="stretch")
    eq = pd.read_csv(results / "equity_base.csv", index_col=0, parse_dates=True).iloc[:, 0]
    spy = pd.read_csv(results / "equity_spy.csv", index_col=0, parse_dates=True).iloc[:, 0]
    line({"Agent (no RL)": eq, "Buy & hold SPY": spy}, r["hard_floor"], log=True)
    if "rl_test" in m:
        st.subheader("RL held-out test period")
        st.dataframe(fmt_metrics(m["rl_test"]), width="stretch")
        a = pd.read_csv(results / "equity_base_test.csv", index_col=0, parse_dates=True).iloc[:, 0]
        b = pd.read_csv(results / "equity_rl_test.csv", index_col=0, parse_dates=True).iloc[:, 0]
        line({"Agent (no RL)": a, "Agent + RL": b}, r["hard_floor"])
    tpath = results / "trades_base.csv"
    if tpath.exists():
        st.subheader("Backtest orders (most recent first)")
        st.dataframe(pd.read_csv(tpath).iloc[::-1].head(200), width="stretch")
else:
    led = Ledger(state_dir)
    mode = st.sidebar.selectbox("Account", ["paper", "live"])
    eq = led.table("equity", mode)
    kill = resolve(r["kill_file"])
    risk_state = json.loads((state_dir / "risk_state.json").read_text()) if (state_dir / "risk_state.json").exists() else {}
    c1, c2, c3, c4 = st.columns(4)
    last_eq = float(eq["equity"].iloc[-1]) if not eq.empty else None
    c1.metric("Equity", f"${last_eq:.2f}" if last_eq else "-")
    c2.metric("Distance to floor", f"${last_eq - r['hard_floor']:.2f}" if last_eq else "-")
    halted = kill.exists() or risk_state.get("halted", False)
    c3.metric("Status", "HALTED" if halted else "Active")
    c4.metric("Day trades (5d)", int(eq["day_trades"].iloc[-1]) if not eq.empty else 0,
              help=f"Limit {r['pdt_max_day_trades']} while equity < ${r['pdt_equity_threshold']:,}")
    if halted:
        st.error(f"Trading halted: {risk_state.get('halt_reason') or kill.read_text()}")
    if eq.empty:
        st.info("No live/paper runs yet. Run `trading-agent daily` during market hours.")
    else:
        s = eq.groupby("session")["equity"].last()
        s.index = pd.to_datetime(s.index)
        line({"Equity": s}, r["hard_floor"])
    st.subheader("Latest signals")
    sig = led.table("signals", mode)
    st.dataframe(sig[sig["session"] == sig["session"].max()] if not sig.empty else sig, width="stretch")
    st.subheader("Orders")
    st.dataframe(led.table("orders", mode).iloc[::-1], width="stretch")
    st.subheader("Runs")
    st.dataframe(led.table("runs", mode).iloc[::-1], width="stretch")
    with st.expander("Risk engine state"):
        st.json(risk_state)
