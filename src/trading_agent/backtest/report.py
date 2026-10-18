"""Static, self-contained HTML report (charts embedded as base64)."""
from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from .metrics import format_table


def _img(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%">'


def _md_table_to_html(md: str) -> str:
    lines = [l for l in md.splitlines() if not l.startswith("|---")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html_report(path: Path, sections: list[tuple[str, dict]], images: list[Path],
                      trades: pd.DataFrame | None, notes: list[str]) -> Path:
    css = ("body{font-family:system-ui,Arial;max-width:1100px;margin:auto;padding:24px;color:#222}"
           "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:4px 10px;"
           "text-align:right}th{background:#f3f3f3}td:first-child{text-align:left}.note{color:#555}")
    parts = [f"<html><head><meta charset='utf-8'><title>Trading agent report</title><style>{css}</style>"
             "</head><body>", "<h1>Risk-managed ETF trading agent: backtest report</h1>",
             f"<p class='note'>Generated {datetime.now():%Y-%m-%d %H:%M}. Out-of-sample walk-forward "
             "results with slippage and spread; past performance does not predict future results.</p>"]
    for title, rows in sections:
        parts += [f"<h2>{html.escape(title)}</h2>", _md_table_to_html(format_table(rows))]
    for img in images:
        parts.append(_img(img))
    if notes:
        parts.append("<h2>Notes</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul>")
    if trades is not None and not trades.empty:
        t = trades.tail(40).copy()
        parts += ["<h2>Last 40 orders (agent, no RL)</h2>", t.to_html(index=False, float_format=lambda x: f"{x:.4f}")]
    parts.append("</body></html>")
    path.write_text("\n".join(parts))
    return path
