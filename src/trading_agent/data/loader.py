"""Daily OHLCV loader: yfinance (free) with a local CSV cache.

Returns adjusted prices (dividends/splits folded in) so returns are total returns.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

log = logging.getLogger("trading_agent.data")


@dataclass
class PriceData:
    open: pd.DataFrame   # index=date, columns=symbols
    close: pd.DataFrame

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def slice(self, end: pd.Timestamp) -> "PriceData":
        return PriceData(self.open.loc[:end], self.close.loc[:end])


def _cache_path(cache_dir: Path, symbols: list[str], start: str) -> Path:
    return cache_dir / f"prices_{'-'.join(sorted(symbols))}_{start}.csv"


def _download_yf(symbols: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbols, start=start, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")
    out = pd.concat({"open": df["Open"], "close": df["Close"]}, axis=1)
    return out


def load_prices(symbols: list[str], start: str, cache_dir: str | Path | None = None,
                refresh: bool = False, max_cache_age_hours: float = 12) -> PriceData:
    """Load daily open/close for `symbols` from `start` to the latest completed session."""
    symbols = [s.upper() for s in symbols]
    raw = None
    path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, symbols, start)
        if path.exists() and not refresh:
            age_h = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
            if age_h < max_cache_age_hours:
                raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    if raw is None:
        log.info("Downloading %s daily prices from %s via yfinance", symbols, start)
        raw = _download_yf(symbols, start)
        if path is not None:
            raw.to_csv(path)
    raw = raw.sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    opens = raw["open"][symbols].astype(float)
    closes = raw["close"][symbols].astype(float)
    # Drop today's partial bar if the download happened before the close settled
    # (New York time, so this is correct on a UTC cloud runner too).
    now_ny = datetime.now(NY)
    today = pd.Timestamp(now_ny.date())
    if closes.index[-1] >= today and now_ny.hour < 17:
        opens, closes = opens.iloc[:-1], closes.iloc[:-1]
    valid = closes.dropna(how="any").index
    return PriceData(opens.loc[valid].ffill(), closes.loc[valid])


def check_freshness(prices: PriceData, max_staleness_days: int, today: date | None = None) -> None:
    """Raise if the latest bar is too old (e.g. data vendor outage) -> safe failure, no trades."""
    today = today or datetime.now(NY).date()
    last = prices.close.index[-1].date()
    if (today - last) > timedelta(days=max_staleness_days):
        raise RuntimeError(f"Stale data: last bar {last}, today {today}. Refusing to trade.")


def load_prices_alpaca(symbols: list[str], start: str, api_key: str, secret_key: str) -> PriceData:
    """Fallback source: Alpaca's free IEX daily bars (split/dividend adjusted)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                           start=pd.Timestamp(start).to_pydatetime(), adjustment=Adjustment.ALL,
                           feed=DataFeed.IEX)
    df = client.get_stock_bars(req).df.reset_index()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    opens = df.pivot(index="date", columns="symbol", values="open")[symbols]
    closes = df.pivot(index="date", columns="symbol", values="close")[symbols]
    now_ny = datetime.now(NY)
    if closes.index[-1] >= pd.Timestamp(now_ny.date()) and now_ny.hour < 17:
        opens, closes = opens.iloc[:-1], closes.iloc[:-1]
    valid = closes.dropna(how="any").index
    return PriceData(opens.loc[valid].ffill(), closes.loc[valid])
