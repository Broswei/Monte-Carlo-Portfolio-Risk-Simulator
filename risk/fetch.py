"""Fetch historical prices based on chosen tickers - allows personalization, requires internet"""

from __future__ import annotations

import pandas as pd

def extract_close(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:

    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        level1 = set(data.columns.get_level_values(1))
        for field in ("Close", "Adj Close"):
            if field in level0:
                return data[field].copy()
            if field in level1:
                return data.xs(field, axis=1, level=1).copy()
        raise ValueError("Downloaded data contained no close prices.")
    
    for field in ("Close", "Adj Close"):
        if field in data.columns:
            out = data[[field]].copy()
            out.columns = [tickers[0]]
            return out
    raise ValueError("Downloaded data contained no close prices.")

def fetch_prices(
    tickers: list[str], 
    period: str = "2y", 
    interval: str = "1d", 
    start: str | None = None, 
    end: str | None = None
) -> pd.DataFrame:

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Install it with: pip install yfinance"
        ) from exc
    
    symbols = [t.strip().upper() for t in tickers if t and t.strip()]
    if len(symbols) < 2:
        raise ValueError("Enter at least two tickers.")
    
    common = dict(interval = interval, auto_adjust = True, progress = True)
    if start:
        data = yf.download(symbols, start = start, end = end, **common)
    else:
        data = yf.download(symbols, period=period, **common)

    if data is None or len(data) == 0:
        raise ValueError("No data returned — check the tickers and date range.")

    prices = extract_close(data, symbols)
    prices = prices.dropna(how="all").ffill().dropna()    

    cols = [s for s in symbols if s in prices.columns]
    prices = prices[cols]
    if prices.shape[1] < 2:
        raise ValueError(
            "Fewer than two tickers returned usable data. Check the symbols."
        )

    prices.index.name = "Date"
    return prices
    
