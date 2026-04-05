from typing import List
import pandas as pd
import yfinance as yf


def get_chart_history_data(symbol: str, period: str = "1d", interval: str = "1m"):
    ticker = yf.Ticker(symbol.upper())
    hist = ticker.history(period=period, interval=interval)

    points: List[dict] = []
    if not hist.empty:
        for idx, row in hist.iterrows():
            close_val = row.get("Close")
            if close_val is None or pd.isna(close_val):
                continue

            py_dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx

            if hasattr(py_dt, "isoformat"):
                time_iso = py_dt.isoformat()
            else:
                time_iso = str(py_dt)

            points.append({
                "time": time_iso,
                "price": float(close_val),
            })

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "points": points,
    }