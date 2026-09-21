"""Relative volume from completed hourly windows, compared with prior sessions."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

SYDNEY = ZoneInfo("Australia/Sydney")
BASELINE_SESSIONS = 20
MIN_BASELINE_SESSIONS = 10
QUOTE_DELAY_MINUTES = 20


def relative_volume(frame, now=None):
    now = (now or datetime.now(timezone.utc)).astimezone(SYDNEY)
    cutoff = now - timedelta(minutes=QUOTE_DELAY_MINUTES)
    if frame.empty:
        return None
    df = frame.copy()
    idx = df.index
    df.index = idx.tz_localize(SYDNEY) if idx.tz is None else idx.tz_convert(SYDNEY)
    df = df[df.index.hour.isin(range(10, 16))]
    df = df[df.index + pd.Timedelta(hours=1) <= cutoff]
    today = df[df.index.date == now.date()]
    if today.empty:
        return None
    hours = set(today.index.hour)
    totals = []
    for day, past in df[df.index.date < now.date()].groupby(df[df.index.date < now.date()].index.date):
        matched = past[past.index.hour.isin(hours)]
        if set(matched.index.hour) == hours and matched.Volume.notna().all():
            totals.append(float(matched.Volume.sum()))
    totals = totals[-BASELINE_SESSIONS:]
    baseline = pd.Series(totals, dtype=float).median()
    if len(totals) < MIN_BASELINE_SESSIONS or not baseline > 0 or today.Volume.isna().any():
        return None
    return {"ratio": float(today.Volume.sum()) / baseline,
            "sessions": len(totals), "through": str(today.index[-1] + pd.Timedelta(hours=1))}


def for_tickers(tickers, now=None):
    if not tickers:
        return {}
    import yfinance as yf
    from .yf_lock import YF_LOCK
    with YF_LOCK:
        raw = yf.download([f"{t}.AX" for t in tickers], period="60d", interval="1h",
                          group_by="ticker", auto_adjust=False, threads=True, progress=False)
    out = {}
    for ticker in tickers:
        try:
            value = relative_volume(raw[f"{ticker}.AX"].dropna(subset=["Close"]), now)
        except (KeyError, TypeError):
            continue
        if value:
            out[ticker] = value
    return out
