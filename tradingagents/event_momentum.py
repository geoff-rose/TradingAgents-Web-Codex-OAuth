"""Does a move backed by NEWS continue, where an unexplained move reverts?
(2026-08-22, testing the user's hypothesis.)

**The hypothesis.** "A company gets FDA clearance and gaps up — that can drive
a few days of gains. I've seen a $0.30 stock go to $3 or $4 in a few days."
Against the measured baseline that gaps *fade* (gap >+3% is followed by a mean
-0.141% rest-of-day), the claim is that news-driven moves are the exception:
they continue because real information is being repriced over days, not hours.

**Why a volume proxy instead of the announcements feed.** The collector's
`announcements` table holds only **3 days** of history (it started 2026-08-19),
so `strategies.fetch_events()` — though wired and correct — has nothing to
backtest against yet. A **volume-confirmed move** is the standard proxy: a
stock moving 8% on 3x its normal volume is almost certainly repricing news; the
same 8% on ordinary volume is drift or thin-market noise. That tests the actual
hypothesis on ten years of data today instead of waiting months, and the real
feed can validate the proxy later as it accumulates.

**Universe is deliberately WIDER than the 26-ticker model universe.** The
tick-granularity finding killed *intraday band trading* on microcaps — it says
nothing about multi-day event-driven moves, and a 30c stock running to $3 has
ample room in ticks. Excluding speculative names here would exclude exactly the
cases the hypothesis is about. Only basic liquidity and history filters apply.

**Entry is the NEXT OPEN**, never the event day's close: you observe the whole
event day, then act. That is executable and carries no path assumption (see the
same-bar artifact note in the asx-dashboard skill for why this matters).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.screener import load_candidates

FORWARD_HORIZONS = (1, 2, 3, 5, 10)
MOVE_THRESHOLD_PCT = 5.0       # what counts as a "big move" worth explaining
KEEP_THRESHOLD_PCT = 3.0       # rows retained from the raw scan (keeps memory sane)
VOLUME_WINDOW = 50
MIN_HISTORY = 300
MIN_MEDIAN_TURNOVER = 200_000  # a move on $20k of turnover is not a repricing


def scan_events(tickers: list[str], period: str = "10y", batch: int = 40) -> pd.DataFrame:
    """One row per ticker-day where |day move| >= KEEP_THRESHOLD_PCT, with the
    volume signature and forward returns from the next open.

    Processed in batches and reduced to event rows immediately -- holding 500
    tickers x 10y of OHLCV at once would be gigabytes on a 2GB box.
    """
    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        data = yf.download([f"{t}.AX" for t in chunk], period=period, interval="1d",
                           group_by="ticker", auto_adjust=True, threads=True, progress=False)
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df) < MIN_HISTORY:
                continue
            close, vol, op = df["Close"], df["Volume"], df["Open"]
            if float((close * vol).tail(250).median() or 0) < MIN_MEDIAN_TURNOVER:
                continue

            day_move = close.pct_change() * 100
            gap = (op - close.shift(1)) / close.shift(1) * 100
            vol_ratio = vol / vol.rolling(VOLUME_WINDOW).median()

            nxt_open = op.shift(-1)
            row = pd.DataFrame({"ticker": t, "day_move_pct": day_move, "gap_pct": gap,
                                "vol_ratio": vol_ratio, "next_open": nxt_open})
            for h in FORWARD_HORIZONS:
                row[f"fwd_{h}d_pct"] = (close.shift(-h) - nxt_open) / nxt_open * 100
            row = row[day_move.abs() >= KEEP_THRESHOLD_PCT]
            frames.append(row.dropna(subset=["vol_ratio", "next_open", "fwd_1d_pct"]))
    return pd.concat(frames) if frames else pd.DataFrame()


def _day_ci(frame: pd.DataFrame, col: str, n: int = 3_000, seed: int = 17) -> dict[str, float]:
    """95% CI resampling whole DAYS. On a strong market day hundreds of names
    move together, so ticker-days are nowhere near independent."""
    if frame.empty:
        return {}
    by_day = frame.groupby(frame.index)[col].mean().to_numpy()
    by_day = by_day[~np.isnan(by_day)]
    if len(by_day) < 5:
        return {}
    rng = np.random.default_rng(seed)
    means = by_day[rng.integers(0, len(by_day), (n, len(by_day)))].mean(axis=1)
    return {"ci_lo": round(float(np.percentile(means, 2.5)), 3),
            "ci_hi": round(float(np.percentile(means, 97.5)), 3),
            "n_days": int(len(by_day))}


VOLUME_BUCKETS = [(0, 1.5, "normal vol (no news)"),
                  (1.5, 3.0, "elevated 1.5-3x"),
                  (3.0, 6.0, "high 3-6x"),
                  (6.0, np.inf, "extreme >6x (news)")]


def analyse(events: pd.DataFrame, direction: str = "up",
            move_threshold: float = MOVE_THRESHOLD_PCT) -> pd.DataFrame:
    """Forward returns by volume bucket for big up (or down) days.

    The comparison that answers the hypothesis is **across rows of this
    table**, not any single row: if news-driven moves continue, the high-volume
    buckets should show progressively better forward returns than the
    normal-volume bucket. If every bucket looks the same, volume (and by proxy
    news) carries no continuation information.
    """
    sub = (events[events.day_move_pct >= move_threshold] if direction == "up"
           else events[events.day_move_pct <= -move_threshold])
    rows = []
    for lo, hi, label in VOLUME_BUCKETS:
        b = sub[(sub.vol_ratio >= lo) & (sub.vol_ratio < hi)]
        if len(b) < 30:
            rows.append({"volume_bucket": label, "n": len(b)})
            continue
        entry = {"volume_bucket": label, "n": len(b),
                 "median_vol_ratio": round(float(b.vol_ratio.median()), 2),
                 "avg_day_move": round(float(b.day_move_pct.mean()), 2)}
        for h in FORWARD_HORIZONS:
            col = f"fwd_{h}d_pct"
            entry[f"fwd{h}d"] = round(float(b[col].mean()), 3)
            entry[f"fwd{h}d_pos%"] = round(100 * float((b[col] > 0).mean()), 1)
        entry.update({f"d1_{k}": v for k, v in _day_ci(b, "fwd_1d_pct").items()})
        entry.update({f"d5_{k}": v for k, v in _day_ci(b, "fwd_5d_pct").items()})
        rows.append(entry)
    return pd.DataFrame(rows)


MOVE_BANDS = [(5, 8, "+5-8%"), (8, 12, "+8-12%"), (12, 20, "+12-20%"), (20, np.inf, "+20%+")]


def analyse_controlled(events: pd.DataFrame, direction: str = "up",
                       horizon: int = 5) -> pd.DataFrame:
    """Volume gradient WITHIN each move-size band -- the control that decides
    whether the headline result means anything.

    In the raw bucketing, the extreme-volume group also has a much larger
    average day move (+15.0% vs +8.1%), so "extreme volume" could simply be
    proxying "huge move" and the apparent news effect would be a size effect
    wearing a disguise. Holding move size roughly fixed and varying only
    volume separates the two. If the volume gradient survives inside each
    band, volume carries information of its own; if it collapses, it never did.
    """
    col = f"fwd_{horizon}d_pct"
    sign = 1 if direction == "up" else -1
    rows = []
    for lo, hi, band in MOVE_BANDS:
        m = events.day_move_pct * sign
        band_rows = events[(m >= lo) & (m < hi)]
        entry = {"move_band": band if direction == "up" else band.replace("+", "-")}
        for vlo, vhi, vlabel in VOLUME_BUCKETS:
            b = band_rows[(band_rows.vol_ratio >= vlo) & (band_rows.vol_ratio < vhi)]
            short = vlabel.split(" ")[0]
            entry[f"n_{short}"] = len(b)
            entry[short] = round(float(b[col].mean()), 3) if len(b) >= 30 else None
        rows.append(entry)
    return pd.DataFrame(rows)


def print_analysis(events: pd.DataFrame, move_threshold: float = MOVE_THRESHOLD_PCT) -> None:
    print("=" * 120)
    print(f"NEWS-PROXY CONTINUATION -- forward returns from the NEXT OPEN after a "
          f"+/-{move_threshold}% day, bucketed by volume")
    print("=" * 120)
    print(f"events scanned: {len(events):,} ticker-days  |  tickers: {events.ticker.nunique()}  "
          f"|  {events.index.min().date()} -> {events.index.max().date()}")
    for direction, label in (("up", f"BIG UP DAYS (>= +{move_threshold}%)"),
                             ("down", f"BIG DOWN DAYS (<= -{move_threshold}%)")):
        t = analyse(events, direction, move_threshold)
        print()
        print(f"--- {label} " + "-" * (100 - len(label)))
        cols = ["volume_bucket", "n", "median_vol_ratio", "avg_day_move",
                "fwd1d", "fwd1d_pos%", "fwd3d", "fwd5d", "fwd5d_pos%", "fwd10d",
                "d5_ci_lo", "d5_ci_hi"]
        cols = [c for c in cols if c in t.columns]
        print(t[cols].to_string(index=False))
    for direction in ("up", "down"):
        for horizon in (5, 10):
            print()
            print(f"--- CONTROL: volume gradient WITHIN move-size bands, {direction} days, "
                  f"fwd{horizon}d " + "-" * 30)
            print(analyse_controlled(events, direction, horizon).to_string(index=False))
    print()
    print("  Read ACROSS volume buckets: if news drives continuation, higher-volume rows")
    print("  should show better forward returns than the normal-volume row. Equal rows")
    print("  mean volume (and by proxy news) carries no continuation information.")


CACHE = "/tmp/claude-0/-root/5bfdc892-ffc0-4d21-84c9-6eabae1d07b5/scratchpad/events.pkl"


if __name__ == "__main__":
    import os
    if os.path.exists(CACHE):
        ev = pd.read_pickle(CACHE)
        print(f"loaded {len(ev):,} cached event rows", flush=True)
    else:
        cands = load_candidates(500)
        tickers = [c.ticker for c in cands]
        print(f"scanning {len(tickers)} tickers...", flush=True)
        ev = scan_events(tickers)
        try:
            ev.to_pickle(CACHE)
        except Exception as exc:
            print(f"(cache write skipped: {exc})", flush=True)
    print_analysis(ev)
