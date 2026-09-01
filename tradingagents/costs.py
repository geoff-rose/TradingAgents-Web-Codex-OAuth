"""All-in round-trip cost, per instrument.

**Brokerage is the small half.** The user's broker charges $2 up to $20,000
and 0.01% above it -- 1bp per side, 2bp round trip on a standard parcel.
Every cost figure used before 2026-08-31 assumed 5bp of brokerage, which was
2.5x too high.

**The spread is the large half, and it varies twentyfold.** Measured
2026-08-31 on 59 days of 15-minute prints: A200 0.007%, CBA 0.006%, BHP
0.068%, WOW 0.106%, REA 0.250%, IPX 0.442%, PNV 0.515%, LRV 0.620%. A single
global cost constant is therefore wrong in both directions at once -- it
condemns an ETF strategy that clears its real 3bp hurdle, and passes a
small-cap strategy that never had a hope against 60bp.

**Estimator.** Roll (1984): absent information, consecutive trade-price changes
are negatively autocovariant purely from bid/ask bounce, so the effective
spread is 2*sqrt(-cov). Where the covariance comes out positive -- trending
prints dominating, common in liquid ETFs whose quotes barely move -- Roll
yields nothing and the ONE-TICK floor is used instead. That floor is a genuine
lower bound: you cannot cross a spread narrower than the minimum increment.

**Why cost is swept, not point-estimated.** Roll is an average over calm and
stressed conditions, and the days a signal fires are rarely the calm ones.
`round_trip_pct` returns the central estimate; callers should still sweep
around it, which is why `gates.cost_sweep` takes a vector.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

# $2 flat up to this parcel size, then a percentage of the value above it.
BROKERAGE_FLAT = 2.0
BROKERAGE_FLAT_LIMIT = 20_000.0
BROKERAGE_PCT_ABOVE = 0.01 / 100
DEFAULT_PARCEL = 20_000.0

# Used when a ticker has no measured spread and no price to derive a tick
# floor from. Deliberately pessimistic -- an unknown instrument is far more
# likely to be illiquid than not.
FALLBACK_SPREAD_PCT = 0.40

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spread_estimates (
    ticker      TEXT PRIMARY KEY,
    price       REAL,
    tick        REAL,
    roll_pct    REAL,          -- NULL when the covariance was positive
    spread_pct  REAL NOT NULL, -- roll, or the tick floor
    method      TEXT NOT NULL,
    n_prints    INTEGER,
    measured_at TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    from .mover_log import DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def tick_for(price: float) -> float:
    """ASX tick schedule, simplified to the three bands that matter."""
    if price < 0.10:
        return 0.001
    if price < 2.00:
        return 0.005
    return 0.010


def brokerage_pct(dollars: float = DEFAULT_PARCEL) -> float:
    """Round-trip brokerage as a percentage of the parcel."""
    if dollars <= 0:
        return 0.0
    one_way = (BROKERAGE_FLAT if dollars <= BROKERAGE_FLAT_LIMIT
               else BROKERAGE_FLAT + (dollars - BROKERAGE_FLAT_LIMIT) * BROKERAGE_PCT_ABOVE)
    return 2 * one_way / dollars * 100


def measure(tickers: Iterable[str], period: str = "59d",
            interval: str = "15m") -> dict[str, Any]:
    """Estimate and store the effective spread for each ticker."""
    import numpy as np
    import yfinance as yf

    from .yf_lock import YF_LOCK
    tickers = sorted({t.upper() for t in tickers if t})
    if not tickers:
        return {"measured": 0}
    syms = [t if t.endswith(".AX") or t.startswith("^") else f"{t}.AX" for t in tickers]
    with YF_LOCK:
        data = yf.download(syms, period=period, interval=interval, group_by="ticker",
                           auto_adjust=False, threads=True, progress=False)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_roll, n_floor, missing = 0, 0, []
    with _connect() as conn:
        for t, sym in zip(tickers, syms):
            try:
                df = data[sym].dropna(subset=["Close"])
            except (KeyError, TypeError):
                missing.append(t)
                continue
            if len(df) < 100:
                missing.append(t)
                continue
            px = float(df["Close"].median())
            tick = tick_for(px)
            r = np.log(df["Close"].astype(float)).diff().dropna().to_numpy()
            cov = float(np.cov(r[1:], r[:-1])[0, 1]) if len(r) > 30 else 1.0
            if cov < 0:
                roll = float(2 * np.sqrt(-cov) * 100)
                spread, method = roll, "roll"
                n_roll += 1
            else:
                roll, spread, method = None, tick / px * 100, "tick_floor"
                n_floor += 1
            conn.execute(
                "INSERT OR REPLACE INTO spread_estimates"
                " (ticker, price, tick, roll_pct, spread_pct, method, n_prints, measured_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (t, round(px, 4), tick, round(roll, 4) if roll else None,
                 round(spread, 4), method, int(len(df)), now))
        conn.commit()
    return {"measured": n_roll + n_floor, "by_roll": n_roll,
            "by_tick_floor": n_floor, "no_data": missing}


def spread_map(tickers: Iterable[str] | None = None) -> dict[str, float]:
    with _connect() as conn:
        if tickers:
            ts = sorted({t.upper() for t in tickers})
            q = ",".join("?" * len(ts))
            rows = conn.execute(
                f"SELECT ticker, spread_pct FROM spread_estimates WHERE ticker IN ({q})",
                ts).fetchall()
        else:
            rows = conn.execute("SELECT ticker, spread_pct FROM spread_estimates").fetchall()
    return {r["ticker"]: float(r["spread_pct"]) for r in rows}


def round_trip_pct(ticker: str | None = None, dollars: float = DEFAULT_PARCEL,
                   price: float | None = None,
                   spreads: dict[str, float] | None = None) -> float:
    """Brokerage plus one full spread (you cross it on the way in and out)."""
    b = brokerage_pct(dollars)
    s = None
    if ticker:
        m = spreads if spreads is not None else spread_map([ticker])
        s = m.get(ticker.upper())
    if s is None and price:
        s = tick_for(price) / price * 100
    if s is None:
        s = FALLBACK_SPREAD_PCT
    return round(b + s, 4)


def floor_fraction(tickers: Iterable[str]) -> float:
    """Share of these tickers whose spread is the TICK FLOOR rather than a
    measurement. A floor is a lower bound, so a strategy that only clears its
    hurdle on floor-derived costs has not actually been shown to clear it."""
    ts = sorted({t.upper() for t in tickers if t})
    if not ts:
        return 0.0
    with _connect() as conn:
        q = ",".join("?" * len(ts))
        rows = conn.execute(
            f"SELECT ticker, method FROM spread_estimates WHERE ticker IN ({q})", ts).fetchall()
    known = {r["ticker"]: r["method"] for r in rows}
    # An unmeasured ticker uses FALLBACK_SPREAD_PCT, which is pessimistic, not
    # a floor -- so it does not count against this.
    floored = sum(1 for t in ts if known.get(t) == "tick_floor")
    return round(floored / len(ts), 3)


def cost_vector(tickers: Iterable[str], prices: Iterable[float] | None = None,
                dollars: float = DEFAULT_PARCEL) -> list[float]:
    """Per-trade round-trip costs, for `gates.cost_sweep`."""
    tickers = list(tickers)
    prices = list(prices) if prices is not None else [None] * len(tickers)
    m = spread_map(tickers)
    return [round_trip_pct(t, dollars, p, spreads=m) for t, p in zip(tickers, prices)]


# ---------------------------------------------------------------------------
# Live bid/ask from IBKR -- the measurement Roll can only approximate
# ---------------------------------------------------------------------------

IBKR_HOST, IBKR_PORT = "127.0.0.1", 4002

_LIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS spread_snapshots (
    ticker      TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    bid         REAL,
    ask         REAL,
    mid         REAL,
    spread_pct  REAL,
    PRIMARY KEY (ticker, captured_at)
);
"""


def capture_live(tickers, client_id: int = 92, timeout: int = 20) -> dict[str, Any]:
    """Snapshot the real bid/ask from IBKR and store it.

    **Why this matters more than the Roll estimate.** Roll cannot resolve a
    spread when consecutive prints trend rather than bounce, which is exactly
    the case for the liquid ETFs -- VAS, A200, STW and CBA all fell back to
    the ONE-TICK FLOOR. That floor is a lower bound, and the VAS overnight
    result flips from beating buy-and-hold by 1.4pp at one tick to losing by
    1.1pp at two. A week of real quotes decides it; nothing else will.

    **Fails loudly.** The gateway can sit at a login dialog with the process
    healthy and no API port open -- which is how `asx-swing-sync` reported
    {"checked":0,"updated":0} for days while IBKR was unreachable. An
    unreachable gateway is an error here, never an empty success.
    """
    import math as _math

    from ib_async import IB, Stock

    def _bad(v) -> bool:
        """NaN is the trap here. ib_async returns nan for an unset quote, and
        `not nan` is False, `nan <= 0` is False, `nan < nan` is False -- so a
        naive guard passes every NaN through and stores an empty row while
        reporting success. Caught 2026-08-31 doing exactly that."""
        return v is None or not isinstance(v, (int, float)) or _math.isnan(v) or v <= 0

    tickers = sorted({t.upper() for t in tickers if t})
    if not tickers:
        return {"error": "no tickers given"}
    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=client_id, timeout=timeout)
    except Exception as exc:
        return {"error": f"IBKR gateway unreachable on {IBKR_HOST}:{IBKR_PORT} "
                         f"({type(exc).__name__}). Is it logged in? A running "
                         f"process is not enough -- check for LOGGED_OUT in "
                         f"`journalctl -u ibgateway`.",
                "captured": 0}
    # Delayed-frozen data. The account has no real-time ASX subscription
    # (error 354), but a spread does not need to be current to be
    # representative -- ask minus bid twenty minutes ago measures the same
    # book. 4 = delayed frozen, which also returns the last quote outside
    # trading hours instead of nothing.
    try:
        ib.reqMarketDataType(4)
    except Exception:
        pass
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, missing = [], []
    try:
        for t in tickers:
            try:
                c = Stock(t, "ASX", "AUD")
                ib.qualifyContracts(c)
                tk = ib.reqMktData(c, "", True, False)
                for _ in range(12):          # up to ~6s for a delayed quote
                    ib.sleep(0.5)
                    if not _bad(tk.bid) and not _bad(tk.ask):
                        break
                bid, ask = tk.bid, tk.ask
                if _bad(bid) or _bad(ask) or ask < bid:
                    missing.append(t)
                    continue
                mid = (bid + ask) / 2
                rows.append((t, now, float(bid), float(ask), round(mid, 4),
                             round((ask - bid) / mid * 100, 4)))
            except Exception:
                missing.append(t)
    finally:
        ib.disconnect()
    with _connect() as conn:
        conn.executescript(_LIVE_SCHEMA)
        conn.executemany(
            "INSERT OR REPLACE INTO spread_snapshots"
            " (ticker, captured_at, bid, ask, mid, spread_pct) VALUES (?,?,?,?,?,?)", rows)
        conn.commit()
    if not rows:
        return {"captured": 0, "no_quote": missing, "at": now,
                "error": "no usable bid/ask returned. Market closed, or the "
                         "account lacks even delayed ASX data."}
    return {"captured": len(rows), "no_quote": missing, "at": now}


def live_spread_map(min_snapshots: int = 3) -> dict[str, float]:
    """Median observed spread per ticker, once there are enough snapshots.

    Median rather than mean: a single capture during an auction or a halt is
    an outlier, not a wider market.
    """
    with _connect() as conn:
        conn.executescript(_LIVE_SCHEMA)
        rows = conn.execute(
            "SELECT ticker, spread_pct FROM spread_snapshots"
            " WHERE spread_pct IS NOT NULL ORDER BY ticker").fetchall()
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["ticker"], []).append(float(r["spread_pct"]))
    import statistics
    return {t: round(statistics.median(v), 4)
            for t, v in by.items() if len(v) >= min_snapshots}
