"""Market snapshot for the ASX dashboard: indices, FX, commodities, and the US
10-year Treasury yield. Cached in-process -- Yahoo rate-limits, and none of
this needs fetching more than once every few minutes.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yfinance as yf

SNAPSHOT_TICKERS = [
    ("^AXJO", "ASX 200"),
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq"),
    ("^DJI", "Dow Jones"),
    # US futures, added 2026-09-07 at the user's request. The cash indices
    # above are frozen from 06:00 Sydney until the US opens at 23:30, so
    # during the whole ASX session they say nothing about what is happening
    # now; the futures trade nearly around the clock and are the live
    # overnight read, the same role SPI 200 plays for the ASX.
    #
    # NQ=F tracks the Nasdaq **100**, while ^IXIC above is the Nasdaq
    # **Composite** -- different indices at different levels (29,565 vs
    # 26,507 when this was added), so they are labelled apart. Reading one
    # against the other as a level or a basis is meaningless.
    #
    # Yahoo's previous_close for these is the contract's OWN prior
    # settlement, so change_pct is the genuine overnight move. That is the
    # distinction the SPI 200 work got wrong first time round: subtracting a
    # cash close from a futures level mixes in a persistent basis that has
    # nothing to do with the overnight move.
    ("ES=F", "S&P 500 (futures)"),
    ("NQ=F", "Nasdaq 100 (futures)"),
    ("AUDUSD=X", "AUD/USD"),
    ("AUDEUR=X", "AUD/EUR"),
    ("USDEUR=X", "USD/EUR"),
    ("BTC-USD", "Bitcoin"),
    ("GC=F", "Gold"),
    ("CL=F", "Crude Oil (WTI)"),
    ("^TNX", "US 10Y Yield"),
]

# SPI 200 futures has no free Yahoo Finance ticker (every guessed symbol
# 404s, Yahoo's search doesn't index it either -- confirmed 2026-08-21).
# Barchart's continuous-contract page embeds the live last price directly in
# the HTML (no API key, auto-resolves to front month) -- confirmed working
# 2026-08-21. Reported as "vs previous ASX 200 close", not vs the future's own
# prior settlement: that's the number that means something (the implied
# overnight move ahead of the next ASX session), per the user's own framing.
#
# The user only checks this once a day (~9am, to see how the overnight
# session went ahead of the ASX open) -- added 2026-08-21, so scraping
# Barchart on every 5-min dashboard poll was pure waste. Refreshed instead by
# a daily systemd timer (asx-spi200-refresh.timer, 08:00 Australia/Sydney,
# an hour ahead of the user's check) hitting POST /api/markets/spi200/refresh,
# persisted to disk so it survives a service restart. get_snapshot() just
# reads the persisted value -- falls back to a live scrape only if that file
# is missing/stale (e.g. right after the feature was deployed, or the timer
# hasn't fired yet), so the number is never simply absent.
_SPI200_URL = "https://www.barchart.com/futures/quotes/AP*0"
_SPI200_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
# Trailing letter (e.g. "8,988.0s") is a Barchart status suffix (seen live
# 2026-08-21, wasn't there when this was first built the same day) -- allow
# and discard it rather than requiring an exact match right up to the quote.
_SPI200_LAST_RE = re.compile(r'"lastPrice"\s*:\s*"([\d,.]+)[a-zA-Z]?"')
# Barchart's own priceChange -- the futures contract's change vs *its own*
# previous settlement, not vs the cash index. This is what "expected open"
# actually needs (see refresh_spi200's docstring for why).
_SPI200_CHANGE_RE = re.compile(r'"priceChange"\s*:\s*"([+-]?[\d,.]+)"')
_SPI200_CACHE_PATH = Path("/opt/tradingagents/data/spi200_cache.json")
_SPI200_STALE_SECONDS = 20 * 3600  # if the daily timer hasn't fired in 20h, self-heal with a live scrape

# Which snapshot rows are futures. The dashboard renders these as their own
# section: they answer a different question from the cash rows -- what is
# happening right now, while Sydney trades and the underlying markets are shut
# -- so mixing them into one grid invited reading a futures level against a
# cash level, which means nothing.
FUTURES_SYMBOLS = {"ES=F", "NQ=F", "AP*0", "AP*0-CHG"}

_SNAPSHOT_TTL = 300  # 5 minutes
_snapshot_cache: dict[str, Any] = {"ts": 0.0, "data": None}

# FRED's daily-Treasury-yield series, one per maturity -- no API key needed
# for the plain CSV export. `months` gives the x-axis position (true linear
# time spacing, matching how yield curve charts are conventionally drawn).
CURVE_SERIES = [
    ("DGS1MO", "1M", 1), ("DGS3MO", "3M", 3), ("DGS6MO", "6M", 6),
    ("DGS1", "1Y", 12), ("DGS2", "2Y", 24), ("DGS3", "3Y", 36),
    ("DGS5", "5Y", 60), ("DGS7", "7Y", 84), ("DGS10", "10Y", 120),
    ("DGS20", "20Y", 240), ("DGS30", "30Y", 360),
]

_CURVE_TTL = 21600  # 6 hours -- FRED itself only publishes once a day
_curve_cache: dict[str, Any] = {"ts": 0.0, "data": None}


def _spi200_scrape() -> tuple[float | None, float | None]:
    """Returns (last, change_pts). `change_pts` is Barchart's own
    priceChange field -- the futures' change vs its own previous settlement,
    e.g. "+12.0" -- distinct from `last` minus the cash index's previous
    close (those two aren't at the same level; SPI trades at a persistent
    basis to XJO, which isn't part of the overnight move)."""
    try:
        r = httpx.get(_SPI200_URL, headers={"User-Agent": _SPI200_UA}, timeout=10)
        r.raise_for_status()
        m_last = _SPI200_LAST_RE.search(r.text)
        m_chg = _SPI200_CHANGE_RE.search(r.text)
        last = float(m_last.group(1).replace(",", "")) if m_last else None
        change_pts = float(m_chg.group(1).replace(",", "")) if m_chg else None
        return last, change_pts
    except Exception:
        return None, None


# IBKR carries the contract Barchart was scraped for: symbol "SPI" on SNFE,
# front month resolved by expiry (APU6 etc.). Delayed data is enough -- this is
# read once a day before the open, and `close` is the contract's own previous
# settlement, which is exactly the basis the "expected open" number needs.
_SPI200_IBKR_CLIENT_ID = 93


def _spi200_ibkr() -> tuple[float | None, float | None]:
    """(last, change_pts) from IB Gateway, or (None, None) if unavailable."""
    import math

    from ib_async import IB, Future

    def bad(v) -> bool:
        return (v is None or not isinstance(v, (int, float))
                or math.isnan(v) or v <= 0)

    ib = IB()
    try:
        ib.connect("127.0.0.1", 4002, clientId=_SPI200_IBKR_CLIENT_ID, timeout=15)
        details = ib.reqContractDetails(
            Future(symbol="SPI", exchange="SNFE", currency="AUD"))
        if not details:
            return None, None
        # front month = nearest expiry still ahead of us
        front = sorted(details,
                       key=lambda d: d.contract.lastTradeDateOrContractMonth)[0].contract
        ib.reqMarketDataType(4)          # delayed-frozen; no real-time ASX subscription
        tk = ib.reqMktData(front, "", True, False)
        for _ in range(24):
            ib.sleep(0.5)
            if not bad(tk.last) and not bad(tk.close):
                break
        last = None if bad(tk.last) else float(tk.last)
        prev = None if bad(tk.close) else float(tk.close)
        change = round(last - prev, 1) if last is not None and prev is not None else None
        return last, change
    except Exception:
        return None, None
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Futures via IB Gateway
# ---------------------------------------------------------------------------
# Yahoo is not usable for these. Measured 2026-09-07 05:23Z, mid-Globex
# session: Yahoo reported ES=F at 7722 with marketState CLOSED, which is
# exactly IBKR's PREVIOUS SETTLEMENT -- it was not tracking the overnight
# session at all, while IBKR delayed showed 7712 against investing.com's live
# 7707.50. Yahoo also stamps these `exchangeDataDelayedBy: 10`, and the
# snapshot cache and browser poll add five minutes each, so the panel could be
# twenty minutes behind and wrong about the session on top.
#
# Fetched by a timer into a cache file rather than inline: a gateway connect
# plus three quotes takes ~7s, which would be added to every dashboard load,
# and refreshing per request would mean ~288 gateway connections a day.
FUTURES_CONTRACTS = [
    # (symbol, exchange, currency, snapshot symbol it replaces)
    ("ES", "CME", "USD", "ES=F"),
    ("NQ", "CME", "USD", "NQ=F"),
    ("SPI", "SNFE", "AUD", "AP*0"),
]
_FUTURES_CACHE_PATH = Path("/opt/tradingagents/data/futures_cache.json")
# Older than this and the panel says so rather than showing a stale number as
# if it were current. The refresh timer runs every 2 minutes.
_FUTURES_STALE_SECONDS = 600
_FUTURES_IBKR_CLIENT_ID = 88


def refresh_futures() -> dict[str, Any]:
    """Fetch every futures contract in ONE gateway session and persist them.

    Returns `error` when the gateway is unreachable or returns nothing, so the
    caller can fail loudly and the dashboard can tell the user rather than
    quietly showing yesterday's number.
    """
    import math

    from ib_async import IB, Future

    def bad(v) -> bool:
        return (v is None or not isinstance(v, (int, float))
                or math.isnan(v) or v <= 0)

    out: dict[str, Any] = {}
    err: str | None = None
    ib = IB()
    try:
        ib.connect("127.0.0.1", 4002, clientId=_FUTURES_IBKR_CLIENT_ID, timeout=15)
        ib.reqMarketDataType(4)          # delayed; no market-data subscription
        for sym, exch, ccy, snap_sym in FUTURES_CONTRACTS:
            try:
                details = ib.reqContractDetails(
                    Future(symbol=sym, exchange=exch, currency=ccy))
                if not details:
                    continue
                front = sorted(
                    details,
                    key=lambda d: d.contract.lastTradeDateOrContractMonth)[0].contract
                tk = ib.reqMktData(front, "", True, False)
                for _ in range(20):
                    ib.sleep(0.5)
                    if not bad(tk.last) and not bad(tk.close):
                        break
                last = None if bad(tk.last) else float(tk.last)
                prev = None if bad(tk.close) else float(tk.close)
                ib.cancelMktData(front)
                if last is None:
                    continue
                out[snap_sym] = {
                    "last": last, "previous_close": prev,
                    "change_pct": ((last - prev) / prev * 100
                                   if prev else None),
                    "change_pts": round(last - prev, 1) if prev else None,
                    "contract": front.localSymbol,
                }
            except Exception as exc:
                err = f"{sym}: {type(exc).__name__}"
    except Exception as exc:
        err = (f"IB Gateway unreachable on 127.0.0.1:4002 ({type(exc).__name__}). "
               f"A running process is not enough -- check for LOGGED_OUT in "
               f"`journalctl -u ibgateway`.")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    data = {"items": out, "fetched_at": time.time(),
            "error": err if not out else (err or None)}
    _FUTURES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FUTURES_CACHE_PATH.write_text(json.dumps(data))
    return data


def _futures_cached() -> dict[str, Any]:
    try:
        return json.loads(_FUTURES_CACHE_PATH.read_text())
    except Exception:
        return {"items": {}, "fetched_at": 0.0,
                "error": "no futures cache yet -- asx-futures-refresh.timer "
                         "has not produced one"}


def refresh_spi200() -> dict[str, Any]:
    """Refresh the SPI 200 quote and persist it. Called once a day by
    asx-spi200-refresh.timer -- not on the dashboard's own poll cycle.

    **IBKR first, Barchart second.** Barchart began returning an empty HTTP 202
    (bot blocking) and the scrape silently produced nulls, so the dashboard row
    read "--" while the timer reported success -- found 2026-09-07. IB Gateway
    is already running for the paper books and carries the same contract, so it
    is both more reliable and one less thing to scrape. The scrape is kept as a
    fallback in case IB Gateway is logged out.

    Returns `source` and, when both fail, `error` -- so the caller can fail
    loudly instead of persisting nulls that look like a quiet market.
    """
    last, change_pts = _spi200_ibkr()
    source = "ibkr"
    if last is None:
        last, change_pts = _spi200_scrape()
        source = "barchart"
    data = {"last": last, "change_pts": change_pts, "fetched_at": time.time(),
            "source": source if last is not None else None}
    if last is None:
        data["error"] = ("SPI 200 unavailable: IB Gateway returned no quote and "
                         "the Barchart scrape returned nothing (it now answers "
                         "with an empty HTTP 202)")
    _SPI200_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SPI200_CACHE_PATH.write_text(json.dumps(data))
    return data


def _spi200_cached() -> dict[str, Any]:
    """Prefer the daily timer's persisted value; only fall back to a live
    scrape if it's missing or the timer hasn't run in a while (self-healing,
    e.g. right after this feature was deployed)."""
    try:
        cached = json.loads(_SPI200_CACHE_PATH.read_text())
        if time.time() - cached["fetched_at"] < _SPI200_STALE_SECONDS:
            return cached
    except Exception:
        pass
    return refresh_spi200()


def _spi200_last() -> float | None:
    return _spi200_cached()["last"]


def get_snapshot() -> dict[str, Any]:
    now = time.monotonic()
    if _snapshot_cache["data"] is not None and now - _snapshot_cache["ts"] < _SNAPSHOT_TTL:
        return _snapshot_cache["data"]

    symbols = [sym for sym, _ in SNAPSHOT_TICKERS]
    tickers = yf.Tickers(" ".join(symbols))
    items = []
    axjo_prev = None
    for sym, label in SNAPSHOT_TICKERS:
        entry = {"symbol": sym, "label": label, "last": None,
                 "previous_close": None, "change_pct": None,
                 "group": "futures" if sym in FUTURES_SYMBOLS else "markets"}
        try:
            info = tickers.tickers[sym].fast_info
            last, prev = info.last_price, info.previous_close
            entry["last"] = last
            entry["previous_close"] = prev
            entry["change_pct"] = (last - prev) / prev * 100 if prev else None
            if sym == "^AXJO":
                axjo_prev = prev
        except Exception:
            pass
        items.append(entry)

    # Override the Yahoo rows with IBKR where the cache is fresh. Yahoo stays
    # as the fallback so the panel degrades to a stale-but-labelled number
    # rather than to nothing.
    fut = _futures_cached()
    fut_age = time.time() - (fut.get("fetched_at") or 0)
    fut_fresh = bool(fut.get("items")) and fut_age < _FUTURES_STALE_SECONDS
    if fut_fresh:
        for entry in items:
            hit = fut["items"].get(entry["symbol"])
            if hit:
                entry.update({k: hit[k] for k in
                              ("last", "previous_close", "change_pct")})
                entry["source"] = "ibkr"
                entry["contract"] = hit.get("contract")
    else:
        for entry in items:
            if entry.get("group") == "futures":
                entry["source"] = "yahoo-fallback"

    spi_hit = fut["items"].get("AP*0") if fut_fresh else None
    spi_cached = _spi200_cached()
    spi_last = spi_hit["last"] if spi_hit else spi_cached["last"]
    spi_vs_xjo_pts = (spi_last - axjo_prev
                      if spi_last is not None and axjo_prev else None)
    items.append({
        "group": "futures",
        "source": "ibkr" if spi_hit else "cache",
        "contract": (spi_hit or {}).get("contract"),
        "symbol": "AP*0", "label": "SPI 200 (futures)",
        "last": spi_last, "previous_close": axjo_prev,
        "change_pct": (spi_vs_xjo_pts / axjo_prev * 100
                       if spi_vs_xjo_pts is not None else None),
    })
    # "Expected open" the way it's actually quoted (e.g. a forum poster the
    # user follows: "-28" for the morning of 2026-08-21) is the futures'
    # OWN overnight point change (vs its own previous settlement), not
    # SPI-last minus XJO's cash close -- those two are at different levels
    # (a persistent futures/cash basis, unrelated to the overnight move) and
    # conflating them inflated this to ~-96 the first time this was built,
    # same day. Barchart's own `priceChange` field already carries the
    # correct number (its own change vs its own last settlement) --
    # extracted by `_spi200_scrape()`, no second data source needed.
    # Confirmed live 2026-08-21 that Yahoo Finance carries no live ASX 24
    # futures quote under any AP+year+month-code convention (only one frozen
    # expired 2017 contract), so Barchart stays the only source either way.
    items.append({
        "group": "futures",
        "symbol": "AP*0-CHG", "label": "Expected Open (SPI overnight move)",
        "last": (spi_hit or spi_cached).get("change_pts"), "previous_close": None,
        "change_pct": None, "is_point_diff": True,
        "source": "ibkr" if spi_hit else "cache",
    })

    # Surfaced on the dashboard. A futures panel that silently shows a stale
    # number is worse than one that says it is stale: the whole point of these
    # rows is what is happening NOW.
    futures_status = {
        "ok": fut_fresh and not fut.get("error"),
        "source": "ibkr" if fut_fresh else "yahoo-fallback",
        "age_seconds": int(fut_age) if fut.get("fetched_at") else None,
        "stale_after_seconds": _FUTURES_STALE_SECONDS,
        "error": fut.get("error") or (
            f"IBKR futures cache is {int(fut_age // 60)} min old "
            f"(stale after {_FUTURES_STALE_SECONDS // 60} min) -- showing "
            f"delayed Yahoo values, which do not track the overnight session"
            if not fut_fresh else None),
    }

    data = {"items": items, "futures_status": futures_status,
            "fetched_at": time.time()}
    _snapshot_cache["ts"] = now
    _snapshot_cache["data"] = data
    return data


def _fred_series(series_id: str) -> list[tuple[str, float]]:
    """[(date, value), ...] oldest first, NaN/blank observations dropped
    (FRED writes "." for missing days -- weekends, holidays). Trimmed to
    ~2 years back -- plenty to find "1 year ago" without pulling the full
    history back to the 1960s on every cache miss."""
    from datetime import date, timedelta
    cosd = (date.today() - timedelta(days=760)).isoformat()
    r = httpx.get(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={cosd}",
        timeout=15,
    )
    r.raise_for_status()
    out = []
    for line in r.text.strip().splitlines()[1:]:
        date, _, value = line.partition(",")
        if value and value != ".":
            out.append((date, float(value)))
    return out


def get_yield_curve() -> dict[str, Any]:
    """Current yield curve, plus the same curve from ~1 year ago for
    comparison -- shape and how it's shifted is the point of a yield curve,
    not any single maturity's level."""
    now = time.monotonic()
    if _curve_cache["data"] is not None and now - _curve_cache["ts"] < _CURVE_TTL:
        return _curve_cache["data"]

    points = []
    latest_date = None
    for series_id, label, months in CURVE_SERIES:
        try:
            obs = _fred_series(series_id)
        except Exception:
            continue
        if not obs:
            continue
        date, current = obs[-1]
        latest_date = date if latest_date is None else max(latest_date, date)
        year_ago = next((v for d, v in reversed(obs[:-1]) if d <= _shift_year(date)), None)
        points.append({
            "maturity": label, "months": months,
            "current": current, "year_ago": year_ago,
        })

    data = {"points": points, "as_of": latest_date}
    _curve_cache["ts"] = now
    _curve_cache["data"] = data
    return data


def _shift_year(date_str: str) -> str:
    y, m, d = date_str.split("-")
    return f"{int(y) - 1}-{m}-{d}"
