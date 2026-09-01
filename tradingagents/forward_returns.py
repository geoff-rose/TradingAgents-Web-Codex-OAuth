"""Forward returns on every logged mover-day -- the research dataset behind
"does a high announcement score actually predict anything" (2026-08-25).

**Why this exists.** `mover_log` recorded what a mover did on its own day and
nothing after, so the two questions actually being asked -- does the classifier
score predict returns, and do big open-to-close moves sustain over days --
had no data to stand on. Worse, the same-day open->close already in the table
measures the ONE horizon where `event_momentum.py` found no edge at all
("3-10 days, never next-day; every volume bucket was negative at 1 day").
Evaluating a classifier against same-day close would be scoring it on the
horizon it was never expected to work at.

**Entry assumption: the mover day's CLOSE.** A mover is identified during or
after its session, so buying at that day's close is the honest earliest entry.
Measuring from the open would credit the strategy with a move it could not
have participated in -- the same look-ahead that the same-bar-exit artifact
introduced into every earlier backtest here.

**Entry at the OPEN, for anything announced before the market opened.** Added
2026-08-31 after IPX. Its US Army task order was released at 08:48 Sydney and
the stock closed down 0.82% -- which reads as a rejected announcement until you
separate the two halves: it GAPPED down 2.62% (its NASDAQ line had fallen 6.37%
the previous session, before the announcement existed) and then rose 1.85% from
the open. Measured close-to-close the announcement looks bad; measured over the
window it was actually tradeable in, it was clearly well received.

A pre-open announcement cannot be traded before the opening auction, so
`prev_close -> close` charges it with an overnight move it had no part in.
`open_close_pct` and `fwdopen_*d_pct` measure from the open instead. Both
framings are stored; neither replaces the other, because for an announcement
released DURING the session the close-to-close number is the right one.

**Benchmark opens come from hourly bars, not daily.** ASX index daily Opens are
carried forward from the previous Close -- verified 2026-08-31, gap exactly
0.0000% on 21 of 21 sessions for ^AXJO, ^AXMJ and ^AXGD alike -- so an index's
daily open-to-close silently equals its full-day move and would over-subtract
from an intraday comparison by the whole overnight gap. The 10:00 Sydney hourly
bar carries a genuine session open (sd 0.32% / 1.25% / 2.26% respectively).
Hourly history reaches ~730 days; beyond that the open-based benchmarks are
left NULL rather than filled with the wrong number.

**Sector-relative matters more than market-relative.** Added 2026-08-31 after
BC8 released FY26 results the classifier scored 90 and fell 6.2% the same
morning -- while the gold index fell 5.14% and the ASX 200 was flat. Against
the ASX 200 that announcement records as a large miss; against its own sector
it barely moved. Benchmarking stock-specific classifiers on a broad index
charges them for sector beta, and the calibration then blames the model for
something it never claimed to predict. `sect_*d_pct` is the ticker's own
sector index over the identical window; see `sectors.py` for how the mapping
is derived. Raw, market and sector are all stored, never pre-subtracted.

**Market-relative matters more than raw.** Both sessions logged so far had
~87% of movers close above their open, which is mostly a fact about a rising
market rather than about stock selection. The ASX 200 return over each
identical window is stored alongside, so any analysis can subtract it. Raw and
market are stored separately rather than pre-subtracted, so both stay
auditable.

**Horizons are TRADING sessions, not calendar days**, and a horizon is filled
only when its bar genuinely exists. A row 3 sessions old has `fwd_1d_pct` and
`fwd_3d_pct` and NULLs beyond; `fwd_bars_available` records how far the data
actually reached, so a partially-matured row is never mistaken for a complete
one that happened to go nowhere.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .mover_log import _connect

logger = logging.getLogger(__name__)

HORIZONS = (1, 3, 5, 10)
MARKET_INDEX = "^AXJO"          # S&P/ASX 200

_COLUMNS: dict[str, str] = {}
for _h in HORIZONS:
    _COLUMNS[f"fwd_{_h}d_pct"] = "REAL"     # close(D+h) vs close(D)
    _COLUMNS[f"mkt_{_h}d_pct"] = "REAL"     # ASX200 over the same window
    _COLUMNS[f"sect_{_h}d_pct"] = "REAL"    # the ticker's SECTOR over the same window
    _COLUMNS[f"fwdopen_{_h}d_pct"] = "REAL"  # entered at the OPEN of D, not its close
_COLUMNS.update({
    # The same-day tradeable reaction, and its benchmarks over the identical
    # window. See the open-entry note in the module docstring.
    "open_close_pct": "REAL",
    "mkt_open_close_pct": "REAL",
    "sect_open_close_pct": "REAL",
    # Which peer index `sect_*` was measured against, and how well the ticker
    # actually tracks it. Stored per row because the mapping is recomputed
    # from data and can change -- an analysis that pools rows benchmarked
    # against different indices without knowing it is not reproducible.
    "sector_index": "TEXT",
    "sector_corr": "REAL",
    "mfe_10d_pct": "REAL",       # best high within 10 sessions, vs close(D)
    "mae_10d_pct": "REAL",       # worst low within 10 sessions, vs close(D)
    "fwd_bars_available": "INTEGER",
    "fwd_updated_at": "TEXT",
    # Which grader produced `ai_score`. Without it, bumping the classifier
    # prompt silently pools two different graders into one calibration -- the
    # same reason asx_signals records model/prompt_version per signal.
    "ai_model": "TEXT",
    "ai_prompt_version": "TEXT",
})


def ensure_columns() -> None:
    with _connect() as conn:
        have = {r["name"] for r in conn.execute("PRAGMA table_info(mover_log)")}
        for col, typ in _COLUMNS.items():
            if col not in have:
                conn.execute(f"ALTER TABLE mover_log ADD COLUMN {col} {typ}")
        conn.commit()


def _bars(symbols: list[str], period: str = "6mo"):
    import yfinance as yf

    from .yf_lock import YF_LOCK
    with YF_LOCK:
        return yf.download(symbols, period=period, interval="1d", group_by="ticker",
                           auto_adjust=False, threads=True, progress=False)


def _session_opens(symbols: list[str], period: str = "3mo") -> dict[str, dict[str, float]]:
    """{symbol: {YYYY-MM-DD: session open}} from the 10:00 Sydney hourly bar.

    Only for INDEX benchmarks. A stock's daily Open is a real opening-auction
    print and needs no such treatment; an index's does not exist.
    """
    import yfinance as yf

    from .yf_lock import YF_LOCK
    out: dict[str, dict[str, float]] = {}
    if not symbols:
        return out
    with YF_LOCK:
        data = yf.download(symbols, period=period, interval="1h", group_by="ticker",
                           auto_adjust=False, threads=True, progress=False)
    for sym in symbols:
        try:
            df = data[sym].dropna(subset=["Open"]).tz_convert("Australia/Sydney")
        except (KeyError, TypeError, AttributeError):
            continue
        first = df[df.index.hour == 10]
        out[sym] = {d.date().isoformat(): float(o)
                    for d, o in zip(first.index, first["Open"])}
    return out


def _series(data, symbol: str):
    """(dates, frame) for one symbol, or (None, None) if absent/empty."""
    try:
        df = data[symbol].dropna(subset=["Close"])
    except (KeyError, TypeError):
        return None, None
    if df.empty:
        return None, None
    return [d.date().isoformat() for d in df.index], df


def compute(days_back: int = 60, force: bool = False) -> dict[str, Any]:
    """Fill forward returns for logged mover-days.

    Re-runnable by design: a row's longer horizons mature over the following
    fortnight, so any row that is not yet complete is recomputed on each run
    rather than being written once and left half-filled forever.
    """
    ensure_columns()
    since = (date.today() - timedelta(days=days_back)).isoformat()
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT ticker, trade_date, close, fwd_bars_available FROM mover_log"
            " WHERE finalized=1 AND trade_date>=? ORDER BY trade_date", (since,))]
    # A row is done once its longest horizon is covered; anything short of
    # that is refreshed so late-arriving sessions get picked up.
    todo = [r for r in rows
            if force or (r.get("fwd_bars_available") or 0) < max(HORIZONS)]
    if not todo:
        return {"updated": 0, "considered": len(rows), "complete": len(rows)}

    tickers = sorted({r["ticker"] for r in todo})
    from .sectors import SECTOR_INDICES, benchmark_index, get_map
    raw_map = get_map(tickers)
    # The BENCHMARK index, which applies the correlation floor -- the stored
    # row carries the best-fit label even when it is too weak to measure against.
    sector_map = {t: {**m, "sector_index": benchmark_index(m)}
                  for t, m in raw_map.items()}
    wanted = sorted({m["sector_index"] for m in sector_map.values()
                     if m.get("sector_index") in SECTOR_INDICES})
    data = _bars([f"{t}.AX" for t in tickers] + [MARKET_INDEX] + wanted)
    # Genuine session opens for the benchmarks (see the docstring: their daily
    # Open is the previous Close carried forward). Best-effort -- a failure
    # here leaves the open-based benchmark columns NULL, never wrong.
    try:
        bench_opens = _session_opens([MARKET_INDEX] + wanted)
    except Exception as exc:
        logger.warning("session-open fetch failed; open-relative benchmarks "
                       "will be NULL: %s", exc)
        bench_opens = {}
    mkt_dates, mkt_df = _series(data, MARKET_INDEX)
    if mkt_dates is None:
        logger.warning("no %s history; market-relative columns will be NULL", MARKET_INDEX)
    sector_series = {sym: _series(data, sym) for sym in wanted}
    unmapped = [t for t in tickers if t not in sector_map]
    if unmapped:
        logger.info("%d tickers have no sector mapping; sect_* will fall back to "
                    "the market index. Run sectors.build_map() to fill them.",
                    len(unmapped))

    def fwd(dates, df, day: str, horizon: int, base_px: float | None = None) -> float | None:
        """% from `day` to `horizon` sessions later, close-to-close by default.
        `base_px` overrides the starting price, for open-entry measurement."""
        if dates is None or day not in dates:
            return None
        i = dates.index(day)
        j = i + horizon
        if j >= len(dates):
            return None
        base = base_px if base_px is not None else float(df.iloc[i]["Close"])
        if not base:
            return None
        return round(100 * (float(df.iloc[j]["Close"]) - base) / base, 4)

    def open_close(dates, df, day: str, sym: str | None = None) -> float | None:
        """Open-to-close % on `day`. For a benchmark index (`sym` given) the
        open comes from the hourly bar, because the daily one is fake."""
        if dates is None or day not in dates:
            return None
        i = dates.index(day)
        if sym is not None:
            o = bench_opens.get(sym, {}).get(day)
        else:
            o = float(df.iloc[i]["Open"])
        if not o:
            return None
        return round(100 * (float(df.iloc[i]["Close"]) - o) / o, 4)

    updated = 0
    with _connect() as conn:
        for r in todo:
            t, day = r["ticker"], r["trade_date"]
            dates, df = _series(data, f"{t}.AX")
            if dates is None or day not in dates:
                continue
            i = dates.index(day)
            base = float(df.iloc[i]["Close"])
            if not base:
                continue
            n_after = len(dates) - 1 - i

            # An unmapped ticker benchmarks against the ASX 200 rather than
            # NULL, so `sect_*` is always populated and always comparable;
            # `sector_index` records which of the two it actually was.
            m = sector_map.get(t) or {}
            sect_sym = m.get("sector_index") or MARKET_INDEX
            s_dates, s_df = sector_series.get(sect_sym, (mkt_dates, mkt_df))

            open_px = float(df.iloc[i]["Open"]) or None

            vals: dict[str, Any] = {"sector_index": sect_sym,
                                    "sector_corr": m.get("sector_corr")}
            for h in HORIZONS:
                vals[f"fwd_{h}d_pct"] = fwd(dates, df, day, h)
                vals[f"mkt_{h}d_pct"] = fwd(mkt_dates, mkt_df, day, h) if mkt_dates else None
                vals[f"sect_{h}d_pct"] = fwd(s_dates, s_df, day, h) if s_dates else None
                vals[f"fwdopen_{h}d_pct"] = fwd(dates, df, day, h, base_px=open_px)
            vals["open_close_pct"] = open_close(dates, df, day)
            vals["mkt_open_close_pct"] = (open_close(mkt_dates, mkt_df, day, MARKET_INDEX)
                                          if mkt_dates else None)
            vals["sect_open_close_pct"] = (open_close(s_dates, s_df, day, sect_sym)
                                           if s_dates else None)

            # Excursions over whatever part of the 10 sessions exists. Both
            # matter: MFE says whether there was ever a profit to take, MAE
            # says whether a stop would have removed you before it arrived.
            window = df.iloc[i + 1:i + 1 + max(HORIZONS)]
            if len(window):
                vals["mfe_10d_pct"] = round(100 * (float(window["High"].max()) - base) / base, 4)
                vals["mae_10d_pct"] = round(100 * (float(window["Low"].min()) - base) / base, 4)
            vals["fwd_bars_available"] = min(n_after, max(HORIZONS))
            vals["fwd_updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

            sets = ", ".join(f"{k}=?" for k in vals)
            conn.execute(f"UPDATE mover_log SET {sets} WHERE ticker=? AND trade_date=?",
                         (*vals.values(), t, day))
            updated += 1
        conn.commit()
    return {"updated": updated, "considered": len(rows),
            "still_maturing": sum(1 for r in todo if True)}


def stamp_grader() -> dict[str, Any]:
    """Record which classifier version produced each row's `ai_score`.

    Backfills existing rows with the CURRENT version, which is correct only
    because the prompt has not changed since these rows were scored. Doing it
    now is what stops the next prompt bump from making them ambiguous.
    """
    ensure_columns()
    try:
        from .asx_signals import PROMPT_VERSION, _MODEL_LABEL
    except Exception:
        return {"updated": 0, "error": "asx_signals unavailable"}
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE mover_log SET ai_model=?, ai_prompt_version=?"
            " WHERE ai_score IS NOT NULL AND ai_prompt_version IS NULL",
            (_MODEL_LABEL, PROMPT_VERSION))
        conn.commit()
    return {"updated": cur.rowcount, "model": _MODEL_LABEL, "prompt_version": PROMPT_VERSION}
