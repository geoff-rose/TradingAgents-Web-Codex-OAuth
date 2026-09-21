"""Pre-open ASX candidate ranking and forward evaluation.

This module is deliberately a ranking and evidence surface, not an order
router.  At about 09:50 Sydney it combines the data already collected by the
dashboard:

* the persisted, tradeable model universe;
* daily OHLCV features calculated strictly before today's session;
* current and immediately-prior ASX announcements and their verified scores;
* the overnight market snapshot and futures cache; and
* the universe's liquidity/range-efficiency and sector map.

The output is up to three *candidates* when the evidence is good enough.  One
or two qualifying names are returned without filling the remaining slots; the
ranker can also abstain.  A forced three-name answer on a bad market day would
be a presentation feature, not a risk control.  There is no claim here that
the score is a probability or that the current research has demonstrated an
edge.

Recommendations are stored separately from ``swing_trades``.  That keeps the
forward record clean and lets the next morning resolve 1/3/5/10-session
returns from the actual opening price without changing the recommendation
after the fact. The recommendation-session result is also recorded from the
previous close to the session close: that is the primary quality measure, while
open-to-close remains the executable entry lens.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv(
    "RECOMMENDATIONS_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "recommendations.db"),
))
SYDNEY = ZoneInfo("Australia/Sydney")
STRATEGY_VERSION = "preopen-ranker-v7-catalyst-continuation"
N_PICKS = 3
# A shortlist is allowed to be empty. A low threshold makes the page look
# productive while quietly promoting ordinary technical leaders with no
# evidence that the next session will be good.
MIN_RECOMMENDATION_SCORE = 60.0
MIN_TECHNICAL_CONFIRMATIONS = 2
HARD_MARKET_GAP_PCT = 1.50
MIN_HISTORY_DAYS = 120
NEWS_POOL_LIMIT = 160
NEWS_MIN_TURNOVER_AUD = 250_000
# The collector's top-500 market-cap universe is the morning recommendation
# band. The existing history and median dollar-turnover checks remain the
# second liquidity control; market-cap rank alone is not a promise that every
# name trades equally well.
RECOMMENDATION_UNIVERSE_RANK_MAX = 500
RECENT_PICK_SESSIONS = 3
CATALYST_CONTINUATION_SESSIONS = 5
CATALYST_CONTINUATION_MIN_SCORE = 55.0
CATALYST_CONTINUATION_MIN_TECHNICAL_CONFIRMATIONS = 2
OUTCOME_HORIZONS = (1, 3, 5, 10)

# A headline is never treated as equivalent to a read-and-scored document.
# It is only a short-lived triage signal so a genuinely fresh, price-sensitive
# release can enter the morning pool before the classifier has finished. These
# patterns are deliberately explicit and conservative; the stored row carries
# `headline_only` so the uncertainty remains visible.
_HEADLINE_POSITIVE = (
    (r"\bhigh[- ]grade\b", 18), (r"\bdiscovery\b", 14),
    (r"\bmaiden\b", 12), (r"\bdrill(?:ing|ing results| results)?\b", 8),
    (r"\bintersections?\b", 10), (r"\bresource (?:upgrade|increase|update)\b", 14),
    (r"\b(?:reserve|resource) upgrade\b", 16), (r"\bstrong\b", 8),
    (r"\brecord\b", 8), (r"\bpositive\b", 8), (r"\bapproved?\b", 11),
    (r"\bapprovals?\b", 10), (r"\battained\b", 7),
    (r"\bcompleted?\b", 6), (r"\bsmelt(?:ed|ing)?\b", 8),
    (r"\bfda\b", 12), (r"\bcontract\b", 10), (r"\bagreement\b", 8),
    (r"\bpartnership\b", 8),
    (r"\bofftake\b", 12), (r"\bproduction\b", 7), (r"\bmilestone\b", 8),
    (r"\bsecured?\b", 7), (r"\bcompletes?\b", 6), (r"\bexpands?\b", 8),
    (r"\bexceptional\b", 10), (r"\bpromising\b", 8),
)
_HEADLINE_NEGATIVE = (
    (r"\bcapital raising\b", 22), (r"\bplacement\b", 18),
    (r"\bright(?:s|s issue)\b", 16), (r"\bdilut(?:e|ion|ive)\b", 18),
    (r"\bdiscount\b", 10), (r"\bconvertible\b", 12),
    (r"\btrading halt\b", 18), (r"\bsuspend(?:ed|sion)\b", 16),
    (r"\bdowngrade\b", 16), (r"\bdefault\b", 18), (r"\boverspend\b", 14),
    (r"\bdelay(?:ed|s)?\b", 10), (r"\btermination\b", 14),
    (r"\bresign(?:ation|s|ed)?\b", 9), (r"\bwithdraw(?:n|s)?\b", 13),
    (r"\bnot proceeding\b", 16), (r"\bguidance cut\b", 18),
    (r"\bloss\b", 8), (r"\bdebt\b", 7), (r"\bproposed issue of securities\b", 16),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL UNIQUE,
    generated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    as_of_date TEXT,
    market_regime_json TEXT NOT NULL,
    data_quality_json TEXT NOT NULL,
    backtest_evidence_json TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES recommendation_runs(id),
    rank INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    company TEXT,
    sector TEXT,
    score REAL NOT NULL,
    decision TEXT NOT NULL,
    reference_price REAL,
    target_price REAL,
    stop_price REAL,
    hold_days INTEGER NOT NULL,
    rationale TEXT NOT NULL,
    components_json TEXT NOT NULL,
    risks_json TEXT NOT NULL,
    data_status TEXT NOT NULL,
    entry_date TEXT,
    entry_prev_close REAL,
    entry_open REAL,
    entry_high REAL,
    entry_close REAL,
    entry_gap_pct REAL,
    entry_session_pct REAL,
    entry_open_to_close_pct REAL,
    fwd_1d_pct REAL,
    fwd_3d_pct REAL,
    fwd_5d_pct REAL,
    fwd_10d_pct REAL,
    mfe_5d_pct REAL,
    mae_5d_pct REAL,
    outcome_updated_at TEXT,
    UNIQUE(run_id, ticker),
    UNIQUE(run_id, rank)
);
CREATE INDEX IF NOT EXISTS idx_recommendation_runs_date
    ON recommendation_runs(trade_date);
CREATE INDEX IF NOT EXISTS idx_recommendations_ticker
    ON recommendations(ticker);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # The database predates the testing-record OHLC fields. CREATE TABLE IF
    # NOT EXISTS does not alter an existing installation, so add them once.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(recommendations)")}
    for column, column_type in (
        ("entry_prev_close", "REAL"),
        ("entry_high", "REAL"),
        ("entry_close", "REAL"),
        ("entry_gap_pct", "REAL"),
        ("entry_session_pct", "REAL"),
        ("entry_open_to_close_pct", "REAL"),
    ):
        if column not in columns:
            conn.execute(f"ALTER TABLE recommendations ADD COLUMN {column} {column_type}")
    conn.commit()
    return conn


def _now() -> datetime:
    return datetime.now(timezone.utc)


def sydney_today() -> str:
    return datetime.now(SYDNEY).date().isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _headline_proxy(headlines: list[str]) -> dict[str, Any]:
    """Triage fresh headlines while a full document score is pending.

    This is intentionally not an LLM substitute. It only decides whether a
    release deserves attention in the morning pool; the result is labelled
    `headline_only` and receives less trust than a verified document score.
    """
    text = " ".join(str(h or "") for h in headlines).lower()
    positive = [pattern for pattern, _ in _HEADLINE_POSITIVE if re.search(pattern, text)]
    negative = [pattern for pattern, _ in _HEADLINE_NEGATIVE if re.search(pattern, text)]
    positive_points = sum(points for pattern, points in _HEADLINE_POSITIVE if pattern in positive)
    negative_points = sum(points for pattern, points in _HEADLINE_NEGATIVE if pattern in negative)
    # Contradictory headlines are not promoted by the proxy. The document
    # classifier must resolve a good result paired with a dilutive raise.
    if positive and negative:
        score, signal = 50, "neutral"
    else:
        score = int(_clip(50 + positive_points - negative_points, 0, 100))
        signal = "positive" if score >= 62 else "negative" if score <= 38 else "neutral"
    return {
        "score": score,
        "signal": signal,
        "positive_terms": positive,
        "negative_terms": negative,
        "confidence": "low" if signal == "neutral" else "headline_only",
    }


def _rank(values: dict[str, float | None], ticker: str) -> float:
    """Cross-sectional percentile rank, with 0.5 for an unavailable value."""
    value = values.get(ticker)
    usable = sorted(float(v) for v in values.values() if _finite(v))
    if value is None or not usable:
        return 0.5
    if len(usable) == 1:
        return 0.5
    below = sum(v < float(value) for v in usable)
    equal = sum(v == float(value) for v in usable)
    return _clip((below + 0.5 * equal) / len(usable))


def _asof_frame(df, trade_date: str):
    """Drop today's partial bar and anything dated after the recommendation."""
    import pandas as pd

    if df is None or df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    return out[out.index < pd.Timestamp(trade_date)].sort_index()


def _technical_features(ticker: str, df, market_df, trade_date: str) -> dict[str, Any] | None:
    """Build causal features from the last completed session."""
    import numpy as np

    df = _asof_frame(df, trade_date)
    market_df = _asof_frame(market_df, trade_date)
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return None
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum.reduce([
        (high - low).to_numpy(float),
        (high - prev_close).abs().to_numpy(float),
        (low - prev_close).abs().to_numpy(float),
    ])
    tr_s = __import__("pandas").Series(tr, index=df.index)
    atr14 = tr_s.rolling(14).mean()
    last = float(close.iloc[-1])
    if not last or not _finite(last):
        return None

    def ret(n: int) -> float | None:
        if len(close) <= n or not close.iloc[-1 - n]:
            return None
        return float((close.iloc[-1] / close.iloc[-1 - n] - 1.0) * 100.0)

    sma20 = float(close.tail(20).mean())
    sma50 = float(close.tail(50).mean())
    high20 = float(high.tail(20).max())
    low10 = float(low.tail(10).min())
    last_range = float(high.iloc[-1] - low.iloc[-1])
    median_prior_volume = float(volume.iloc[:-1].tail(20).median()) if len(volume) > 20 else 0.0
    volume_ratio = float(volume.iloc[-1] / median_prior_volume) if median_prior_volume else None
    rsi_delta = close.diff().tail(14)
    gains = rsi_delta.clip(lower=0).mean()
    losses = (-rsi_delta.clip(upper=0)).mean()
    rsi = float(100 - 100 / (1 + gains / losses)) if losses > 0 else (100.0 if gains > 0 else 50.0)

    market_ret5 = None
    market_ret20 = None
    if market_df is not None and not market_df.empty and len(market_df) > 20:
        mc = market_df["Close"].astype(float)
        if len(mc) > 5 and mc.iloc[-6]:
            market_ret5 = float((mc.iloc[-1] / mc.iloc[-6] - 1.0) * 100.0)
        if mc.iloc[-21]:
            market_ret20 = float((mc.iloc[-1] / mc.iloc[-21] - 1.0) * 100.0)
    ret20 = ret(20)
    ret5 = ret(5)
    rel5 = ret5 - market_ret5 if ret5 is not None and market_ret5 is not None else None
    rel20 = ret20 - market_ret20 if ret20 is not None and market_ret20 is not None else None
    return {
        "ticker": ticker,
        "as_of_date": str(df.index[-1].date()),
        "price": round(last, 6),
        "ret_5_pct": ret5,
        "ret_20_pct": ret20,
        "ret_60_pct": ret(60),
        "market_ret_5_pct": market_ret5,
        "market_ret_20_pct": market_ret20,
        "relative_ret_5_pct": rel5,
        "relative_ret_20_pct": rel20,
        "sma20": sma20,
        "sma50": sma50,
        "above_sma20": last > sma20,
        "above_sma50": last > sma50,
        "drawdown_20d_pct": (last / high20 - 1.0) * 100.0 if high20 else None,
        "distance_from_10d_low_pct": (last / low10 - 1.0) * 100.0 if low10 else None,
        "atr14_pct": float(atr14.iloc[-1] / last * 100.0) if _finite(atr14.iloc[-1]) else None,
        "last_range_pct": last_range / last * 100.0 if last else None,
        "volume_ratio": volume_ratio,
        "rsi14": rsi,
    }


def _market_context() -> dict[str, Any]:
    """Read the existing market/futures snapshot and produce an abstain flag."""
    out: dict[str, Any] = {
        "snapshot_available": False,
        "futures_fresh": False,
        "spi_expected_open_pct": None,
        "us_futures_avg_pct": None,
        "hard_abstain": False,
        "reasons": [],
    }
    try:
        from .markets import get_snapshot

        snap = get_snapshot()
        out["snapshot_available"] = True
        out["fetched_at"] = snap.get("fetched_at")
        fs = snap.get("futures_status") or {}
        out["futures_fresh"] = bool(fs.get("ok"))
        items = {i.get("symbol"): i for i in snap.get("items", [])}
        vix = items.get("^VIX") or {}
        audusd = items.get("AUDUSD=X") or {}
        us10y = items.get("^TNX") or {}
        out["vix_level"] = vix.get("last") if _finite(vix.get("last")) else None
        out["vix_change_pct"] = vix.get("change_pct") if _finite(vix.get("change_pct")) else None
        out["audusd_level"] = audusd.get("last") if _finite(audusd.get("last")) else None
        out["audusd_change_pct"] = audusd.get("change_pct") if _finite(audusd.get("change_pct")) else None
        if _finite(us10y.get("last")):
            out["us10y_level_pct"] = round(float(us10y["last"]), 4)
            out["us10y_change_bps"] = (
                round((float(us10y["last"]) - float(us10y["previous_close"])) * 100.0, 2)
                if _finite(us10y.get("previous_close")) else None
            )
        else:
            out["us10y_level_pct"] = None
            out["us10y_change_bps"] = None
        xjo = items.get("^AXJO") or {}
        expected = items.get("AP*0-CHG") or {}
        xjo_prev = xjo.get("previous_close")
        points = expected.get("last")
        if _finite(points) and _finite(xjo_prev) and float(xjo_prev):
            out["spi_expected_open_pct"] = round(float(points) / float(xjo_prev) * 100.0, 3)
        us = [items[s].get("change_pct") for s in ("ES=F", "NQ=F")
              if s in items and _finite(items[s].get("change_pct"))]
        if us:
            out["us_futures_avg_pct"] = round(sum(float(x) for x in us) / len(us), 3)
        asia = [items[s].get("change_pct") for s in
                ("^HSI", "^N225", "^KS11", "000001.SS", "^STI")
                if s in items and _finite(items[s].get("change_pct"))]
        out["asia_avg_pct"] = round(sum(float(x) for x in asia) / len(asia), 3) if asia else None
        out["asia_markets"] = {
            s: round(float(items[s]["change_pct"]), 3)
            for s in ("^HSI", "^N225", "^KS11", "000001.SS", "^STI")
            if s in items and _finite(items[s].get("change_pct"))
        }
        commodity_symbols = {
            "gold": "GC=F", "silver": "SI=F", "copper": "HG=F",
            "oil": "CL=F", "brent": "BZ=F", "iron_ore": "TIO=F",
        }
        out["commodities"] = {}
        for name, symbol in commodity_symbols.items():
            item = items.get(symbol) or {}
            if _finite(item.get("change_pct")):
                out["commodities"][name] = {
                    "symbol": symbol,
                    "label": item.get("label") or name.replace("_", " ").title(),
                    "change_pct": round(float(item["change_pct"]), 3),
                    "last": item.get("last"),
                }
        # The sector board is a previous-session confirmation of local beta.
        # It is useful even before the cash market opens, but failure must not
        # block the rest of the pre-open ranking.
        try:
            from .sectors import board
            b = board()
            out["sector_board"] = {
                r["label"]: {
                    "day_pct": r.get("day_pct"),
                    "vs_market_pct": r.get("vs_market_pct"),
                    "symbol": r.get("symbol"),
                }
                for r in b.get("sectors", []) if r.get("label")
            }
            out["sector_board_as_of"] = b.get("as_of")
        except Exception as exc:
            out["sector_board"] = {}
            out["sector_board_error"] = str(exc)[:120]
    except Exception as exc:
        out["error"] = str(exc)[:160]
        out["reasons"].append("market snapshot unavailable")

    gap = out.get("spi_expected_open_pct")
    if _finite(gap) and abs(float(gap)) >= HARD_MARKET_GAP_PCT:
        out["hard_abstain"] = True
        out["reasons"].append(f"SPI-implied move {float(gap):+.2f}% exceeds {HARD_MARKET_GAP_PCT:.2f}%")
    us_avg = out.get("us_futures_avg_pct")
    if _finite(us_avg) and float(us_avg) <= -2.0:
        out["hard_abstain"] = True
        out["reasons"].append(f"US futures average {float(us_avg):+.2f}% is under -2.00%")
    vix_level = out.get("vix_level")
    if _finite(vix_level) and float(vix_level) >= 40.0:
        out["hard_abstain"] = True
        out["reasons"].append(f"VIX {float(vix_level):.1f} is at panic-risk level")
    if not out["futures_fresh"]:
        out["reasons"].append("futures cache is stale or unavailable")
    return out


def _announcement_context(trade_date: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Collect current announcements and a bounded catalyst continuation lane.

    A price-sensitive milestone can keep moving a stock for several sessions
    after release. Continuation context is limited to five prior ASX sessions,
    requires a scored price-sensitive document, and is labelled separately so
    it cannot masquerade as fresh news.
    """
    from .asx_feed import recent_announcements, tradeable_session_for
    from .asx_signals import _worth_a_call, attach_signals, get_ticker_signals

    now = _now()
    dated_rows: list[tuple[str, dict[str, Any]]] = []
    prior_sessions: list[str] = []
    try:
        for row in recent_announcements(limit=3000, universe_only=False):
            stamp = row.get("released_at") or row.get("seen_at") or ""
            session = tradeable_session_for(stamp)
            if not session or session > trade_date:
                continue
            try:
                if datetime.fromisoformat(stamp).astimezone(timezone.utc) > now:
                    continue
            except (TypeError, ValueError):
                pass
            dated_rows.append((session, row))
        prior_sessions = sorted(
            {session for session, _ in dated_rows if session < trade_date},
            reverse=True,
        )[:CATALYST_CONTINUATION_SESSIONS]
        allowed_sessions = {trade_date, *prior_sessions}
        rows = [row for session, row in dated_rows if session in allowed_sessions]
        attach_signals(rows)
    except Exception as exc:
        logger.warning("announcement context failed: %s", exc)
        rows = [row for session, row in dated_rows if session == trade_date]

    nets: dict[str, dict[str, Any]] = {}
    try:
        nets = get_ticker_signals(recent_sessions=CATALYST_CONTINUATION_SESSIONS + 1)
    except Exception:
        pass

    by_ticker: dict[str, dict[str, Any]] = {}
    for ticker, ticker_rows in _group_rows(rows).items():
        current = [r for r in ticker_rows if tradeable_session_for(
            r.get("released_at") or r.get("seen_at")) == trade_date]
        continuation = [r for r in ticker_rows if tradeable_session_for(
            r.get("released_at") or r.get("seen_at")) in prior_sessions]
        # Routine ASX paperwork is deliberately not sent through the paid AI
        # queue. It must not make an otherwise clean ticker look as if its
        # material news is still pending, nor should its wording drive the
        # headline proxy.
        material = [r for r in current if _worth_a_call(r)]
        scored = [r for r in material if _finite(r.get("score"))]
        pending = [r for r in material if r.get("score") is None
                   and r.get("evidence_status") not in ("document_unavailable", "classification_failed")]
        headlines = [r.get("headline") for r in material if r.get("headline")]
        proxy = _headline_proxy(headlines) if headlines else {
            "score": 50, "signal": "neutral", "positive_terms": [],
            "negative_terms": [], "confidence": "none",
        }
        continuation_material = [
            r for r in continuation
            if _worth_a_call(r) and r.get("price_sensitive")
        ]
        continuation_scored = [r for r in continuation_material if _finite(r.get("score"))]
        continuation_dates = {
            tradeable_session_for(r.get("released_at") or r.get("seen_at"))
            for r in continuation_scored
        }
        net = nets.get(ticker)
        score = None
        reason = None
        status = "no_recent_news"
        session = None
        catalyst_age_sessions = None
        if net and net.get("session_date") == trade_date:
            score = int(net["score"])
            reason = net.get("reason")
            status = "current_scored"
            session = trade_date
        elif current and pending:
            if proxy["signal"] in {"positive", "negative"}:
                score = proxy["score"]
                reason = "headline triage only; document score pending"
                status = f"headline_proxy_{proxy['signal']}"
            else:
                status = "pending_announcement"
        elif current and len(scored) == 1:
            score = int(scored[0]["score"])
            reason = scored[0].get("signal_reason")
            status = "current_document_scored"
            session = trade_date
        elif current and len(scored) > 1:
            status = "awaiting_net_score"
        elif continuation_scored:
            # Prefer the cached net for the same older session. Otherwise use
            # the strongest scored document from the latest continuation day.
            if net and net.get("session_date") in continuation_dates:
                session = net["session_date"]
                score = int(net["score"])
                reason = net.get("reason")
            else:
                session = max(continuation_dates)
                latest = [r for r in continuation_scored if tradeable_session_for(
                    r.get("released_at") or r.get("seen_at")) == session]
                best = max(latest, key=lambda r: abs(float(r["score"]) - 50.0))
                score = int(best["score"])
                reason = best.get("signal_reason")
            status = "catalyst_continuation"
            try:
                catalyst_age_sessions = prior_sessions.index(session) + 1
            except ValueError:
                pass

        all_context = current + continuation
        by_ticker[ticker] = {
            "score": score,
            "reason": reason,
            "status": status,
            "session_date": session,
            "catalyst_continuation": status == "catalyst_continuation",
            "catalyst_session_date": session if status == "catalyst_continuation" else None,
            "catalyst_age_sessions": catalyst_age_sessions,
            "n_announcements": len(current),
            "n_material_announcements": len(material),
            "n_continuation_announcements": len(continuation_material),
            "pending": bool(pending),
            "halt": any(bool(r.get("is_halt")) for r in current),
            "price_sensitive": any(bool(r.get("price_sensitive")) for r in all_context),
            "headline_signal": proxy["signal"] if current else "none",
            "headline_score": proxy["score"] if current else None,
            "headline_terms": proxy["positive_terms"] + proxy["negative_terms"],
            "universe_rank": min(
                (int(r["universe_rank"]) for r in all_context
                 if r.get("universe_rank") is not None),
                default=None,
            ),
            "latest_at": max(
                (r.get("released_at") or r.get("seen_at") or "" for r in all_context),
                default=None,
            ),
            "n_price_sensitive": sum(1 for r in all_context if r.get("price_sensitive")),
            "headline": next((r.get("headline") for r in scored if r.get("headline")),
                              next((r.get("headline") for r in current),
                                   next((r.get("headline") for r in continuation_scored), None))),
        }
    continuation_tickers = sorted(
        t for t, value in by_ticker.items() if value["catalyst_continuation"]
    )
    current_rows = [row for session, row in dated_rows if session == trade_date]
    return by_ticker, {
        "n_announcements": len(current_rows),
        "n_tickers": len(by_ticker),
        "n_pending_tickers": sum(1 for v in by_ticker.values() if v["pending"]),
        "catalyst_continuation_tickers": continuation_tickers,
        "catalyst_continuation_sessions": CATALYST_CONTINUATION_SESSIONS,
        "as_of": now.isoformat(timespec="seconds"),
    }


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ticker = (row.get("ticker") or "").strip().upper()
        if ticker:
            out.setdefault(ticker, []).append(row)
    return out


def _recent_pick_dates(trade_date: str) -> dict[str, str]:
    """Return tickers recommended in the last few completed morning runs.

    This is a freshness control, not an outcome signal: yesterday's pick is
    not allowed to occupy today's scarce slots merely because its old chart
    still ranks well. A new, material catalyst can explicitly earn a repeat.
    """
    with _connect() as conn:
        dates = [r[0] for r in conn.execute(
            "SELECT trade_date FROM recommendation_runs WHERE trade_date < ? "
            "ORDER BY trade_date DESC LIMIT ?", (trade_date, RECENT_PICK_SESSIONS))]
        if not dates:
            return {}
        marks = ",".join("?" * len(dates))
        rows = conn.execute(
            "SELECT r.ticker, u.trade_date FROM recommendations r "
            "JOIN recommendation_runs u ON u.id=r.run_id "
            f"WHERE u.trade_date IN ({marks})", dates).fetchall()
    # The query covers several sessions; dictionary insertion order follows
    # SQLite's result order only accidentally, so keep the newest date per name
    # explicitly.
    out: dict[str, str] = {}
    for row in rows:
        if row[0] not in out or row[1] > out[row[0]]:
            out[row[0]] = row[1]
    return out


def _fresh_catalyst(news: dict[str, Any]) -> bool:
    status = str(news.get("status") or "")
    return status in {"current_scored", "current_document_scored"} or status.startswith("headline_proxy")


def _positive_catalyst(news: dict[str, Any]) -> bool:
    """Whether fresh news is strong enough to substitute for chart confirmation."""
    if not _fresh_catalyst(news):
        return False
    if news.get("headline_signal") == "positive":
        return True
    score = news.get("score")
    return _finite(score) and float(score) >= 60.0


def _technical_confirmations(features: dict[str, Any]) -> int:
    """Count independent, causal confirmations for the full-session objective."""
    if not features:
        return 0
    recent_relative = features.get("relative_ret_5_pct")
    medium_relative = features.get("relative_ret_20_pct")
    if not _finite(recent_relative):
        recent_relative = features.get("ret_5_pct")
    if not _finite(medium_relative):
        medium_relative = features.get("ret_20_pct")
    pullback = (
        bool(features.get("above_sma50"))
        and _finite(features.get("drawdown_20d_pct"))
        and -12.0 <= float(features["drawdown_20d_pct"]) <= -1.0
    )
    return sum((
        bool(features.get("above_sma20")) and bool(features.get("above_sma50")),
        _finite(recent_relative) and float(recent_relative) > 0.0,
        _finite(medium_relative) and float(medium_relative) > 0.0,
        pullback,
    ))


def _macro_component(sector: str, market: dict[str, Any]) -> tuple[float, str]:
    """Translate overnight commodity and sector moves into a small factor.

    This is beta context, not a price target. The factor is centred at 7.5/15
    and only moves a few points, so a commodity move can support a name but
    cannot rescue a technically weak, illiquid, or negatively flagged stock.
    """
    label = (sector or "").lower()
    if "gold" in label:
        exposure = {"gold": 0.65, "silver": 0.25, "copper": 0.10}
    elif any(x in label for x in ("material", "resource")):
        exposure = {"iron_ore": 0.50, "copper": 0.30, "gold": 0.20}
    elif "energy" in label:
        exposure = {"oil": 0.70, "brent": 0.30}
    elif any(x in label for x in ("technology", "communication")):
        exposure = {"global_equity": 1.0}
    else:
        exposure = {}

    commodities = market.get("commodities") or {}
    weighted, total = 0.0, 0.0
    details = []
    for name, weight in exposure.items():
        if name == "global_equity":
            change = market.get("us_futures_avg_pct")
        else:
            change = (commodities.get(name) or {}).get("change_pct")
        if not _finite(change):
            continue
        change = float(change)
        weighted += weight * math.tanh(change / 2.5)
        total += weight
        if abs(change) >= 0.20:
            details.append(f"{name.replace('_', ' ')} {change:+.1f}%")
    commodity_signal = weighted / total if total else 0.0

    board = market.get("sector_board") or {}
    board_row = board.get(sector)
    if board_row is None:
        board_row = next((v for k, v in board.items() if k.lower() in label or label in k.lower()), None)
    sector_move = (board_row or {}).get("vs_market_pct")
    sector_signal = math.tanh(float(sector_move) / 3.0) if _finite(sector_move) else 0.0

    # Risk-regime modifiers are deliberately small. VIX, rates, FX and Asia
    # should change the ranking, not overwhelm a company-specific catalyst.
    regime_adjustment = 0.0
    vix = market.get("vix_level")
    if _finite(vix):
        if float(vix) >= 28.0:
            regime_adjustment -= min(3.0, (float(vix) - 25.0) / 5.0)
            details.append(f"VIX {float(vix):.1f}")
        elif float(vix) <= 15.0:
            regime_adjustment += 0.5
    rate_bps = market.get("us10y_change_bps")
    if _finite(rate_bps) and abs(float(rate_bps)) >= 5.0:
        if any(x in label for x in ("technology", "communication", "reit")):
            regime_adjustment -= math.copysign(min(1.5, abs(float(rate_bps)) / 10.0), float(rate_bps))
        details.append(f"US 10Y {float(rate_bps):+.0f}bp")
    asia = market.get("asia_avg_pct")
    if _finite(asia) and any(x in label for x in ("material", "resource", "gold", "energy")):
        regime_adjustment += _clip(float(asia) / 2.0, -1.0, 1.0)
        if abs(float(asia)) >= 0.4:
            details.append(f"Asia {float(asia):+.1f}%")

    component = _clip(7.5 + 5.0 * commodity_signal + 2.5 * sector_signal
                      + regime_adjustment, 0.0, 15.0)
    if details:
        direction = "supports" if commodity_signal > 0.05 else "pressures" if commodity_signal < -0.05 else "mixed"
        reason = f"{direction} {sector} exposure ({', '.join(details)})"
    elif _finite(sector_move) and abs(float(sector_move)) >= 0.5:
        reason = f"{sector} sector is {float(sector_move):+.1f}% vs ASX 200"
    else:
        reason = "no strong overnight commodity or sector-beta impulse"
    return round(component, 2), reason


def _candidate_rows(trade_date: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return scored candidates plus data-quality and market context.

    The persisted model universe remains the stable technical base. For the
    morning decision, both that base and the fresh-news expansion are capped
    at the live top-500 universe. Names outside this recommendation band
    cannot become buy candidates.
    """
    from .backtest import fetch_daily_history
    from .screener import Candidate, bulk_daily, compute_metrics, get_model_universe, load_candidates
    from .sectors import get_map

    model_universe_all = get_model_universe(tradeable_only=True)
    # Restrict the stable technical basket to the same live recommendation
    # universe used by the fresh-news expansion.
    catalog = {c.ticker: c for c in load_candidates(limit=RECOMMENDATION_UNIVERSE_RANK_MAX)}
    model_universe = [r for r in model_universe_all if r["ticker"] in catalog]
    if not model_universe:
        raise RuntimeError("the tradeable model universe is empty")
    ann, ann_quality = _announcement_context(trade_date)
    recent_picks = _recent_pick_dates(trade_date)

    # A fresh current-session price-sensitive release, a directional headline
    # proxy, or a scored catalyst in the continuation lane is allowed to enter
    # the technical pool if it is in the collector's live top-500 universe.
    # Routine filings are not allowed to expand the pool by themselves.
    news_pool = [
        ticker for ticker, news in ann.items()
        if (news.get("n_announcements", 0) or news.get("n_continuation_announcements", 0))
        and news.get("universe_rank") is not None
        and int(news["universe_rank"]) <= RECOMMENDATION_UNIVERSE_RANK_MAX
        and (
            (news.get("catalyst_continuation")
             and _finite(news.get("score"))
             and float(news["score"]) >= CATALYST_CONTINUATION_MIN_SCORE)
            or (not news.get("catalyst_continuation")
                and (news.get("price_sensitive")
                     or news.get("headline_signal") in {"positive", "negative"}
                     or (_fresh_catalyst(news) and _finite(news.get("score"))
                         and abs(float(news["score"]) - 50.0) >= 10.0)))
        )
    ]
    news_pool.sort(key=lambda t: (
        0 if ann[t].get("catalyst_continuation") else 1,
        0 if (ann[t].get("headline_signal") == "positive") else 1,
        -(ann[t].get("headline_score") or 50),
        ann[t].get("universe_rank") or 9999,
        t,
    ))
    base_tickers = {r["ticker"] for r in model_universe}
    extra_tickers = [t for t in news_pool if t not in base_tickers and t in catalog][:NEWS_POOL_LIMIT]
    tickers = [r["ticker"] for r in model_universe] + extra_tickers
    histories = bulk_daily(tickers, period="2y")
    market_df = fetch_daily_history("^AXJO", period="2y", market="US")
    market = _market_context()
    sector_map = get_map(tickers)
    metrics = {r["ticker"]: r for r in model_universe}
    extra_kept = 0
    extra_excluded_liquidity = 0
    for ticker in extra_tickers:
        c = catalog[ticker]
        metric = compute_metrics(c, histories.get(ticker)) if histories.get(ticker) is not None else {}
        if (not metric.get("usable")
                or metric.get("n_days", 0) < MIN_HISTORY_DAYS
                or (metric.get("median_turnover_aud") or 0) < NEWS_MIN_TURNOVER_AUD):
            extra_excluded_liquidity += 1
            continue
        metrics[ticker] = {
            "ticker": ticker,
            "company": c.company,
            "bucket": c.cap_bucket,
            "sector": c.sector,
            "variance_ratio": metric.get("variance_ratio"),
            "median_turnover_aud": metric.get("median_turnover_aud"),
            "range_efficiency": metric.get("range_efficiency"),
        }
        extra_kept += 1

    tickers = [t for t in tickers if t in metrics]
    features: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        f = _technical_features(ticker, histories.get(ticker), market_df, trade_date)
        if f:
            features[ticker] = f

    ret5 = {t: f.get("ret_5_pct") for t, f in features.items()}
    ret20 = {t: f.get("ret_20_pct") for t, f in features.items()}
    rel5 = {t: f.get("relative_ret_5_pct") for t, f in features.items()}
    rel20 = {t: f.get("relative_ret_20_pct") for t, f in features.items()}
    efficiencies = {t: metrics[t].get("range_efficiency") for t in features}
    turnovers = {t: math.log1p(float(metrics[t].get("median_turnover_aud") or 0)) for t in features}
    scored: list[dict[str, Any]] = []
    for ticker, f in features.items():
        m = metrics[ticker]
        news = ann.get(ticker) or {"score": None, "status": "no_recent_news", "pending": False, "halt": False}
        sm = sector_map.get(ticker) or {}
        sector = sm.get("sector_name") or m.get("sector") or m.get("bucket") or "Unknown"

        # The recommendation is judged on the whole session, so use both a
        # recent and a medium-term close-to-close signal. The 5-session leg
        # prevents a name with an old 20-day run from dominating after its
        # full-session behaviour has started to deteriorate.
        trend_component = 5.0 * _rank(ret5, ticker) + 10.0 * _rank(ret20, ticker)
        relative_component = 4.0 * _rank(rel5, ticker) + 6.0 * _rank(rel20, ticker)
        setup_component = 0.0
        if f["above_sma50"]:
            setup_component += 4.0
        if f["above_sma20"]:
            setup_component += 2.0
        dd = f.get("drawdown_20d_pct")
        if f["above_sma50"] and dd is not None and -12.0 <= dd <= -1.0:
            setup_component += 6.0
        elif f["above_sma20"] and (f.get("ret_5_pct") or 0) > 0:
            setup_component += 2.0
        if f.get("rsi14") is not None and 35 <= f["rsi14"] <= 65:
            setup_component += 0.0  # neutral RSI is a quality flag, not alpha

        liquidity_component = 10.0 * (
            0.6 * _rank(efficiencies, ticker) + 0.4 * _rank(turnovers, ticker)
        )
        nscore = news.get("score")
        if nscore is None and news.get("headline_signal") in {"positive", "negative"}:
            nscore = news.get("headline_score")
        if nscore is None:
            # No release is not bearish, but it is not evidence of a
            # directional edge either. Keeping this below the old neutral
            # contribution makes catalyst-backed names win on equal charts.
            news_component = 6.0
        else:
            news_component = 14.0 + (float(nscore) - 50.0) * 0.32
            if news.get("status") == "prior_session_scored":
                news_component *= 0.75
            if news.get("status", "").startswith("headline_proxy"):
                news_component *= 0.75
            if news.get("status") == "catalyst_continuation":
                # Older catalysts retain signal, but their weight decays by
                # session so a continuation can support a name without
                # overpowering fresh evidence.
                age = max(1, int(news.get("catalyst_age_sessions") or 1))
                news_component *= max(0.55, 1.0 - 0.10 * (age - 1))
        news_component = _clip(news_component, 0.0, 30.0)

        macro_component, macro_reason = _macro_component(sector, market)

        # Guard against the common failure mode where a high 20-day rank is
        # just a stretched move that is already reversing. These penalties
        # use only completed bars and therefore remain causal at 09:50.
        risk_penalty = 0.0
        rsi = f.get("rsi14")
        if _finite(rsi) and float(rsi) > 72.0:
            risk_penalty += min(4.0, (float(rsi) - 72.0) * 0.25)
        atr_pct = float(f.get("atr14_pct") or 0.0)
        if atr_pct > 4.5:
            risk_penalty += min(4.0, (atr_pct - 4.5) * 0.8)
        if _finite(f.get("ret_5_pct")) and float(f["ret_5_pct"]) < -2.0:
            risk_penalty += min(3.0, (-float(f["ret_5_pct"]) - 2.0) * 0.35)
        if (f.get("drawdown_20d_pct") is not None
                and float(f["drawdown_20d_pct"]) > -1.0
                and not _positive_catalyst(news)):
            risk_penalty += 1.5

        vr = f.get("volume_ratio")
        event_component = 10.0 * _clip(((float(vr) if _finite(vr) else 0.8) - 0.8) / 4.2)
        repeat_penalty = 0.0
        if ticker in recent_picks and not _fresh_catalyst(news):
            # A strong fresh release can justify a repeat. A stale chart alone
            # cannot: otherwise the ranker simply rediscovers yesterday's
            # winners and fails the user's goal of a genuinely fresh list.
            repeat_penalty = 8.0
        score = round(trend_component + relative_component + setup_component
                      + liquidity_component + news_component + macro_component
                      + event_component - repeat_penalty - risk_penalty, 2)

        target_pct = _clip(max(1.5 * atr_pct, 2.0), 2.0, 8.0)
        stop_pct = _clip(max(1.0 * atr_pct, 1.5), 1.5, 6.0)
        ref = float(f["price"])
        risks = []
        if news.get("pending"):
            risks.append("fresh announcement is headline-triaged; document score is pending")
        if news.get("status") == "no_recent_news":
            risks.append("no fresh announcement catalyst")
        if news.get("status") == "catalyst_continuation":
            risks.append(
                f"catalyst continuation from {news.get('catalyst_session_date')}; "
                "older than a current-session release"
            )
        if (m.get("variance_ratio") or 0) >= 1:
            risks.append("price behaviour is trending rather than mean-reverting")
        if f.get("rsi14", 50) > 75:
            risks.append("extended RSI")
        if market.get("reasons"):
            risks.extend(market["reasons"][:1])
        if news.get("halt"):
            risks.append("recent halt")
        if news.get("status", "").startswith("headline_proxy"):
            risks.append("headline proxy may miss offsetting detail in the document")
        if repeat_penalty:
            risks.append(f"recent pick on {recent_picks[ticker]}; no fresh catalyst for repeat")
        if risk_penalty:
            risks.append(f"extension/volatility guard deducted {risk_penalty:.1f} points")
        if liquidity_component < 3.0:
            risks.append("lower relative liquidity than the rest of the morning pool")

        data_status = "verified" if news.get("status") in {
            "current_scored", "current_document_scored", "prior_session_scored",
            "catalyst_continuation", "no_recent_news"
        } and f.get("as_of_date") else "limited"
        scored.append({
            "ticker": ticker,
            "company": m.get("company") or ticker,
            "sector": sector,
            "bucket": m.get("bucket"),
            "score": score,
            "reference_price": round(ref, 4),
            "target_price": round(ref * (1 + target_pct / 100.0), 4),
            "stop_price": round(ref * (1 - stop_pct / 100.0), 4),
            "hold_days": 5,
            "rationale": _rationale(ticker, f, news, score, market, macro_reason),
            "components": {
                "trend": round(trend_component, 2),
                "relative_strength": round(relative_component, 2),
                "setup": round(setup_component, 2),
                "liquidity": round(liquidity_component, 2),
                "news": round(news_component, 2),
                "macro": round(macro_component, 2),
                "event_volume": round(event_component, 2),
                "risk_guard": round(-risk_penalty, 2),
            },
            "risks": risks,
            "data_status": data_status,
            "features": f,
            "news": news,
            "macro_reason": macro_reason,
            "recent_pick_date": recent_picks.get(ticker),
            "repeat_penalty": repeat_penalty,
            "technical_confirmations": _technical_confirmations(f),
            "risk_penalty": round(risk_penalty, 2),
        })

    scored.sort(key=lambda r: (-r["score"], r["ticker"]))
    return scored, {
        "universe_size": len(tickers),
        "model_universe_size": len(model_universe),
        "model_universe_size_before_rank_cap": len(model_universe_all),
        "recommendation_universe_rank_max": RECOMMENDATION_UNIVERSE_RANK_MAX,
        "news_pool_size": len(news_pool),
        "news_pool_candidates": extra_kept,
        "news_pool_excluded_liquidity": extra_excluded_liquidity,
        "history_rows": len(features),
        "history_missing": len(tickers) - len(features),
        "technical_as_of": max((f["as_of_date"] for f in features.values()), default=None),
        "technical_age_days": (
            (datetime.fromisoformat(trade_date).date()
             - datetime.fromisoformat(max((f["as_of_date"] for f in features.values()), default=trade_date)).date()).days
            if features else None
        ),
        "announcements": ann_quality,
        "recent_pick_sessions": RECENT_PICK_SESSIONS,
        "recent_pick_tickers": sorted(recent_picks),
        "sector_map_rows": sum(1 for t in tickers if t in sector_map),
    }, market


def _rationale(ticker: str, f: dict[str, Any], news: dict[str, Any], score: float,
               market: dict[str, Any], macro_reason: str | None = None) -> str:
    bits = [f"Composite evidence score {score:.1f}/100"]
    if f.get("above_sma20") and f.get("above_sma50"):
        bits.append("price is above its 20d and 50d averages")
    elif f.get("above_sma50"):
        bits.append("price remains above its 50d average")
    if _finite(f.get("ret_5_pct")):
        bits.append(f"recent 5-session return {float(f['ret_5_pct']):+.1f}%")
    if _finite(f.get("relative_ret_20_pct")):
        bits.append(f"20-session relative strength {float(f['relative_ret_20_pct']):+.1f}% vs ASX 200")
    if news.get("status") == "catalyst_continuation":
        bits.append(
            f"scored price-sensitive catalyst from {news.get('catalyst_session_date')} "
            f"remains in the {news.get('catalyst_age_sessions')}-session continuation window"
        )
    if _finite(f.get("drawdown_20d_pct")) and -12 <= float(f["drawdown_20d_pct"]) <= -1:
        bits.append(f"controlled pullback {float(f['drawdown_20d_pct']):+.1f}% from 20d high")
    if news.get("score") is not None:
        if news.get("status", "").startswith("headline_proxy"):
            bits.append(f"fresh headline proxy {news['score']}/100 ({news.get('headline') or 'directional release'})")
        else:
            bits.append(f"announcement score {news['score']}/100 ({news.get('status')})")
    elif news.get("status") == "no_recent_news":
        bits.append("no fresh announcement score, so news contributes no directional edge")
    if news.get("headline") and news.get("status") in {"current_scored", "current_document_scored"}:
        bits.append(f"fresh release: {news['headline']}")
    if _finite(f.get("volume_ratio")) and float(f["volume_ratio"]) >= 2:
        bits.append(f"prior-session volume {float(f['volume_ratio']):.1f}x its 20d median")
    if macro_reason and macro_reason != "no strong overnight commodity or sector-beta impulse":
        bits.append(macro_reason)
    return "; ".join(bits) + "."


def _select_top(scored: list[dict[str, Any]], market: dict[str, Any]) -> list[dict[str, Any]]:
    """Select at most one name per sector before filling the three slots."""
    selected: list[dict[str, Any]] = []
    sectors: set[str] = set()
    for row in scored:
        news = row.get("news") or {}
        if row["score"] < MIN_RECOMMENDATION_SCORE:
            continue
        # A positive fresh headline can enter as a clearly labelled,
        # lower-confidence catalyst candidate. An unscored neutral release is
        # not enough; an unscored negative headline is rejected.
        if news.get("halt"):
            continue
        # A current score in the bearish band is not rescued by a good chart;
        # this slightly wider cutoff also keeps mildly negative fresh releases
        # out of the buy shortlist while leaving neutral news rankable.
        if news.get("score") is not None and float(news["score"]) <= 45:
            continue
        if news.get("pending") and news.get("headline_signal") != "positive":
            continue
        if news.get("headline_signal") == "negative":
            continue
        if news.get("catalyst_continuation"):
            # Continuation is deliberately stricter than fresh news: the
            # catalyst must still be scored and price must confirm it rather
            # than merely bouncing on the old headline.
            if (float(news.get("score") or 0) < CATALYST_CONTINUATION_MIN_SCORE
                    or (news.get("catalyst_age_sessions") or 99) > CATALYST_CONTINUATION_SESSIONS):
                continue
        # A fresh positive catalyst can justify a name before its chart has
        # confirmed. Without one, require at least two causal confirmations;
        # otherwise a single old trend rank can become a recommendation.
        features = row.get("features") or {}
        if news.get("catalyst_continuation"):
            if (_technical_confirmations(features) < CATALYST_CONTINUATION_MIN_TECHNICAL_CONFIRMATIONS
                    or not features.get("above_sma50")
                    or not _finite(features.get("relative_ret_5_pct"))
                    or float(features["relative_ret_5_pct"]) <= 0.0):
                continue
        if features and row.get("technical_confirmations", _technical_confirmations(features)) < MIN_TECHNICAL_CONFIRMATIONS:
            if not _positive_catalyst(news):
                continue
        # Rotate the live list. A name can repeat only when it has a fresh
        # current-session catalyst; otherwise the ranker is merely recycling
        # yesterday's technical leaders.
        if row.get("recent_pick_date") and not _fresh_catalyst(news):
            continue
        if row["sector"] in sectors:
            continue
        selected.append(row)
        sectors.add(row["sector"])
        if len(selected) >= N_PICKS:
            break

    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
        row["decision"] = "WATCH_ONLY" if market.get("hard_abstain") else "BUY_CANDIDATE"
    return selected


def _backtest_evidence() -> dict[str, Any]:
    """The current research verdict travels with every morning's output."""
    return {
        "status": "not_validated",
        "primary_outcome": "previous_close_to_session_close",
        "execution_outcome": "session_open_to_session_close",
        "summary": (
            "Existing dashboard tests have not demonstrated a reliable multi-day timing edge. "
            "The pullback and fitted range-model results failed their random-entry/selection gates; "
            "the overnight weakness result is narrow, cost-sensitive and regime-dependent. "
            "Recommendation quality is therefore judged first on the full session from the prior "
            "close, with open-to-close retained as the executable entry lens."
        ),
        "gates": ["realistic fills", "randomised-entry null", "buy-and-hold comparison"],
        "link": "/backtest",
    }


def generate(trade_date: str | None = None, force: bool = False) -> dict[str, Any]:
    """Generate and persist one idempotent pre-open run."""
    trade_date = trade_date or sydney_today()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM recommendation_runs WHERE trade_date=?", (trade_date,)
        ).fetchone()
        if existing and not force:
            return get_run(trade_date)

    scored, quality, market = _candidate_rows(trade_date)
    selected = _select_top(scored, market)
    market_abstain = bool(market.get("hard_abstain"))
    if market_abstain:
        status = "abstain_market"
        note = "Market-wide abstain: " + "; ".join(market.get("reasons") or ["risk regime"])
    elif not quality.get("history_rows"):
        status = "abstain_data_unavailable"
        note = ("No completed technical price history was available from the market-data provider; "
                "no picks were generated and no stale candidates were carried forward.")
    elif (quality.get("technical_age_days") is not None
          and quality["technical_age_days"] > 3):
        status = "abstain_stale_data"
        note = (f"Latest completed price bar is {quality['technical_age_days']} days old; "
                "no forced picks on a stale session.")
    elif not selected:
        status = "abstain_insufficient"
        note = "No candidate cleared the data and risk filters; no forced pick."
    else:
        status = "ready" if len(selected) == N_PICKS else "ready_partial"
        note = (f"{len(selected)} diversified candidate(s) cleared the transparent rank and risk "
                "filters; fewer than three is acceptable when the remaining names do not qualify.")

    # Never call a row a BUY when the run abstained. The candidates remain
    # visible as a research shortlist, but the decision is explicit.
    if not status.startswith("ready"):
        for row in selected:
            row["decision"] = "WATCH_ONLY"
    generated_at = _now().isoformat(timespec="seconds")
    as_of_date = quality.get("technical_as_of")
    with _connect() as conn:
        if existing:
            run_id = int(existing["id"])
            conn.execute("DELETE FROM recommendations WHERE run_id=?", (run_id,))
            conn.execute(
                "UPDATE recommendation_runs SET generated_at=?, status=?, strategy_version=?,"
                " as_of_date=?, market_regime_json=?, data_quality_json=?, backtest_evidence_json=?, note=?"
                " WHERE id=?",
                (generated_at, status, STRATEGY_VERSION, as_of_date, _json(market),
                 _json(quality), _json(_backtest_evidence()), note, run_id),
            )
        else:
            cur = conn.execute(
                "INSERT INTO recommendation_runs (trade_date,generated_at,status,strategy_version,"
                "as_of_date,market_regime_json,data_quality_json,backtest_evidence_json,note)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (trade_date, generated_at, status, STRATEGY_VERSION, as_of_date,
                 _json(market), _json(quality), _json(_backtest_evidence()), note),
            )
            run_id = int(cur.lastrowid)
        for row in selected:
            conn.execute(
                "INSERT INTO recommendations (run_id,rank,ticker,company,sector,score,decision,"
                "reference_price,target_price,stop_price,hold_days,rationale,components_json,"
                "risks_json,data_status,entry_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, row["rank"], row["ticker"], row["company"], row["sector"], row["score"],
                 row["decision"], row["reference_price"], row["target_price"], row["stop_price"],
                 row["hold_days"], row["rationale"], _json(row["components"]), _json(row["risks"]),
                 row["data_status"], trade_date),
            )
        conn.commit()
    return get_run(trade_date)


def _row_to_json(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for key in ("components_json", "risks_json"):
        if key not in out:
            continue
        dest = key.removesuffix("_json")
        out[dest] = _loads(out.pop(key), {} if dest == "components" else [])
    return out


def get_run(trade_date: str | None = None) -> dict[str, Any]:
    requested_trade_date = trade_date or datetime.now(SYDNEY).date().isoformat()
    with _connect() as conn:
        if trade_date:
            run = conn.execute(
                "SELECT * FROM recommendation_runs WHERE trade_date=?", (trade_date,)
            ).fetchone()
        else:
            run = conn.execute("SELECT * FROM recommendation_runs ORDER BY trade_date DESC LIMIT 1").fetchone()
        if not run:
            return {
                "run": None,
                "recommendations": [],
                "history": [],
                "requested_trade_date": requested_trade_date,
                "is_current": False,
            }
        run_out = dict(run)
        # The no-argument endpoint is used by the dashboard for "today". Do
        # not let the last completed Friday run look like a fresh Monday run
        # when the scheduled job is still running or has failed.
        run_out["requested_trade_date"] = requested_trade_date
        run_out["is_current"] = run["trade_date"] == requested_trade_date
        for key in ("market_regime_json", "data_quality_json", "backtest_evidence_json"):
            run_out[key.removesuffix("_json")] = _loads(run_out.pop(key), {})
        recs = [_row_to_json(r) for r in conn.execute(
            "SELECT * FROM recommendations WHERE run_id=? ORDER BY rank", (run["id"],)
        )]
        history = []
        for hist in conn.execute(
            "SELECT id,trade_date,status,note FROM recommendation_runs "
            "ORDER BY trade_date DESC LIMIT 30"
        ):
            item = dict(hist)
            item["recommendations"] = [_row_to_json(r) for r in conn.execute(
                "SELECT ticker,rank,decision,entry_date,entry_prev_close,entry_open,entry_high,entry_close,"
                "entry_gap_pct,entry_session_pct,entry_open_to_close_pct,"
                "fwd_1d_pct,fwd_3d_pct,fwd_5d_pct,"
                "fwd_10d_pct,mfe_5d_pct,mae_5d_pct FROM recommendations WHERE run_id=? ORDER BY rank",
                (hist["id"],),
            )]
            history.append(item)
    return {"run": run_out, "recommendations": recs, "history": history}


def resolve_outcomes(days_back: int = 30) -> dict[str, Any]:
    """Resolve the session split and mature forward returns for each pick."""
    from .backtest import fetch_daily_history

    since = (datetime.now(SYDNEY).date() - timedelta(days=days_back)).isoformat()
    sydney_now = datetime.now(SYDNEY)
    sydney_today = sydney_now.date().isoformat()
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT r.id,u.trade_date,r.ticker,r.reference_price FROM recommendations r "
            "JOIN recommendation_runs u ON u.id=r.run_id "
            "WHERE (r.entry_prev_close IS NULL OR r.entry_open IS NULL OR r.entry_high IS NULL "
            "OR r.entry_close IS NULL OR r.entry_session_pct IS NULL) "
            "AND r.entry_date>=?", (since,)
        )]
    updated = 0
    for row in rows:
        try:
            df = fetch_daily_history(row["ticker"], period="3mo", market="AU")
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        dates = [d.date().isoformat() for d in df.index]
        after = [d for d in dates if d >= row["trade_date"]]
        if not after:
            continue
        entry_day = after[0]
        i = dates.index(entry_day)
        raw_entry, raw_high, raw_close = (df.iloc[i][column]
                                          for column in ("Open", "High", "Close"))
        if not all(_finite(value) for value in (raw_entry, raw_high, raw_close)):
            continue
        entry, high, close = (float(value) for value in (raw_entry, raw_high, raw_close))
        raw_prev_close = df.iloc[i - 1]["Close"] if i > 0 else None
        prev_close = float(raw_prev_close) if _finite(raw_prev_close) else None
        gap_pct = ((entry / prev_close - 1.0) * 100.0
                   if prev_close else None)
        session_pct = ((close / prev_close - 1.0) * 100.0
                       if prev_close else None)
        open_to_close_pct = (close / entry - 1.0) * 100.0 if entry else None
        # Never persist today's bar while ASX is still trading (or before its
        # open). A dated row can exist in the provider before its High/Close
        # is final, so the clock check is part of the data-quality guard.
        if entry_day > sydney_today or (entry_day == sydney_today and sydney_now.time() < time(16, 0)):
            continue
        # Wait for a following daily bar before recording anything. That means
        # the selected session is complete, so High and Close cannot be an
        # intraday partial value when the testing record is viewed.
        if i >= len(df) - 1:
            # A completed session's OHLC is useful immediately even when a
            # later bar has not arrived yet. Forward returns remain NULL until
            # their required future bars exist.
            with _connect() as conn:
                conn.execute(
                    "UPDATE recommendations SET entry_date=?,entry_prev_close=?,entry_open=?,"
                    "entry_high=?,entry_close=?,entry_gap_pct=?,entry_session_pct=?,"
                    "entry_open_to_close_pct=? WHERE id=?",
                    (entry_day, prev_close, entry, high, close, gap_pct,
                     session_pct, open_to_close_pct, row["id"]),
                )
                conn.commit()
            updated += 1
            continue
        vals: dict[str, Any] = {
            "entry_date": entry_day,
            "entry_prev_close": prev_close,
            "entry_open": entry,
            "entry_high": high,
            "entry_close": close,
            "entry_gap_pct": gap_pct,
            "entry_session_pct": session_pct,
            "entry_open_to_close_pct": open_to_close_pct,
        }
        for horizon in OUTCOME_HORIZONS:
            j = i + horizon
            vals[f"fwd_{horizon}d_pct"] = (
                round((float(df.iloc[j]["Close"]) / entry - 1.0) * 100.0, 4)
                if j < len(df) else None
            )
        end = min(i + 5, len(df) - 1)
        vals["mfe_5d_pct"] = round((float(df.iloc[i:end + 1]["High"].max()) / entry - 1.0) * 100.0, 4)
        vals["mae_5d_pct"] = round((float(df.iloc[i:end + 1]["Low"].min()) / entry - 1.0) * 100.0, 4)
        vals["outcome_updated_at"] = _now().isoformat(timespec="seconds")
        with _connect() as conn:
            conn.execute(
                "UPDATE recommendations SET entry_date=?,entry_prev_close=?,entry_open=?,"
                "entry_high=?,entry_close=?,entry_gap_pct=?,entry_session_pct=?,"
                "entry_open_to_close_pct=?,"
                "fwd_1d_pct=?,fwd_3d_pct=?,"
                "fwd_5d_pct=?,fwd_10d_pct=?,mfe_5d_pct=?,mae_5d_pct=?,outcome_updated_at=? WHERE id=?",
                (vals["entry_date"], vals["entry_prev_close"], vals["entry_open"],
                 vals["entry_high"], vals["entry_close"], vals["entry_gap_pct"],
                 vals["entry_session_pct"], vals["entry_open_to_close_pct"],
                 vals["fwd_1d_pct"], vals["fwd_3d_pct"],
                 vals["fwd_5d_pct"], vals["fwd_10d_pct"], vals["mfe_5d_pct"], vals["mae_5d_pct"],
                 vals["outcome_updated_at"], row["id"]),
            )
            conn.commit()
        updated += 1
    return {"updated": updated, "considered": len(rows)}


def generate_and_resolve(trade_date: str | None = None, force: bool = False) -> dict[str, Any]:
    resolved = resolve_outcomes()
    result = generate(trade_date, force=force)
    result["resolved_outcomes"] = resolved
    return result
