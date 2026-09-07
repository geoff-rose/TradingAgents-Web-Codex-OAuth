"""FastAPI web server for TradingAgents."""

from __future__ import annotations

import asyncio
import datetime as _dt
import threading
import json
import os
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="TradingAgents Web")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Reports are stored under ~/.tradingagents/logs (respects TRADINGAGENTS_RESULTS_DIR override)
_HOME = Path.home() / ".tradingagents"
LOGS_DIR = Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", str(_HOME / "logs")))
COMPANY_DIR = _HOME / "company_info"
COMPANY_DIR.mkdir(exist_ok=True)

# Auth config
_SECRET_KEY = "138e54e633b69c30efb52175f90b80adf92bb646d02a5216b6666db167bec88c"
# Was hardcoded here despite /etc/tradingagents/env already defining
# TRADINGAGENTS_WEB_PASSWORD -- that env var was silently ignored (found
# 2026-08-21 while wiring up site-wide auth). Now actually read, falling back
# to the old hardcoded value only so existing deployments don't break if the
# env var isn't set.
_ANALYSIS_PASSWORD = os.environ.get("TRADINGAGENTS_WEB_PASSWORD", "gIZMRdploEpq0K7V")
_SESSION_COOKIE = "ta_session"
_SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days
_signer = URLSafeTimedSerializer(_SECRET_KEY)


def _is_authenticated(request: Request) -> bool:
    token = request.cookies.get(_SESSION_COOKIE)
    if not token:
        return False
    try:
        _signer.loads(token, max_age=_SESSION_MAX_AGE)
        return True
    except (BadSignature, SignatureExpired):
        return False


# Site-wide auth, added 2026-08-21. Previously only /analysis and /sync
# checked auth individually -- everything else (including data-bearing APIs
# like /api/reports, /api/asx/feed, /api/research/*) was reachable by anyone
# who found the URL. A single middleware is used rather than annotating every
# route by hand, so a future route is protected by default instead of by
# remembering to add a check -- that's exactly how /api/analyze and
# /api/stream ended up unprotected despite /analysis itself redirecting.
_PUBLIC_PATHS = {"/login", "/api/login", "/api/auth/status", "/api/logout"}

# Called by systemd timers / asxbrief's own webhook from localhost, with no
# browser session to present. Exempted only when the caller is local;
# otherwise falls through to the normal auth check like everything else.
_LOCAL_OR_AUTH_PATHS = {
    "/api/performance/refresh", "/api/performance/ingest", "/api/asx/signals/refresh",
    # Read-only progress for the background classifier; local-only so load can
    # be watched from the box without a browser session.
    "/api/asx/signals/status", "/api/scanner/finalize", "/api/scanner/scan",
    "/api/scanner/forward-returns", "/api/eod/scan", "/api/eod/resolve",
    "/api/eod/open-positions", "/api/eod/close-positions",
    "/api/gap-reversion/scan", "/api/gap-reversion/open-positions",
    "/api/gap-reversion/close-positions", "/api/markets/sector-map/rebuild",
    "/api/patterns/signal-outcomes/collect", "/api/costs/capture-spreads",
    "/api/patterns/openhigh-review", "/api/patterns/tuning-lane/score",
    "/api/asx/signals/health", "/api/health/freshness",
    "/api/markets/spi200/refresh", "/api/markets/futures/refresh", "/api/swing/propose", "/api/swing/sync",
}


@app.middleware("http")
async def require_auth(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static") or path in _PUBLIC_PATHS:
        return await call_next(request)
    if path in _LOCAL_OR_AUTH_PATHS and request.client and request.client.host in ("127.0.0.1", "::1"):
        return await call_next(request)
    if _is_authenticated(request):
        return await call_next(request)
    # Page routes redirect to the login page; API routes 401. Keyed off the
    # path, not an Accept header -- curl and other tools don't send
    # `Accept: text/html` the way browsers do, so that heuristic silently
    # 401s a plain `curl http://host/` instead of redirecting it.
    if request.method == "GET" and not path.startswith("/api/"):
        return RedirectResponse("/login", status_code=302)
    return JSONResponse({"detail": "Unauthorized"}, status_code=401)


# Report fields in display order; covers both new (md files) and legacy (JSON keys) formats
_REPORT_FIELDS = [
    ("final_trade_decision",  "Decision"),
    ("market_report",         "Market"),
    ("sentiment_report",      "Sentiment"),
    ("news_report",           "News"),
    ("fundamentals_report",   "Fundamentals"),
    ("short_interest_report", "Short Interest"),
    ("investment_plan",       "Research"),
    ("trader_investment_plan","Trader"),
]

_jobs: Dict[str, asyncio.Queue] = {}
_executor = ThreadPoolExecutor(max_workers=4)

# Nodes to suppress in the progress feed
_SKIP_PREFIXES = ("Msg Clear ", "tools_", "__")

# Team groupings matching the CLI's progress table
_NODE_TEAMS = {
    "Market Analyst":       "Analyst Team",
    "Social Analyst":       "Analyst Team",
    "News Analyst":         "Analyst Team",
    "Fundamentals Analyst": "Analyst Team",
    "Bull Researcher":      "Research Team",
    "Bear Researcher":      "Research Team",
    "Research Manager":     "Research Team",
    "Trader":               "Trading Team",
    "Aggressive Analyst":   "Risk Management",
    "Neutral Analyst":      "Risk Management",
    "Conservative Analyst": "Risk Management",
    "Portfolio Manager":    "Risk Management",
}


class AnalyzeRequest(BaseModel):
    ticker: str
    date: str
    provider: str = "xai-grok"
    deep_model: str = "gpt-5.4"
    quick_model: str = "gpt-5.4-mini"
    research_depth: int = 1
    analysts: List[str] = ["market", "social", "news", "fundamentals", "short"]


def _normalize_ticker(ticker: str) -> str:
    """Append .AX if the ticker has no exchange suffix (no dot)."""
    t = ticker.strip().upper()
    return t if '.' in t else t + '.AX'


def _run_analysis(request: AnalyzeRequest, emit: Callable[[Any], None]) -> None:
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        ticker = _normalize_ticker(request.ticker)

        config = {
            **DEFAULT_CONFIG,
            "llm_provider": request.provider,
            "deep_think_llm": request.deep_model,
            "quick_think_llm": request.quick_model,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
        }

        emit({"type": "status", "message": f"Initialising agents for {ticker}…"})

        ta = TradingAgentsGraph(selected_analysts=request.analysts, config=config)

        original_stream = ta.graph.stream

        def instrumented_stream(state, **kwargs):
            # Use combined stream modes: "updates" gives node names, "values" gives full state.
            # We yield only the values chunks to the caller so debug-mode merge works correctly.
            combined_kwargs = {**kwargs, "stream_mode": ["updates", "values"]}
            for mode, data in original_stream(state, **combined_kwargs):
                if mode == "updates":
                    for node_name in data:
                        if any(node_name.startswith(p) for p in _SKIP_PREFIXES):
                            continue
                        team = _NODE_TEAMS.get(node_name, "")
                        emit({"type": "node", "name": node_name, "team": team})
                elif mode == "values":
                    yield data  # pass full-state chunk to the caller unchanged

        ta.graph.stream = instrumented_stream
        ta.debug = True  # force the streaming code path in _run_graph

        result, _signal = ta.propagate(ticker, request.date)

        emit({
            "type": "complete",
            "result": {
                "final_trade_decision":  result.get("final_trade_decision") or "",
                "market_report":         result.get("market_report") or "",
                "sentiment_report":      result.get("sentiment_report") or "",
                "news_report":           result.get("news_report") or "",
                "fundamentals_report":   result.get("fundamentals_report") or "",
                "investment_plan":       result.get("investment_plan") or "",
                "trader_investment_plan":result.get("trader_investment_plan") or "",
            },
        })

    except Exception as exc:
        emit({"type": "error", "message": str(exc), "detail": traceback.format_exc()})
    finally:
        emit(None)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=(static_dir / "home.html").read_text())


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _is_authenticated(request):
        return RedirectResponse("/analysis", status_code=302)
    return HTMLResponse(content=(static_dir / "login.html").read_text())


@app.post("/api/login")
async def do_login(password: str = Form(...)):
    if password != _ANALYSIS_PASSWORD:
        return RedirectResponse("/login?error=1", status_code=302)
    token = _signer.dumps("authenticated")
    response = RedirectResponse("/analysis", status_code=302)
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        max_age=_SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/api/auth/status")
async def auth_status(request: Request):
    return {"authenticated": _is_authenticated(request)}


@app.get("/api/logout")
async def do_logout():
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(_SESSION_COOKIE)
    return response


@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    if not _is_authenticated(request):
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse(content=(static_dir / "analysis.html").read_text())


@app.get("/reports", response_class=HTMLResponse)
async def reports_page():
    return HTMLResponse(content=(static_dir / "reports.html").read_text())


@app.get("/api/reports")
async def list_reports():
    """Return all saved reports as [{ticker, date}] sorted newest first."""
    if not LOGS_DIR.exists():
        return []

    entries = []
    for ticker_dir in sorted(LOGS_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        # New format: logs/{ticker}/{date}/reports/*.md
        for date_dir in sorted(ticker_dir.iterdir(), reverse=True):
            if not date_dir.is_dir() or date_dir.name == "TradingAgentsStrategy_logs":
                continue
            if (date_dir / "reports").exists():
                entries.append({"ticker": ticker, "date": date_dir.name})

        # Legacy format: logs/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json
        legacy_dir = ticker_dir / "TradingAgentsStrategy_logs"
        if legacy_dir.exists():
            for f in sorted(legacy_dir.glob("full_states_log_*.json"), reverse=True):
                date = f.stem.replace("full_states_log_", "")
                entries.append({"ticker": ticker, "date": date})

    return entries


@app.get("/api/reports/{ticker}/{date}")
async def get_report(ticker: str, date: str):
    """Return report content for a given ticker and date."""
    ticker_dir = LOGS_DIR / ticker

    # New format
    reports_dir = ticker_dir / date / "reports"
    if reports_dir.exists():
        data: Dict[str, str] = {}
        for key, _ in _REPORT_FIELDS:
            # try exact key name, then trader_investment_plan variant
            for stem in (key, key.replace("trader_investment_plan", "trader_investment_decision")):
                md = reports_dir / f"{stem}.md"
                if md.exists():
                    data[key] = md.read_text()
                    break
        return data

    # Legacy JSON format
    legacy = ticker_dir / "TradingAgentsStrategy_logs" / f"full_states_log_{date}.json"
    if legacy.exists():
        raw = json.loads(legacy.read_text())
        return {
            "final_trade_decision":  raw.get("final_trade_decision") or "",
            "market_report":         raw.get("market_report") or "",
            "sentiment_report":      raw.get("sentiment_report") or "",
            "news_report":           raw.get("news_report") or "",
            "fundamentals_report":   raw.get("fundamentals_report") or "",
            "short_interest_report": raw.get("short_interest_report") or "",
            "investment_plan":       raw.get("investment_plan") or "",
            # legacy key name differs
            "trader_investment_plan": raw.get("trader_investment_plan")
                                   or raw.get("trader_investment_decision") or "",
        }

    raise HTTPException(status_code=404, detail="Report not found")


@app.get("/api/company/{code}")
async def get_company(code: str):
    """Return cached company info for a ticker code (e.g. BHP, not BHP.AX)."""
    code = code.upper().replace(".AX", "")
    path = COMPANY_DIR / f"{code}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No cached company data")
    return json.loads(path.read_text())


@app.get("/api/performance")
async def get_performance():
    """Return all recommendations and summary stats from the performance DB."""
    from tradingagents.performance.db import get_performance_data
    return get_performance_data()


@app.post("/api/performance/refresh")
async def refresh_performance():
    """Trigger snapshot fill for all pending recommendations (background)."""
    from tradingagents.performance.db import refresh_all_snapshots
    loop = asyncio.get_running_loop()
    updated = await loop.run_in_executor(_executor, refresh_all_snapshots)
    return {"updated": updated}


@app.post("/api/performance/ingest")
async def ingest_performance():
    """Import any report files from logs dir not yet in the performance DB."""
    from tradingagents.performance.db import ingest_from_logs
    loop = asyncio.get_running_loop()
    counts = await loop.run_in_executor(_executor, ingest_from_logs)
    return counts


@app.get("/performance", response_class=HTMLResponse)
async def performance_page():
    html = (Path(__file__).parent / "static" / "performance.html").read_text()
    return HTMLResponse(html)


@app.get("/asx", response_class=HTMLResponse)
async def asx_dashboard_page():
    html = (Path(__file__).parent / "static" / "asx.html").read_text()
    return HTMLResponse(html)


@app.get("/api/asx/feed")
async def asx_feed(limit: int = 100, kind: str = "all", universe_only: bool = False,
                    group: bool = False, day: str | None = None,
                    ticker: str | None = None):
    """`day=today` (or an explicit YYYY-MM-DD) restricts to one ASX session,
    measured in Sydney local time -- see asx_feed.sydney_day_bounds for why
    the UTC date is the wrong boundary here."""
    from tradingagents.asx_feed import recent_announcements, sydney_today
    from tradingagents.asx_signals import attach_signals, group_announcements
    from tradingagents.sectors import MIN_CORR, get_map
    # A ticker search spans sessions, so the day filter is dropped for it --
    # otherwise searching for a company that announced yesterday returns
    # nothing while looking like it worked.
    items = recent_announcements(limit=limit, kind=kind, universe_only=universe_only,
                                 session_date=None if ticker else day,
                                 ticker=ticker)
    attach_signals(items)
    resolved = None if ticker else (sydney_today() if day == "today" else day)
    out = group_announcements(items) if group else items

    # Sector comes from the correlation map, so an unmapped ticker (or one
    # that correlates with nothing and falls back to the index) is shown as
    # blank rather than being given a sector it does not actually track --
    # the whole point of the column is to explain a move, and a wrong label
    # explains it wrongly.
    smap = get_map(sorted({r["ticker"] for r in out if r.get("ticker")}))
    for r in out:
        m = smap.get(r.get("ticker")) or {}
        r["sector"] = m.get("sector_name")
        r["sector_index"] = m.get("sector_index")
        r["sector_corr"] = m.get("sector_corr")
        # Whether the label is strong enough to also be a valid benchmark --
        # the page greys the weak ones rather than hiding them.
        r["sector_weak"] = bool(m) and (m.get("sector_corr") or 0) < MIN_CORR
    return {"items": out, "grouped": group, "session_date": resolved,
            "ticker": ticker.strip().upper() if ticker else None}


@app.get("/api/asx/health")
async def asx_health_endpoint():
    from tradingagents.asx_feed import health
    from tradingagents.asx_signals import provider_status
    # Classifier provider is part of health: when its credential lapses the
    # feed keeps working and scores silently stop appearing, which reads as
    # "nothing notable today" rather than "the classifier is down". That is
    # exactly what happened on 2026-08-23 when the xAI credits ran out.
    return {**health(), "classifier": provider_status()}


# Classification became a long job once each announcement's PDF started being
# fetched (~6s each: interstitial GET + PDF GET + a much larger LLM prompt).
# asxbrief's webhook has a **5 second** timeout, so a synchronous endpoint now
# times out on the very first announcement. Run it in the background and answer
# the webhook immediately; the single-flight guard stops runs piling up, since
# the collector fires every 5 minutes and a large batch takes longer than that.
_signals_run_lock = threading.Lock()
_signals_last_result: dict[str, Any] = {}


def _classify_in_background(limit: int) -> None:
    from tradingagents.asx_signals import classify_pending
    if not _signals_run_lock.acquire(blocking=False):
        return
    try:
        _signals_last_result.update(classify_pending(limit))
        # Per-ticker netting must follow classification, not run separately:
        # a ticker's net score is only meaningful once all of its announcements
        # for the session have been scored.
        from tradingagents.asx_signals import refresh_ticker_signals
        _signals_last_result["ticker_netting"] = refresh_ticker_signals()
        _signals_last_result["finished_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds")
    except Exception as exc:
        _signals_last_result.update({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        _signals_run_lock.release()


@app.get("/api/health/freshness")
async def health_freshness():
    """503 when a job reported success but the table it owns did not advance.

    Exit codes cannot catch this: five jobs here have returned success while
    writing nothing. This compares each unit's last successful run against the
    last write to the data it owns, so a hollow run is visible in
    `systemctl --failed` within the hour.

    Jobs whose output is legitimately intermittent are excluded by name, with
    the reason, in freshness.SKIPPED -- a monitor that fires on a normal quiet
    day gets ignored, and then it protects nothing."""
    from tradingagents.freshness import check
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(_executor, check)
    if not data["ok"]:
        raise HTTPException(status_code=503, detail=data)
    return data


@app.get("/api/asx/signals/health")
async def asx_signals_health(hours: int = 6, min_attempts: int = 20):
    """503 when the classifier has been attempting calls and none succeed.

    The refresh endpoint is fire-and-forget, so a provider outage returns HTTP
    200 with an empty result and the timer reports success -- which is how
    gpt-5.4 being cut off for ChatGPT-account Codex auth ran for a full trading
    day unnoticed (2026-09-06, 778 calls, zero successes). This is the third
    failure of that shape here, after asx-swing-sync's {"checked":0} during an
    IBKR logout and the spread capture storing NaN rows.

    A non-200 makes `asx-signals-health.service` go red, so the outage shows up
    in `systemctl --failed` instead of needing someone to notice missing scores.
    """
    import sqlite3 as _sq
    from datetime import datetime, timedelta, timezone
    from tradingagents.asx_signals import DB_PATH
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)
             ).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _sq.connect(f"file:{DB_PATH}?mode=ro", uri=True) as c:
        row = c.execute("SELECT COUNT(*), COALESCE(SUM(ok),0), MAX(error)"
                        " FROM llm_calls WHERE called_at >= ?", (since,)).fetchone()
    attempts, ok, last_err = row[0], row[1], row[2]
    body = {"window_hours": hours, "attempts": attempts, "succeeded": ok,
            "last_error": (last_err or "")[:300]}

    # Record quota alongside liveness. Until 2026-09-07 nothing here knew what
    # the subscription limits even were, so "will this batch fit" was guesswork
    # -- and the guess was 42% low. One cheap call an hour (~1.7% of a day's
    # allowance) buys a usage history and an early warning. Never fatal on its
    # own: a deliberate batch run legitimately drives usage high, and a unit
    # that goes red during normal work stops being read.
    try:
        from tradingagents.quota import snapshot
        body["quota"] = snapshot()
    except Exception as exc:                     # never let telemetry break the check
        body["quota"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    if attempts >= min_attempts and ok == 0:
        raise HTTPException(status_code=503, detail={
            "error": f"classifier has made {attempts} calls in {hours}h with ZERO "
                     f"successes -- the provider is refusing every request",
            **body})
    return {"ok": True, **body}


@app.post("/api/asx/signals/refresh")
async def asx_signals_refresh(limit: int = 60, wait: bool = False):
    """`wait=true` runs synchronously (manual/testing use). The default is
    fire-and-forget so asxbrief's 5s webhook timeout is never the thing that
    decides whether announcements get classified."""
    from tradingagents.asx_signals import classify_pending
    loop = asyncio.get_running_loop()
    if wait:
        return await loop.run_in_executor(_executor, classify_pending, limit)
    if _signals_run_lock.locked():
        return {"status": "already running", "last_result": _signals_last_result}
    loop.run_in_executor(_executor, _classify_in_background, limit)
    return {"status": "started", "limit": limit, "last_result": _signals_last_result}


@app.get("/api/asx/signals/usage")
async def asx_signals_usage(days: int = 14):
    """Actual provider call counts per day -- what the classifier really costs,
    as opposed to what the cached-row counts imply."""
    from tradingagents.asx_signals import usage_summary
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, usage_summary, days)


@app.get("/api/asx/signals/status")
async def asx_signals_status():
    from tradingagents.announcement_body import fetch_stats
    return {"running": _signals_run_lock.locked(),
            "last_result": _signals_last_result,
            "body_cache": fetch_stats()}


@app.get("/research", response_class=HTMLResponse)
async def research_page():
    html = (Path(__file__).parent / "static" / "research.html").read_text()
    return HTMLResponse(html)


class FocusTickerRequest(BaseModel):
    ticker: str


@app.get("/api/research/universe")
async def research_universe_get():
    from tradingagents.research_universe import bar_coverage, get_universe
    data = get_universe()
    coverage = bar_coverage()
    for group in ("focus", "recommended"):
        for row in data[group]:
            row["bars"] = coverage.get(row["ticker"])
    return data


@app.post("/api/research/focus")
async def research_focus_add(req: FocusTickerRequest):
    from tradingagents.research_universe import add_focus_ticker
    result = add_focus_ticker(req.ticker)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.delete("/api/research/ticker/{ticker}")
async def research_ticker_remove(ticker: str):
    from tradingagents.research_universe import remove_ticker
    result = remove_ticker(ticker)
    if not result["ok"]:
        raise HTTPException(status_code=404, detail=f"{ticker.upper()} not in research universe")
    return result


@app.post("/api/research/recommend")
async def research_recommend():
    from tradingagents.research_universe import run_recommend
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, run_recommend)


@app.post("/api/research/backfill")
async def research_backfill_start(years: float = 0.25):
    from tradingagents.research_universe import get_universe, start_backfill
    universe = get_universe()
    tickers = [r["ticker"] for r in universe["focus"] + universe["recommended"]]
    if not tickers:
        raise HTTPException(status_code=400, detail="research universe is empty -- add focus tickers or recommend some first")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, start_backfill, tickers, years)


@app.get("/api/research/backfill/status")
async def research_backfill_status():
    from tradingagents.research_universe import backfill_status
    return backfill_status()


@app.get("/api/research/backfill/estimate")
async def research_backfill_estimate(years: float = 0.25):
    from tradingagents.research_universe import backfill_estimate_hours, get_universe
    universe = get_universe()
    n = len(universe["focus"]) + len(universe["recommended"])
    return {"n_tickers": n, "years": years, "estimate_hours": backfill_estimate_hours(n, years)}


@app.get("/api/markets/snapshot")
async def markets_snapshot():
    from tradingagents.markets import get_snapshot
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, get_snapshot)


@app.post("/api/markets/futures/refresh")
async def markets_futures_refresh():
    """Refresh ES/NQ/SPI from IB Gateway into the futures cache.

    503 when nothing came back, so asx-futures-refresh.service goes red and the
    dashboard's own status line tells the user the panel has fallen back to
    delayed Yahoo values."""
    from tradingagents.markets import refresh_futures
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(_executor, refresh_futures)
    if not data.get("items"):
        raise HTTPException(status_code=503, detail=data)
    return data


@app.post("/api/markets/spi200/refresh")
async def markets_spi200_refresh():
    """503 when no quote could be obtained, so asx-spi200-refresh.service goes
    red rather than persisting nulls. Barchart started answering with an empty
    HTTP 202 at some point before 2026-09-07 and the dashboard row read "--"
    for days while the timer reported success."""
    from tradingagents.markets import refresh_spi200
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(_executor, refresh_spi200)
    if data.get("last") is None:
        raise HTTPException(status_code=503, detail=data)
    return data


@app.get("/api/markets/sectors")
async def markets_sectors():
    """S&P/ASX sector indices with day/week/month moves, sorted by the day.
    The ASX 200 comes back as `market` so a sector move can be read against
    the index without a second request."""
    from tradingagents.sectors import board
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, board)


@app.post("/api/patterns/signal-outcomes/collect")
async def patterns_collect_outcomes(days: int = 30):
    """Snapshot prices for every scored ticker-session. Runs daily alongside
    forward-returns: horizons mature over the following fortnight, so this is
    re-run rather than written once."""
    from tradingagents.signal_outcomes import collect
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: collect(days=days))


@app.post("/api/patterns/openhigh-review")
async def patterns_openhigh_review(session: str = "", top_n: int = 5,
                                   window: int = 3, store: bool = True):
    """Rank a session's movers by open-to-high and check the news in front of
    them against what the classifier scored it.

    Looks for the classifier's unmeasured failure: a LOW or neutral score
    followed by a big upward move. Called by asx-openhigh-review.timer at 20:45
    Sydney -- after the 20:15 mover-finalize run has written the true high, and
    the ten-minute signals refresh has scored the day's announcements.

    Returns the base rate for the session alongside the top N. The base rate is
    not decoration: most scores sit at 50, so a big mover with an unremarkable
    score is the default outcome, not evidence."""
    from tradingagents.openhigh_review import review
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, lambda: review(session=session, top_n=top_n,
                                  window=window, store=store))


@app.get("/api/patterns/openhigh-review")
async def patterns_openhigh_stored(session: str = ""):
    """What the nightly review stored, read-only. The dashboard uses this so
    rendering never depends on a recompute that could disagree with the row the
    timer actually wrote."""
    from tradingagents.openhigh_review import stored
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: stored(session=session))


@app.post("/api/patterns/tuning-lane/score")
async def tuning_lane_score(session: str = "", limit: int = 400):
    """Score a session under the active tuned classifier version.

    Called daily by asx-tuning-lane.timer. A no-op until a tuned version is
    minted, so the lane costs no quota until it actually diverges from
    production."""
    from tradingagents.tuning_lane import score_session
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, lambda: score_session(session=session, limit=limit))


@app.get("/api/patterns/tuning-lane")
async def tuning_lane_report(horizon: str = "fwd_5d_pct"):
    """Every registered classifier version with its OUT-OF-SAMPLE record.

    In-sample rows are excluded by construction: a tuned version measured on
    outcomes it was tuned against wins by fitting, not by skill."""
    from tradingagents.tuning_lane import report
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: report(horizon=horizon))


@app.get("/api/patterns/openhigh-flags")
async def patterns_openhigh_flags(limit: int = 20):
    """Flagged rows across sessions -- the accumulating record. One session
    cannot show a classifier problem; this is what makes a real test possible
    later."""
    from tradingagents.openhigh_review import recent
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: recent(limit=limit))


@app.get("/api/patterns/hypotheses")
async def patterns_hypotheses(limit: int = 100):
    """The multiple-testing ledger: every hypothesis run, its verdict, and
    which gate killed it."""
    from tradingagents.hypothesis import history
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: history(limit=limit))


@app.post("/api/costs/capture-spreads")
async def costs_capture_spreads(limit: int = 60):
    """Snapshot real bid/ask from IBKR for the most-traded names. Must run
    during ASX continuous trading -- a quote taken outside it is frozen and
    wider than the book you would actually cross."""
    from tradingagents.costs import capture_live
    from tradingagents.gap_reversion import _all_tickers
    from tradingagents.gap_study import _universe

    def run():
        ts = ["VAS", "A200", "STW", "IVV", "NDQ"] + _universe(limit) + list(_all_tickers())
        return capture_live(sorted(set(ts)))

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, run)


@app.get("/api/costs/spreads")
async def costs_spreads():
    """Measured spreads: the Roll/floor estimate and the live median beside it."""
    from tradingagents.costs import live_spread_map, spread_map
    loop = asyncio.get_running_loop()
    est, live = await asyncio.gather(
        loop.run_in_executor(_executor, spread_map),
        loop.run_in_executor(_executor, live_spread_map))
    keys = sorted(set(est) | set(live))
    return {"rows": [{"ticker": k, "estimated_pct": est.get(k),
                      "live_median_pct": live.get(k)} for k in keys],
            "n_live": len(live)}


@app.get("/api/patterns/ic")
async def patterns_ic(min_names: int = 5):
    """Information Coefficient of the announcement classifier -- per-date rank
    correlation between score and forward return, averaged over dates. See
    tradingagents/factor_ic.py for why per-date rather than pooled."""
    from tradingagents.factor_ic import compute
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: compute(min_names=min_names))


@app.get("/api/markets/sector-map")
async def markets_sector_map():
    """The ticker -> sector-index assignments and how well each ticker tracks
    the index it was given (`sector_corr`)."""
    from tradingagents.sectors import get_map
    loop = asyncio.get_running_loop()
    m = await loop.run_in_executor(_executor, get_map)
    rows = sorted(m.values(), key=lambda r: (r["sector_name"] or "", r["ticker"]))
    return {"n": len(rows), "rows": rows}


@app.post("/api/markets/sector-map/rebuild")
async def markets_sector_map_rebuild(limit: int = 500):
    """Recompute the ticker -> sector mapping from the last year of returns.
    Slow (a few minutes for 500 names) and only needs running occasionally --
    sector membership drifts over quarters, not days."""
    from tradingagents.gap_study import _universe
    from tradingagents.mover_log import _connect as mover_connect
    from tradingagents.sectors import announcing_tickers, build_map

    def run():
        with mover_connect() as conn:
            logged = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM mover_log")]
        # Everything that can appear anywhere on the dashboard: logged movers,
        # the scored universe, and every ticker the announcement feed shows.
        return build_map(sorted(set(logged) | set(_universe(limit))
                                | set(announcing_tickers())))

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, run)


@app.get("/api/markets/yield-curve")
async def markets_yield_curve():
    from tradingagents.markets import get_yield_curve
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, get_yield_curve)


@app.get("/swing", response_class=HTMLResponse)
async def swing_page():
    html = (Path(__file__).parent / "static" / "swing.html").read_text()
    return HTMLResponse(html)


@app.get("/api/swing/universe")
async def swing_universe_get():
    from tradingagents.research_universe import get_universe
    from tradingagents.swing_db import get_universe as get_swing_universe, set_universe_tickers
    focus_tickers = [r["ticker"] for r in get_universe()["focus"]]
    set_universe_tickers(focus_tickers)
    enabled = {r["ticker"]: bool(r["enabled"]) for r in get_swing_universe()}
    return {"tickers": [{"ticker": t, "enabled": enabled.get(t, False)} for t in focus_tickers]}


class SwingUniverseToggle(BaseModel):
    enabled: bool


@app.post("/api/swing/universe/{ticker}")
async def swing_universe_toggle(ticker: str, req: SwingUniverseToggle):
    from tradingagents.swing_db import set_enabled
    set_enabled(ticker, req.enabled)
    return {"ok": True}


@app.get("/api/swing/settings")
async def swing_settings_get():
    from tradingagents.swing_db import get_range_model_quantiles, get_trade_dollars
    entry_q, target_q = get_range_model_quantiles()
    return {"trade_dollars": get_trade_dollars(), "entry_q": entry_q, "target_q": target_q}


class SwingSettings(BaseModel):
    trade_dollars: float | None = None
    entry_q: float | None = None
    target_q: float | None = None


@app.post("/api/swing/settings")
async def swing_settings_set(req: SwingSettings):
    from tradingagents.swing_db import (
        get_range_model_quantiles, set_range_model_quantiles, set_trade_dollars,
    )
    try:
        if req.trade_dollars is not None:
            set_trade_dollars(req.trade_dollars)
        if req.entry_q is not None or req.target_q is not None:
            cur_entry, cur_target = get_range_model_quantiles()
            set_range_model_quantiles(req.entry_q or cur_entry, req.target_q or cur_target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.get("/api/swing/trades")
async def swing_trades_get(status: str | None = None):
    from tradingagents.swing_db import list_trades, pnl_summary
    return {"trades": list_trades(status), "pnl": pnl_summary()}


@app.post("/api/swing/trades/{trade_id}/approve")
async def swing_trade_approve(trade_id: int):
    from tradingagents.swing import approve_trade
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, approve_trade, trade_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "approve failed"))
    return result


@app.post("/api/swing/trades/{trade_id}/reject")
async def swing_trade_reject(trade_id: int):
    from tradingagents.swing import reject_trade
    result = reject_trade(trade_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "reject failed"))
    return result


@app.post("/api/swing/propose")
async def swing_propose():
    """Daily job (asx-swing-propose.timer): proposes new range-model entries
    for flat tickers AND re-quotes today's target for any already-open
    range-model position -- both are "once a day" concerns, run together."""
    from tradingagents.swing import propose_daily_range_model, refresh_open_targets
    loop = asyncio.get_running_loop()
    proposals = await loop.run_in_executor(_executor, propose_daily_range_model)
    refreshed = await loop.run_in_executor(_executor, refresh_open_targets)
    return {"proposals": proposals, "target_refresh": refreshed}


@app.post("/api/swing/propose-legacy-heuristic")
async def swing_propose_legacy():
    """Kept for reference/backtest parity -- not called by the live timer.
    See swing.py's module docstring for why the heuristic strategy is no
    longer proposed by default."""
    from tradingagents.swing import propose_daily
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, propose_daily)


@app.post("/api/swing/sync")
async def swing_sync():
    from tradingagents.swing import sync_all
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, sync_all)


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page():
    html = (Path(__file__).parent / "static" / "backtest.html").read_text()
    return HTMLResponse(html)


@app.get("/api/backtest/strategies")
async def backtest_strategies():
    from tradingagents.strategies import STRATEGIES
    out = [
        {"key": key, "name": s.name, "description": s.description,
         "default_params": s.default_params, "event_driven": s.event_driven}
        for key, s in STRATEGIES.items()
    ]
    from tradingagents.overnight import CONDITIONS as _OVERNIGHT_CONDITIONS
    out.append({
        "key": "overnight_hold", "name": "Overnight hold (buy close, sell next open)",
        "description": ("Buys the closing auction and sells the next opening auction across the whole "
                        "model universe. Motivated by the 2026-08-22 finding that ~94% of these names' "
                        "return accrues overnight (+94.0% raw) against +6.6% intraday. Reports a SPREAD "
                        "SWEEP (the per-night edge is the same order as the round trip) and a condition "
                        "null that fixes each ticker's night count and randomises which nights, so the "
                        "unconditional anomaly can't be mistaken for conditional skill."),
        "default_params": {"condition": "after_big_down_day", "period": "3y",
                            "conditions_available": list(_OVERNIGHT_CONDITIONS)},
        "event_driven": False,
    })
    out.append({
        "key": "range_model", "name": "Range model (AU swing universe)",
        "description": "Live fitted quantile-regression daily-refresh strategy -- runs over the whole "
                        "swing universe together, not a single ticker (see asx-dashboard skill for full "
                        "detail and the 2026-08-22 finding that it hasn't demonstrated timing edge).",
        "default_params": {"entry_q": 0.1, "target_q": 0.7}, "event_driven": False,
    })
    return {"strategies": out}


class BacktestRunRequest(BaseModel):
    ticker: str | None = None
    market: str = "AU"
    strategy: str
    params: dict = {}
    period: str = "10y"


@app.post("/api/backtest/run")
async def backtest_run(req: BacktestRunRequest):
    loop = asyncio.get_running_loop()
    if req.strategy == "range_model":
        from tradingagents.backtest_report import run_and_report_range_model
        from tradingagents.swing_db import enabled_tickers
        tickers = req.params.get("tickers") or enabled_tickers()
        entry_q = req.params.get("entry_q", 0.1)
        target_q = req.params.get("target_q", 0.7)
        result = await loop.run_in_executor(_executor, run_and_report_range_model, tickers, entry_q, target_q)
    elif req.strategy == "overnight_hold":
        from tradingagents.backtest_report import run_and_report_overnight
        from tradingagents.screener import get_model_universe
        tickers = req.params.get("tickers") or [
            r["ticker"] for r in get_model_universe(tradeable_only=True)
        ]
        condition = req.params.get("condition", "after_big_down_day")
        period = req.params.get("period", req.period or "3y")
        result = await loop.run_in_executor(
            _executor, run_and_report_overnight, tickers, condition, period,
        )
    else:
        from tradingagents.backtest_report import run_and_report
        if not req.ticker:
            raise HTTPException(status_code=400, detail="ticker is required for this strategy")
        result = await loop.run_in_executor(
            _executor, run_and_report, req.ticker, req.strategy, req.params, req.period, req.market,
        )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# --- Movers scanner + momentum paper trades (2026-08-22) ---------------------
_scan_cache: dict[str, Any] = {"at": 0.0, "data": None}
# 10 minutes, matching asx-scanner-refresh.timer's cadence (see that unit) --
# the timer keeps this cache warm on a schedule regardless of whether anyone
# has the page open, so a page-triggered request almost always just reads
# what the timer already fetched rather than doing its own yfinance round trip.
_SCAN_TTL_SECONDS = 600


@app.get("/scanner", response_class=HTMLResponse)
async def scanner_page():
    return HTMLResponse((static_dir / "scanner.html").read_text())


@app.get("/watchlist", response_class=HTMLResponse)
async def watchlist_page():
    return HTMLResponse((static_dir / "watchlist.html").read_text())


@app.get("/api/watchlist")
async def watchlist_list():
    """Watchlist rows joined to a delayed quote. The quote fetch is the slow
    part, so it goes to the executor like every other yfinance call here."""
    from tradingagents.watchlist import items_with_quotes
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, items_with_quotes)


@app.post("/api/watchlist")
async def watchlist_add(payload: dict):
    from tradingagents.watchlist import add
    try:
        return add(
            ticker=payload.get("ticker", ""), company=payload.get("company"),
            source=payload.get("source"), ai_score=payload.get("ai_score"),
            headline=payload.get("headline"), url=payload.get("url"),
            note=payload.get("note"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/watchlist/{ticker}")
async def watchlist_remove(ticker: str):
    from tradingagents.watchlist import remove
    return remove(ticker)


@app.get("/api/watchlist/stream")
async def watchlist_stream():
    """Server-Sent Events feed of watchlist quotes.

    One shared poller backs every connected browser (see quote_hub), so extra
    tabs cost nothing. `X-Accel-Buffering: no` matters if this ever sits behind
    nginx -- without it the proxy buffers the stream and the page appears
    frozen while events pile up upstream.
    """
    from tradingagents.quote_hub import event_stream
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.get("/api/watchlist/quote/{ticker}")
async def watchlist_quote(ticker: str):
    """Single-ticker quote for the Trade button's price prefill. Kept off the
    list endpoint on purpose: it makes an extra per-ticker call for the book,
    which is only worth paying for at the moment a trade is being opened."""
    from tradingagents.watchlist import get_quote_detail
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, get_quote_detail, ticker)


_scan_lock = asyncio.Lock()


@app.get("/api/scanner/scan")
async def scanner_scan(force: bool = False):
    """One scan at a time, process-wide.

    Without the lock two overlapping requests both missed the cache and both
    ran a scan, which collided inside yfinance and 500'd -- what the page
    showed as "Scan failed" on 2026-08-25. The cache is re-checked after the
    lock is acquired so the second caller gets the first one's fresh result
    instead of running a duplicate 30-60s download.
    """
    import time as _time
    from tradingagents.scanner import scan

    def _fresh() -> bool:
        return bool(_scan_cache["data"]) and (_time.time() - _scan_cache["at"]) < _SCAN_TTL_SECONDS

    if not force and _fresh():
        return {**_scan_cache["data"], "cached": True}
    async with _scan_lock:
        # Re-check: another request may have finished a scan while we queued.
        if not force and _fresh():
            return {**_scan_cache["data"], "cached": True}
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(_executor, scan)
        # Never cache a fetch that read nothing. "No movers" and "the download
        # failed" both come back as zero rows, and caching the latter pins an
        # empty table on the page for the whole TTL.
        if data.get("n_bars_read", 0) > 0 or not data.get("n_candidates"):
            _scan_cache.update({"at": _time.time(), "data": data})
        return {**data, "cached": False}


class MomentumOpenRequest(BaseModel):
    ticker: str
    entry_price: float
    dollars: float = 5000.0
    stop_pct: float | None = 8.0
    hold_days: int | None = 10
    note: str | None = None
    setup: dict | None = None
    # 'AU' | 'US'. Omitted, the market is detected server-side and stored --
    # the page never has to know, but can override when it does.
    market: str | None = None


@app.post("/api/scanner/trades")
async def momentum_open(req: MomentumOpenRequest):
    from tradingagents.momentum_db import open_trade
    res = open_trade(req.ticker, req.entry_price, req.dollars,
                     req.stop_pct, req.hold_days, req.note, req.setup,
                     market=req.market)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "could not open trade"))
    return res


class MomentumCloseRequest(BaseModel):
    exit_price: float
    reason: str = "manual"


@app.post("/api/scanner/trades/{trade_id}/close")
async def momentum_close(trade_id: int, req: MomentumCloseRequest):
    from tradingagents.momentum_db import close_trade
    res = close_trade(trade_id, req.exit_price, req.reason)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "could not close trade"))
    return res


@app.get("/api/scanner/trades")
async def momentum_trades():
    from tradingagents.momentum_db import list_trades, mark_to_market, summary
    loop = asyncio.get_running_loop()
    open_rows = await loop.run_in_executor(_executor, mark_to_market)
    return {"open": open_rows,
            "closed": [t for t in list_trades(status="closed")],
            "summary": summary()}


@app.post("/api/scanner/forward-returns")
async def scanner_forward_returns(days_back: int = 60, force: bool = False):
    """Fill/refresh forward returns on logged mover-days. Idempotent, and
    meant to run daily: a row's longer horizons only mature over the
    following fortnight, so rows are revisited until D+10 exists."""
    from tradingagents.forward_returns import compute, stamp_grader
    loop = asyncio.get_running_loop()
    stamped = await loop.run_in_executor(_executor, stamp_grader)
    result = await loop.run_in_executor(_executor, compute, days_back, force)
    return {**result, "grader_stamped": stamped.get("updated", 0)}


@app.get("/api/eod/scan")
async def eod_scan(threshold: float = 6.0, limit: int = 500, store: bool = False):
    """Names running above `threshold`x normal volume-to-this-hour. `store=true`
    is what the 15:45 timer calls; the page reads without storing so a casual
    look does not pollute the forward record."""
    from tradingagents.eod_volume import scan
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: scan(
        universe_limit=limit, threshold=threshold, store=store))


@app.post("/api/eod/resolve")
async def eod_resolve(days_back: int = 10):
    from tradingagents.eod_volume import resolve
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, resolve, days_back)


@app.post("/api/eod/open-positions")
async def eod_open_positions(max_positions: int = 10, dollars: float = 20000.0):
    """Book the day's top candidates at the ACTUAL closing price. Run after the
    close, not at 15:55 -- the closing auction price does not exist yet then."""
    from tradingagents.eod_volume import open_positions
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, lambda: open_positions(max_positions=max_positions, dollars=dollars))


@app.post("/api/eod/close-positions")
async def eod_close_positions(days_back: int = 10):
    """Exit open paper positions at the next opening auction."""
    from tradingagents.eod_volume import close_positions
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, close_positions, days_back)


@app.get("/api/eod/book")
async def eod_book(days: int = 90):
    from tradingagents.eod_volume import book
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, book, days)


@app.get("/api/eod/record")
async def eod_record(days: int = 90):
    """The live forward record of what the scan picked and how it did."""
    from tradingagents.eod_volume import record
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, record, days)


@app.post("/api/gap-reversion/scan")
async def gap_reversion_scan(threshold: float = -3.0):
    """Frozen-cohort names that gapped down past `threshold` at today's open.
    Called by the 11:20 timer; the gap itself is a settled fact by then."""
    from tradingagents.gap_reversion import scan
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: scan(threshold=threshold))


@app.post("/api/gap-reversion/open-positions")
async def gap_reversion_open(dollars: float = 20000.0, max_positions: int = 5):
    """Book today's gap-down candidates at the last complete 5m bar. Entry is
    mid-morning, NOT the open -- the open price is not tradeable on this signal
    (the auction clears before the gap is observable)."""
    from tradingagents.gap_reversion import open_positions
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: open_positions(
        dollars=dollars, max_positions=max_positions))


@app.post("/api/gap-reversion/close-positions")
async def gap_reversion_close(days_back: int = 5):
    """Exit at the same day's close -- holding overnight would be a different
    strategy and would pick up the drift the other paper book is testing."""
    from tradingagents.gap_reversion import close_positions
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, close_positions, days_back)


@app.get("/api/gap-reversion/book")
async def gap_reversion_book(days: int = 180):
    from tradingagents.gap_reversion import book
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, book, days)


@app.get("/api/scanner/outcomes")
async def scanner_outcomes(days: int = 30, date: str | None = None,
                           provisional: bool = False):
    """`date=YYYY-MM-DD` narrows the report to a single session (what the
    calendar's day cells select); omitted, it stays the rolling window.
    `provisional=true` additionally includes that day's not-yet-finalized
    rows, so today can be watched mid-session -- ignored without `date`, so
    the rolling statistic can never pick up a partial session."""
    from tradingagents.mover_log import outcomes, pending_finalize_count
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, outcomes, days, date, provisional)
    result["pending_finalize_today"] = pending_finalize_count()
    return result


@app.get("/api/scanner/outcomes/calendar")
async def scanner_outcomes_calendar(start: str | None = None, end: str | None = None):
    """Per-day roll-up that paints the gap-outcomes calendar. Bounds are
    optional -- the page asks for everything so month navigation can grey out
    the months either side of the logged range without another round trip."""
    from tradingagents.mover_log import daily_summary
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, daily_summary, start, end)


@app.post("/api/scanner/finalize")
async def scanner_finalize():
    """Re-fetches today's true close/high/low and marks rows finalized.
    Called by asx-mover-finalize.timer at 16:15 Sydney (after the 16:00
    close), but also callable by hand -- running it mid-session just
    re-stamps "last traded so far" harmlessly."""
    from tradingagents.mover_log import finalize_today
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, finalize_today)


@app.get("/sync", response_class=HTMLResponse)
async def sync_page(request: Request):
    if not _is_authenticated(request):
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse(content=(static_dir / "sync.html").read_text())


@app.get("/api/sync/local-reports")
async def list_local_reports(request: Request):
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    from datetime import date as _date
    import re as _re
    entries = []
    if LOGS_DIR.exists():
        for p in sorted(LOGS_DIR.iterdir()):
            if not p.is_dir() or p.name.endswith(".log"):
                continue
            ticker = p.name
            if '.' not in ticker:  # skip bare tickers (e.g. AAPL, BET) that lack an exchange suffix
                continue
            latest = None
            # New format date dirs
            for d in p.iterdir():
                if not d.is_dir() or d.name == "TradingAgentsStrategy_logs":
                    continue
                if _re.match(r"\d{4}-\d{2}-\d{2}", d.name):
                    if latest is None or d.name > latest:
                        latest = d.name
            # Legacy JSON logs
            legacy = p / "TradingAgentsStrategy_logs"
            if legacy.exists():
                for f in legacy.glob("full_states_log_*.json"):
                    d = f.stem.replace("full_states_log_", "")
                    if latest is None or d > latest:
                        latest = d
            entries.append({"ticker": ticker, "latest_date": latest})
    return {"tickers": entries}


@app.post("/api/sync")
async def run_sync(request: Request):
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    selected: List[str] = body.get("tickers", [])  # empty = all

    # Host and key come from the environment, never from source. The remote
    # dropped the hardcoded IP and PEM path in July (dba8be2); this copy of
    # the file predated that and still carried both. The key name is only
    # reconnaissance on its own -- no key material was ever committed -- but a
    # `root@<ip>` target in source tells an attacker exactly what to point at.
    pem_env = os.environ.get("TRADINGAGENTS_SYNC_PEM")
    if not pem_env:
        raise HTTPException(
            status_code=503,
            detail="Sync is not configured: set TRADINGAGENTS_SYNC_PEM to the "
                   "SSH key path and TRADINGAGENTS_SYNC_HOST to user@host.")
    pem = Path(pem_env).expanduser()
    if not pem.exists():
        raise HTTPException(status_code=503,
                            detail=f"TRADINGAGENTS_SYNC_PEM does not exist: {pem}")
    host = os.environ.get("TRADINGAGENTS_SYNC_HOST")
    if not host:
        raise HTTPException(
            status_code=503,
            detail="Sync is not configured: set TRADINGAGENTS_SYNC_HOST to user@host.")
    remote_path = os.environ.get("TRADINGAGENTS_SYNC_REMOTE_PATH",
                                 "/root/.tradingagents/logs/")
    remote = f"{host}:{remote_path}"
    local = str(LOGS_DIR) + "/"

    cmd = [
        "rsync", "--archive", "--checksum", "--human-readable",
        "--stats", "--exclude=*.tmp", "--exclude=*.bak",
        "-e", f"ssh -i {pem} -o StrictHostKeyChecking=no",
    ]

    if selected:
        # Include only selected tickers; exclude everything else
        for ticker in selected:
            cmd += [f"--include={ticker}/***"]
        cmd += ["--exclude=*"]

    cmd += [local, remote]

    async def generate():
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        yield f"data: {json.dumps({'type': 'start', 'cmd': ' '.join(cmd[-2:])})}\n\n"
        async for line in proc.stdout:
            text = line.decode(errors="replace").rstrip()
            if text:
                yield f"data: {json.dumps({'type': 'line', 'text': text})}\n\n"
        await proc.wait()
        yield f"data: {json.dumps({'type': 'done', 'code': proc.returncode})}\n\n"

        # After a successful sync, trigger DB ingest on the remote server
        if proc.returncode == 0:
            try:
                yield f"data: {json.dumps({'type': 'line', 'text': 'Updating remote performance DB…'})}\n\n"
                ingest_proc = await asyncio.create_subprocess_exec(
                    "ssh",
                    "-i", str(pem),
                    "-o", "StrictHostKeyChecking=no",
                    host,
                    "curl -s -X POST http://localhost:7777/api/performance/ingest",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                out, _ = await ingest_proc.communicate()
                result = out.decode(errors="replace").strip()
                yield f"data: {json.dumps({'type': 'line', 'text': f'DB ingest: {result}'})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'line', 'text': f'DB ingest failed: {exc}'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/analyze")
async def start_analysis(request: AnalyzeRequest):
    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _jobs[job_id] = queue
    loop = asyncio.get_running_loop()

    def emit(event):
        loop.call_soon_threadsafe(queue.put_nowait, event)

    loop.run_in_executor(_executor, _run_analysis, request, emit)
    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
async def stream_results(job_id: str):
    queue = _jobs.get(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def generate():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    yield 'data: {"type":"done"}\n\n'
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            _jobs.pop(job_id, None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
