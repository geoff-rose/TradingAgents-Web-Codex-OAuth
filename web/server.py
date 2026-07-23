"""FastAPI web server for TradingAgents."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
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

# Auth config — must be set in the environment, never hardcoded here
_SECRET_KEY = os.environ.get("TRADINGAGENTS_WEB_SECRET_KEY")
_ANALYSIS_PASSWORD = os.environ.get("TRADINGAGENTS_WEB_PASSWORD")
if not _SECRET_KEY or not _ANALYSIS_PASSWORD:
    raise RuntimeError(
        "TRADINGAGENTS_WEB_SECRET_KEY and TRADINGAGENTS_WEB_PASSWORD must both be set "
        "in the environment before starting the web app."
    )
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

    sync_host = os.environ.get("TRADINGAGENTS_SYNC_HOST")  # e.g. "user@your-server.example.com"
    sync_pem = os.environ.get("TRADINGAGENTS_SYNC_PEM")  # path to the SSH private key
    if not sync_host or not sync_pem:
        raise HTTPException(
            status_code=500,
            detail="TRADINGAGENTS_SYNC_HOST and TRADINGAGENTS_SYNC_PEM must both be set "
                   "in the environment to use sync.",
        )
    pem = Path(sync_pem).expanduser()
    sync_remote_path = os.environ.get("TRADINGAGENTS_SYNC_REMOTE_PATH", "/root/.tradingagents/logs/")

    body = await request.json()
    selected: List[str] = body.get("tickers", [])  # empty = all

    remote = f"{sync_host}:{sync_remote_path}"
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
                    sync_host,
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
