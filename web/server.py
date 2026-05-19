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

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="TradingAgents Web")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Reports are stored under ~/.tradingagents/logs (respects TRADINGAGENTS_RESULTS_DIR override)
_HOME = Path.home() / ".tradingagents"
LOGS_DIR = Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", str(_HOME / "logs")))

# Report fields in display order; covers both new (md files) and legacy (JSON keys) formats
_REPORT_FIELDS = [
    ("final_trade_decision",  "Decision"),
    ("market_report",         "Market"),
    ("sentiment_report",      "Sentiment"),
    ("news_report",           "News"),
    ("fundamentals_report",   "Fundamentals"),
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
    deep_model: str = "gpt-5.5"
    quick_model: str = "gpt-5.4-mini"
    research_depth: int = 1
    analysts: List[str] = ["market", "social", "news", "fundamentals"]


def _run_analysis(request: AnalyzeRequest, emit: Callable[[Any], None]) -> None:
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = {
            **DEFAULT_CONFIG,
            "llm_provider": "openai-codex",
            "deep_think_llm": request.deep_model,
            "quick_think_llm": request.quick_model,
            "max_debate_rounds": request.research_depth,
            "max_risk_discuss_rounds": request.research_depth,
        }

        emit({"type": "status", "message": f"Initialising agents for {request.ticker.upper()}…"})

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

        result, _signal = ta.propagate(request.ticker.upper(), request.date)

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
    return HTMLResponse(content=(static_dir / "index.html").read_text())


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
            "investment_plan":       raw.get("investment_plan") or "",
            # legacy key name differs
            "trader_investment_plan": raw.get("trader_investment_plan")
                                   or raw.get("trader_investment_decision") or "",
        }

    raise HTTPException(status_code=404, detail="Report not found")


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
