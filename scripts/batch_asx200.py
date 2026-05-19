#!/usr/bin/env python3
"""Batch runner: analyse all ASX 200 stocks using the openai-codex OAuth provider.

Usage:
    uv run python scripts/batch_asx200.py [--workers N] [--date YYYY-MM-DD] [--depth 1|3|5]

Defaults: workers=3, date=today, depth=1 (Shallow)
Results are saved to ~/.tradingagents/logs/ and can be viewed at /reports in the web UI.
Interrupt with Ctrl+C — already-completed reports are skipped on resume.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from io import StringIO
from pathlib import Path

import requests

# Make TradingAgents importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.default_config import DEFAULT_CONFIG

LOGS_DIR = Path(os.getenv("TRADINGAGENTS_RESULTS_DIR",
                          str(Path.home() / ".tradingagents" / "logs")))

# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_asx200_tickers() -> list[str]:
    """Scrape the ASX 200 constituent list from Wikipedia."""
    import pandas as pd
    url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    # Table 2 has columns: Code, Company, Sector, Market Cap, Headquarters
    df = tables[2]
    codes = df["Code"].dropna().astype(str).tolist()
    return [c + ".AX" for c in codes]


def report_exists(ticker: str, trade_date: str) -> bool:
    """Return True if a completed report already exists for this ticker+date."""
    base = LOGS_DIR / ticker
    if (base / trade_date / "reports" / "final_trade_decision.md").exists():
        return True
    if (base / "TradingAgentsStrategy_logs" / f"full_states_log_{trade_date}.json").exists():
        return True
    return False


# ── Worker ───────────────────────────────────────────────────────────────────

_lock   = threading.Lock()
_counts = {"done": 0, "failed": 0, "skipped": 0}
_total  = 0
_start  = 0.0


def _status_line() -> str:
    done    = _counts["done"]
    failed  = _counts["failed"]
    skipped = _counts["skipped"]
    finished = done + failed + skipped
    pct  = finished / _total * 100 if _total else 0
    elapsed = time.time() - _start
    rate = finished / elapsed if elapsed > 0 else 0
    remaining = (_total - finished) / rate if rate > 0 else 0
    eta = f"{remaining/60:.0f}m" if remaining > 60 else f"{remaining:.0f}s"
    return (f"[{finished}/{_total} {pct:.0f}%]  "
            f"done={done}  failed={failed}  skipped={skipped}  "
            f"elapsed={elapsed/60:.1f}m  eta={eta}")


def run_ticker(ticker: str, trade_date: str, config: dict) -> tuple[str, str, float]:
    if report_exists(ticker, trade_date):
        with _lock:
            _counts["skipped"] += 1
            print(f"  SKIP   {ticker:<12}  {_status_line()}", flush=True)
        return ticker, "skipped", 0.0

    t0 = time.time()
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        ta = TradingAgentsGraph(config=config)
        ta.propagate(ticker, trade_date)
        elapsed = time.time() - t0
        with _lock:
            _counts["done"] += 1
            print(f"  DONE   {ticker:<12}  {elapsed:>5.0f}s  {_status_line()}", flush=True)
        return ticker, "done", elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        with _lock:
            _counts["failed"] += 1
            print(f"  FAIL   {ticker:<12}  {str(exc)[:60]}  {_status_line()}", flush=True)
        return ticker, "failed", elapsed


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _total, _start

    parser = argparse.ArgumentParser(description="Batch ASX 200 analysis")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent analyses (default 3; raise carefully to avoid rate limits)")
    parser.add_argument("--date", default=str(date.today()),
                        help="Trade date YYYY-MM-DD (default: today)")
    parser.add_argument("--depth", type=int, choices=[1, 3, 5], default=1,
                        help="Research depth: 1=Shallow, 3=Medium, 5=Deep (default 1)")
    parser.add_argument("--deep-model",  default="gpt-5.5")
    parser.add_argument("--quick-model", default="gpt-5.4-mini")
    args = parser.parse_args()

    depth_label = {1: "Shallow", 3: "Medium", 5: "Deep"}[args.depth]

    print("=" * 60)
    print("  TradingAgents — ASX 200 Batch Runner")
    print("=" * 60)
    print(f"  Date:       {args.date}")
    print(f"  Depth:      {depth_label} ({args.depth} debate round{'s' if args.depth > 1 else ''})")
    print(f"  Workers:    {args.workers}")
    print(f"  Quick LLM:  {args.quick_model}")
    print(f"  Deep LLM:   {args.deep_model}")
    print(f"  Output:     {LOGS_DIR}")
    print()

    print("Fetching ASX 200 tickers from Wikipedia...", flush=True)
    tickers = fetch_asx200_tickers()
    print(f"Got {len(tickers)} tickers.\n")

    # Pre-check which are already done
    todo    = [t for t in tickers if not report_exists(t, args.date)]
    already = len(tickers) - len(todo)
    if already:
        print(f"  {already} reports already exist — will skip them.")
    print(f"  {len(todo)} to run.\n")

    if not todo:
        print("Nothing to do. All reports already exist.")
        return

    est_min_lo = len(todo) * 3 / args.workers
    est_min_hi = len(todo) * 7 / args.workers
    print(f"Estimated time: {est_min_lo/60:.1f}–{est_min_hi/60:.1f} hours "
          f"(assuming 3–7 min each at {args.workers} workers).")
    print("Press Ctrl+C at any time — progress is saved and resumable.\n")

    config = {
        **DEFAULT_CONFIG,
        "llm_provider":          "openai-codex",
        "deep_think_llm":        args.deep_model,
        "quick_think_llm":       args.quick_model,
        "max_debate_rounds":     args.depth,
        "max_risk_discuss_rounds": args.depth,
    }

    _total = len(tickers)   # includes already-done (they'll be counted as skipped)
    _start = time.time()

    results: list[tuple[str, str, float]] = []

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_ticker, t, args.date, config): t for t in tickers}
            for future in as_completed(futures):
                results.append(future.result())

    except KeyboardInterrupt:
        print("\nInterrupted. Waiting for in-flight analyses to finish...", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - _start
    done_results  = [(t, e) for t, s, e in results if s == "done"]
    fail_results  = [(t, e) for t, s, e in results if s == "failed"]

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Completed:   {_counts['done']}")
    print(f"  Failed:      {_counts['failed']}")
    print(f"  Skipped:     {_counts['skipped']}")
    print(f"  Total time:  {total_elapsed/60:.1f} minutes")
    if done_results:
        times = [e for _, e in done_results]
        print(f"  Per-stock:   avg {sum(times)/len(times):.0f}s  "
              f"min {min(times):.0f}s  max {max(times):.0f}s")
    if fail_results:
        print(f"\n  Failed tickers:")
        for t, _ in fail_results:
            print(f"    {t}")
    print()
    print(f"View reports at: http://localhost:7777/reports")


if __name__ == "__main__":
    main()
