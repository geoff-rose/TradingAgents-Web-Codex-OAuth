#!/usr/bin/env python3
"""Batch runner: analyse ASX stocks with automatic provider fallback.

Usage:
    uv run python scripts/batch_asx200.py [--workers N] [--date YYYY-MM-DD] [--depth 1|3|5]

By default uses xai-grok as the primary provider and hermes-claude as the
fallback. When the primary provider's quota is exhausted (usage_limit_reached),
the runner automatically switches every remaining ticker to the fallback provider
without stopping the batch.

Rate-limit errors (RPM/TPM) are retried with exponential backoff on the same
provider before switching.

Reads tickers from ~/.tradingagents/asx300_tickers.txt. Falls back to fetching
ASX 200 from Wikipedia if the file doesn't exist.

Interrupt with Ctrl+C — already-completed reports are skipped on resume.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.default_config import DEFAULT_CONFIG

LOGS_DIR = Path(os.getenv("TRADINGAGENTS_RESULTS_DIR",
                          str(Path.home() / ".tradingagents" / "logs")))
_TICKER_FILE = Path.home() / ".tradingagents" / "asx300_tickers.txt"

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_tickers() -> list[str]:
    if _TICKER_FILE.exists():
        tickers = []
        with open(_TICKER_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    tickers.append(line)
        print(f"Loaded {len(tickers)} tickers from {_TICKER_FILE}")
        return tickers

    print(f"Ticker file not found at {_TICKER_FILE}")
    print("Falling back to ASX 200 from Wikipedia (run scripts/fetch_asx300.py to build the full list).")
    import pandas as pd
    url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text), flavor="lxml")
    df = tables[2]
    codes = df["Code"].dropna().astype(str).tolist()
    return [c + ".AX" for c in codes]


def report_exists(ticker: str, trade_date: str) -> bool:
    """Return True if a completed report less than 7 days old exists for this ticker."""
    base = LOGS_DIR / ticker
    if not base.exists():
        return False

    cutoff = date.today() - timedelta(days=1)

    # New format: logs/{ticker}/{YYYY-MM-DD}/reports/final_trade_decision.md
    for d in base.iterdir():
        if not d.is_dir() or d.name == "TradingAgentsStrategy_logs":
            continue
        try:
            if date.fromisoformat(d.name) >= cutoff:
                if (d / "reports" / "final_trade_decision.md").exists():
                    return True
        except ValueError:
            continue

    # Legacy format: logs/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json
    legacy_dir = base / "TradingAgentsStrategy_logs"
    if legacy_dir.exists():
        for f in legacy_dir.glob("full_states_log_*.json"):
            try:
                report_date = date.fromisoformat(f.stem.replace("full_states_log_", ""))
                if report_date >= cutoff:
                    return True
            except ValueError:
                continue

    return False


# ── Worker state ─────────────────────────────────────────────────────────────

_lock    = threading.Lock()
_counts  = {"done": 0, "failed": 0, "skipped": 0}
_total   = 0
_start   = 0.0

# Active config — read by every worker on each attempt. Swapped atomically
# (under _lock) when the primary provider's quota is exhausted.
_active_config: dict = {}
_fallback_config: dict | None = None
_on_fallback = False   # True once we've switched providers
_force = False         # When True, skip the report_exists check in run_ticker


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
    provider = _active_config.get("llm_provider", "?")
    return (f"[{finished}/{_total} {pct:.0f}%]  "
            f"done={done}  failed={failed}  skipped={skipped}  "
            f"elapsed={elapsed/60:.1f}m  eta={eta}  [{provider}]")


_RATE_BACKOFF = [15, 30, 60, 120, 300]   # seconds between retries on RPM/TPM limits


def _is_quota_exhausted(exc: Exception) -> bool:
    """True for hard quota errors that won't resolve by waiting (switch provider)."""
    msg = str(exc).lower()
    return "usage_limit_reached" in msg or "insufficient_quota" in msg


def _is_rate_limit(exc: Exception) -> bool:
    """True for soft rate limits that resolve after a short wait (retry same provider)."""
    msg = str(exc).lower()
    # Standard 429 / rate_limit errors
    if ("429" in msg or "rate_limit" in msg) and not _is_quota_exhausted(exc):
        return True
    # xAI streaming burst error: API sends an error event before response.created
    # when too many concurrent requests hit the endpoint simultaneously.
    if "expected to have received" in msg and "response.created" in msg:
        return True
    return False


def _switch_to_fallback() -> bool:
    """Swap _active_config to the fallback. Returns True if switch happened."""
    global _active_config, _on_fallback
    with _lock:
        if _on_fallback or _fallback_config is None:
            return False
        _active_config = dict(_fallback_config)
        _on_fallback = True
        fb = _active_config["llm_provider"]
        print(f"\n  *** Quota exhausted — switching to fallback provider: {fb} ***\n",
              flush=True)
        return True


def run_ticker(ticker: str, trade_date: str) -> tuple[str, str, float]:
    if not _force and report_exists(ticker, trade_date):
        with _lock:
            _counts["skipped"] += 1
            print(f"  SKIP   {ticker:<12}  {_status_line()}", flush=True)
        return ticker, "skipped", 0.0

    t0 = time.time()
    with _lock:
        print(f"  START  {ticker:<12}  {_status_line()}", flush=True)

    rate_attempts = 0

    while True:
        config = dict(_active_config)   # snapshot current provider config
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
            if _is_quota_exhausted(exc):
                switched = _switch_to_fallback()
                if switched or _on_fallback:
                    # Retry immediately on the new provider
                    with _lock:
                        print(f"  RETRY  {ticker:<12}  retrying on fallback provider",
                              flush=True)
                    continue
                # No fallback available — fail
                elapsed = time.time() - t0
                with _lock:
                    _counts["failed"] += 1
                    print(f"  FAIL   {ticker:<12}  quota exhausted, no fallback  "
                          f"{_status_line()}", flush=True)
                return ticker, "failed", elapsed

            elif _is_rate_limit(exc) and rate_attempts < len(_RATE_BACKOFF):
                wait = _RATE_BACKOFF[rate_attempts]
                rate_attempts += 1
                with _lock:
                    print(f"  WAIT   {ticker:<12}  rate-limited, retrying in {wait}s  "
                          f"(attempt {rate_attempts})", flush=True)
                time.sleep(wait)
                continue

            else:
                elapsed = time.time() - t0
                with _lock:
                    _counts["failed"] += 1
                    print(f"  FAIL   {ticker:<12}  {str(exc)[:60]}  {_status_line()}",
                          flush=True)
                return ticker, "failed", elapsed


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _total, _start, _active_config, _fallback_config, _force

    parser = argparse.ArgumentParser(description="Batch ASX analysis with provider fallback")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent analyses (default 3)")
    parser.add_argument("--date", default=str(date.today()),
                        help="Trade date YYYY-MM-DD (default: today)")
    parser.add_argument("--depth", type=int, choices=[1, 3, 5], default=1,
                        help="Research depth: 1=Shallow, 3=Medium, 5=Deep (default 1)")
    # Primary provider
    parser.add_argument("--provider",     default="xai-grok")
    parser.add_argument("--deep-model",   default="grok-4-fast-non-reasoning")
    parser.add_argument("--quick-model",  default="grok-4-fast-non-reasoning")
    # Fallback provider (used when primary quota is exhausted)
    parser.add_argument("--fallback-provider",    default=None,
                        help="Provider to switch to when primary quota is exhausted")
    parser.add_argument("--fallback-deep-model",  default="claude-sonnet-4-6")
    parser.add_argument("--fallback-quick-model", default="claude-haiku-4-5")
    parser.add_argument("--ticker-file", default=None,
                        help="Path to a text file of tickers to run (one per line). Overrides the default asx300 list.")
    parser.add_argument("--force", action="store_true",
                        help="Run all tickers even if a recent report exists.")
    args = parser.parse_args()

    depth_label = {1: "Shallow", 3: "Medium", 5: "Deep"}[args.depth]

    print("=" * 60)
    print("  TradingAgents — ASX Batch Runner")
    print("=" * 60)
    print(f"  Date:            {args.date}")
    print(f"  Depth:           {depth_label} ({args.depth} debate round{'s' if args.depth > 1 else ''})")
    print(f"  Workers:         {args.workers}")
    print(f"  Primary:         {args.provider}  ({args.quick_model} / {args.deep_model})")
    print(f"  Fallback:        {args.fallback_provider or 'none'}"
          + (f"  ({args.fallback_quick_model} / {args.fallback_deep_model})" if args.fallback_provider else ""))
    print(f"  Tickers:         {_TICKER_FILE}")
    print(f"  Output:          {LOGS_DIR}")
    print()

    if args.ticker_file:
        with open(args.ticker_file) as f:
            tickers = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        print(f"Loaded {len(tickers)} tickers from {args.ticker_file}")
    else:
        tickers = load_tickers()
    print()

    _force  = args.force
    todo    = tickers if args.force else [t for t in tickers if not report_exists(t, args.date)]
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

    base_cfg = {
        **DEFAULT_CONFIG,
        "max_debate_rounds":       args.depth,
        "max_risk_discuss_rounds": args.depth,
    }
    _active_config = {
        **base_cfg,
        "llm_provider":   args.provider,
        "deep_think_llm": args.deep_model,
        "quick_think_llm": args.quick_model,
    }
    _fallback_config = {
        **base_cfg,
        "llm_provider":   args.fallback_provider,
        "deep_think_llm": args.fallback_deep_model,
        "quick_think_llm": args.fallback_quick_model,
    } if args.fallback_provider else None

    _total = len(tickers)
    _start = time.time()

    results: list[tuple[str, str, float]] = []

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_ticker, t, args.date): t for t in tickers}
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
    if _on_fallback:
        print(f"  Note: switched to fallback provider ({args.fallback_provider}) "
              f"mid-run due to primary quota exhaustion")
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
