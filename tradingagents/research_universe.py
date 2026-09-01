"""Read/write access to asxbrief's `research_universe` table -- the ~50-ticker
basket for phase-2 pattern research (distinct from `universe`, the top-500
table used to filter the ASX dashboard's announcement feed).

**This is a deliberate exception to the read-only pattern in `asx_feed.py`.**
Ticker management (add/remove focus picks, regenerate recommendations) is a
low-frequency admin action, not a hot path, so writing directly to asxbrief's
shared SQLite file (WAL mode, one writer at a time) is simpler than adding an
HTTP API to a project that's otherwise a pure CLI collector. Schema/writes
for every other table stay asxbrief's exclusively.
"""

from __future__ import annotations

import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .asx_feed import DB_PATH

FOCUS_LIMIT = 10
RECOMMENDED_LIMIT = 40

# The asxbrief CLI, run as its own service user -- reused here for the two
# operations that need real compute (yfinance scoring, IBKR connection)
# rather than reimplementing them against the shared db from this side.
ASXBRIEF_VENV_PYTHON = "/opt/asxbrief/.venv/bin/asxbrief"
ASXBRIEF_CONFIG = "/opt/asxbrief/config.toml"
ASXBRIEF_CWD = "/opt/asxbrief"


def _connect(readonly: bool = False) -> sqlite3.Connection:
    mode = "ro" if readonly else "rwc"
    conn = sqlite3.connect(f"file:{DB_PATH}?mode={mode}", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def get_universe() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"focus": [], "recommended": [], "focus_limit": FOCUS_LIMIT, "recommended_limit": RECOMMENDED_LIMIT}
    with _connect(readonly=True) as conn:
        rows = conn.execute(
            "SELECT ticker, kind, added_at, volatility, dollar_volume, notes "
            "FROM research_universe ORDER BY kind, ticker"
        ).fetchall()
    focus = [dict(r) for r in rows if r["kind"] == "focus"]
    recommended = sorted(
        (dict(r) for r in rows if r["kind"] == "recommended"),
        key=lambda r: -(r["volatility"] or 0),
    )
    return {
        "focus": focus, "recommended": recommended,
        "focus_limit": FOCUS_LIMIT, "recommended_limit": RECOMMENDED_LIMIT,
    }


def add_focus_ticker(ticker: str) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    if not ticker:
        return {"ok": False, "error": "empty ticker"}
    with _connect() as conn:
        existing = {r["ticker"] for r in conn.execute(
            "SELECT ticker FROM research_universe WHERE kind='focus'"
        ).fetchall()}
        if ticker in existing:
            return {"ok": False, "error": f"{ticker} is already a focus ticker"}
        if len(existing) >= FOCUS_LIMIT:
            return {"ok": False, "error": f"already at the {FOCUS_LIMIT}-ticker focus limit"}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn.execute(
            "INSERT OR REPLACE INTO research_universe (ticker, kind, added_at) VALUES (?, 'focus', ?)",
            (ticker, now),
        )
        conn.commit()
    return {"ok": True}


def remove_ticker(ticker: str) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM research_universe WHERE ticker=?", (ticker,))
        conn.commit()
    return {"ok": cur.rowcount > 0}


def bar_coverage() -> dict[str, dict[str, Any]]:
    """{ticker: {count, earliest, latest}} for every ticker currently in the
    research universe -- lets the dashboard show backfill progress without
    SSHing in."""
    if not DB_PATH.exists():
        return {}
    with _connect(readonly=True) as conn:
        tickers = [r["ticker"] for r in conn.execute("SELECT ticker FROM research_universe").fetchall()]
        if not tickers:
            return {}
        placeholders = ",".join("?" * len(tickers))
        rows = conn.execute(
            f"SELECT ticker, COUNT(*) n, MIN(ts) earliest, MAX(ts) latest "
            f"FROM bars WHERE ticker IN ({placeholders}) GROUP BY ticker",
            tickers,
        ).fetchall()
    return {r["ticker"]: {"count": r["n"], "earliest": r["earliest"], "latest": r["latest"]} for r in rows}


def run_recommend() -> dict[str, Any]:
    """Shells out to `asxbrief research recommend` (as the asxbrief user) --
    this does real work (yfinance scoring across ~500 candidates, ~30-60s),
    not something to reimplement against the shared db from this side."""
    t0 = time.monotonic()
    proc = subprocess.run(
        ["sudo", "-u", "asxbrief", ASXBRIEF_VENV_PYTHON, "-c", ASXBRIEF_CONFIG, "research", "recommend"],
        cwd=ASXBRIEF_CWD, capture_output=True, text=True, timeout=180,
    )
    return {
        "ok": proc.returncode == 0,
        "output": proc.stdout if proc.returncode == 0 else (proc.stdout + proc.stderr),
        "duration_s": round(time.monotonic() - t0, 1),
    }


BACKFILL_UNIT = "asxbrief-ibkr-backfill"


def backfill_estimate_hours(n_tickers: int, years: float) -> float:
    """Rough wall-clock estimate at current pacing (1-day chunks, 2s delay,
    2 bar types) -- shown in the UI so nobody triggers this expecting minutes."""
    trading_days_per_year = 250
    request_delay_s = 2.0
    return round((n_tickers + 1) * years * trading_days_per_year * 2 * request_delay_s / 3600, 1)


# journalctl emits this on STDOUT -- once per requested line -- when the unit
# file no longer exists. `start_backfill` uses `systemd-run --collect`, which
# removes the transient unit as soon as it exits, so every completed backfill
# leaves a journal whose unit cannot be resolved. With `-n 30` that produced
# thirty identical "Failed to open ..." lines and pushed the actual backfill
# output out of the tail entirely, which read on the page as the backfill
# having failed thirty times when nothing had gone wrong at all.
_JOURNAL_NOISE = (
    "Failed to open /run/systemd/transient/",
    "-- No entries --",
)


def _clean_journal(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines()
             if ln.strip() and not any(n in ln for n in _JOURNAL_NOISE)]
    return "\n".join(lines)


def backfill_status() -> dict[str, Any]:
    """Is a backfill currently running, and what has it logged so far.

    Backed by a real transient systemd unit (see `start_backfill`), not a
    pidfile -- a pidfile-tracked plain subprocess was tried first and died
    silently the first time `tradingagents.service` was restarted for an
    unrelated reason, because `start_new_session=True` detaches a process
    from its controlling terminal but NOT from systemd's cgroup: restarting
    the parent service kills the whole cgroup, descendants included,
    regardless of session leadership. A separate systemd unit has its own
    cgroup and survives that.
    """
    result = subprocess.run(["systemctl", "is-active", BACKFILL_UNIT], capture_output=True, text=True)
    running = result.stdout.strip() == "active"
    log_tail = ""
    try:
        log_result = subprocess.run(
            ["journalctl", "-u", BACKFILL_UNIT, "-n", "30", "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=10,
        )
        log_tail = _clean_journal(log_result.stdout)
    except Exception:
        pass
    return {"running": running, "log_tail": log_tail,
            "has_log": bool(log_tail.strip())}


def start_backfill(tickers: list[str] | None = None, years: float = 0.25) -> dict[str, Any]:
    """Launches `asxbrief ibkr-backfill` as its own transient systemd unit
    (`systemd-run --unit ... --collect`) and returns immediately -- this can
    run for hours (see `backfill_estimate_hours`), far longer than any HTTP
    request or executor-thread call should block for, and it must survive
    this web app's own service being restarted. Poll `backfill_status()` for
    progress. Only one run at a time; refuses to start a second while one is
    already active.
    """
    status = backfill_status()
    if status["running"]:
        return {"ok": False, "error": "a backfill is already running"}

    inner_cmd = ["sudo", "-u", "asxbrief", ASXBRIEF_VENV_PYTHON, "-c", ASXBRIEF_CONFIG,
                 "ibkr-backfill", "--years", str(years)]
    if tickers:
        inner_cmd += ["--tickers", ",".join(tickers)]

    run_cmd = [
        "systemd-run", "--unit", BACKFILL_UNIT, "--collect",
        f"--working-directory={ASXBRIEF_CWD}",
        "--description=asxbrief IBKR bar backfill",
    ] + inner_cmd
    proc = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout).strip()}
    return {"ok": True, "unit": BACKFILL_UNIT,
            "estimate_hours": backfill_estimate_hours(len(tickers) if tickers else RECOMMENDED_LIMIT + FOCUS_LIMIT, years)}
