"""Codex subscription quota, read from the response headers and kept.

**Why this exists.** Nothing in this system knew what its own limits were.
Planning the 2026-09-07 A/B rescore meant guessing how many calls would fit,
and the guess was 42% low -- measured cost per call is ~0.071% of the 5-hour
window for a full announcement prompt, not the 0.050% estimated from shorter
ones. A recorded history turns that into arithmetic.

The numbers come from headers the Codex backend already returns
(`x-codex-primary-used-percent` over a 300-minute window,
`x-codex-secondary-used-percent` over 10080 minutes) -- there is no usage API
to call, so a single trivial request is the cheapest way to read them.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quota_log (
    captured_at       TEXT PRIMARY KEY,
    model             TEXT,
    plan              TEXT,
    limit_class       TEXT,
    primary_pct       INTEGER,
    primary_window_m  INTEGER,
    primary_reset_s   INTEGER,
    secondary_pct     INTEGER,
    secondary_window_m INTEGER,
    secondary_reset_s INTEGER
);
"""


def _connect() -> sqlite3.Connection:
    from .mover_log import DB_PATH
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.executescript(_SCHEMA)
    return conn


def snapshot(store: bool = True) -> dict[str, Any]:
    """Read the current quota headers, optionally recording them."""
    import httpx

    from . import codex_oauth as C

    _r, _w, headers_for = C._token_helpers()
    tok = C.ensure_fresh_token()
    hd = dict(headers_for(tok))
    hd["Authorization"] = f"Bearer {tok}"
    hd["Content-Type"] = "application/json"
    with httpx.Client(timeout=45) as cl:
        resp = cl.post(C.BASE_URL + "/responses", headers=hd, json={
            "model": C.DEFAULT_MODEL,
            "input": [{"role": "user", "content": "hi"}],
            "store": False, "stream": True})
    H = resp.headers
    gi = lambda k, d=0: int(H.get(k, d) or d)
    row = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": C.DEFAULT_MODEL,
        "plan": H.get("x-codex-plan-type"),
        "limit_class": H.get("x-codex-active-limit"),
        "primary_pct": gi("x-codex-primary-used-percent"),
        "primary_window_m": gi("x-codex-primary-window-minutes"),
        "primary_reset_s": gi("x-codex-primary-reset-after-seconds"),
        "secondary_pct": gi("x-codex-secondary-used-percent"),
        "secondary_window_m": gi("x-codex-secondary-window-minutes"),
        "secondary_reset_s": gi("x-codex-secondary-reset-after-seconds"),
    }
    if store:
        with _connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO quota_log VALUES (?,?,?,?,?,?,?,?,?,?)",
                tuple(row[k] for k in (
                    "captured_at", "model", "plan", "limit_class",
                    "primary_pct", "primary_window_m", "primary_reset_s",
                    "secondary_pct", "secondary_window_m", "secondary_reset_s")))
            db.commit()
    return row


def history(limit: int = 48) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM quota_log ORDER BY captured_at DESC LIMIT ?", (limit,))]
    finally:
        conn.close()
