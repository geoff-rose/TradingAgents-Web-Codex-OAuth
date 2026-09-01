"""Fetch and cache ASX announcement PDF text, so the classifier reads the
document instead of guessing from the headline (2026-08-23, user request).

**Why**: the classifier previously saw only a headline. The user caught
"First Drawdown Proceeds Received under Gold Stream" scoring 78 (Buy) when the
body makes clear it is the first drawdown of an already-announced facility --
administrative, not news. A tightened prompt fixes that *specific* pattern, but
only the body can distinguish "$2m received" from "$200m received", or a
placement at a 5% discount from one at 40%. Headlines are a lossy summary
written by the company.

**The interstitial gotcha** (learned the hard way in asxbrief, repeated here so
it isn't rediscovered): `announcements.url` is NOT a PDF. It is a terms-of-use
page on `www.asx.com.au` carrying the real PDF URL in a hidden `pdfURL` form
field, on a *different* subdomain. Fetch the page, extract the field, then
fetch that.

**Cached in TradingAgents' own signals DB, not written back to asxbrief's.**
The collector owns `announcements`; this app reading it is fine, writing to it
is not. A derived cache keyed by fingerprint keeps ownership unambiguous and
means a re-classify never re-fetches.

**Load**: one interstitial GET plus one PDF GET per announcement, once ever.
Politeness delay between fetches, a byte cap so a 300-page annual report can't
blow memory, and failures cache as empty so a broken document is not retried
forever. The user asked to watch load while this runs -- `fetch_stats()`
reports what it has done.
"""

from __future__ import annotations

import io
import re
import sqlite3
import time
from typing import Any

import httpx

PDF_URL_RE = re.compile(r'name="pdfURL"\s+value="([^"]+)"')

MAX_PDF_BYTES = 8 * 1024 * 1024      # skip pathological documents
MAX_TEXT_CHARS = 12_000              # what we keep; see truncation note below
FETCH_DELAY_SECONDS = 0.5            # be a polite client to ASX
REQUEST_TIMEOUT = 25.0
USER_AGENT = "tradingagents-asx-classifier/1.0"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS announcement_bodies (
    fingerprint TEXT PRIMARY KEY,
    text        TEXT,
    n_chars     INTEGER,
    fetched_at  TEXT NOT NULL,
    error       TEXT
);
"""


def _connect() -> sqlite3.Connection:
    from tradingagents.asx_signals import DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def get_cached(fingerprint: str) -> str | None:
    """Returns cached text, '' for a known-failed fetch, or None if never tried."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT text FROM announcement_bodies WHERE fingerprint=?", (fingerprint,),
        ).fetchone()
    return None if row is None else (row["text"] or "")


def _store(fingerprint: str, text: str, error: str | None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO announcement_bodies (fingerprint, text, n_chars, fetched_at, error) "
            "VALUES (?,?,?,?,?)",
            (fingerprint, text, len(text or ""),
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), error),
        )
        conn.commit()


def extract_text(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def fetch_body(fingerprint: str, announcement_url: str,
               client: httpx.Client | None = None) -> str:
    """Fetch one announcement's text, using the cache. Returns '' on failure --
    callers fall back to the headline rather than skipping the announcement,
    because a document we cannot read is not a reason to leave it unscored."""
    cached = get_cached(fingerprint)
    if cached is not None:
        return cached

    own_client = client is None
    client = client or httpx.Client(timeout=REQUEST_TIMEOUT,
                                    headers={"User-Agent": USER_AGENT},
                                    follow_redirects=True)
    text, error = "", None
    try:
        page = client.get(announcement_url)
        page.raise_for_status()
        pdf_url = PDF_URL_RE.search(page.text)
        if not pdf_url:
            error = "no pdfURL field in interstitial page"
        else:
            resp = client.get(pdf_url.group(1))
            resp.raise_for_status()
            if len(resp.content) > MAX_PDF_BYTES:
                error = f"pdf too large ({len(resp.content)} bytes)"
            else:
                text = extract_text(resp.content).strip()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        if own_client:
            client.close()

    # Truncation keeps the head of the document deliberately: ASX announcements
    # front-load the material facts (title, summary, key figures) and tail off
    # into boilerplate, disclaimers and director bios. Keeping the head is
    # strictly better than keeping a middle slice.
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    _store(fingerprint, text, error)
    time.sleep(FETCH_DELAY_SECONDS)
    return text


def fetch_stats() -> dict[str, Any]:
    """What the cache holds -- for watching load while this beds in."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) errs, "
            "SUM(CASE WHEN COALESCE(text,'')='' THEN 1 ELSE 0 END) empty, "
            "AVG(n_chars) avg_chars, MIN(fetched_at) first, MAX(fetched_at) last "
            "FROM announcement_bodies"
        ).fetchone()
        errors = [dict(r) for r in conn.execute(
            "SELECT fingerprint, error FROM announcement_bodies WHERE error IS NOT NULL "
            "ORDER BY fetched_at DESC LIMIT 5")]
    return {
        "cached": row["n"] or 0,
        "errors": row["errs"] or 0,
        "empty_text": row["empty"] or 0,
        "avg_chars": round(row["avg_chars"] or 0),
        "first_fetch": row["first"], "last_fetch": row["last"],
        "recent_errors": errors,
    }
