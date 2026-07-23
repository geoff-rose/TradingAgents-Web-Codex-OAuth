"""AFR (Australian Financial Review) headline store and fetcher.

Headlines are captured from AFR's publicly-listed news sitemap and stored in a
local SQLite database so that weekly reports can access headlines that appeared
earlier in the week (the live sitemap only carries ~190 most-recent articles).

robots.txt listing: https://www.afr.com/sitemaps/news/brands/afr

Typical workflow
----------------
1. ``scripts/scan_afr.py`` runs on a cron (e.g. 3× daily) to ingest new
   headlines from the sitemap and tag them against a configurable watchlist.
2. ``fetch_afr_headlines()`` queries the DB for tagged headlines in a date
   range. Falls back to a live sitemap fetch if the DB has no rows for that
   period (e.g. first run before any scan has completed).
"""

from __future__ import annotations

import logging
import re
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_SITEMAP_URL = "https://www.afr.com/sitemaps/news/brands/afr"
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_SM_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
_NEWS_NS = "http://www.google.com/schemas/sitemap-news/0.9"


# ---------------------------------------------------------------------------
# Sentiment classification
# ---------------------------------------------------------------------------

# Multi-word phrases are checked first (weight 2) before single words (weight 1).
# Ordering within each list doesn't matter — all matches are scored.
_NEGATIVE_PHRASES = [
    "profit warning", "write-down", "writedown", "job cuts", "class action",
    "share price falls", "share price drops", "regulatory action",
    "below expectations", "worse than expected", "misses estimates",
    "misses forecast", "cuts dividend", "dividend cut", "dividend suspended",
    "trading halt", "forced sale", "debt default", "credit downgrade",
]
_POSITIVE_PHRASES = [
    "record profit", "record earnings", "record revenue", "record sales",
    "beats expectations", "beats estimates", "beats forecast",
    "above expectations", "better than expected", "raises dividend",
    "dividend increase", "special dividend", "share buyback",
    "new contract", "wins contract", "major contract", "strategic review",
    "credit upgrade",
]

_NEGATIVE_WORDS = {
    # price action
    "falls", "fell", "drops", "dropped", "sinks", "sank", "slides", "slid",
    "tumbles", "tumbled", "plunges", "plunged", "crashes", "crashed",
    "collapses", "collapsed", "slumps", "slumped", "dives", "dived",
    "retreats", "retreated", "weakens", "weakened", "declines", "declined",
    "selloff", "sell-off",
    # earnings / results
    "miss", "misses", "missed", "disappoints", "disappointing", "disappointed",
    "shortfall", "deficit", "impairment", "write-off", "writeoff", "loss",
    "losses",
    # analyst
    "downgrade", "downgrades", "downgraded", "underperform", "underweight",
    # corporate negatives
    "layoffs", "redundancies", "restructuring", "recall", "suspended",
    "suspension", "investigation", "probe", "fine", "fined", "penalty",
    "lawsuit", "sued", "breach", "fraud", "scandal", "rejects", "cancelled",
    "abandoned", "halted", "halts",
    # macro
    "recession", "downturn", "slowdown", "crisis", "default", "fears",
    "concern", "concerns",
}
_POSITIVE_WORDS = {
    # price action
    "rises", "rose", "gains", "gained", "surges", "surged", "climbs",
    "climbed", "jumps", "jumped", "rallies", "rallied", "soars", "soared",
    "advances", "advanced", "lifts", "lifted", "rebounds", "rebounded",
    "rally",
    # earnings / results
    "profit", "profits", "beat", "beats", "exceeded", "exceeds", "record",
    "dividend", "dividends",
    # analyst
    "upgrade", "upgrades", "upgraded", "outperform", "overweight",
    # corporate positives
    "acquisition", "acquires", "merger", "deal", "partnership", "expands",
    "expansion", "buyback", "approved", "wins", "contract",
    # macro
    "growth", "recovery", "improvement",
}


def classify_headline(title: str) -> str:
    """Return 'positive', 'negative', or 'neutral' for a headline string.

    Phrases (2+ words) are weighted 2× over single words. The final label is
    determined by the sign of the net score; ties → 'neutral'.
    """
    t = title.lower()
    score = 0

    for phrase in _NEGATIVE_PHRASES:
        if phrase in t:
            score -= 2
    for phrase in _POSITIVE_PHRASES:
        if phrase in t:
            score += 2

    # Tokenise on non-word chars so "falls" != "waterfalls"
    words = set(re.findall(r"\b\w+\b", t))
    score -= len(words & _NEGATIVE_WORDS)
    score += len(words & _POSITIVE_WORDS)

    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


_SENTIMENT_LABEL = {
    "positive": "[+]",
    "negative": "[-]",
    "neutral":  "[~]",
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _default_db_path() -> Path:
    from tradingagents.default_config import DEFAULT_CONFIG
    return Path(DEFAULT_CONFIG["data_cache_dir"]) / "afr_headlines.db"


def open_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and initialise if necessary) the AFR headlines database."""
    path = Path(db_path) if db_path else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS afr_headlines (
            url        TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            pub_date   TEXT,
            scraped_at TEXT NOT NULL,
            sentiment  TEXT
        );
        CREATE TABLE IF NOT EXISTS afr_headline_tickers (
            url    TEXT NOT NULL,
            ticker TEXT NOT NULL,
            PRIMARY KEY (url, ticker)
        );
        CREATE INDEX IF NOT EXISTS idx_aht_ticker ON afr_headline_tickers(ticker);
        CREATE INDEX IF NOT EXISTS idx_ah_pub_date ON afr_headlines(pub_date);
    """)
    # Migrate existing DB that predates the sentiment column
    try:
        conn.execute("ALTER TABLE afr_headlines ADD COLUMN sentiment TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Sitemap fetching and parsing
# ---------------------------------------------------------------------------

def _fetch_sitemap(timeout: float = 15.0) -> str | None:
    req = Request(
        _SITEMAP_URL,
        headers={
            "User-Agent": _UA,
            "Accept": "application/xml,text/xml,*/*;q=0.8",
            "Accept-Language": "en-AU,en;q=0.9",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("AFR sitemap fetch failed: %s", exc)
        return None


def _parse_sitemap(xml_text: str) -> list[dict]:
    """Return list of {url, title, pub_date} dicts from sitemap XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("AFR sitemap parse error: %s", exc)
        return []

    entries = []
    for url_el in root.findall(f"{{{_SM_NS}}}url"):
        loc = url_el.findtext(f"{{{_SM_NS}}}loc", default="")
        news_el = url_el.find(f"{{{_NEWS_NS}}}news")
        if news_el is None:
            continue
        pub_date_str = news_el.findtext(f"{{{_NEWS_NS}}}publication_date", default="")
        title = news_el.findtext(f"{{{_NEWS_NS}}}title", default="")
        if not title or not loc:
            continue

        pub_date: str | None = None
        if pub_date_str:
            try:
                dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                pub_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        entries.append({"url": loc, "title": title, "pub_date": pub_date})
    return entries


# ---------------------------------------------------------------------------
# Ticker matching
# ---------------------------------------------------------------------------

def build_search_terms(ticker: str, company_name: str | None) -> list[str]:
    """Return lower-cased terms to match against headline text."""
    symbol = ticker.upper().removesuffix(".AX")
    terms: list[str] = [symbol.lower()]
    if company_name:
        stopwords = {
            "the", "and", "for", "inc", "ltd", "pty", "plc", "corp", "group",
            "limited", "holdings", "australia", "australian", "company",
        }
        for word in re.split(r"[\s\-/]+", company_name):
            w = re.sub(r"[^\w]", "", word).lower()
            if len(w) >= 3 and w not in stopwords:
                terms.append(w)
    return list(dict.fromkeys(terms))  # dedupe, preserve order


def headline_matches(title: str, url: str, terms: list[str]) -> bool:
    """Return True if any search term appears in the headline title or URL."""
    t = title.lower()
    u = url.lower()
    return any(term in t or term in u for term in terms)


# ---------------------------------------------------------------------------
# Ingestion (used by scanner script)
# ---------------------------------------------------------------------------

def ingest_sitemap(
    conn: sqlite3.Connection,
    watchlist: dict[str, str],
) -> dict:
    """Fetch the live sitemap, store new headlines, and tag against watchlist.

    ``watchlist`` maps ticker → company_name (may be empty string).

    Returns a summary dict with keys ``new_headlines``, ``new_tags``.
    """
    xml_text = _fetch_sitemap()
    if xml_text is None:
        return {"new_headlines": 0, "new_tags": 0}

    entries = _parse_sitemap(xml_text)
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    new_headlines = 0
    new_tags = 0

    ticker_terms = {
        ticker: build_search_terms(ticker, name)
        for ticker, name in watchlist.items()
    }

    for entry in entries:
        sentiment = classify_headline(entry["title"])
        cur = conn.execute(
            "INSERT OR IGNORE INTO afr_headlines (url, title, pub_date, scraped_at, sentiment) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry["url"], entry["title"], entry["pub_date"], now, sentiment),
        )
        if cur.rowcount:
            new_headlines += 1

        for ticker, terms in ticker_terms.items():
            if headline_matches(entry["title"], entry["url"], terms):
                cur2 = conn.execute(
                    "INSERT OR IGNORE INTO afr_headline_tickers (url, ticker) VALUES (?, ?)",
                    (entry["url"], ticker),
                )
                if cur2.rowcount:
                    new_tags += 1

    conn.commit()

    _backfill_sentiment(conn)
    _retag_untagged(conn, ticker_terms)

    return {"new_headlines": new_headlines, "new_tags": new_tags}


def _backfill_sentiment(conn: sqlite3.Connection) -> None:
    """Classify any headlines that have no sentiment yet (migration path)."""
    rows = conn.execute(
        "SELECT url, title FROM afr_headlines WHERE sentiment IS NULL"
    ).fetchall()
    for url, title in rows:
        conn.execute(
            "UPDATE afr_headlines SET sentiment = ? WHERE url = ?",
            (classify_headline(title), url),
        )
    if rows:
        conn.commit()


def _retag_untagged(conn: sqlite3.Connection, ticker_terms: dict[str, list[str]]) -> int:
    """Tag stored headlines that have no ticker assignment yet."""
    rows = conn.execute("""
        SELECT h.url, h.title
        FROM afr_headlines h
        LEFT JOIN afr_headline_tickers t ON h.url = t.url
        WHERE t.url IS NULL
    """).fetchall()

    added = 0
    for url, title in rows:
        for ticker, terms in ticker_terms.items():
            if headline_matches(title, url, terms):
                conn.execute(
                    "INSERT OR IGNORE INTO afr_headline_tickers (url, ticker) VALUES (?, ?)",
                    (url, ticker),
                )
                added += 1
    conn.commit()
    return added


# ---------------------------------------------------------------------------
# Query (used by news analyst)
# ---------------------------------------------------------------------------

def fetch_afr_headlines(
    ticker: str,
    company_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 15,
    db_path: str | Path | None = None,
    timeout: float = 15.0,
) -> str:
    """Return AFR headlines relevant to ``ticker`` as a prompt-ready string.

    Queries the local DB first. Falls back to the live sitemap if the DB has no
    rows for the requested period (e.g. scanner has never run).
    """
    path = Path(db_path) if db_path else _default_db_path()

    if path.exists():
        try:
            conn = open_db(path)
            rows = _query_db(conn, ticker, start_date, end_date, limit)
            conn.close()
            if rows:
                return _format_results(ticker, rows)
            logger.debug(
                "AFR DB has no rows for %s in %s–%s; falling back to live sitemap",
                ticker, start_date, end_date,
            )
        except Exception as exc:
            logger.warning("AFR DB query failed: %s; falling back to live sitemap", exc)

    return _fetch_live(ticker, company_name, start_date, end_date, limit, timeout)


def _query_db(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> list[dict]:
    params: list = [ticker]
    where = ["t.ticker = ?"]
    if start_date:
        where.append("h.pub_date >= ?")
        params.append(start_date)
    if end_date:
        where.append("h.pub_date <= ?")
        params.append(end_date)
    params.append(limit)

    sql = f"""
        SELECT h.title, h.pub_date, h.url, h.sentiment
        FROM afr_headlines h
        JOIN afr_headline_tickers t ON h.url = t.url
        WHERE {" AND ".join(where)}
        ORDER BY h.pub_date DESC
        LIMIT ?
    """
    return [
        {
            "title": r[0],
            "date": r[1] or "unknown date",
            "url": r[2],
            "sentiment": r[3] or "neutral",
        }
        for r in conn.execute(sql, params).fetchall()
    ]


def _format_results(ticker: str, rows: list[dict]) -> str:
    pos = sum(1 for r in rows if r["sentiment"] == "positive")
    neg = sum(1 for r in rows if r["sentiment"] == "negative")
    neu = len(rows) - pos - neg

    lines = [
        f"AFR (Australian Financial Review) — {len(rows)} relevant headlines "
        f"({pos} positive · {neg} negative · {neu} neutral):",
        f"  Legend: [+] positive  [-] negative  [~] neutral",
        "",
    ]
    for r in rows:
        tag = _SENTIMENT_LABEL.get(r["sentiment"], "[~]")
        lines.append(f"  {tag} [{r['date']}] {r['title']}")
        lines.append(f"       {r['url']}")
    return "\n".join(lines)


def _fetch_live(
    ticker: str,
    company_name: str | None,
    start_date: str | None,
    end_date: str | None,
    limit: int,
    timeout: float,
) -> str:
    """Fallback: search the live sitemap without touching the DB."""
    xml_text = _fetch_sitemap(timeout=timeout)
    if xml_text is None:
        return "<AFR headlines unavailable: network error>"

    entries = _parse_sitemap(xml_text)
    terms = build_search_terms(ticker, company_name)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

    results: list[dict] = []
    for e in entries:
        if e["pub_date"]:
            d = datetime.strptime(e["pub_date"], "%Y-%m-%d").date()
            if start_dt and d < start_dt:
                continue
            if end_dt and d > end_dt:
                continue
        if not headline_matches(e["title"], e["url"], terms):
            continue
        results.append({
            "title": e["title"],
            "date": e["pub_date"] or "unknown date",
            "url": e["url"],
            "sentiment": classify_headline(e["title"]),
        })
        if len(results) >= limit:
            break

    if not results:
        sym = ticker.upper().removesuffix(".AX")
        return f"<no AFR headlines found for {sym}>"
    return _format_results(ticker, results)
