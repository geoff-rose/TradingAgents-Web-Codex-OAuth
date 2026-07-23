"""AFR headline scanner — fetch the AFR news sitemap and store headlines in the DB.

Run this on a cron to capture headlines throughout the day. The AFR sitemap
carries only ~190 most-recent articles, so running 2–3× daily ensures nothing
is missed before it rolls off.

Usage
-----
    # Scan using the default watchlist config (~/.tradingagents/afr_watchlist.json)
    uv run python scripts/scan_afr.py

    # Scan for specific tickers only
    uv run python scripts/scan_afr.py --tickers BHP.AX CBA.AX WES.AX A2M.AX

    # Show stored headlines for a ticker
    uv run python scripts/scan_afr.py --query BHP.AX --start 2026-05-01

Watchlist config
----------------
Create ``~/.tradingagents/afr_watchlist.json`` with a list of ASX tickers:

    ["BHP.AX", "CBA.AX", "WES.AX", "ANZ.AX", "A2M.AX", "WOW.AX"]

Company names are resolved automatically from yfinance on first use and cached
in ``~/.tradingagents/afr_company_names.json`` to avoid repeated API calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yfinance as yf

from tradingagents.dataflows.afr import ingest_sitemap, open_db, _query_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HOME = Path.home() / ".tradingagents"
_WATCHLIST_PATH = _HOME / "afr_watchlist.json"
_COMPANY_NAMES_CACHE = _HOME / "afr_company_names.json"


def load_watchlist(tickers_override: list[str] | None = None) -> list[str]:
    if tickers_override:
        return [t.upper() for t in tickers_override]
    if _WATCHLIST_PATH.exists():
        with open(_WATCHLIST_PATH) as f:
            return [t.upper() for t in json.load(f)]
    logger.warning(
        "No watchlist found at %s and no --tickers argument supplied.\n"
        "Create the file with a JSON list of ASX tickers, e.g.:\n"
        '  ["BHP.AX", "CBA.AX", "WES.AX"]',
        _WATCHLIST_PATH,
    )
    return []


def load_company_names_cache() -> dict[str, str]:
    if _COMPANY_NAMES_CACHE.exists():
        with open(_COMPANY_NAMES_CACHE) as f:
            return json.load(f)
    return {}


def save_company_names_cache(cache: dict[str, str]) -> None:
    _HOME.mkdir(parents=True, exist_ok=True)
    with open(_COMPANY_NAMES_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def resolve_company_names(tickers: list[str]) -> dict[str, str]:
    """Return ticker → company_name dict, using cache + yfinance for misses."""
    cache = load_company_names_cache()
    updated = False
    result: dict[str, str] = {}

    for ticker in tickers:
        if ticker in cache:
            result[ticker] = cache[ticker]
            continue
        try:
            info = yf.Ticker(ticker).info
            name = info.get("shortName") or info.get("longName") or ""
            logger.info("Resolved %s → %s", ticker, name or "(no name)")
        except Exception as exc:
            logger.warning("Could not resolve company name for %s: %s", ticker, exc)
            name = ""
        cache[ticker] = name
        result[ticker] = name
        updated = True

    if updated:
        save_company_names_cache(cache)
    return result


def cmd_scan(args: argparse.Namespace) -> None:
    tickers = load_watchlist(args.tickers)
    if not tickers:
        sys.exit(1)

    watchlist = resolve_company_names(tickers)
    logger.info("Scanning AFR sitemap for %d tickers: %s", len(tickers), ", ".join(tickers))

    conn = open_db()
    summary = ingest_sitemap(conn, watchlist)
    conn.close()

    logger.info(
        "Done — %d new headlines, %d new ticker tags",
        summary["new_headlines"], summary["new_tags"],
    )


def cmd_query(args: argparse.Namespace) -> None:
    conn = open_db()
    rows = _query_db(conn, args.query.upper(), args.start, args.end, args.limit)
    conn.close()

    if not rows:
        print(f"No AFR headlines found for {args.query.upper()} in the requested range.")
        return
    print(f"AFR headlines for {args.query.upper()} ({len(rows)} rows):\n")
    for r in rows:
        print(f"  [{r['date']}] {r['title']}")
        print(f"    {r['url']}")


def cmd_watchlist(args: argparse.Namespace) -> None:
    """Print or save a watchlist."""
    if args.save:
        _HOME.mkdir(parents=True, exist_ok=True)
        with open(_WATCHLIST_PATH, "w") as f:
            json.dump([t.upper() for t in args.save], f, indent=2)
        print(f"Watchlist saved to {_WATCHLIST_PATH}")
    else:
        if _WATCHLIST_PATH.exists():
            with open(_WATCHLIST_PATH) as f:
                tickers = json.load(f)
            print(f"Watchlist ({len(tickers)} tickers): {', '.join(tickers)}")
        else:
            print(f"No watchlist file at {_WATCHLIST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AFR headline scanner")
    sub = parser.add_subparsers(dest="cmd")

    # scan (default)
    p_scan = sub.add_parser("scan", help="Fetch sitemap and store headlines (default)")
    p_scan.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Override watchlist with these tickers")

    # query
    p_query = sub.add_parser("query", help="Query stored headlines for a ticker")
    p_query.add_argument("query", metavar="TICKER")
    p_query.add_argument("--start", metavar="YYYY-MM-DD")
    p_query.add_argument("--end", metavar="YYYY-MM-DD")
    p_query.add_argument("--limit", type=int, default=30)

    # watchlist
    p_wl = sub.add_parser("watchlist", help="View or set the default watchlist")
    p_wl.add_argument("--save", nargs="+", metavar="TICKER",
                      help="Save a new watchlist")

    args = parser.parse_args()

    # Default subcommand is scan (so cron can call the script with no args)
    if args.cmd is None or args.cmd == "scan":
        if args.cmd is None:
            args.tickers = None
        cmd_scan(args)
    elif args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "watchlist":
        cmd_watchlist(args)


if __name__ == "__main__":
    main()
