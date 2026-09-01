#!/usr/bin/env python3
"""Fetch company info for all ASX tickers from api.asxshort.app and cache locally.

Usage:
    uv run python scripts/fetch_company_info.py [--force]

Saves JSON files to ~/.tradingagents/company_info/{CODE}.json
Skips tickers that already have a cached file unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TICKER_FILE = Path.home() / ".tradingagents" / "asx300_tickers.txt"
OUT_DIR     = Path.home() / ".tradingagents" / "company_info"
API_BASE    = "https://api.asxshort.app/api/v1/stocks"
HEADERS     = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
DELAY       = 0.3   # seconds between requests — be polite


def fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch even if already cached")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    tickers = [
        line.strip()
        for line in TICKER_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    total = len(tickers)
    done = skipped = failed = 0

    for i, ticker in enumerate(tickers, 1):
        code = ticker.replace(".AX", "").upper()
        path = OUT_DIR / f"{code}.json"

        if path.exists() and not args.force:
            skipped += 1
            continue

        try:
            data = fetch(f"{API_BASE}/{code}")
            path.write_text(json.dumps(data, indent=2))
            done += 1
            print(f"  [{i}/{total}] {code:10} ✓  {data.get('name', '')[:40]}")
        except urllib.error.HTTPError as e:
            failed += 1
            print(f"  [{i}/{total}] {code:10} HTTP {e.code}", file=sys.stderr)
        except Exception as e:
            failed += 1
            print(f"  [{i}/{total}] {code:10} ERROR: {e}", file=sys.stderr)

        time.sleep(DELAY)

    print(f"\nDone: {done} fetched, {skipped} skipped (cached), {failed} failed")
    print(f"Data saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
