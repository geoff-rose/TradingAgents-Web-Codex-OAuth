#!/usr/bin/env python3
"""Download the current ASX 300 constituent list and save to a text file.

Data sources (both from SSGA / State Street Global Advisors):
  - SPDR S&P/ASX 200 Fund (STW)          — top 200 by float-adj market cap
  - SPDR S&P/ASX Small Ordinaries (SSO)  — ranks ~101-300 (mid-caps)

Both ETFs publish daily holdings Excel files publicly on ssga.com.
The union of both lists closely matches the official S&P/ASX 300 index.

Usage
-----
    uv run python scripts/fetch_asx300.py
    uv run python scripts/fetch_asx300.py --output /custom/path/tickers.txt

Run this monthly (or quarterly) to keep the list current with index rebalances.
"""

from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

_DEFAULT_OUTPUT = Path.home() / ".tradingagents" / "asx300_tickers.txt"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
_SSGA_BASE = "https://www.ssga.com/library-content/products/fund-data/etfs/apac"
_SOURCES = {
    "STW (ASX 200)":          f"{_SSGA_BASE}/holdings-daily-au-en-stw.xlsx",
    "SSO (Small Ordinaries)": f"{_SSGA_BASE}/holdings-daily-au-en-sso.xlsx",
}


def fetch_holdings(name: str, url: str) -> list[str]:
    """Download an SSGA holdings Excel and return a list of ASX tickers (e.g. BHP.AX)."""
    print(f"Fetching {name}...", end=" ", flush=True)
    r = requests.get(
        url,
        headers={"User-Agent": _UA, "Referer": "https://www.ssga.com/au/"},
        timeout=20,
    )
    r.raise_for_status()

    df = pd.read_excel(BytesIO(r.content), skiprows=4)

    # Ticker column is "BHP-AU" format — convert to "BHP.AX"
    tickers = []
    for raw in df["Ticker"].dropna().astype(str):
        raw = raw.strip()
        if raw.endswith("-AU"):
            tickers.append(raw[:-3] + ".AX")
        elif raw in ("-", "nan", ""):
            continue
        else:
            tickers.append(raw)  # keep as-is (cash/other rows will be filtered later)

    # Keep only tickers that look like real ASX codes (letters/numbers + .AX)
    tickers = [t for t in tickers if t.endswith(".AX") and len(t) <= 10]

    print(f"{len(tickers)} tickers.")
    return tickers


def main() -> None:
    parser = argparse.ArgumentParser(description="Download current ASX 300 ticker list from SSGA ETF holdings")
    parser.add_argument(
        "--output", "-o",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output file (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_tickers: list[str] = []
    seen: set[str] = set()

    for name, url in _SOURCES.items():
        tickers = fetch_holdings(name, url)
        for t in tickers:
            if t not in seen:
                seen.add(t)
                all_tickers.append(t)

    print(f"\nCombined unique tickers: {len(all_tickers)}")

    # Show top 10 and bottom 10 as a sanity check
    print("\nFirst 10:", ", ".join(all_tickers[:10]))
    print("Last 10: ", ", ".join(all_tickers[-10:]))

    with open(output_path, "w") as f:
        f.write(f"# ASX 300 ticker list\n")
        f.write(f"# Source: SSGA SPDR STW (ASX 200) + SSO (Small Ordinaries)\n")
        f.write(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# Total: {len(all_tickers)} tickers\n")
        for ticker in all_tickers:
            f.write(ticker + "\n")

    print(f"\nSaved {len(all_tickers)} tickers to {output_path}")


if __name__ == "__main__":
    main()
