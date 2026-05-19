"""HotCopper forum scraper for ASX ticker sentiment.

HotCopper (hotcopper.com.au) is the dominant Australian retail investor
forum. Each ASX stock has a dedicated page listing recent threads, each
of which carries an optional user-applied sentiment badge (Bullish /
Bearish / empty).

No API key required — the stock pages are publicly accessible.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_BASE = "https://hotcopper.com.au/asx/{symbol}/"
_UA   = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

_HTML_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
                  "&#039;": "'", "&apos;": "'", "&nbsp;": " "}


def _decode_entities(text: str) -> str:
    for ent, char in _HTML_ENTITIES.items():
        text = text.replace(ent, char)
    return re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)


def _parse_date(title_attr: str) -> str:
    """Convert HotCopper date title attribute to a human-readable string.

    Formats seen:
      ''               → today
      '15/08/24'       → 2024-08-15
      'Yesterday, ...' → Yesterday
      'Wednesday, ...' → Wednesday  (within current week)
      '18/05/26'       → 2026-05-18 (ISO-like DD/MM/YY)
    """
    s = title_attr.strip()
    if not s:
        return "Today"
    if s.lower().startswith("yesterday"):
        return "Yesterday"
    # Day-of-week prefix: "Wednesday, 09:58"
    days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    if any(s.lower().startswith(d) for d in days):
        return s.split(",")[0].strip()
    try:
        return datetime.strptime(s, "%d/%m/%y").strftime("%Y-%m-%d")
    except ValueError:
        return s


def fetch_hotcopper_posts(ticker: str, limit: int = 20, timeout: float = 12.0) -> str:
    """Scrape recent HotCopper threads for ``ticker`` and return a formatted
    plaintext block ready for prompt injection.

    Strips the ``.AX`` exchange suffix automatically. Returns a descriptive
    placeholder string on any failure so callers never need to handle None.
    """
    symbol = ticker.upper().removesuffix(".AX").lower()
    url    = _BASE.format(symbol=symbol)
    req    = Request(url, headers={
        "User-Agent":      _UA,
        "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
        "Referer":         "https://hotcopper.com.au/",
    })

    try:
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("HotCopper fetch failed for %s: %s", ticker, exc)
        return f"<hotcopper unavailable: {type(exc).__name__}>"

    # Parse by independently extracting subject cells and date cells then
    # zipping them — avoids fragile multi-column row regexes.

    # subject-td blocks contain the thread title link and optional badge.
    # Announcements use class="subject-td has-black-link" — match by prefix.
    subject_blocks = re.findall(
        r'<td[^>]*class="subject-td[^"]*"[^>]*>(.*?)</td>', html, re.S
    )

    # Each thread row has exactly one stats-td with a title attribute (the date).
    # The reply/view cells don't carry a title attr, so this gives one per thread.
    date_titles_deduped = re.findall(
        r'<td[^>]*class="stats-td[^"]*"[^>]*title="([^"]*)"', html
    )

    threads: list[dict] = []

    for subject_html, date_title in zip(subject_blocks, date_titles_deduped):
        if len(threads) >= limit:
            break

        # Thread title — prefer the <a title="..."> attribute; fall back to text
        title_m = re.search(
            r'<a href="/threads/[^"]+"[^>]*title="([^"]+)"', subject_html
        )
        if not title_m:
            # plain text fallback
            title_m = re.search(r'<a href="/threads/[^"]+"[^>]*>([^<]+)<', subject_html)
        if not title_m:
            continue
        title = _decode_entities(title_m.group(1).strip())

        # Badge sentiment (Bullish / Bearish / "")
        badge_m = re.search(r'class="thread-badge"[^>]+title="([^"]*)"', subject_html, re.I)
        badge = badge_m.group(1).strip() if badge_m else ""

        is_ann = title.startswith("Ann:")
        post_date = _parse_date(date_title)

        threads.append({
            "title":  title,
            "badge":  badge,
            "date":   post_date,
            "is_ann": is_ann,
        })

    if not threads:
        return f"<no HotCopper threads found for {ticker.upper().removesuffix('.AX')}>"

    # Tally badge sentiment
    bullish  = sum(1 for t in threads if t["badge"].lower() == "bullish")
    bearish  = sum(1 for t in threads if t["badge"].lower() == "bearish")
    no_badge = len(threads) - bullish - bearish
    ann_count = sum(1 for t in threads if t["is_ann"])

    lines = [
        f"HotCopper — {len(threads)} recent threads for "
        f"{ticker.upper().removesuffix('.AX')} "
        f"({bullish} Bullish · {bearish} Bearish · {no_badge} untagged · "
        f"{ann_count} ASX announcements):"
    ]
    for t in threads:
        tag = f" [{t['badge']}]" if t["badge"] else ""
        ann = " [Ann]" if t["is_ann"] else ""
        lines.append(f"  [{t['date']}{tag}{ann}] {t['title']}")

    return "\n".join(lines)
