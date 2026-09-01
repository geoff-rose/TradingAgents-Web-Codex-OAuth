"""Process-wide lock serialising yfinance downloads.

`yf.download(..., threads=True)` is not safe to call concurrently from more
than one thread in the same process: it mutates a shared internal cache, and
two overlapping calls raise `RuntimeError: dictionary changed size during
iteration` from somewhere deep inside yfinance. Observed 2026-08-25 at 10:41
Sydney, when a forced scan and the browser's own scan overlapped -- the
browser's request 500'd and the page showed "Scan failed".

Everything in this project that downloads bars runs inside the web server's
ThreadPoolExecutor, so a plain `threading.Lock` is the right primitive (an
asyncio lock would only guard the event loop, not the worker threads).

This serialises downloads rather than making them safe to interleave; a second
caller waits. That is the correct trade here -- these are 30-60s bulk pulls
whose results are cached, so waiting is cheaper than failing.
"""

from __future__ import annotations

import threading

YF_LOCK = threading.RLock()
