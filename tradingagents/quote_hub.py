"""Shared live-quote fan-out for the watchlist page (2026-08-25, user request).

The watchlist used to repaint on a 10-minute timer. This pushes prices to the
browser as they change instead, the way a broker page behaves -- no refresh,
no polling from the client.

**One poller, many viewers.** Every connected browser reads from a single
refresh loop rather than triggering its own fetch. Two tabs open must not mean
two sets of provider calls; with IBKR later it would also mean two market-data
subscriptions against a limited number of lines.

**The loop only runs while someone is watching.** It starts on the first
subscriber and stops after the last one disconnects, so an idle dashboard
costs nothing.

**Deliberately source-agnostic.** It calls `watchlist.get_quotes()`, the same
seam the rest of the app uses, so pointing that at IBKR streaming later makes
this page live without touching anything here. The only thing that changes is
how fast meaningful updates arrive: against the current ~20-minute delayed
daily bars a "live" page is honest about being live-updating, not live-priced.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# How often the shared loop re-reads quotes. Matched to the current source:
# yfinance daily bars do not move faster than about a minute, so polling
# harder would burn requests to redraw identical numbers. A streaming source
# would push instead of poll and this becomes a fallback tick.
REFRESH_SECONDS = 20.0

# Sent when nothing has changed, so proxies and browsers keep the connection
# open rather than treating a quiet market as a dead socket.
HEARTBEAT_SECONDS = 25.0


class QuoteHub:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._latest: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    # -- subscription ------------------------------------------------------
    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        async with self._lock:
            self._subscribers.add(q)
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._run())
        if self._latest:
            # Seed immediately so a new tab paints from cache instead of
            # sitting blank until the next refresh.
            q.put_nowait(("snapshot", self._latest))
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(q)
            if not self._subscribers and self._task:
                self._task.cancel()
                self._task = None

    @property
    def n_subscribers(self) -> int:
        return len(self._subscribers)

    # -- the shared loop ---------------------------------------------------
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                from tradingagents import watchlist as wl
                items = await loop.run_in_executor(None, wl.list_items)
                tickers = [i["ticker"] for i in items]
                quotes = (await loop.run_in_executor(None, wl.get_quotes, tickers)
                          if tickers else {})

                # Only send what actually moved. A watchlist mostly sits still
                # between ticks, and repainting every row every refresh would
                # make the flash-on-change highlight meaningless.
                changed = {t: q for t, q in quotes.items() if self._latest.get(t) != q}
                gone = [t for t in self._latest if t not in quotes]
                self._latest = quotes
                if changed or gone:
                    await self._broadcast("update", {"quotes": changed, "removed": gone})
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed fetch must not kill the stream; the next tick retries.
                logger.warning("quote refresh failed", exc_info=True)
            await asyncio.sleep(REFRESH_SECONDS)

    async def _broadcast(self, kind: str, payload: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait((kind, payload))
            except asyncio.QueueFull:
                # A browser that has stopped reading (backgrounded tab, dead
                # socket) must not stall everyone else's updates.
                logger.debug("dropping update for a slow subscriber")


HUB = QuoteHub()


async def event_stream(request: Any = None):
    """SSE generator: a snapshot first, then deltas, with heartbeats."""
    q = await HUB.subscribe()
    try:
        yield f"event: hello\ndata: {json.dumps({'refresh_seconds': REFRESH_SECONDS})}\n\n"
        while True:
            try:
                kind, payload = await asyncio.wait_for(q.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            yield f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"
    finally:
        await HUB.unsubscribe(q)
