"""Is this market trading right now?

**Computed from published exchange hours, not asked of the data provider.**
Yahoo carries a `marketState` field, but it reported CLOSED for ES=F in the
middle of a Globex session on 2026-09-07 -- the same bug that had the dashboard
showing a settlement price as if it were live. It also costs one HTTP request
per ticker, where this costs nothing.

**Holidays are not handled.** That needs an exchange calendar per market, and
the failure mode is mild and in the safe direction for a status light: on a
public holiday the dot says "open" while the market is shut, so a flat quote
looks unexplained rather than a live quote looking stale. The same limitation
is documented in `asx_feed.tradeable_session_for`.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

# symbol -> (timezone, [(open, close), ...] in local time, label)
# Ranges are same-day; a lunch break is simply two ranges.
CASH_SESSIONS: dict[str, tuple[str, list[tuple[time, time]]]] = {
    "^STI":      ("Asia/Singapore", [(time(9, 0), time(12, 0)), (time(13, 0), time(17, 0))]),
    "^HSI":      ("Asia/Hong_Kong", [(time(9, 30), time(12, 0)), (time(13, 0), time(16, 0))]),
    "^N225":     ("Asia/Tokyo",     [(time(9, 0), time(11, 30)), (time(12, 30), time(15, 30))]),
    "^KS11":     ("Asia/Seoul",     [(time(9, 0), time(15, 30))]),
    "000001.SS": ("Asia/Shanghai",  [(time(9, 30), time(11, 30)), (time(13, 0), time(15, 0))]),
    "^AXJO":     ("Australia/Sydney", [(time(10, 0), time(16, 0))]),
}

_CHICAGO = ZoneInfo("America/Chicago")
_SYDNEY = ZoneInfo("Australia/Sydney")


def _cme_open(now: datetime) -> bool:
    """CME Globex: continuous from Sunday 17:00 CT to Friday 16:00 CT, with a
    daily maintenance halt from 16:00 to 17:00 CT."""
    t = now.astimezone(_CHICAGO)
    wd = t.weekday()                      # Mon=0 .. Sun=6
    if wd == 5:                           # Saturday
        return False
    if wd == 6 and t.time() < time(17, 0):   # Sunday before the reopen
        return False
    if wd == 4 and t.time() >= time(16, 0):  # Friday after the close
        return False
    return not (time(16, 0) <= t.time() < time(17, 0))   # daily halt


def _spi_open(now: datetime) -> bool:
    """ASX 24 SPI 200: day session 09:50-16:30 Sydney, night session 17:10
    through 07:00 the following morning, Monday to Friday."""
    t = now.astimezone(_SYDNEY)
    wd, clock = t.weekday(), t.time()
    if wd < 5 and time(9, 50) <= clock < time(16, 30):
        return True
    if wd < 5 and clock >= time(17, 10):          # night session opens
        return True
    if wd in (1, 2, 3, 4, 5) and clock < time(7, 0):  # night session continues
        return True
    return False


def is_open(symbol: str, now: datetime | None = None) -> bool | None:
    """True/False when the schedule is known, None when it is not -- callers
    must show "unknown" rather than guessing a colour."""
    now = now or datetime.now(ZoneInfo("UTC"))
    if symbol in ("ES=F", "NQ=F"):
        return _cme_open(now)
    if symbol in ("AP*0", "AP*0-CHG"):
        return _spi_open(now)
    sched = CASH_SESSIONS.get(symbol)
    if not sched:
        return None
    tzname, ranges = sched
    local = now.astimezone(ZoneInfo(tzname))
    if local.weekday() >= 5:
        return False
    return any(a <= local.time() < b for a, b in ranges)


def status(symbol: str, now: datetime | None = None) -> dict[str, Any]:
    open_now = is_open(symbol, now)
    return {"is_open": open_now,
            "session": ("open" if open_now else "closed") if open_now is not None
                       else "unknown"}
