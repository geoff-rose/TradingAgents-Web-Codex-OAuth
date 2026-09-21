"""Is this market trading right now?

**Computed from published exchange hours, not asked of the data provider.**
Yahoo carries a `marketState` field, but it reported CLOSED for ES=F in the
middle of a Globex session on 2026-09-07 -- the same bug that had the dashboard
showing a settlement price as if it were live. It also costs one HTTP request
per ticker, where this costs nothing.

Exchange holidays are handled with `exchange_calendars` (with the published
clock below as a safe fallback if that package cannot be imported). This keeps
the light from calling a public holiday an open session.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
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
    "^GSPC":     ("America/New_York", [(time(9, 30), time(16, 0))]),
    "^IXIC":     ("America/New_York", [(time(9, 30), time(16, 0))]),
    "^DJI":      ("America/New_York", [(time(9, 30), time(16, 0))]),
}

_CHICAGO = ZoneInfo("America/Chicago")
_SYDNEY = ZoneInfo("Australia/Sydney")

# The package carries exchange-specific holidays and lunch breaks that a
# weekday-only check cannot. Keep the clock tables above as a fallback because
# the status light must remain non-fatal if a deployment is missing the package.
_CALENDAR_NAMES = {
    "^STI": "XSES", "^HSI": "XHKG", "^N225": "XTKS", "^KS11": "XKRX",
    "000001.SS": "XSHG", "^AXJO": "XASX",
    "^GSPC": "XNYS", "^IXIC": "XNYS", "^DJI": "XNYS",
}
_calendar_cache: dict[str, Any] = {}


def _exchange_open(calendar_name: str, now: datetime) -> bool | None:
    """Return an exchange-calendar answer, or None when unavailable."""
    try:
        import exchange_calendars as xc
        import pandas as pd

        calendar = _calendar_cache.get(calendar_name)
        if calendar is None:
            calendar = xc.get_calendar(calendar_name)
            _calendar_cache[calendar_name] = calendar
        stamp = pd.Timestamp(now)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        return bool(calendar.is_open_on_minute(stamp.floor("min"), ignore_breaks=False))
    except Exception:
        return None


def _asx_session_exists(now: datetime) -> bool | None:
    """Whether the relevant ASX trading day exists for the SPI session."""
    try:
        import exchange_calendars as xc

        calendar = _calendar_cache.get("XASX")
        if calendar is None:
            calendar = xc.get_calendar("XASX")
            _calendar_cache["XASX"] = calendar
        local = now.astimezone(_SYDNEY)
        session_day = (local.date() - timedelta(days=1)
                       if local.time() < time(7, 0) else local.date())
        return bool(calendar.is_session(session_day))
    except Exception:
        return None


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
    if time(16, 0) <= t.time() < time(17, 0):             # daily halt
        return False
    # CMES supplies holidays and special closes; the manual clock above keeps
    # the one-hour Globex maintenance break explicit.
    exchange_answer = _exchange_open("CMES", now)
    return True if exchange_answer is None else exchange_answer


def _spi_open(now: datetime) -> bool:
    """ASX 24 SPI 200: day session 09:50-16:30 Sydney, night session 17:10
    through 07:00 the following morning, Monday to Friday."""
    t = now.astimezone(_SYDNEY)
    wd, clock = t.weekday(), t.time()
    manual = False
    if wd < 5 and time(9, 50) <= clock < time(16, 30):
        manual = True
    elif wd < 5 and clock >= time(17, 10):          # night session opens
        manual = True
    elif wd in (1, 2, 3, 4, 5) and clock < time(7, 0):  # night continues
        manual = True
    if not manual:
        return False
    exchange_answer = _asx_session_exists(now)
    return True if exchange_answer is None else exchange_answer


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
    exchange_answer = _exchange_open(_CALENDAR_NAMES[symbol], now)
    if exchange_answer is not None:
        return exchange_answer
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
