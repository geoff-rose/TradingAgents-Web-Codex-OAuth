"""Edge cases for the open/closed indicators.

Expectations for SPI were checked against IBKR's own `tradingHours` for APU6
rather than reasoned about: 20260904:1710-20260905:0700, 20260907:0950-1630,
20260907:1710-20260908:0700. There is no Sunday-evening session, so early
Monday morning is CLOSED -- the first version of this test asserted otherwise
and the code was right.
"""
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, "/opt/tradingagents")
from tradingagents.market_hours import is_open

UTC = ZoneInfo("UTC")


def at(tz, s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(
        tzinfo=ZoneInfo(tz)).astimezone(UTC)


CASES = [
    (at("America/Chicago", "2026-09-05 12:00"), "ES=F", False, "Saturday"),
    (at("America/Chicago", "2026-09-06 12:00"), "ES=F", False, "Sunday pre-reopen"),
    (at("America/Chicago", "2026-09-06 18:00"), "ES=F", True, "Sunday post-reopen"),
    (at("America/Chicago", "2026-09-08 16:30"), "ES=F", False, "daily halt"),
    (at("America/Chicago", "2026-09-08 15:59"), "ES=F", True, "just before halt"),
    (at("America/Chicago", "2026-09-11 16:30"), "ES=F", False, "Friday close"),
    (at("Asia/Tokyo", "2026-09-07 11:45"), "^N225", False, "Tokyo lunch"),
    (at("Asia/Tokyo", "2026-09-07 13:00"), "^N225", True, "Tokyo afternoon"),
    (at("Asia/Tokyo", "2026-09-07 15:45"), "^N225", False, "after Tokyo close"),
    (at("Asia/Hong_Kong", "2026-09-07 12:30"), "^HSI", False, "HK lunch"),
    (at("Asia/Shanghai", "2026-09-07 12:00"), "000001.SS", False, "Shanghai lunch"),
    (at("Asia/Seoul", "2026-09-07 15:00"), "^KS11", True, "Seoul, no lunch break"),
    (at("Asia/Singapore", "2026-09-05 10:00"), "^STI", False, "Saturday"),
    # per IBKR tradingHours: no Sunday-evening session, so Monday 03:00 is shut
    (at("Australia/Sydney", "2026-09-07 03:00"), "AP*0", False, "no Sun-night session"),
    (at("Australia/Sydney", "2026-09-08 03:00"), "AP*0", True, "Mon-night session runs to Tue 07:00"),
    (at("Australia/Sydney", "2026-09-12 03:00"), "AP*0", True, "Fri-night session runs to Sat 07:00"),
    (at("Australia/Sydney", "2026-09-07 16:45"), "AP*0", False, "between day and night"),
    (at("Australia/Sydney", "2026-09-07 18:00"), "AP*0", True, "night session"),
    (at("Australia/Sydney", "2026-09-05 12:00"), "AP*0", False, "Saturday midday"),
    (at("Australia/Sydney", "2026-09-07 12:00"), "^AXJO", True, "ASX mid-session"),
]

if __name__ == "__main__":
    bad = 0
    for when, sym, exp, why in CASES:
        got = is_open(sym, when)
        if got != exp:
            bad += 1
            print(f"  FAIL {sym:<10} {why:<38} expected={exp} got={got}")
    assert is_open("BTC-USD") is None, "unknown symbol must be None, not a guess"
    print(f"  {len(CASES) - bad}/{len(CASES)} passed")
    sys.exit(1 if bad else 0)
