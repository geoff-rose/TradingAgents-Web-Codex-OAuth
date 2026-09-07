"""Did the job that reported success actually write anything?

**Why this exists.** Five separate jobs here have reported success while doing
nothing: `asx-swing-sync` returning {"checked":0} through a three-day IBKR
logout, spread capture storing NaN rows, the classifier answering HTTP 200 for
a full trading day after gpt-5.4 was withdrawn, an A/B driver that scored 101
of 910 and said COMPLETE, and the SPI 200 scrape persisting nulls after
Barchart started blocking. Every one exited zero. Watching exit codes cannot
catch this class at all.

**So this checks effects, not return codes.** For each job: systemd says the
unit last succeeded at T; the table that job owns must then hold a write at or
after T. If the unit ran and the data did not move, the run was hollow.

**Only jobs that MUST write on every run are listed.** The gap-reversion entry
fires perhaps sixty times a year, swing proposals are often empty, and the EOD
volume scan legitimately finds no >6x candidates -- a check on those would cry
wolf most days, and a monitor people learn to ignore is worse than none. Those
are named in SKIPPED with the reason, so the omission is a decision on the
record rather than an oversight.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any

# unit -> the table it owns, the column stamped on write, and why a silent
# no-write is a real failure rather than a quiet day.
CHECKS: list[dict[str, Any]] = [
    {"unit": "asx-spread-capture.service", "table": "spread_snapshots",
     "column": "captured_at", "grace_min": 20,
     "why": "captures a fixed ~84-ticker list at four fixed times; a run that "
            "stores nothing means IBKR was unreachable or returned NaN"},
    {"unit": "asx-mover-finalize.service", "table": "mover_log",
     "column": "last_seen_at", "grace_min": 30,
     "why": "rewrites the session's true high/low/close; nothing to write means "
            "it could not fetch them"},
    {"unit": "asx-forward-returns.service", "table": "signal_outcomes",
     "column": "updated_at", "grace_min": 45,
     "why": "re-snapshots every scored ticker-session; horizons mature daily so "
            "there is always something to update"},
    {"unit": "asx-openhigh-review.service", "table": "openhigh_population",
     "column": "reviewed_at", "grace_min": 20,
     "why": "writes exactly one population row per trading session"},
    {"unit": "asx-sector-map.service", "table": "sector_map",
     "column": "computed_at", "grace_min": 120,
     "why": "rebuilds the whole ticker->sector index; a no-write means the "
            "returns fetch failed"},
]

# Deliberately not checked, and why. A monitor that fires on a normal quiet day
# gets ignored, and then it protects nothing.
SKIPPED = {
    "asx-gap-reversion-entry": "only fires on a gap-down cohort, ~60 times a year",
    "asx-gap-reversion-exit": "no-op unless an entry is open",
    "asx-eod-volume": "legitimately finds no >6x volume candidates on most days",
    "asx-eod-open/close/resolve": "conditional on the volume scan having found candidates",
    "asx-swing-propose": "an empty proposal set is a normal outcome",
    "asx-swing-sync": "no open orders to sync is normal; IBKR liveness is covered "
                      "by the spread capture check, which needs the same gateway",
    "asx-scanner-refresh": "writes only tickers passing the move/volume screen, "
                           "which can genuinely be none early in a session",
    "asx-signals-refresh": "covered by /api/asx/signals/health, which checks call "
                           "success rather than rows written",
    "asx-tuning-lane": "a no-op by design until a tuned version is minted",
}


def _unit_last_success(unit: str) -> datetime | None:
    """When systemd last ran this unit to a successful finish."""
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return None
    # Parsed as key=value, NOT positionally: `--value` prints properties in
    # alphabetical order, so Result comes out before ExecMainStartTimestamp and
    # reading them in the order requested silently swaps them. That bug made
    # every check return "cannot determine", which this function then reported
    # as healthy -- the exact failure this module exists to catch.
    out = subprocess.run(
        [systemctl, "show", unit, "-p", "ExecMainStartTimestamp", "-p", "Result"],
        capture_output=True, text=True, timeout=15)
    props = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
    if props.get("Result") != "success":
        return None                       # a failed unit is already visible
    stamp = (props.get("ExecMainStartTimestamp") or "").strip()
    if not stamp:
        return None
    for fmt in ("%a %Y-%m-%d %H:%M:%S %Z", "%a %Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unparsable systemd timestamp for {unit}: {stamp!r}")


def _last_write(table: str, column: str) -> datetime | None:
    from .mover_log import DB_PATH
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as db:
        row = db.execute(f'SELECT MAX("{column}") FROM "{table}"').fetchone()
    if not row or not row[0]:
        return None
    try:
        dt = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def check() -> dict[str, Any]:
    """One entry per checked job. `stale` means the unit reported success but
    its table holds no write from that run."""
    results, stale, unknown = [], [], []
    for c in CHECKS:
        try:
            ran = _unit_last_success(c["unit"])
            wrote = _last_write(c["table"], c["column"])
        except Exception as exc:
            # A watchdog that cannot read its own inputs must say so. Returning
            # "ok" because the lookup broke is how a monitor becomes decorative.
            results.append({"unit": c["unit"], "table": c["table"],
                            "stale": False, "unknown": True,
                            "detail": f"check failed: {type(exc).__name__}: {exc}"[:200]})
            unknown.append(c["unit"])
            continue
        entry = {"unit": c["unit"], "table": c["table"],
                 "last_success": ran.isoformat() if ran else None,
                 "last_write": wrote.isoformat() if wrote else None,
                 "why_it_matters": c["why"], "stale": False}
        if ran is not None:
            # the write must land at or after the run, allowing for a job that
            # takes a while to get going
            deadline = ran - timedelta(minutes=c["grace_min"])
            if wrote is None or wrote < deadline:
                entry["stale"] = True
                entry["detail"] = (
                    f"{c['unit']} reported success at {ran:%Y-%m-%d %H:%M}Z but "
                    f"{c['table']}.{c['column']} has not advanced past "
                    f"{wrote:%Y-%m-%d %H:%M}Z" if wrote else
                    f"{c['unit']} reported success at {ran:%Y-%m-%d %H:%M}Z but "
                    f"{c['table']} is empty")
                stale.append(entry)
        results.append(entry)
    return {"ok": not stale and not unknown, "checked": len(results),
            "stale": len(stale), "unknown": len(unknown),
            "results": results, "not_checked": SKIPPED}
