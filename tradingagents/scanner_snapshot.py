"""Durable latest movers scan; reading never downloads market data."""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scanner_snapshot.db"


def latest():
    if not DB_PATH.exists():
        return {"available": False, "rows": [], "n_rows": 0}
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT payload FROM snapshot WHERE id=1").fetchone()
    return {**json.loads(row[0]), "available": True, "cached": True} if row else {
        "available": False, "rows": [], "n_rows": 0,
    }


def save(data):
    # A quiet session is valid; an unsuccessful download must not replace it.
    if data.get("n_bars_read", 0) <= 0:
        return False
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS snapshot (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute("INSERT OR REPLACE INTO snapshot VALUES (1, ?)", (json.dumps(data),))
    return True
