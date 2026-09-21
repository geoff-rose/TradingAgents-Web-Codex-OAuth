"""Tests for the `/setups` scanner: indicators, each detector on a series
that must fire and one that must not, the no-lookahead property, and the
scan -> store -> resolve -> scorecard path on a temp DB."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tradingagents import ta_detectors as D
from tradingagents import ta_indicators as I
from tradingagents import ta_setups as S


def _frame(closes, vols=None, spread=0.01, seed=0) -> pd.DataFrame:
    """OHLCV frame on business days from a close path. High/Low bracket the
    close by `spread`; Open is the prior close."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    rng = np.random.default_rng(seed)
    if vols is None:
        vols = np.full(n, 1_000_000.0)
    vols = np.asarray(vols, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = closes * (1 + spread)
    lows = closes * (1 - spread)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                         "Volume": vols}, index=idx)


def _noise(n, base=10.0, amp=0.002, seed=1):
    rng = np.random.default_rng(seed)
    return base * np.cumprod(1 + rng.normal(0, amp, n))


def _ids(dets, state=None):
    return {(d.setup_id, d.state) for d in dets if state is None or d.state == state}


class IndicatorTests(unittest.TestCase):
    def test_ema_seed_is_sma(self):
        s = pd.Series(np.arange(1, 31, dtype=float))
        e = I.ema(s, 10)
        self.assertTrue(np.isnan(e.iloc[8]))
        self.assertAlmostEqual(e.iloc[9], 5.5)
        self.assertAlmostEqual(e.iloc[10], 11 * (2 / 11) + 5.5 * (9 / 11))

    def test_rsi_bounds(self):
        up = pd.Series(np.arange(1, 40, dtype=float))
        self.assertAlmostEqual(I.rsi_wilder(up).iloc[-1], 100.0)
        down = pd.Series(np.arange(40, 1, -1, dtype=float))
        self.assertLess(I.rsi_wilder(down).iloc[-1], 1.0)

    def test_bollinger_population_std(self):
        s = pd.Series(np.arange(1, 41, dtype=float))
        b = I.bollinger(s, 20, 2.0)
        sd = np.std(np.arange(21, 41, dtype=float))
        self.assertAlmostEqual(b["bb_upper"].iloc[-1] - b["bb_mid"].iloc[-1], 2 * sd)

    def test_swing_pivots_known_at(self):
        # zigzag: peaks at 10, 29; troughs at 19, 39 (segments share their joins)
        c = np.concatenate([np.linspace(10, 12, 11), np.linspace(12, 9, 10)[1:],
                            np.linspace(9, 13, 11)[1:], np.linspace(13, 8, 11)[1:],
                            np.linspace(8, 10, 11)[1:]])
        df = _frame(c, spread=0.0)
        pv = I.swing_pivots(df, k=5)
        highs = pv[pv.kind == 1].idx.tolist()
        lows = pv[pv.kind == -1].idx.tolist()
        self.assertIn(10, highs)
        self.assertIn(29, highs)
        self.assertIn(19, lows)
        self.assertIn(39, lows)
        self.assertTrue((pv.known_at == pv.idx + 5).all())


class DetectorTests(unittest.TestCase):
    """Each test builds >= 260 bars of quiet drift, then the shape."""

    N0 = 270

    def _base(self, seed=1, base=10.0):
        return list(_noise(self.N0, base=base, seed=seed))

    def test_donchian_forming_and_near_miss(self):
        c = self._base()
        hh = float(I.indicator_frame(_frame(c + [c[-1]]))["hh20"].iloc[-1])
        c_form = c + [hh * 0.995]
        df = _frame(c_form, vols=[1e6] * self.N0 + [2e6])
        self.assertIn(("donchian20_up", "forming"), _ids(D.detect_all("T", df)))
        # same price, thin volume: no
        df2 = _frame(c_form, vols=[1e6] * self.N0 + [0.8e6])
        self.assertNotIn(("donchian20_up", "forming"), _ids(D.detect_all("T", df2)))
        # close through the level on volume: confirmed
        df3 = _frame(c + [hh * 1.02], vols=[1e6] * self.N0 + [2e6])
        self.assertIn(("donchian20_up", "confirmed"), _ids(D.detect_all("T", df3)))

    def test_rsi_turn_up(self):
        c = self._base()
        drop = list(np.linspace(c[-1], c[-1] * 0.80, 12))  # 12 down days -> RSI < 35
        c2 = c + drop + [drop[-1] * 1.01]
        dets = D.detect_all("T", _frame(c2))
        self.assertIn(("rsi_turn_up", "forming"), _ids(dets))
        # still falling: no
        c3 = c + drop + [drop[-1] * 0.99]
        self.assertNotIn(("rsi_turn_up", "forming"), _ids(D.detect_all("T", _frame(c3))))

    def test_rsi_exhaust(self):
        c = self._base()
        run = list(np.linspace(c[-1], c[-1] * 1.30, 14))   # RSI well above 70, stretched
        c2 = c + run + [run[-1] * 0.99]
        self.assertIn(("rsi_exhaust", "forming"), _ids(D.detect_all("T", _frame(c2))))
        c3 = c + run + [run[-1] * 1.01]
        self.assertNotIn(("rsi_exhaust", "forming"), _ids(D.detect_all("T", _frame(c3))))

    def test_ema20_reclaim(self):
        c = self._base()
        # gentle uptrend, a two-day dip under EMA20, then the first close back above
        up = list(c[-1] * np.cumprod(1 + np.full(30, 0.004)))
        ind = I.indicator_frame(_frame(c + up))
        e = float(ind["ema20"].iloc[-1])
        c2 = c + up + [e * 0.99, e * 0.985]
        ind2 = I.indicator_frame(_frame(c2))
        e2 = float(ind2["ema20"].iloc[-1])
        c3 = c2 + [e2 * 1.012]
        self.assertIn(("ema20_reclaim", "confirmed"), _ids(D.detect_all("T", _frame(c3))))
        # forming: within 1% below on an up bar
        c4 = c2 + [e2 * 0.995]
        self.assertIn(("ema20_reclaim", "forming"), _ids(D.detect_all("T", _frame(c4))))
        # a second close above is not a fresh reclaim
        c5 = c3 + [c3[-1] * 1.003]
        self.assertNotIn(("ema20_reclaim", "confirmed"), _ids(D.detect_all("T", _frame(c5))))

    def test_macd_hist_cross(self):
        c = self._base()
        up0 = list(np.linspace(c[-1], c[-1] * 1.10, 15))
        down = list(np.linspace(up0[-1], up0[-1] * 0.85, 25))
        up = list(np.linspace(down[-1], down[-1] * 1.12, 14))
        dets = D.detect_history("T", _frame(c + up0 + down + up), start_idx=self.N0)
        self.assertIn(("macd_hist_up", "forming"), _ids(dets))
        self.assertIn(("macd_hist_up", "confirmed"), _ids(dets))
        self.assertIn(("macd_hist_down", "confirmed"), _ids(dets))

    def test_pullback_ema20(self):
        c = self._base()
        up = list(c[-1] * np.cumprod(1 + np.full(70, 0.004)))
        df = _frame(c + up)
        ind = I.indicator_frame(df)
        e = float(ind["ema20"].iloc[-1])
        # three down days into EMA20
        c2 = c + up + [up[-1] * 0.99, up[-1] * 0.98, e * 1.005]
        self.assertIn(("pullback_ema20", "forming"), _ids(D.detect_all("T", _frame(c2))))
        c3 = c2 + [c2[-1] * 1.03]
        self.assertIn(("pullback_ema20", "confirmed"), _ids(D.detect_all("T", _frame(c3))))
        # no prior uptrend: no
        flat = list(_noise(70, base=c[-1], amp=0.001, seed=3))
        c4 = c + flat + [flat[-1] * 0.99, flat[-1] * 0.98, flat[-1] * 0.985]
        self.assertNotIn(("pullback_ema20", "forming"), _ids(D.detect_all("T", _frame(c4))))

    def test_bb_squeeze(self):
        c = self._base(seed=2)
        # 40 bars of near-zero range, then expansion up
        env = np.linspace(1, 0.2, 40)      # contracting range: the bandwidth low is at the end
        tight = list(c[-1] * (1 + 0.0005 * env * np.sin(np.arange(40) * 1.3)))
        c2 = c + tight + [tight[-1] * 1.004, tight[-1] * 1.02]
        ids = _ids(D.detect_all("T", _frame(c2)))
        self.assertTrue(("bb_squeeze_up", "confirmed") in ids or ("bb_squeeze_up", "forming") in ids)
        c3 = c + tight + [tight[-1] * 0.996, tight[-1] * 0.98]
        ids3 = _ids(D.detect_all("T", _frame(c3)))
        self.assertTrue(("bb_squeeze_down", "confirmed") in ids3 or ("bb_squeeze_down", "forming") in ids3)
        self.assertNotIn(("bb_squeeze_up", "confirmed"), ids3)

    def test_double_bottom(self):
        c = self._base(seed=4)
        b = c[-1]
        w = (list(np.linspace(b, b * 0.88, 10)) + list(np.linspace(b * 0.88, b * 0.97, 10))
             + list(np.linspace(b * 0.97, b * 0.885, 10)) + list(np.linspace(b * 0.885, b * 0.94, 8)))
        df = _frame(c + w, spread=0.0)
        self.assertIn(("double_bottom", "forming"), _ids(D.detect_all("T", df)))
        df2 = _frame(c + w + [b * 0.985], spread=0.0)
        self.assertIn(("double_bottom", "confirmed"), _ids(D.detect_all("T", df2)))
        # a V (single low) is not a double bottom
        v = list(np.linspace(b, b * 0.88, 15)) + list(np.linspace(b * 0.88, b * 0.94, 23))
        self.assertNotIn(("double_bottom", "forming"), _ids(D.detect_all("T", _frame(c + v, spread=0.0))))

    def test_double_top(self):
        c = self._base(seed=5)
        b = c[-1]
        m = (list(np.linspace(b, b * 1.12, 10)) + list(np.linspace(b * 1.12, b * 1.03, 10))
             + list(np.linspace(b * 1.03, b * 1.115, 10)) + list(np.linspace(b * 1.115, b * 1.06, 8)))
        self.assertIn(("double_top", "forming"), _ids(D.detect_all("T", _frame(c + m, spread=0.0))))
        self.assertIn(("double_top", "confirmed"),
                      _ids(D.detect_all("T", _frame(c + m + [b * 1.015], spread=0.0))))

    def test_asc_triangle(self):
        c = self._base(seed=6)
        b = c[-1]
        r = b * 1.10
        # rising lows 0.90, 0.95, 0.98 against a flat top at r; each leg ~7 bars
        legs = [(b * 0.90, r), (b * 0.95, r * 0.995), (b * 0.98, r * 1.002)]
        path = []
        for lo, hi in legs:
            path += list(np.linspace(lo, hi, 7)) + list(np.linspace(hi, lo * 1.02, 7)[1:])
        # last approach into the level
        path += list(np.linspace(path[-1], r * 0.985, 8)[1:])
        df = _frame(c + path, spread=0.0)
        self.assertIn(("asc_triangle", "forming"), _ids(D.detect_all("T", df)))
        df2 = _frame(c + path + [r * 1.02], spread=0.0)
        self.assertIn(("asc_triangle", "confirmed"), _ids(D.detect_all("T", df2)))

    def test_bull_flag(self):
        c = self._base(seed=7)
        b = c[-1]
        pole = list(np.linspace(b, b * 1.15, 8))
        flag = list(np.linspace(b * 1.15, b * 1.12, 7))
        vols = [1e6] * self.N0 + [3e6] * 8 + [1e6] * 7
        df = _frame(c + pole + flag, vols=vols, spread=0.0)
        self.assertIn(("bull_flag", "forming"), _ids(D.detect_all("T", df)))
        df2 = _frame(c + pole + flag + [b * 1.16], vols=vols + [2e6], spread=0.0)
        self.assertIn(("bull_flag", "confirmed"), _ids(D.detect_all("T", df2)))
        # heavy flag volume: no
        df3 = _frame(c + pole + flag, vols=[1e6] * self.N0 + [3e6] * 8 + [3e6] * 7, spread=0.0)
        self.assertNotIn(("bull_flag", "forming"), _ids(D.detect_all("T", df3)))

    def test_liquidity_filter(self):
        c = self._base()
        thin = _frame(c + [max(c[-20:]) * 1.05], vols=[1e3] * (self.N0 + 1))   # $10k/day
        self.assertEqual(D.detect_all("T", thin), [])

    def test_no_lookahead(self):
        c = list(_noise(330, base=20.0, amp=0.02, seed=11))
        vols = np.random.default_rng(3).uniform(5e5, 3e6, 330)
        df = _frame(c, vols=vols)
        full = D.detect_history("T", df, start_idx=D.MIN_BARS)
        by_date = {}
        for d in full:
            by_date.setdefault(d.date, set()).add((d.setup_id, d.state, d.level))
        for t in range(D.MIN_BARS, 330, 7):
            sub = df.iloc[:t + 1]
            live = D.detect_all("T", sub)
            date = str(sub.index[-1].date())
            self.assertEqual({(d.setup_id, d.state, d.level) for d in live},
                             by_date.get(date, set()), f"mismatch at bar {t}")


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        self.p_db = patch.object(S, "DB_PATH", self.db)
        self.p_db.start()
        self.today = "2024-03-15"
        c = list(_noise(270, base=10.0, seed=1))
        hh = max(c[-20:]) * 1.02
        n = 271
        idx = pd.bdate_range(end=self.today, periods=n)
        breakout = _frame(c + [hh * 1.02], vols=[1e6] * 270 + [2e6])
        breakout.index = idx
        quiet = _frame(list(_noise(271, base=5.0, amp=0.001, seed=9)))
        quiet.index = idx
        self.frames = {"AAA": breakout, "BBB": quiet}

    def tearDown(self):
        self.p_db.stop()
        self.tmp.cleanup()

    def _extend(self, frames, n=25):
        out = {}
        for t, df in frames.items():
            last = float(df["Close"].iloc[-1])
            extra_c = last * np.cumprod(1 + np.full(n, 0.002 if t == "AAA" else -0.001))
            ext = _frame(list(df["Close"]) + list(extra_c), vols=list(df["Volume"]) + [1e6] * n)
            ext.index = pd.bdate_range(start=df.index[0], periods=len(ext))
            out[t] = ext
        return out

    def test_scan_store_resolve_scorecard(self):
        with patch.object(S, "_load_bars", return_value={k: v.copy() for k, v in self.frames.items()}), \
             patch("tradingagents.asx_feed.sydney_today", return_value=self.today), \
             patch("tradingagents.market_hours.is_open", return_value=True), \
             patch("tradingagents.symbols.company_name", return_value="Co"), \
             patch.object(S, "_ensure_today_bar", side_effect=lambda fr, d: {t: "daily_partial" for t in fr}):
            out = S.scan(limit=2, store=True)
        self.assertTrue(out["available"])
        ids = {(d["ticker"], d["setup_id"], d["state"]) for d in out["detections"]}
        self.assertIn(("AAA", "donchian20_up", "confirmed"), ids)
        self.assertEqual(out["run"]["n_liquid"], 2)
        self.assertEqual(out["run"]["bar_provisional"], 1)
        for d in out["detections"]:
            self.assertEqual(d["first_in_window"], 1)

        # Resolve after 25 more sessions.
        later = self._extend(self.frames)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        fake_now = datetime(2024, 4, 22, 17, 10, tzinfo=ZoneInfo("Australia/Sydney"))
        with patch.object(S, "_load_bars", return_value=later), \
             patch.object(S, "_index_bars", return_value={}), \
             patch.object(S, "_sydney_now", return_value=fake_now), \
             patch("tradingagents.sectors.get_map", return_value={}):
            res = S.resolve(days_back=60)
        self.assertGreater(res["resolved"], 0)
        self.assertEqual(res["control_dates"], 1)
        with S._connect() as conn:
            row = dict(conn.execute("SELECT * FROM ta_detections WHERE ticker='AAA'"
                                    " AND setup_id='donchian20_up'").fetchone())
            ctrl = dict(conn.execute("SELECT * FROM ta_control").fetchone())
        self.assertIsNotNone(row["fwd_5d_pct"])
        self.assertIsNotNone(row["fwd_20d_pct"])
        self.assertEqual(row["state_at_close"], "confirmed")
        self.assertEqual(row["fwd_bars_available"], 25)
        self.assertEqual(ctrl["n"], 2)
        self.assertEqual(ctrl["date"], self.today)

        sc = S.scorecard()
        cell = next(c for c in sc["cells"] if c["setup_id"] == "donchian20_up" and c["state"] == "confirmed")
        self.assertEqual(cell["badge"], "insufficient")
        self.assertEqual(cell["live"]["n"], 1)
        self.assertEqual(cell["close_check"]["same_rate"], 1.0)
        with patch.object(S, "_sydney_now", return_value=fake_now):
            h = S.history(days=60)
        self.assertTrue(any(r["ticker"] == "AAA" for r in h["rows"]))

    def test_scan_dedups_against_prior_session(self):
        with patch.object(S, "_load_bars", return_value={k: v.copy() for k, v in self.frames.items()}), \
             patch("tradingagents.asx_feed.sydney_today", return_value=self.today), \
             patch("tradingagents.market_hours.is_open", return_value=True), \
             patch("tradingagents.symbols.company_name", return_value="Co"), \
             patch.object(S, "_ensure_today_bar", side_effect=lambda fr, d: {t: "daily_partial" for t in fr}):
            S.scan(limit=2, store=True)
            with S._connect() as conn:
                conn.execute("UPDATE ta_detections SET scan_date='2024-03-14'")
                conn.execute("UPDATE ta_scan_runs SET scan_date='2024-03-14'")
                conn.commit()
            out = S.scan(limit=2, store=True)
        d = next(x for x in out["detections"] if x["ticker"] == "AAA" and x["setup_id"] == "donchian20_up")
        self.assertEqual(d["first_in_window"], 0)

    def test_skips_when_market_closed(self):
        with patch("tradingagents.market_hours.is_open", return_value=False), \
             patch("tradingagents.asx_feed.sydney_today", return_value="2024-03-16"):
            out = S.scan(limit=2, store=True)
        self.assertEqual(out.get("skipped"), "market closed")

    def test_backfill_and_hypothesis_contract(self):
        c = list(_noise(400, base=20.0, amp=0.02, seed=11))
        vols = np.random.default_rng(3).uniform(5e5, 3e6, 400)
        frames = {"AAA": _frame(c, vols=vols), "BBB": _frame(list(_noise(400, base=8.0, amp=0.015, seed=12)), vols=vols)}
        with patch.object(S, "_index_bars", return_value={}), \
             patch("tradingagents.sectors.get_map", return_value={}), \
             patch("tradingagents.symbols.company_name", return_value="Co"):
            out = S.backfill(store=True, run_gates=False, frames=frames)
        self.assertGreater(out["n_detections"], 0)
        self.assertGreater(out["n_control_dates"], 100)
        with S._connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM ta_detections WHERE source='backfill'").fetchone()[0]
            n_first = conn.execute("SELECT SUM(first_in_window) FROM ta_detections WHERE source='backfill'").fetchone()[0]
        self.assertEqual(n, out["n_detections"])
        self.assertLess(n_first, n)
        # Pick the busiest cell and check the hypothesis contract holds.
        with S._connect() as conn:
            sid, st = conn.execute(
                "SELECT setup_id, state FROM ta_detections WHERE source='backfill'"
                " GROUP BY setup_id, state ORDER BY COUNT(*) DESC LIMIT 1").fetchone()
        df = S.trades_frame(sid, st)
        self.assertTrue({"date", "ticker", "side", "gross_pct", "price", "is_train"} <= set(df.columns))
        self.assertEqual(df["is_train"].nunique(), 2)
        from tradingagents.hypothesis import run as run_h
        res = run_h(S.hypothesis_for(sid, st), store=False)
        self.assertIn(res["verdict"], {"rejected", "survived", "inconclusive"})
        names = {g["name"] for g in res["gates"]}
        self.assertIn("delayed_entry", names)
        self.assertIn("out_of_sample", names)


class ServerWiringTests(unittest.TestCase):
    def test_local_paths_and_page(self):
        import os
        with patch.dict(os.environ, {'TRADINGAGENTS_SESSION_SECRET': 'x' * 48,
                                     'TRADINGAGENTS_WEB_PASSWORD': 'test-only'}):
            import web.server as server
        for p in ("/api/setups/scan", "/api/setups/resolve", "/api/setups/backfill"):
            self.assertIn(p, server._LOCAL_OR_AUTH_PATHS)
        html = (Path(server.__file__).parent / "static" / "setups.html").read_text()
        self.assertIn('href="/setups"', html)


if __name__ == "__main__":
    unittest.main()
