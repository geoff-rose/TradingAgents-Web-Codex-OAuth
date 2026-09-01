# TradingAgents — full session handoff (2026-08-24 → 2026-08-28)

Written to be read cold by another assistant. Covers everything done in this
session: UI work, bug fixes, data corrections, infrastructure, and research.

---

## 1. The system

FastAPI app at `/opt/tradingagents`, port 7777, unit `tradingagents.service`.
Static HTML in `web/static/` (no framework — plain HTML + inline JS), logic in
`tradingagents/`. Auth is a single shared password (`/etc/tradingagents/env`,
`TRADINGAGENTS_WEB_PASSWORD`) with a signed session cookie; a middleware 401s
`/api/*` and redirects page routes to `/login`.

**Databases**
- `data/swing.db` — app data: `mover_log`, `momentum_trades`, `watchlist`,
  `swing_*`, `gap_study_sessions`, `gap_study_variants`
- `~/.tradingagents/asx_signals.db` — LLM classifier cache: `signals`,
  `ticker_signals`, `llm_calls`
- `/opt/asxbrief/data/asx.db` — **separate collector** (asxbrief service),
  read-only from this app: `announcements`, `universe` (top 500 by market cap),
  `short_positions` (ASIC, 2.27M rows, 2010→), `bars`, `day_shape`

**Pages**: Scanner, Watchlist, ASX (announcement stream), Patterns (`/research`),
Backtest, Swing, Performance, Reports, Sync.

Context: ASX-focused, user in Sydney. yfinance is the price source (~20 min
delayed); IBKR paper gateway exists for order work.

---

## 2. Page-by-page changes

### Scanner (`/scanner`)

**Movers table** — live scan of the top 500 for moves ≥5% (watch) / ≥8% + 3x
volume + news (FULL).
- Added **Prev close**, **Open**, and **Move % vs open** columns. Previously only
  Last and Move % were shown, so the arithmetic wasn't checkable. Both percentages
  are measured from the *previous close*; headers say so explicitly
  (`Move % / vs prev close`, `Gap % / open vs prev close`).
- **All 12 columns now sortable**; default remains the scan's own ranking
  (FULL first, then move+vol, then move size). Missing values sink rather than
  sorting as zero.
- **"+ Watch" button** per row → adds to the watchlist with company, AI score,
  headline and URL captured.
- Action buttons **pinned to the right edge** (`position: sticky`) — the added
  columns pushed them off-screen entirely at 1440px.

**Gap outcomes table** — one row per logged mover-day, "did the move hold past
the open".
- **Calendar** for picking a session; opens on today by default; falls back to the
  most recent session with data (weekends/holidays).
- Today's in-progress rows shown with a `live` badge. Provisional rows are only
  included when a *single date* is selected — the rolling 30-day statistic is
  finalized-only so a half-session can't skew it.
- Added **Setup** (FULL/move+vol/watch, derived from stored fields using the
  scanner's own imported thresholds), **Low**, **Open→High %**, **News**,
  **AI score** columns.
- **Setup filter** (All / FULL / move+vol / watch). Summary tiles recompute from
  the *filtered* rows, not the API's all-rows figures.
- **Horizon toggle**: "Same day" ⇄ "Forward returns" (D+1/3/5/10, MFE, MAE,
  market-relative). Both views are driven by one column spec — three
  hand-maintained lists (header/body/colspan) had caused repeated off-by-one bugs.

Paper trades **moved off this page** to the Watchlist. The trade dialog stays
(the movers' "Paper trade" button needs it) and now confirms with a link.

### Watchlist (`/watchlist`) — new page

- Add from Scanner or the announcement feed, or type a ticker.
- Columns: Ticker, **Company**, Added, Source, Prev close, Open, High, Low, Last,
  Move %, AI, Why (headline), actions.
- **Live SSE price push** — `/api/watchlist/stream`, cells update in place with a
  direction flash, `● live` badge. Prices are still ~20 min delayed: what's live is
  the *delivery*, not the price. Says so in the UI.
- **Trade button** → paper-trade dialog prefilled: ticker, $20,000 (editable),
  entry from the ask where usable else last trade, source `watchlist`, AI score.
  Conviction and note deliberately left blank.
- **Paper trades section** (moved here) with an **All / ASX only / US only**
  filter; summary recomputes per filter.

### ASX announcement stream (`/asx`)

- **"Today only" toggle**, default on. Filters on the *Sydney* session day, not the
  UTC date — a 7:30am pre-open announcement is 21:30 UTC the previous day, so a UTC
  filter would drop exactly the announcements that matter.
- **Company column** added (sortable). `announcements.company` is NULL for every
  row asxbrief has ever written, so names come from `COALESCE(a.company, u.company)`
  via the `universe` join. Out-of-universe tickers show a dash.
- **"+ Watch" button** per company row.
- Fixed grouping: was grouping on the UTC date, splitting every session in two at
  10:00 Sydney (a company announcing pre-open and mid-morning appeared as two rows
  on two dates). Now grouped on the Sydney session.

### Patterns (`/research`)

- Renamed from "Pattern Research" (title + `<h1>`) to **Patterns** to match the nav.
- **Backfill log fixed**: the status box showed 30 identical
  `Failed to open /run/systemd/transient/...` lines. That is *journalctl's own
  message on stdout*, once per requested line, because `systemd-run --collect`
  deletes the transient unit on exit. Filtered, with an explicit message when the
  journal has nothing attributable.

---

## 3. New modules

| module | purpose |
|---|---|
| `watchlist.py` | watchlist CRUD + quotes; `get_quotes()` is the swappable price seam (yfinance today, IBKR later) |
| `quote_hub.py` | one shared SSE poller fanning out to all connected browsers; starts on first subscriber, stops on last |
| `forward_returns.py` | D+1/3/5/10 + market + MFE/MAE on every mover-day; daily timer |
| `symbols.py` | market resolution (AU/US) and company-name lookup |
| `gap_study.py` | the gap/volume backtests (see §6) |
| `yf_lock.py` | process-wide lock; yfinance is not thread-safe |

---

## 4. Bugs found and fixed

These matter more than the features — several were silently corrupting data.

1. **Stale-bar bug (severe).** `scan()` took `df.iloc[-1]` as "today" without
   checking the bar's date. Before yfinance publishes today's bar, that is
   *yesterday* — so the page reported yesterday's move as live (SKC showed +12%,
   its previous session's move), and `log_movers()` wrote it into `mover_log`
   under today's date. **37 of 39 rows** under 2026-08-25 were byte-identical
   copies of the 24th. Fixed: rows carry `bar_date` and are logged under it;
   stale rows are flagged in the UI. 35 duplicates deleted after user approval.
2. **Write-once fields couldn't self-heal.** `prev_close`/`open` were written once
   and never updated, so a row created from a stale bar kept yesterday's prices
   forever while `high`/`low`/`close` updated — a row mixing two sessions. Also
   `high`/`low` used `max()`/`min()` against the stored value, which can only widen
   a range, permanently absorbing a wrong extreme (NXL held a low of 1.635 after
   opening at 1.97). Now every price field refreshes from the bar.
3. **Finalize ran once and missed late arrivals.** `finalize_today()` fired at 16:15
   Sydney, but the scanner keeps logging after the close, so anything first seen at
   16:21 was never finalized and silently dropped out of the statistics. Now
   `finalize_pending()` sweeps every unfinalized past session, stamping each row
   from *its own* date's bar, and the timer runs three passes (16:15 / 20:15 / 09:15).
4. **yfinance thread-safety.** Two concurrent `yf.download` calls raise
   `RuntimeError: dictionary changed size during iteration` — this 500'd the scan
   endpoint ("Scan failed") and then *cached the empty result* for 10 minutes.
   Fixed with `YF_LOCK`, an asyncio lock on the endpoint, and never caching a fetch
   that read zero bars (`n_bars_read`).
5. **US tickers could never mark to market.** `mark_to_market` hardcoded `.AX`, so
   5 of 8 open paper trades (BE, CCJ, OKLO, INTC, UUUU) sat with no price and no
   P&L indefinitely. Market is now resolved once at entry and stored.
6. **`n_finalized` counted all rows**, so a fully provisional day reported itself as
   fully settled.
7. **`sortBy` used the implicit global `event`**, throwing if ever called
   programmatically.
8. **`max-width` on an inline `<a>`** did nothing (doesn't apply to non-replaced
   inline elements), letting one headline stretch a table to 389px.

---

## 5. LLM cost work

Classifier: `asx_signals.py`, provider chosen by `ASX_SIGNALS_PROVIDER`
(`codex_oauth.py` / `grok_oauth.py`, deliberately interface-compatible:
`is_configured` / `ask` / `is_quota_error` / `is_auth_error` + a quota/auth
exception pair). A Gemini module would slot in identically — Gemini CLI personal
OAuth gives **1,000 req/day free** (note: an unpaid *API key* gives fewer, 250/day).

**The dominant cost was redundant re-netting.** `refresh_ticker_signals()` runs
after every `classify_pending` — every 10 minutes, 144×/day — and re-computed the
net score for every ticker unconditionally. One observed run: `classified: 1,
ticker_netting: 15`. Now skipped when the scored fingerprint set is unchanged under
the same model and prompt version → the same run costs **0**. Verified it still
re-nets when a new announcement arrives and when `PROMPT_VERSION` is bumped.

Also added a narrow headline skip-list (Appendix 3G, notification re unquoted
securities, cleansing notice — ~2.6% of calls) with ASX's price-sensitive flag as
an absolute override. Deliberately narrow: substantial-holding and director's
interest notices are *kept* at the user's instruction.

**Usage is now measured, not inferred**: `_ask_counted()` is the single door all
provider calls pass through, logging to `llm_calls` (including failures, which
still consume quota). `/api/asx/signals/usage` + a readout on the ASX page. Row
counts in `signals`/`ticker_signals` are upserts and badly undercount — which is
how thousands of redundant calls stayed invisible.

Steady state after the fix: ~284 classifications + ~200 nets/day.

---

## 6. Research findings

All backtests: asxbrief top 500, **2 years of hourly bars** (yfinance serves ~2y
hourly / ~60d 15m / 7d 1m for ASX). **Survivorship bias is present and not
removable** — the universe is *today's* top 500.

1. **Buying a gap-up at the open loses.** 230k sessions. Gap >3%, buy open, +6%
   target, exit 15:00 → negative in every gap bucket, worse as the gap grows.
   Hit-rate rises monotonically with gap size (1.3% → 24.8%) while return *falls*:
   capped upside, open downside. % of sessions closing positive is 38–42% on gap-ups
   vs 46% with no gap — gapping up makes the rest of the day *worse*.
2. **A stop fixes the failure mode but not the strategy.** Tightening the stop
   improves the long monotonically (−0.774% → −0.027% at −2% on >8% gaps), but only
   reaches break-even by shrinking the stop until the measurement stops being valid
   (see §7).
3. **Shorting the gap is positive but small**: ~+0.46% gross/trade, stable across
   every stop level, 7/9 quarters positive, not outlier-driven (trimming the 5 best
   and worst *raises* it). 94% of names are borrowable per ASIC `short_positions`.
   **The edge inverts with short interest**: +0.999% on lightly-shorted names,
   +0.114% on heavily-shorted — short squeezes. Recent quarters decaying.
4. **Overnight is where the return is.** Buy close → sell next open beats every
   intraday exit: **+0.19%** vs +0.06% at 11:00. Holding to 11:00 gives back 69% of
   the overnight gain; win rate falls from 54.9% to ~50%. (16% of observations show
   exactly 0.0000% — stocks that didn't trade — and must be excluded.) Agrees with
   `overnight.py`'s earlier daily-bar work (~94% of total return accrues overnight).
5. **Volume-conditioned overnight roughly doubles it.** Prior-day volume >6x the
   20-day median → **+0.366%**/night, 8/9 quarters positive, survives trimming the
   50 best and worst. Using only volume *through 15:00* (what a 15:45 scan actually
   has) → **+0.337%**, so it is genuinely tradeable. ~4,150 events in 2y (~8/week
   across 500 names); ~21% don't trade overnight at all.
6. **Announcement-conditioned tests are not yet possible.** asxbrief's
   `announcements` table starts 2026-08-19 (~8 sessions). `event_momentum.py` hit
   the same wall and established volume as the news proxy.
7. **Movers with volume do NOT continue next day** — `event_momentum.py`, 10y:
   "the edge showed up over 3–10 days, never next-day; every volume bucket was
   negative at 1 day." Three independent methods now agree that the session
   immediately following a mover is the bad window.

---

## 7. Methodological traps hit — read before running further studies

- **Look-ahead via the conditioning variable.** A study conditioned on the *gap
  day's own full-session volume* showed +0.775%. That volume is unknown at the
  10:00 entry. Moving the signal to the prior day flipped it to **−0.374%**. Always
  ask what is known *at the entry instant*.
- **Optima at the edge of a swept range are not optima.** Happened twice (tightest
  stop, highest take-profit). Extend the range or state the optimum is unknown.
- **Tight stops stop measuring the strategy.** As the stop tightened, same-bar
  ambiguity (one bar spanning target and stop) rose 22× to 2.2%; at −0.5% the
  "edge" was an artifact of the tie-break. Ties are always resolved as the STOP.
- **Baselines are mandatory for one-sided measures.** Open→high is ≥0 by
  definition, so it looks positive on random days.
- **A monotonic gradient across buckets is the evidence**, not one good cell.
- **Check outlier dependence and time stability** on anything that looks good.
- **Intraday timing**: ASX hourly bars are stamped with their START in Sydney,
  10:00–16:00; the 16:00 bar is the closing auction (O=H=L=C). Only ~43% of daily
  volume has traded by 15:00, so a partial-day volume ratio must be compared with
  prior days' volume *to the same hour*, never scaled by an assumed profile.

---

## 8. Infrastructure / ops

- **IB Gateway** (`ibgateway.service`, IBC, paper account) was failing to log in and
  had **credentials exposed in `ps` output** (IBC passes `--user`/`--pw` as argv)
  and in a world-readable `config.ini`. Now: `/etc/ibkr.env` (600) is the single
  source, a runtime ini is generated per start at `/run/ibkr/config.ini` (tmpfs,
  600), argv is clean, and the API is listening on 4002.
  **The password was exposed and should be rotated.**
- **Timers**: scanner refresh 10min · signals refresh 10min · mover finalize
  16:15/20:15/09:15 Sydney · forward returns 17:00 Sydney · spi200 · swing sync.
- **Bid/ask**: yfinance ASX bid/ask is *crossed* pre-open (07:00–10:00 auction) —
  that is correct market state, not bad data, and matches a broker screen. It
  becomes sane in continuous trading (BHP 67.58/67.59), but is ~20 min delayed, so
  the guard still matters for the first 20 minutes. `_ask_is_usable()` rejects
  crossed/missing/stale asks and the UI says which side the price came from.

---

## 9. Open threads

- **Build the end-of-day >6x-volume scanner** — validated (+0.337% with the signal
  available at 15:45), not yet built. Would run ~15:45 Sydney, list top-500 names
  running >6x their 20-day median volume *to the same hour*, buy in the closing
  auction, sell in the next opening auction.
- **Classifier calibration** needs ~1 month of accumulation (39 scored mover-days
  so far; buckets of n=15/17/7 are far too small). Forward returns are accruing.
- **Reconcile** gap-shorting being positive with the volume-conditioned overnight
  *long* being positive — different windows, worth understanding jointly.
- **Paper-trade log** has an INTC entry at $20.00 against ~$87 that distorts summary
  stats.
- **Watchlist quotes are ASX-only** (`get_quotes` appends `.AX`); US watchlist names
  would show "no quote". The resolver exists if that needs extending.
- **Live IBKR quotes** would need an ASX market-data subscription; the `get_quotes`
  seam and the SSE plumbing are already in place for it.
