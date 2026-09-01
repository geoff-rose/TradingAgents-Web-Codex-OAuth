---
name: asx-dashboard
description: Architecture, file map, and conventions for the ASX trading dashboard pages inside this TradingAgents app — /asx (market snapshot, yield curve, announcement feed, AI Buy/Sell signals), /research (phase-2 pattern-research ticker basket, backfill control), /swing (simulated swing trading: daily heuristic proposals, semi-automatic paper-IBKR bracket orders, P&L tracking), /backtest (strategy library, fill models, and the statistical honesty layer that gates any claimed edge), and /scanner (live movers scan + momentum paper-trade log). Load before adding a panel, data source, strategy, or feature to any of these, or debugging why one isn't updating.
---

# ASX Dashboard

A market dashboard bolted onto the existing TradingAgents FastAPI app (chosen
over a standalone service to reuse its hosting/port/auth — see the project
memory for that decision). Everything lives in this repo except the
announcement/bar data, which comes from a sibling project (asx-collector).
Three pages so far, same conventions throughout (see below): `/asx`,
`/research`, and `/swing`.

## The whole picture

```
asx-collector repo (/root/asx-collector, deployed to /opt/asxbrief)
  -> asxbrief.service polls ASX every 5 min -> /opt/asxbrief/data/asx.db (SQLite)
  -> also holds: universe (top-500 by mkt cap), research_universe (~50-ticker
     basket), bars (IBKR 1-min), short_positions (ASIC)
       |
       v
tradingagents/asx_feed.py          -- read-only reader of announcements/universe
tradingagents/research_universe.py -- read/write for research_universe (see below -- deliberate exception)
tradingagents/markets.py           -- yfinance snapshot + FRED yield curve + SPI200 scrape (own data)
tradingagents/asx_signals.py       -- AI Buy/Sell/Hold + score, cached in ~/.tradingagents/asx_signals.db
tradingagents/grok_oauth.py        -- Grok API via SuperGrok/X OAuth (not an API key -- see below)
tradingagents/swing_db.py          -- swing-trade proposals/positions/P&L, own db (data/swing.db)
tradingagents/swing_signal.py      -- daily ATR-pullback heuristic (yfinance daily OHLC)
tradingagents/swing_ibkr.py        -- paper-IBKR bracket order placement + fill-status sync
tradingagents/swing.py             -- orchestration: propose_daily/approve_trade/reject_trade/sync_all
       |
       v
web/server.py   -- /asx + /api/asx/*, /api/markets/*; /research + /api/research/*; /swing + /api/swing/*
       |
       v
web/static/asx.html, web/static/research.html, web/static/swing.html  -- vanilla
                         JS, no build step, no CDN libs, hand-rolled inline SVG
                         charts (matches the rest of this app's static pages --
                         see performance.html for the same pattern). GitHub-dark
                         palette (--bg:#0d1117 etc, copied from performance.html,
                         NOT home.html's purple palette).
```

## `/research` — the phase-2 ticker basket page (added 2026-08-21)

Lets the user manage the ~50-ticker basket for IBKR bar backfill and
day-shape pattern research: up to 10 self-picked "focus" tickers (add/remove
directly) plus up to 40 "recommended" ones (regenerated on demand, scored by
volatility+liquidity — see `asxbrief-phase2` skill for the exact method and
why volatility is weighted higher). Also shows per-ticker bar-count/date-range
coverage, and controls a background backfill job (see below). The
"Identified Patterns" section is a deliberate placeholder — there's no
clustering analysis yet to show; don't remove the placeholder copy explaining
why, and don't build a fake/empty chart just to fill the space.

**`tradingagents/research_universe.py` writes directly to asxbrief's shared
SQLite file** — the one deliberate exception to `asx_feed.py`'s read-only
pattern. Justified because ticker add/remove is a low-frequency admin action,
not a hot path, and asxbrief has no HTTP API of its own to call instead
(it's a pure CLI collector). Don't extend this write-access pattern to
anything higher-frequency without reconsidering the tradeoff.

**Backfill runs as its own transient systemd unit, not a detached subprocess
and not an HTTP-blocking call.** At current IBKR pacing (1-day chunks, 2s
delay, 2 bar types), even a 90-day/50-ticker backfill is ~3 hours — full
5-year is ~65-70 hours. `start_backfill()` uses
`systemd-run --unit asxbrief-ibkr-backfill --collect ...`; `backfill_status()`
checks `systemctl is-active` and tails `journalctl -u asxbrief-ibkr-backfill`.
The page polls `GET /api/research/backfill/status` every 15s.

**Learned the hard way, 2026-08-21**: the first version used
`subprocess.Popen(..., start_new_session=True)` with a PID file. It died
silently ~10 minutes into a real run, the first time `tradingagents.service`
was restarted for an unrelated reason (an auth fix). `start_new_session=True`
(`setsid`) only detaches a process from its controlling terminal/session —
it does **not** move it to a different systemd cgroup. Restarting a service
kills its entire cgroup, descendants included, regardless of session
leadership. A real separate systemd unit has its own cgroup and genuinely
survives the parent app being restarted — don't revert to the subprocess
version even though it looks simpler; it silently breaks the very first time
someone restarts `tradingagents.service` while a backfill is running, and
nothing in the failure mode looks like a crash (no traceback, no error --
the log just stops).

**If asked to extend this dashboard with a new panel/data source, this file map is
where it goes** — don't invent a new page or a new service unless the feature
is genuinely unrelated to markets/announcements.

## `/swing` — simulated swing trading (added 2026-08-21)

Daily heuristic scan of a user-picked subset of the Focus Tickers (Patterns
page) for a "likely small gain" setup; user reviews and approves each
proposal; approved trades become a real bracket order on the **paper** IBKR
account (see /etc/ibkr.env), tracked through fill to a closed, realized-P&L row.
Fixed $ per trade (default $20,000 AUD), user-editable on the page (added
2026-08-21) — `swing_db.get_trade_dollars()`/`set_trade_dollars()`, backed by
a generic `swing_settings` key/value table; `propose_daily()` reads it fresh
each scan, so a change applies to the next day's proposals, not
retroactively. **Semi-automatic by design**: nothing
reaches IBKR without a per-trade approval click — this was an explicit
choice over full automation, see the project memory.

**Why a simple daily-OHLC heuristic instead of the phase-2 day-shape/
clustering engine**: that engine needs per-ticker intraday bar history to
find recurring patterns, and the user's 6 focus tickers had **zero** bar
history the day this was built (just added, backfill hadn't reached them
yet). Even backfilled, IBKR's own historical 1-min depth for smaller ASX
names tops out around 3.5 months (see asxbrief-phase2 skill) — not enough to
trust a clustered pattern without serious overfitting risk on 5 tickers. This
heuristic has **no proven edge either** — the Swing page's own P&L table is
what's supposed to judge that, the same honest standard the (still unproven)
AI announcement-signal score is held to. Don't oversell this as validated;
it's an MVP, explicitly chosen over waiting for data that doesn't exist yet.

**The heuristic** (`swing_signal.py::evaluate()`): long-only pullback-in-
uptrend on daily OHLC (`yfinance`, `{ticker}.AX`, ~4mo history) — qualifies
if `close > SMA20` (uptrend filter) AND `close` is within 2% of the trailing
10-day low (a pullback), giving entry=last close, target=entry+1.5×ATR14,
stop=entry-1.0×ATR14, only kept if the target implies a 1-6% gain (the "small
gain" band). All constants are named at the top of the module — tune there,
not inline.

**Order mechanics** (`swing_ibkr.py`): a standard 3-leg bracket order (parent
limit buy, take-profit limit sell, stop-loss stop sell, `GTC` — a swing hold
sits for however many days it takes, not just today) via `ib_async`, same
paper Gateway as the phase-2 backfill (`127.0.0.1:4002`) but its own
`clientId=71` so a brief connect-act-disconnect cycle here never collides
with a concurrent backfill session. **Two real gotchas hit and fixed while
building this, both worth knowing before touching this file again:**
1. **`ReadOnlyApi=yes` in `/opt/ibkr/ibc/config.ini`** (set back when this
   Gateway only did read-only bar backfill) silently discards any order an
   API client submits — no exception, no error event, just nothing reaches
   IBKR's book. Flipped to `no` and Gateway restarted 2026-08-21 specifically
   to enable this feature. If a future session finds orders aren't landing
   with no visible error, check this first.
2. **Direct-to-ASX contract routing gets orders discarded by IBKR itself**
   (error 10311/201 — a precautionary API-order-routing restriction). Fixed
   by using `Stock(ticker, "SMART", "AUD", primaryExchange="ASX")` instead of
   `Stock(ticker, "ASX", "AUD")` — SMART routing is IBKR's own standard
   pattern for API stock orders and isn't subject to that block. Confirmed
   live: a real bracket order (parent+target+stop, correctly OCA-linked via
   `parentId`) sat resting and visible via `reqAllOpenOrders()` from a fresh
   connection, then cleanly cancelled via `reqGlobalCancel()`.
3. Also: enabling `ReadOnlyApi=no` made IBKR require the "this is not a
   brokerage account" paper-trading disclaimer to be dismissed before an API
   client can connect at all (read-only connections didn't need this).
   Initially handled with a manual `xdotool` click each Gateway restart (same
   screenshot technique as the original Gateway login debugging, see
   asxbrief-phase2 skill) — **fixed properly 2026-08-21** per the phase-3
   spec's §8.1: `AcceptNonBrokerageAccountWarning=yes` in
   `/opt/ibkr/ibc/config.ini` makes IBC auto-accept the dialog on every
   login/restart, no manual step needed. Also set `AllowBlindTrading=yes`
   at the same time (dismisses the no-market-data-subscription order
   warning, which would otherwise silently block orders the same way).
   Confirmed live: a fresh `ib_async` connection right after a Gateway
   restart succeeds with zero manual intervention, dialog never appears.
   `AcceptIncomingConnectionAction=accept` was left as-is rather than
   switched to `reject`+allowlist (the spec's alternative suggestion) since
   `TrustedTwsApiClientIPs=127.0.0.1` already restricts this the same way.

**Sync, not a monitor loop**: because it's a real bracket order, IBKR's own
order book handles "wait until target or stop, however many days that
takes" — nothing here needs to poll price and decide when to exit.
`asx-swing-sync.timer` (every 15 min, all day) just asks IBKR for each
tracked order id's status and reconciles: parent filled → `open`
(`entered_at`/`entry_fill_price` set); a bracket child filled → `closed`
(`exit_reason` = `target`/`stop`, `pnl` computed from fill price × shares).

**Lifecycle**: `proposed` →(user approves)→ `submitted` →(parent fills)→
`open` →(target or stop fills)→ `closed`. Or `proposed` →(user rejects)→
`rejected` at any point before approval. `swing_db.has_active_trade()` blocks
a new proposal for a ticker that already has a non-terminal trade, so
`propose_daily()` never pyramids into the same name.

**Systemd**: `asx-swing-propose.timer` (Mon-Fri 08:20 Australia/Sydney, ~1h40
before the ASX open, after the 08:05 SPI200 refresh) hits `POST
/api/swing/propose`. `asx-swing-sync.timer` (`*:0/15`, every 15 min
around the clock) hits `POST /api/swing/sync`. Both localhost-only
(`_LOCAL_OR_AUTH_PATHS` in server.py), same pattern as `asx-signals-refresh`
and `asx-spi200-refresh`. The approve/reject endpoints are **not** in that
set — they go through normal cookie-session auth, since a human is meant to
be the one clicking them.

**Future**: a Momentum/News page is planned as a *separate*, later page —
don't fold same-day news-driven trading into this one; Swing is specifically
the multi-day heuristic hold.

## `tradingagents/backtest.py` — heuristic backtest (added 2026-08-21, phase 3)

Built per `/root/asxbrief-phase3-spec-v2.md` §3 — that document's reasoning
is the source of truth for *why* a backtest exists alongside forward paper
testing (short version: at ~40-60 trades/year across 5 tickers, forward
testing alone needs ~5 years to separate real edge from noise; a backtest
over years of daily history gets that sample size immediately, at the cost
of needing an honest fill model since it can't observe real depth). Run via
`.venv/bin/python -m tradingagents.backtest`.

**Point-in-time signal generation, not a call to `swing_signal.evaluate()`**:
`generate_signals()` re-implements the exact same rule (imports the same
constants from `swing_signal.py` so there's one source of truth for the
thresholds) as a day-by-day walk-forward scan, so every day's decision only
sees data available up to and including that day's close — `evaluate()`
itself always fetches live/latest data and can't be reused directly for
historical replay. No-pyramiding is enforced the same way
`swing_db.has_active_trade` does live.

**Two fill models, always reported side by side**: `naive` (fills on mere
touch — the number to distrust) and `realistic` (requires trading through
the level by one tick, approximating worst-case queue position — still an
approximation given only daily OHLC; minimum-volume-through-level is NOT
implemented, daily bars don't carry volume-at-price). If naive and realistic
disagree sharply, that gap is simulation artefact, not edge.

**ASX tick table verified live** (`TICK_TABLE` in this file) against ASX's
own price-steps page, not assumed — the phase-3 spec explicitly flagged this
as disputed and decisive. Confirmed: below 10c → 0.1c tick; 10c up to and
including $2.00 → 0.5c tick; above $2.00 → 1c tick.

**Same-bar target/stop ambiguity**: when a day's range contains both target
and stop, daily OHLC can't tell which filled first. Resolved via real
intraday bars (`asxbrief`'s `bars` table) when available, otherwise the
conservative default (assume stop) — never silently assumes target. As of
2026-08-21 the current focus tickers have no intraday backfill started at
all yet, so this will read at or near 100% conservative-default until that
changes.

**First real run's result — a genuine finding, not a bug** (verified by hand
against individual CBA trades before trusting it): across the 6 current
focus tickers over ~10 years, the heuristic only ever fires on **CBA (30
signals) and PNV (9 signals) — zero for BRN/INR/MEI/TTT**. Traced this to a
real, non-bug cause: those four are sub-25c speculative names whose
ATR14-implied target (`1.5×ATR`) almost always exceeds the 1-6% "small gain"
band on the high side (checked BRN by hand: 11 uptrend-pullback days in 10
years, every one implying an 8-19% target, never 1-6%) — the gain band
structurally excludes genuinely volatile penny stocks from this heuristic
entirely, regardless of history length. On the ~37 trades that did fire
(concentrated in just 2 names): **win rate ~24-27%, average return per trade
~-0.8% to -0.9%, and the randomised-entry permutation test placed the actual
result at roughly the 2nd percentile of the null** — i.e. *worse* than a
random entry held for the same length of time, not merely "no edge." Don't
soften this in any summary of it; it's the honest result of the first real
test the heuristic has been given, exactly the kind of check the project's
own AI-announcement-score has still never had this rigorously.

**Deferred from this first pass** (flagged in-code, not silently skipped):
walk-forward with multiple rolling splits (only a single held-out-12-months
split exists so far); multiple-comparisons correction (White's Reality
Check / Hansen's SPA); Monte Carlo for risk (drawdown distribution, cost
sensitivity, cross-sectional bootstrap over tickers) — spec §3.4, explicitly
for risk/sizing, not for testing whether edge exists. Also: `MAX_HOLD_DAYS`
(60 trading days) is a backtest-only modelling cap: the **live** system has
no equivalent — a submitted bracket order sits GTC indefinitely and
`swing_db.has_active_trade` blocks new signals for that ticker the whole
time, meaning a ticker could get permanently stuck waiting for a price that
never returns. That's a real, separate operational gap this backtest
surfaced, not something it fixes.

**CBA was removed from the swing universe 2026-08-21** ("wasn't meant to be
in this" -- the user's own words). `backtest.py`'s `__main__` reads
`swing_db.enabled_tickers()`, not `research_universe`'s full focus list --
don't revert that, it's what caused CBA to appear in the first backtest run
by accident. Live swing universe as of 2026-08-21: BRN, INR, MEI, PNV, TTT.

## `tradingagents/range_model.py` — fitted range model (phase-3 §6, added 2026-08-21)

Replaces `swing_signal.py`'s heuristic as the **live** proposal source (the
heuristic is kept for reference/backtest-baseline comparison only -- see
`swing.py`'s `propose_daily()`, no longer called by the live timer). Built
after the backtest showed the heuristic loses to random entries; per the
spec's own gate ("a fitted model that can't beat the baseline out of sample
should be discarded"), this was validated in `backtest.py`-style out-of-
sample simulation *before* being wired into live order placement.

**What it predicts**: pooled quantile regression (`statsmodels.QuantReg`,
full pooling across the universe -- see below for why) on next-day low and
next-day high, each as a % of that day's close. Features are all causal
(computed from data up to and including the day's own close, no lookahead):
`ret_std_5/10/20` (realised vol at three windows), `atr_ratio_short_long`
(14d vs 50d ATR -- vol regime), `range_vs_avg20`, `close_pos_in_range`
(where in the day's own range the close landed), `volume_ratio_20`,
`gap_propensity` (this ticker's own expanding-window average gap size).
**Full pooling (shared coefficients, no per-ticker terms) is the
deliberate starting point**, not a shortcut -- spec §2.6: "per-stock
coefficients only once a stock has earned them." ~10-12k pooled rows across
5 tickers over ~10 years makes full pooling the correctly-shrunk choice for
now; a per-ticker refinement is a legitimate future step once there's
evidence a specific stock's dynamics genuinely diverge, not something to add
pre-emptively.

**Two real bugs found and fixed while building this — both worth reading
before touching `simulate_daily_refresh()` again:**
1. **Target anchored to the wrong reference price.** First version computed
   the daily-refreshed target as `today's_close × (1 + predicted_high_pct)`
   -- using *today's* close, which is what the model natively predicts
   relative to. But on a losing trade where price keeps drifting down, that
   "target" drifts down with it and can fall below the actual entry price,
   so a modest bounce off a falling price got logged as a profitable target
   hit when it was actually a loss. Caught by hand-checking individual
   trades (e.g. entry=$1.090, "target" exit at $1.070 -- a loss mislabeled
   as a win). Fixed: `predict_pct()` returns the raw % move; the exit-refresh
   loop applies it to the **fixed entry price**, never to the day's own
   close. `predict()` (used for a fresh, flat-position entry quote) correctly
   still anchors to that day's close -- that's the right frame for a *new*
   order, just the wrong one for judging an *existing* position's profit.
2. **First evaluation used naive touch-fill with zero costs**, unlike the
   heuristic's backtest (which reported naive AND realistic, with cost
   sweeps). Produced a suspiciously good result (extremely high trade
   frequency, +492% held-out total return) that didn't survive fixing this:
   `simulate_daily_refresh(mode=...)` now supports 'naive'/'realistic' the
   same way `backtest.py` does, and `evaluate_daily_refresh()` reports a
   cost-adjusted net return alongside gross. **Always check both bugs are
   still fixed if this file is ever refactored** -- a "too good" result here
   is the same red flag it was for the heuristic.

**Calibration is checked, not assumed**: `check_calibration()` verifies the
q-th quantile prediction is exceeded by the actual outcome roughly (1-q) of
the time, out of sample. First fit: well-calibrated across the board (e.g.
`low_q0.1` expected 0.10, actual 0.101). Good calibration alone does NOT
imply a profitable trading rule -- that's exactly what the fill-model/cost
bugs above demonstrated; calibration and P&L must both be checked
independently.

**First sweep (held-out ~12 months, realistic fills)**: shallow/frequent
settings (entry_q≈0.3-0.5, near-symmetric target) are net-negative after
costs -- essentially scalping noise. Deep, selective pullback entries do
much better on raw returns: `entry_q=0.1` (only the bottom ~10% of modelled
next-day lows) with `target_q=0.7` gave 40 trades, 70% win rate, +43.2% gross
/ +23.2% net (at an assumed 0.2%-side spread).

**Critical correction, 2026-08-22 -- initially reported this as "beats the
heuristic," which is true but incomplete and was called out correctly by the
user**: `evaluate_daily_refresh()` computed gross/net returns but did NOT run
the result through `backtest.py`'s own `permutation_test()` -- the exact
randomised-entry null the heuristic itself was judged against. The user
caught this directly: "you said the heuristic lost to random entries. You
didn't say the range model beat them... that's more fundamental than the
multiple-comparisons issue." They were right. Fixed by adding
`_to_trade_results()` (adapts range_model's trade dicts to `TradeResult` so
the SAME permutation machinery is reused, not reimplemented) and wiring it
into every `evaluate_daily_refresh()` call.

**Result: nothing in the tested grid clears the bar.** Every (entry_q,
target_q) combination tried -- including (0.1, 0.7) -- lands at the 12th-60th
percentile of the randomised-entry null (need >95th to claim real timing
skill; ≤5th would itself be a distinct, informative "worse than random"
finding, per the same two-tailed framing used for the heuristic). **The
honest conclusion is materially different from "beats the heuristic
baseline"**: (0.1, 0.7)'s positive gross/net return is more plausibly just
capturing these tickers' general upward drift during the held-out window
(deep pullbacks happening to precede a broadly rising period) than genuine
entry/exit timing skill. The heuristic's failure was a real, demonstrated
anti-pattern (buying into actual breakdowns, confirmed by hand on individual
trades); the range model's improvement over it is "less actively
counterproductive," not a demonstrated edge. **Don't present (0.1, 0.7), or
any config, as beating the null until one actually does** -- the
multiple-comparisons caveat from the first sweep still applies on top of
this, not instead of it.

This doesn't mean discard the model -- it means the honest status is
"doesn't yet show demonstrated timing edge," which is exactly what forward
paper testing is for gathering more evidence on. Any future config change
(different entry_q/target_q, new features, per-stock refinement) MUST be
checked against this same permutation null before being reported as an
improvement -- a positive-looking backtest number alone is not sufficient
evidence any more, for this model or the next one.

**Three follow-up checks, 2026-08-22 (user asked all three unprompted after
the correction above -- good instinct to build a habit of asking):**

1. **Bracket resolution audit.** Added `ambiguous_same_bar`/
   `ambiguous_resolved_by` fields to `simulate_daily_refresh()`'s trade
   dicts (previously computed internally but discarded) so this is
   checkable, not just assumed. Result at (0.1, 0.7): **0% of the 40 trades
   had same-bar target+stop overlap** -- confirmed genuine, not a broken
   check, by hand-tracing individual PNV trades day-by-day (the
   hit_target/hit_stop booleans evaluate correctly and independently; the
   target-stop spread at this config is wide enough relative to typical
   daily ranges that simultaneous breach essentially never occurred in this
   sample). The 70% win rate is not inflated by an optimistic same-bar
   assumption here, because the ambiguity never arose. Re-check this if
   entry_q/target_q move closer together (a tighter target-stop spread
   makes same-bar overlap more likely) or if intraday backfill for these
   tickers ever starts (would let real bars resolve ambiguity instead of
   defaulting to 'stop').
2. **Fill-rate diagnostic.** `simulate_daily_refresh()` now returns
   `(trades, {quote_attempts, fills})` -- every caller updated
   (`evaluate_daily_refresh()` surfaces per-ticker `fill_diagnostics`).
   Result: **1.3-6.2% fill rate at entry_q=0.1**, dropping to 0.4-4.1% at
   entry_q=0.05 -- well below the raw ~10%/5% quantile calibration, which is
   the expected and desired effect of the realistic (through-by-a-tick)
   fill requirement actually biting on top of an already-rare deep-pullback
   quote. Confirms the harsher fill model isn't being silently bypassed.
3. **Plateau vs. peak -- a full 5x5 grid** (entry_q ∈ {0.05,0.1,0.15,0.2,0.25}
   × target_q ∈ {0.5,0.6,0.7,0.8,0.9}), run via `N_PERMUTATIONS=2000` for
   speed (still consistent with the n=10000 single-cell result at (0.1,
   0.7): 59.7th percentile both ways). **Two different shapes, opposite
   implications:**
   - Along **target_q** at fixed entry_q=0.05: 69.3 → 80.0 → **88.9** → 82.7
     → 67.2 -- a genuine hump centered at target_q≈0.7, flanked by decent
     neighbours on both sides. This is what a real plateau looks like; the
     center (0.7) is defensible, not cherry-picked.
   - Along **entry_q** at fixed target_q=0.7: **88.9** (0.05) → 59.7 (0.1) →
     36.4 (0.15) → 20.9 (0.2) → 9.3 (0.25) -- monotonically *improving* as
     entry_q shrinks toward the smallest value tested, while trade count
     *shrinks* alongside it (23-27 trades at 0.05 vs 88-112 at 0.25). That's
     the signature of an edge effect / finite-sample artefact, not a real
     optimum -- smaller, more extreme cutoffs produce noisier estimates that
     can look better purely by chance. **Do not chase entry_q below 0.05
     chasing a higher percentile; the pattern would very likely continue for
     the wrong reason.**
   - **Net recommendation at the time**: `entry_q≈0.1, target_q≈0.6-0.7` (the
     plateau's center in the trustworthy dimension) over the grid's literal
     maximum at (0.05, 0.7) -- see the definitive follow-up below, which
     closed this question completely rather than leaving it at "neither
     clears 95."

**Definitive close-out, 2026-08-22 (phase-4 spec `/root/asxbrief-phase4-backtest-page.md`
§12.1/§12.6, and `bootstrap_confidence_interval()`/`selection_permutation_test()`
added to `backtest.py` to do it):**

- **95% CI on (0.1, 0.7)'s mean return: [-0.96%, +3.02%] -- crosses zero.**
  Can't even be confident the true mean is positive, independent of the
  timing-vs-drift question already settled above.
- **95% CI on the grid's actual best cell (0.05, 0.7): [+0.31%, +4.76%] --
  entirely positive**, which looked like it might be a real finding in
  isolation.
- **But `selection_permutation_test()` (White's Reality Check in substance --
  reuses `_draw_null_avg()`, no model refitting needed, cheap) is what
  actually answers the question that matters: is "the best cell in a 25-cell
  grid search" itself distinguishable from what pure noise would produce?**
  Result: **the real best (+2.74%) sits at the 3.4th percentile of the
  best-of-grid-under-null distribution, whose median is +5.02% -- HIGHER
  than what we actually found.** A blind grid search over pure noise
  typically finds a better-looking "best cell" than this one did. This is
  not "inconclusive" -- it's a clear demonstration that the entire
  grid-search-and-pick-the-best procedure produced a result indistinguishable
  from (if anything, slightly worse than) chance.
- **This is the headline finding, supersedes every earlier "promising
  region"/"defensible plateau" framing in this section.** No config from this
  grid -- not (0.1, 0.7), not the grid's actual maximum, not the plateau's
  center -- has demonstrated real timing edge once the selection procedure
  itself is corrected for. Report it exactly this bluntly if asked to
  summarize; this is a stronger, more definitive negative result than "still
  needs more data," and should reset expectations for what forward paper
  testing is actually validating (data collection towards a future, larger
  universe/re-fit, not confirmation of a promising strategy).
- `bootstrap_confidence_interval()` and `selection_permutation_test()` are
  now general-purpose additions to `backtest.py` -- reusable for the phase-4
  `/backtest` page's strategy library, not one-off scripts. Any future grid
  search over this or any other strategy MUST run through
  `selection_permutation_test()` before its best cell is reported as a
  finding -- a single cell's own permutation percentile is not sufficient
  once more than one configuration was tried.

**Three more checks, 2026-08-22 (user again asked exactly the right
questions unprompted -- before fully accepting the 3.4th-percentile
finding above):**

1. **Plumbing check -- does the null path apply the same costs/fill model
   as the real run?** Traced it: no cost asymmetry (both real
   `gross_return_pct` and the null's close-to-close draw are pure,
   cost-free price-level returns). But a real, different issue was found:
   `_draw_null_avg()` sampled from a ticker's FULL fetched history (e.g. 10
   years) while every real trade came only from the ~12-month held-out
   window -- not apples-to-apples. Fixed (`min_date` param on
   `permutation_test()`/`selection_permutation_test()`, auto-derived from
   the real trades' earliest `entry_fill_date` if not given, so every
   existing caller gets the fix for free). Effect: (0.1, 0.7)'s percentile
   moved from 59.7 to 66.4 -- the held-out window had *less* free drift
   available by chance than the full decade, so the fix nudges the real
   result slightly better, not worse. Conclusion unchanged.
2. **Mirror-image test -- enter on strength instead of weakness.** Added
   `direction` param to `simulate_daily_refresh()` ('mean_reversion'
   default, unchanged; 'momentum' entries use the HIGH quantile model at
   `1 - entry_q` instead of the LOW quantile model, with the fill direction
   flipped to require price trading UP through the entry -- a breakout
   trigger, not a dip). Ran the identical 25-cell grid. **Result: best
   momentum cell (entry_q=0.2, target_q=0.9) averages +1.73% (90 trades, CI
   [-0.06%, +3.60%] -- barely crosses zero) but its
   `selection_permutation_test()` percentile is 16.8 -- squarely
   "indistinguishable from noise," same conclusion as mean-reversion, just
   not as extreme. This does NOT redirect the finding toward "this universe
   is momentum-natured and we had the direction backwards" (that would have
   needed something near the 95th+ percentile). It confirms the *other*
   hypothesis instead: **daily-horizon timing is a dead end in both
   directions on this universe**, not a sign-flip miss.
3. **Variance ratio, Var(5d)/(5×Var(1d))**, phase-3 §4's own screener
   metric (<1 = mean-reverting). All 5 tickers: **BRN 0.985, INR 0.887, MEI
   1.089, PNV 0.926, TTT 0.933 over the full 10y** -- and even more clearly
   mean-reverting specifically over the held-out window (0.627-0.862 across
   all five). **The screener would NOT have excluded these tickers** --
   they're exactly the right character for a mean-reversion strategy. The
   "no demonstrated edge" finding is therefore not a stock-selection
   failure; it's a real negative result on well-suited names.

**Debugging note worth keeping**: while chasing this down, a `Monitor`
running `... | grep -v IterationLimitWarning | grep -v "warnings.warn"`
appeared to hang indefinitely on the (0.2, 0.9)-shaped cell across three
separate attempts (5, 10, then 15-minute budgets, all exhausted with zero
output past a certain point). Root cause: `grep` fully buffers its own
stdout when writing to a pipe (not a TTY), and interleaved stderr warnings
from statsmodels apparently prevented the buffer from flushing at a point
that looked like "stuck." The actual computation was never slow -- run
without the grep filter and with `python -u`, the full 25-cell grid
completed in 28 seconds. **Never pipe a long-running Python script's output
through `grep` without `--line-buffered` when watching it via Monitor** --
redirect to a file and read it, or use `--line-buffered`, or this exact
false "hang" will recur.

**The user has a cheaper trading platform than IBKR and said not to over-
index on IBKR's own cost assumptions** (2026-08-21) -- report gross returns
prominently alongside any cost-adjusted figure, don't let one broker's fee
assumption gate a go/no-go decision.

`entry_q`/`target_q` are user-editable on the Swing page (`swing_db
get_range_model_quantiles()`/`set_range_model_quantiles()`, defaults 0.1/0.7)
-- the user explicitly expects to keep refining these, not treat them as
final.

## Live daily-refresh order mechanic (wired in 2026-08-21)

The user's explicit instruction: "change order mechanic to daily refresh
otherwise its pointless" (in response to being told the fitted model's
whole value is a fresh daily min/max estimate, which pairs naturally with a
daily-refreshed order rather than the heuristic's long-sitting GTC bracket).

**New `swing_trades.strategy` column** (`'heuristic'` vs `'range_model'`,
migration in `swing_db.py`) lets both lifecycles coexist in history. New
lifecycle for `'range_model'` trades, different from the heuristic's:

```
proposed --(approve)--> submitted --(entry fills same day)--> open --(target or stop fills)--> closed
                              |
                              +--(entry expires unfilled, TIF=DAY)--> expired
```

- **Entry is a single DAY limit order** (`swing_ibkr.place_day_entry()`),
  not part of a bracket -- it expires automatically at the exchange if
  unfilled, which is exactly the semantics a daily-refreshed entry needs (an
  unfilled quote shouldn't carry over to tomorrow at a stale price).
  `sync_all()` transitions `submitted`→`expired` when IBKR reports the entry
  order `Cancelled` (not filled by end of day) -- tomorrow's proposal is a
  fresh one, not a retry of the same row.
- **Once filled, a standalone GTC stop is placed for the first time**
  (`swing_ibkr.place_stop()`, called from `sync_all()` at the moment of
  fill) using `stop_price` computed and stored on the row *at proposal
  time* (`entry_price - STOP_ATR_MULT × atr_at_entry`, same method
  `backtest.py` uses) -- fixed for the life of the trade. Risk control
  should never be a moving target, only the take-profit side is refreshed.
- **The target is refreshed daily** by `swing.py::refresh_open_targets()`:
  for every `open` range_model trade whose `target_order_date` isn't today,
  cancel the existing target day-order and place a fresh one
  (`swing_ibkr.refresh_target_order()`, one connection covers both the
  cancel and the new placement) at today's freshly-modelled level --
  **anchored to the fixed entry price**, not today's close (see bug #1
  above; the same anchoring mistake in live code would have the same
  consequence it did in the backtest).
- Both the daily proposal pass (`swing.propose_daily_range_model()`) and the
  daily target refresh (`refresh_open_targets()`) are called together from
  `POST /api/swing/propose` (still hit by `asx-swing-propose.timer`, no
  timer/systemd changes needed -- only what the endpoint does server-side
  changed). `POST /api/swing/sync` (still `asx-swing-sync.timer`, every 15
  min) only polls IBKR order status and never touches the model -- cheap to
  run frequently.
- The old heuristic path is reachable at `POST
  /api/swing/propose-legacy-heuristic` for reference/testing, not called by
  any timer.

**Verified live end-to-end 2026-08-21** (test order, then cleaned up): a
real DAY LMT BUY order reached IBKR's book (`tif='DAY'`, correct price,
correct quantity), `sync_all()` correctly left it as `submitted` (not
falsely progressed) while unfilled, and `swing_ibkr.cancel_order()` cleanly
removed it. Confirmed via the same fresh-connection-lookup discipline used
for the original bracket-order verification.

## `/backtest` — strategy testing with a statistical honesty layer (phase 4, added 2026-08-22)

Built from `/root/asxbrief-phase4-backtest-page.md`. The page exists to test
arbitrary strategies over arbitrary tickers **and to make a small-sample
result impossible to mistake for a real one** — spec §1's framing: "an equity
curve from 25 trades looks exactly as convincing as one from 2,500."

```
tradingagents/strategies.py       -- named strategy library + STRATEGIES registry + fetch_events()
tradingagents/backtest_report.py  -- run_and_report() / run_and_report_range_model(): the honesty layer
web/server.py                     -- GET /backtest, GET /api/backtest/strategies, POST /api/backtest/run
web/static/backtest.html          -- Configuration / Trade Count & Statistics / Equity Curve /
                                     Sub-Periods / Trade List, plus a banners section
```

### The strategy interface — and why `events` is in it from day one

Every strategy implements
`generate_signals(ticker, df, params, events=None) -> list[Signal]` and reuses
`backtest.py`'s `Signal`/`TradeResult`/`simulate_trade()` **unchanged**.
`simulate_trade()` already handles fill/exit/tick-rounding/same-bar-ambiguity/
costs generically for any (entry, target, stop) proposal, so **adding a
strategy must never require touching the fill model**. If a change seems to,
that's a signal the strategy is being modelled wrong, not that the fill model
needs a special case.

`events` is an optional `pd.DataFrame` of `{event_date, event_type, detail}`
rows, wired to real announcement data via `fetch_events()` (the phase-1
collector's db — not a stub). It was added at the user's explicit design
instruction ("make sure the strategy interface can express event-driven
strategies, not just price-pattern ones... retrofitting it later would be
painful"). **Don't build event-driven strategies as a second parallel
system** — that's the whole point of the seam.

Registered strategies (`STRATEGIES` dict): `pullback_in_uptrend` (the legacy
live heuristic, kept for parity), `sma_crossover`, `rsi_mean_reversion`,
`n_day_breakout`, `announcement_reaction` (the only `event_driven=True` one),
plus a special-cased `range_model` key handled by
`run_and_report_range_model()` — the range model's daily-refresh mechanic
(a fresh entry/target quoted every day) genuinely doesn't fit the one-static-
`Signal`-per-setup shape, so it gets its own entry point rather than being
forced into the registry.

No-pyramiding is enforced identically in every strategy via the
`blocked_until` pattern (no new signal for a ticker until `MAX_HOLD_DAYS`
elapse), matching how live `swing_db.has_active_trade` blocks overlapping
proposals. Keep that pattern when adding a strategy.

### What `run_and_report()` returns, and which gates are live

Every run reports: trade count + an `insufficient_sample` banner below 30
trades, **both** fill models (realistic and naive touch-fill, side by side —
the gap between them was a real bug source, see the range-model section
above), bootstrap CI, sub-period breakdown, max drawdown, buy-and-hold return
and equity curve, time-in-market and an exposure-adjusted return, the
randomised-entry permutation test, and the full trade list.

`passes_both_gates` requires **both** ≥95th percentile on the randomised-entry
null **and** beating buy-and-hold on an exposure-adjusted basis (spec §6.3b) —
the null alone can't catch "the strategy just captured drift," which is
exactly the failure mode that made the range model look good in its first
evaluation.

**Sub-periods rather than recency weighting** (spec §2), deliberately: a decay
of 8%/6%/2%/-3% blends to +1.8% under any weighting scheme and the actual
finding — the decay — disappears. `sub_period_stats()` buckets trades by how
long before the dataset's end they entered (0-6m/6-12m/1-2y/2-3y/3y+).

### Spec sections NOT yet built (don't assume these are covered)

- **§6.3 selection permutation (White's Reality Check)** — `backtest.py`
  has `selection_permutation_test()` and it produced the project's most
  important negative result (see the range-model section), but it is **not
  wired into `run_and_report()` or the page**. Any parameter sweep driven
  from this page therefore currently has no multiple-comparisons gate on it.
  This is the single most important gap, because the page makes sweeping easy.
- **§6.4 random-basket control**, **§6.5 walk-forward** — not built.
- **§8 promote-to-forward-experiment**, **§9 meta-tracker**, **§10 schema** —
  not built. The meta-tracker (logging every backtest run so the hit rate of
  the whole research process is visible) is called "the most valuable output
  here" by the spec; it doesn't exist yet.
- **§13 regime work / §14 universe expansion** — see the range diagnostics
  section below and the project memory.

## `tradingagents/range_diagnostics.py` — the §13.1 width-vs-centre diagnosis (added 2026-08-22)

Phase-4 spec §13 lays out an order — **diagnose → expand the universe →
regime classifier → new regressors** — and is explicit that adding features
to a ~40-trade sample is dangerous. This module is the diagnose step. Run it
with `uv run python -m tradingagents.range_diagnostics`; `diagnose(tickers)`
returns the same thing as a dict.

**The question**: `range_model.py` predicts next-day low and high. Split its
out-of-sample skill into the *width* of tomorrow's range and its *centre*.
The spec's hypothesis was that width is forecastable (volatility clusters)
and centre is not, which would fully explain the null trading results.

### Three methodology points that changed the answer

1. **Level calibration is mandatory here, not cosmetic.** The model's native
   width prediction is `median(next_high) - median(next_low)`, biased *low*
   by construction — next-day highs are right-skewed, lows left-skewed, so
   each median sits nearer zero than its mean. Uncalibrated it predicted 4.74%
   mean width against a realised 6.62% and *lost to a constant*, which says
   nothing about conditional information. Every predictor (model and
   persistence baseline alike) now gets one additive offset fitted on the
   **training split only** — `median(actual_train) - median(pred_train)` —
   so the MAE comparison measures conditional information and nothing else.
   Don't remove this and read the raw numbers as a finding.
2. **Persistence is a worse baseline than a constant on this universe**
   (MAE 2.649 vs 2.375). On names whose whole day spans 2-3 ticks, one day's
   range is mostly quantisation noise. So "+0.159 skill vs persistence" is the
   flattering number and **"+0.062 vs constant" is the honest headline** —
   lead with the constant.
3. **The shuffle null is a low bar, by construction.** Permuting predictions
   across held-out rows makes the model *worse than a constant* (null median
   -0.068), so anything with a trace of signal clears the 95th percentile.
   Treat "clears the shuffle null" as "the pairing is not accidental," never
   as "this is tradeable" — those are different claims, and conflating them
   is the mistake this project already made once with the range model.

### The result (held-out year to 2026-08-20, 1,275 rows)

Both targets carry a little genuine conditional information — width skill
+0.062 vs constant (95% CI [+0.039, +0.086]), centre +0.035
([+0.019, +0.050]), both at the 100th percentile of the shuffle null; centre
direction accuracy 57.2% against a 50.7% majority base rate. So the spec's
"centre is pure noise" hypothesis is *not* literally right.

**But `economic_significance()` settles it, and this is the number to quote:**

| ticker | med close | 1 tick | width gain | centre gain | day range |
|---|---|---|---|---|---|
| BRN | $0.160 | 3.12% | 0.14 ticks | 0.03 ticks | **2.0 ticks** |
| INR | $0.145 | 3.45% | 0.12 ticks | 0.03 ticks | **2.3 ticks** |
| MEI | $0.180 | 2.78% | 0.15 ticks | 0.03 ticks | **2.6 ticks** |
| PNV | $1.055 | 0.47% | 0.89 ticks | 0.19 ticks | 10.6 ticks |
| TTT | $0.230 | 2.17% | 0.19 ticks | 0.04 ticks | **3.0 ticks** |

Every forecast gain on every ticker is a *fraction of one tick*. Statistical
and economic significance diverge violently here and only the second one
decides anything.

**The structural finding matters more than either skill score**: four of the
five names average **2.0-3.0 ticks of range for an entire day**. Buying a low
band and selling a high band inside a 2-tick day is not a hard forecasting
problem, it is arithmetically impossible once a spread is crossed. PNV
($1.06, 10.6-tick days) is the only name where range trading is even
structurally expressible — and it is the one the phase-3 heuristic backtest
found 0-for-7 on.

### What this redirects

**This is a universe-composition problem before it is a modelling problem.**
It re-orders the spec's own sequence: §14/§16 (expand to 30 tickers, and
§16.1's "do not make it 30 speculatives") comes **before** §13.2's regime
classifier, not after it. A regime classifier fitted on 2-tick days would be
measuring tick noise no matter how well specified it is. Note this also
independently vindicates §16.1 — the current universe is exactly the
"30 speculatives" failure mode, at n=5.

Do not read this as "the range model is broken." It forecasts about as well
as such a model reasonably can; it is pointed at instruments whose price grid
is too coarse for the answer to be usable.

### §13.1 re-run on the expanded universe (2026-08-22) — the hypothesis now confirms cleanly

Re-running `range_diagnostics.diagnose()` on the 26 tradeable names (52,006
training rows, 6,630 held-out, vs 10,790/1,275 before) gives the sharpest
result this project has produced, and it is exactly what phase-4 §13.1
predicted:

| | old 5-ticker universe | new 26-ticker universe |
|---|---|---|
| width skill vs constant | +0.062 [+0.039, +0.086] | **+0.198 [+0.189, +0.207]** |
| width Spearman | +0.352 | **+0.569** |
| width gain in ticks | 0.15 | **3.98 — economically meaningful** |
| centre skill vs zero | +0.035 [+0.019, +0.050] | **+0.0002 [−0.0023, +0.0026] — crosses zero** |
| centre direction accuracy | 57.2% (base 50.7%) | **52.1% (base 51.0%)** |
| centre gain in ticks | 0.03 | 0.01 |
| min daily range | **2.0 ticks** | 10.6 ticks |

**Width is forecastable and now usefully so; the centre is not forecastable at
all.** That is the spec's §13.1 hypothesis, confirmed.

**It also reinterprets the old result.** The apparent centre "skill" of +0.035
on the microcaps was an artefact of tick quantisation — when a whole day spans
2-3 price levels, the midpoint of that day is largely determined by the same
coarse grid the model was fitting, which manufactures correlation that has
nothing to do with forecasting. On a proper universe it vanishes entirely.
Don't cite the old +0.035 centre figure as evidence of anything.

**This is the "sized correctly, no timing edge" diagnosis in its pure form**,
and it is exactly the condition §13.2's reframe was written for: the model can
size a band and has no idea where the centre will go. Regime classification
(is today mean-reverting or trending?) is now the right next step, and it is
now worth doing — on 30 tickers with 10+ ticks of room, a regime classifier is
measuring behaviour rather than tick noise.

**A caution the numbers here demonstrate directly**: centre skill of +0.0002
still "clears the 95th percentile" of the shuffle null, because permuting
predictions scores *worse* than a constant, so the null's whole distribution
sits below zero. The shuffle null answers "is the pairing accidental," never
"is this tradeable." Read the CI and the tick conversion. The verdict block in
`print_diagnosis()` is data-driven for this reason — it no longer hardcodes
the original finding.

## `tradingagents/screener.py` — universe expansion to 30 (phase-4 §14/§16, added 2026-08-22)

`uv run python -m tradingagents.screener` prints the screen; add `--persist`
to write the result. `build_universe()` returns the same as a dict.

### The one thing to understand before touching this

**Percentage range is the wrong screen; ticks are the right one.** The
intuitive filter — "keep stocks that move at least 5% in a day" — is what the
user proposed and it is actively harmful on ASX microcaps. Measured on the
live top-500:

- **54 names pass a 5%-range screen but fail an 8-tick screen.** BRN/INR/MEI/
  TTT are in this group: 6.4-7.0% daily range, 73-79% of days moving ≥5%, and
  only 1.6-2.8 ticks of actual room. They look like the most volatile names on
  the board and cannot be range-traded at all.
- **237 names pass the tick screen but fail the 5% screen.** PNV is in this
  group — only 26.5% of its days move ≥5%, but 8.15 ticks of room, and it is
  the only one of the old five that was ever tradeable.

So the 5% screen would have kept exactly the four names that don't work and
discarded the one that does. `pct_days_range_ge_5pct` is computed and
displayed **only** so this comparison stays visible; never rank on it.

Ranking is on `range_efficiency` = ATR14 / (spread + 2×brokerage), which
already blends volatility against what a round trip costs. Ranking on raw
volatility re-imports the exact problem the module exists to prevent.

### Data-source gotchas

- **Tick band comes from the collector's `universe.price` (current,
  unadjusted), not from yfinance history.** yfinance returns split/dividend-
  adjusted OHLC; computing a tick band from an adjusted 2023 close puts a
  stock in the wrong band. Range *percentages* are scale-invariant so they can
  come from adjusted data safely — the tick conversion cannot. Don't
  "simplify" this by using one price source for both.
- **Spread is estimated as one tick, which is optimistic.** No real spread
  history exists yet (phase-3 §5.1's preopen/spread capture is what would fix
  it). Every cost-sensitive number is a best case, and the optimism is worst
  for the small bucket where real spreads are widest.
- **`universe.industry` is entirely NULL**, so sectors come from yfinance
  `.info`, fetched only for names that already passed the numeric screen
  (~150 lookups, ~30s) rather than all 500.
- **TTT is no longer in the top-500** and is carried explicitly; `build_universe()`
  re-fetches any `EXISTING_UNIVERSE` name that has dropped out.
- **Not screened, because the data isn't local**: live scheme/takeover/
  suspension status, and cash runway for the speculative end. Both are on the
  spec's list. Eyeball the shortlist before trading anything — BET's
  acquisition is the standing reminder.

### The selected universe (run 2026-08-22)

30 names — 5 existing (continuity) + 8 large + 9 mid + 8 small, per §16.2,
spread across 8 sectors with per-bucket sector caps so "nine miners" can't
happen. **Effective independent positions: 16.05 of 30** (mean pairwise
correlation 0.147) — reported per §16.4 so the headline count never flatters
the real cross-sectional power.

- large: PME WTC LNW BSL SGH CSL MIN CAR
- mid: 360 SRL MP1 CDA HUB JBH SEK COH SGM
- small: SKS EQT TPW EOL LYL C79 AYA CUV
- existing: PNV (tradeable) + MEI INR BRN TTT (**flagged `tradeable=0`**)

The four old microcaps stay in the pool because pooled fitting benefits from
the rows and the spec asks for continuity, but `model_universe.tradeable` is
0 for them so nothing downstream mistakes "in the model universe" for "worth
placing an order against."

### Storage — and the separation that must not be collapsed

Written to a **new `model_universe` table** in the collector's db, not to
`research_universe`. That table is written from both sides (this app's
`/research` page *and* asxbrief's own `research recommend`, which does
`DELETE FROM research_universe WHERE kind=?`), and `bar_coverage()` reads
every row in it regardless of kind — so adding rows there would silently
change what `/research` displays.

**`swing_universe` (what actually gets orders placed against it) is
untouched, deliberately** (spec §16.5). Research universe and traded universe
are different things; expanding one must never silently expand the other.
Verified after the run: swing universe is still the original six rows with
five enabled.

### What still has to happen (spec §16.6)

The pooled range model must be **re-fitted on the new universe and every
existing result re-run through the full gate** — permutation, buy-and-hold,
selection-permutation. **The `(0.1, 0.7)` finding was established on 5
tickers and does not transfer.** Only the §13.1 diagnostic has been re-run so
far (see below); no trading configuration has been re-validated on these 30.

## `tradingagents/refit_gate.py` — the §16.6 re-fit and full gate (added 2026-08-22)

`uv run python -m tradingagents.refit_gate`. Re-fits the pooled range model on
the expanded universe and runs every configuration through all four gates.
Fits **once** and reuses that fit for all 25 cells — `range_model.sweep()`
re-fits per cell, which was merely wasteful at 5 tickers and unusable at 26;
sharing one fit is also strictly more correct, since differences between cells
are then attributable to (entry_q, target_q) alone.

### Result, 26 tickers, held-out 2025-08-21 → 2026-08-20 (255 days), conservative same-bar

**Every one of the 25 cells is negative. 0 of 25 clear the gates.**

- avg gross return per trade ranges **−0.878% to −0.004%**; no cell is positive
- most cells sit at the **0.0-3rd percentile** of the randomised-entry null —
  `permutation_test()`'s own two-tailed reading calls this *evidence the
  timing is WORSE than random entry*, not merely absent
- **selection permutation (White's Reality Check): the best cell (0.1, 0.9)
  returned −0.004% and sits at the 0.0th percentile of the best-of-grid noise
  null, whose median is +0.698%** — a blind grid search over pure noise would
  essentially always have done better
- universe buy-and-hold over the same window: **+39.34%**

Trade counts are 314-1,224 per cell, so this is no longer a small-sample
result. **The conclusion is now strongly supported rather than merely
unrefuted**: next-day mean-reversion band trading on this universe is
actively worse than random entry, and far worse than owning the names.

### `_null_draw_batch` — why the permutation tests got 645x faster

The null was drawn one trade at a time through pandas indexing: fine at ~40
trades, dominant at ~640 (measured **272s per cell**, i.e. ~4 hours for the
grid plus selection test). `backtest._null_draw_batch()` generates all `n`
draws per trade in one numpy call. Verified against the loop version on a real
637-trade cell before adoption — means within 1.6σ of their difference, sd and
5/50/95th percentiles matching. `_draw_null_avg()` is retained as the
**reference implementation** the fast path is verified against; change it
first if the null's semantics ever change.

`n` is 10,000 for both the per-cell and selection tests (it was only ever
lowered because of the speed problem, which no longer exists).

### Operational notes

- Long runs go through **`systemd-run --unit=ta-refit-gate --collect`**, the
  same pattern `research_universe.start_backfill()` uses, so they survive
  independently of any shell. Wait on `systemctl is-active`, never on
  `pgrep -f <module name>` — that matches the watching shell's own command
  line and waits on itself forever (a real hour lost to this on 2026-08-22).
- `run_full_gate()` logs every phase with a timestamp. A long job that prints
  nothing until the end is undiagnosable if it dies — which is exactly what
  happened on the first attempt.
- `per_ticker_for_cell()` breaks one cell down per ticker. Read it only for
  "is the aggregate carried by one or two names", never as a shortlist —
  picking the best-looking names out of it is the selection problem one level
  down, and no per-ticker null is corrected for it.

## The same-bar exit artifact (found and fixed 2026-08-22) — read this before trusting any backtest number

**The single most consequential bug found in this project.** It invalidated a
result that had already passed every gate, and the gates could not have caught
it.

### What it was

`simulate_trade()` scanned for exits starting on the bar that filled the
entry, with an explicit comment justifying it ("a wide-range day can both fill
the entry and resolve target/stop"). `simulate_daily_refresh()` did the same.
A daily OHLC bar carries no path information, so crediting a target on the
entry bar silently assumes **the low preceded the high**. On a day whose low
is below the entry and whose high is above the target, the simulator booked a
guaranteed win — regardless of whether that ordering actually occurred.

### Why it stayed hidden and then exploded

On the old 5-ticker microcap universe entries almost never filled (1.3-6.2%),
so it was close to harmless. `screener.py` then selected the new universe
explicitly for **wide daily ranges** — precisely the condition where a bar's
low and high straddle both levels. Measured on the 26-ticker universe at
(0.1, 0.7):

| | optimistic (old) | conservative (fixed) |
|---|---|---|
| trades | 637 | 561 |
| same-bar exits | **82%** | 11% (stop-outs only) |
| win rate | 80.1% | 57.2% |
| avg return | **+1.343%** | **−0.366%** |
| total | +855.6% | −205.4% |

Splitting the optimistic run showed the artifact carried everything: same-bar
trades averaged **+1.84%** (86% win) while genuine multi-day trades underneath
averaged **−0.90%** (53% win) — the latter matching every honest result this
project has produced.

### Why every gate passed it anyway — the important lesson

`permutation_test()` draws the null as close-to-close returns held for the
trade's own `holding_days`. A same-bar trade has `holding_days = 0`, so the
null held **one bar close-to-close** while the "real" trade captured an
intrabar low-to-high excursion. The null cannot compete by construction, so
every cell scored the 100th percentile and 21 of 25 cells "cleared all gates."

**A permutation test comparing two different things will always look
significant.** When a null is passed trivially and uniformly, suspect the
comparison before believing the result. That symptom — every cell at 100.0 —
is the tell to watch for.

### The fix

`same_bar_exit` on both `simulate_trade()` and `simulate_daily_refresh()`,
defaulting to **`"conservative"`**: on the entry bar a **stop** may trigger
(assume the adverse path), a **target** may not. `"optimistic"` reproduces the
old behaviour and exists only for comparison — it is an upper bound, never a
headline.

**Every backtest number produced before this date went through the optimistic
path**, including all `/backtest` page strategies (they share
`simulate_trade`) and the phase-3 range-model results. Treat them as
flattered until re-run. The heuristic's original failure is unaffected in
direction (it was already negative) but its magnitude was understated.

## `tradingagents/open_range.py` — open-entry intraday, and the overnight-drift finding (added 2026-08-22)

Built to test a strategy the user correctly pointed out had **never actually
been tested**: not "buy the dip" but "form a view on the day, enter inside the
expected range, exit higher" — e.g. prev close $10.00, expect a good day, buy
$10.20 and out at $10.45. `range_model.py` only ever buys a low quantile
*below* the close, so it is a mean-reversion engine and this is a different
strategy entirely.

### Why open-entry is measurable where the range model was contaminated

The user also made the methodological point that settles this. The same-bar
artifact exists because a resting limit order below the market has an
**unknown fill time** within the day, so crediting a same-bar target assumes
the low preceded the high. An **open entry has a known fill time** — you are
in at the start of the bar, so everything in that bar happens after you.
Same-day target fills are then legitimate, and the only remaining ambiguity
is target-vs-stop ordering, which a stop bounds conservatively (assume the
stop went first). Resolution order per day:

  high >= target and low > stop   -> TARGET (unambiguous)
  low <= stop and high < target   -> STOP   (unambiguous)
  both touched                    -> STOP   (conservative)
  neither                         -> exit at the close

### A data bug that would have produced a fake finding

The first version used `^AXJO`'s open gap as the "market opened hot" signal
and reported a +6.50pp lift. **It was meaningless**: Yahoo reports the ASX
index's Open as the previous Close on **88% of days**, so the gap was exactly
0.000 almost always, and the ~10 non-zero days were feed quirks rather than
hot opens. **Never use `^AXJO` Open for anything.** `^GSPC` is clean (0%
exactly-zero gaps), and individual ASX stock opens are clean too (0.4-2.5%
exactly-zero, normal) — so the open-entry study itself is sound, only the
index signal was broken.

Replacements, both causally clean: `us_overnight_signal()` (prior US session's
S&P close-to-close, z-scored, shifted forward one day so a session's return is
never used on or before the session that produced it) and
`cross_sectional_open_signal()` (median open-gap across the universe — known
at the instant of entry, since entry *is* the open).

### Two reporting flaws fixed at the same time

1. **Break-even was quoted against the full sample.** It only applies to
   trades that actually resolved at a target or stop, and 43-79% of days exit
   at the close instead. Now reported as `target_share_of_resolved_pct`
   against `break_even_of_resolved_pct`.
2. **Statistics were treated as if ticker-days were independent.** 26 tickers
   on one day share that day's market move; treating them as 26 observations
   shrinks the interval by ~sqrt(26) for free. `_day_clustered_bootstrap()`
   resamples **whole days**. This is the same concern the screener's
   "effective independent positions" reports, and it bites far harder within
   a single day.

### Result

Direction is consistent — signal days beat all days in **4 of 4**
configurations (target-rate lift +0.72 to +1.94pp; avg-net lift +0.06 to
+0.18pp), which is what a real US-lead effect should look like. But only ONE
cell has a day-clustered CI excluding zero (`us_overnight`, +5%/-3%: avg net
**+0.0421%** per trade, CI [+0.0200, +0.2634], 97 signal days) — and that is
1 of 4 configurations tested, uncorrected; Bonferroni alpha for 4 tests is
0.0125 against a 95% interval.

**The context that settles it**: trading every signal day at that best cell
returns **+4.08% over three years**. Equal-weight buy-and-hold over the same
three years returned **+357%**.

### The finding that actually matters — overnight drift

Splitting three years of returns for the 26-ticker universe:

| | mean per ticker |
|---|---|
| **overnight** (prev close -> open) | **+97.9%** |
| **intraday** (open -> close) | **+6.6%** |

17 of 26 tickers have overnight > intraday; PNV is +195.8% overnight against
**-182.6%** intraday. (Summed simple returns, so not compounding-exact — the
comparison is directional, but the gap is far too large to be an artifact of
that.)

**Roughly 94% of the return in these names accrues while the market is
closed.** Any strategy that buys at the open and exits the same day is
structurally excluded from where essentially all the money is made — no
forecast skill required to explain the result, and none can rescue it. This
is the well-documented overnight/night-effect anomaly, confirmed on this
specific universe.

**How to apply**: this is an argument about *when to hold*, not *what to
predict*, and it should reframe strategy design before any further intraday
work. If a future session proposes another day-trading variant on these
names, this table is the first thing to put in front of it.

## `tradingagents/overnight.py` — overnight hold, and why the daytime session loses (added 2026-08-22)

On the `/backtest` page as **`overnight_hold`** (`run_and_report_overnight()`
in `backtest_report.py`, special-cased like `range_model` because there is no
target and no stop — the entry is the closing auction, the exit is the next
opening auction, and the outcome is whatever the gap turns out to be).
Standalone: `uv run python -m tradingagents.overnight`.

### Why the daytime session does so badly — two compounding reasons

The user pushed back on this reasonably ("things can trend up all day when
things are hot"). Measured on 19,734 ticker-days over 3 years, the intuition
is **backwards**: opening gaps fade, they do not continue.

| opening gap | rest-of-day mean | median | % days up |
|---|---|---|---|
| < −3% | **+0.372%** | +0.429% | 52.6% |
| −3 to −1% | +0.320% | +0.307% | 55.7% |
| −1 to +1% | −0.020% | 0.000% | 47.2% |
| +1 to +3% | −0.173% | −0.278% | 42.1% |
| > +3% | **−0.141%** | **−0.489%** | **41.2%** |

Cleanly monotonic. Multi-day is the same shape: after a day up >5% the next
day is positive only **44.6%** of the time; after a day down >5% it averages
**+0.84%**. This is exactly the memory bias spec §13.2c warned about — the
stocks that ran 5% on a hot day are vivid, the ones that gapped up and bled
all day are forgettable, and there are more of the second kind.

Combined with the overnight split (**+94.0% raw overnight vs +6.6% intraday**;
dividends account for only 3.9pp of it, so it is not an adjustment artifact),
the daytime session loses on both counts: the drift isn't there, and the gaps
revert.

### Results

Buy every close, sell the next open, 26 tickers, 3 years:

| condition | n | avg gross/night | net @0.05% | net @0.10% | condition null |
|---|---|---|---|---|---|
| all_nights | 19,786 | +0.1287% | −0.0713% | −0.1713% | (control) |
| after_down_day | 9,340 | +0.1811% | −0.0189% | −0.1189% | **100.0th** |
| **after_big_down_day** (≤−3%) | 1,787 | **+0.3196%** | **+0.1196%** | **+0.0196%** | **98.4th** |
| after_up_day | 9,710 | +0.0681% | −0.1319% | −0.2319% | 0.0th |
| closed_near_high | 4,706 | −0.0413% | −0.2413% | −0.3413% | 0.0th |

**`after_big_down_day` is the first thing in this project to stay positive
after a realistic spread.** Buying weakness clears the null; buying strength
sits at the 0th percentile of it — worse than picking nights at random. The
coherence matters: it is the same reversal the intraday gap table shows, so
this reads as one effect seen twice rather than a lucky cell.

### The null design is the point — don't weaken it

`condition_null_test()` holds each ticker's **night count** fixed and
randomises **which** nights. That puts both the unconditional overnight
anomaly *and* the ticker mix inside the null. It matters: the null median for
`after_big_down_day` is **+0.2098%**, well above the unconditional +0.1287%,
because big down days concentrate in the higher-volatility names that have
more overnight drift anyway. Testing against zero, or against the
unconditional mean, would both have overstated the edge. The real lift beyond
ticker mix is ~+0.11pp, not ~+0.19pp.

### Caveats that must travel with this result

1. **4 conditions tested.** `after_big_down_day` at the 98.4th percentile is
   marginal under a Bonferroni threshold (~98.75th for 4 tests);
   `after_down_day` at 100.0th is not. Run a selection correction before
   promoting either.
2. **Costs decide it.** The edge (+0.32%/night) is the same order as a round
   trip. It survives to a 0.10% half-spread and dies at 0.25%. That is fine
   for the liquid large caps (one tick ≈ 0.01-0.05% of price) and **not** for
   the small end. Any deployment must be per-ticker cost-aware, not universe-wide.
3. **Buy-and-hold still wins on raw total return** (+357% over the same 3
   years). The overnight strategy's appeal, if any, is exposure — it is in
   the market a fraction of the time — not raw return. Do not present it as
   beating buy-and-hold.
4. **Walk-forward now run** — see the section below. Seven of ten windows
   predate the discovery period; the effect survives, with three real
   qualifications (crash-regime failure, possible recent decay, thin margins).

### Walk-forward: does the overnight condition survive out of sample? (2026-08-22)

`overnight.walk_forward()` — 10 years, 12-month windows. Nothing is *fitted*
here, so the risk isn't parameter overfitting; it's that the condition was
**chosen after looking at the gap table** on a 3-year window. Seven of these
ten windows predate that discovery period entirely.

`after_big_down_day` (buy close / sell next open after a day ≤ −3%):

| window | n | trades/day | trade-wtd | day-wtd | net@0.05 | pctile |
|---|---|---|---|---|---|---|
| 2016-17 | 382 | 1.94 | +0.664 | +0.525 | +0.464 | **100.0** |
| 2017-18 | 329 | 1.86 | +0.456 | +0.445 | +0.256 | **99.8** |
| 2018-19 | 487 | 2.55 | +0.482 | +0.565 | +0.282 | **100.0** |
| **2019-20** | **729** | **3.66** | **−0.092** | +0.401 | −0.292 | **0.0** |
| 2020-21 | 415 | 2.41 | +0.176 | +0.138 | −0.024 | 35.1 |
| 2021-22 | 759 | 3.74 | +0.416 | +0.622 | +0.216 | **99.6** |
| 2022-23 | 546 | 2.66 | +0.663 | +1.118 | +0.463 | **100.0** |
| 2023-24 | 477 | 2.34 | +0.463 | +0.633 | +0.263 | **100.0** |
| 2024-25 | 548 | 2.73 | +0.313 | +0.508 | +0.113 | 62.0 |
| 2025-26 | 749 | 3.65 | +0.232 | +0.129 | +0.032 | 61.2 |

**6 of 10 clear the 95th percentile of their own null; 9 of 10 positive;
median percentile 99.7.** The control `after_up_day` clears **1 of 10**
(median percentile 15.7) — so the buy-weakness/sell-strength asymmetry is
systematic across a decade, not an artifact of the window it was found in.
This is the strongest result the project has produced.

**Three qualifications that must travel with it:**

1. **It fails in crash regimes, exactly when exposure peaks.** 2019-20 (COVID)
   is the one negative window and it sits at the **0.0th percentile**. Look at
   why: `trades_per_day` was 3.66, near the highest in the sample, while
   trade-weighted return (−0.092) fell far below day-weighted (+0.401). In a
   broad selloff many tickers breach −3% on the same night, so capital is
   maximally deployed precisely when the effect inverts. **This is correlated
   exposure at the worst possible moment**, and it is invisible in the
   day-weighted figure. Any deployment needs a market-wide abstain
   (phase-3 §7.2's second layer) before it is safe.
2. **Possible recent decay.** The two most recent windows (62.0, 61.2) do not
   clear, despite being the period the condition was chosen on. Could be
   crowding, could be noise — two windows can't distinguish them. Watch it.
3. **Thin after costs.** Positive at a 0.05% half-spread in 8 of 10 windows,
   but the margin is small and it dies at 0.25%. Liquid names only.

**Reporting note — trade-weighted vs day-weighted.** These are different
quantities and both are reported. The day-clustered CI is built on day means
and brackets the **day-weighted** figure; an earlier version printed it beside
the trade-weighted mean, producing a CI that did not contain its own point
estimate (the 2019-20 row made this obvious: −0.092 with a CI of
[+0.133, +0.667]). **Trade-weighted is the deployment-relevant number** — it
is what capital spread across every qualifying name actually earns — and the
gap between the two is a direct read on concentration risk.

## `tradingagents/event_momentum.py` — does news-driven strength continue? (added 2026-08-22)

Tests the user's hypothesis: a move backed by real news continues over days,
where an unexplained move reverts. Their example was a $0.30 stock running to
$3-4 on FDA clearance.

**Volume as a news proxy, because the real feed is empty.** The collector's
`announcements` table holds **3 days** of history (started 2026-08-19), so
`strategies.fetch_events()` — wired and correct — has nothing to backtest
against. A move on 6x+ normal volume is almost certainly repricing news; the
same move on ordinary volume is thin-market noise. That runs on ten years
today. **Universe is the full top-500, not the 26-ticker model universe** —
the tick finding killed *intraday band trading* on microcaps and says nothing
about multi-day event moves, and excluding speculatives would exclude the very
cases the hypothesis is about. Entry is the **next open**, never the event
day's close.

### The phenomenon is real

Within +8-12% up moves (move size held fixed, so this is not size in disguise),
forward return from the next open rises monotonically with volume:

| volume | fwd 5d | fwd 10d |
|---|---|---|
| normal (~1x) | −0.014% | +1.191% |
| elevated (2x) | +0.190% | +1.164% |
| high (4x) | +0.749% | +1.983% |
| **extreme (>6x)** | **+1.827%** | **+2.783%** |

Extreme-minus-normal, day-clustered: **+2.284pp at 5d (95% CI [+1.202,
+3.428])** and **+1.932pp at 10d ([+0.651, +3.296])** — both significant.
Same monotone shape in the +12-20% band. **No effect at all in +5-8% moves** —
the mechanism needs a substantial move *and* volume confirmation.

Note the continuation is **not next-day**: every bucket is negative at 1 day.
The immediate fade still happens; the repricing plays out over 3-10 days.

### But the tradeable window and the significant window don't overlap

The edge decays monotonically with price and dies above 50c (extreme-vs-normal
at ≥50c: +1.232pp, CI [−0.359, +2.859], ns). Against the round trip you must
cross — a momentum entry is an aggressive order, passive resting is not
available when chasing a move that already happened:

| price band | n | 1 tick | round trip | edge fwd10d | verdict |
|---|---|---|---|---|---|
| under 5c | 168 | 5.00% | 10.00% | +5.33% | cost > edge |
| 5-20c | 377 | 2.69% | 5.39% | +3.99% | cost > edge |
| 20-50c | 330 | 1.61% | 3.23% | +2.27% | cost > edge |
| **50c-$2** | 340 | 0.48% | 0.96% | +2.41% | survives cost |
| over $2 | 239 | 0.25% | 0.50% | +0.31% | cost > edge |

And the one surviving cell fails its own baseline test. Candidate (+8-12%,
>6x vol, 50c-$2, 10d): **+2.321%, CI [+0.515, +4.193]** — beats zero. But
against *any* ±3% mover in the same price band (+1.048%): **difference
+1.273pp, CI [−0.593, +3.149] — NOT significant** at n=340.

**Verdict: the effect is real but the tradeable window and the statistically
significant window do not overlap.** Where it is strong (sub-50c) the spread
eats it; where costs are low enough (50c-$2) it is indistinguishable from
simply buying any mover in that band. Do not report this as a working
strategy, and do not report it as a null result either — the phenomenon is
solidly demonstrated, the exploitation is not.

### Payoff shape — the user's anecdote is accurate

Extreme bucket, 10 days: mean +2.783%, **median 0.000%, only 46.1% positive**,
p90 +20.8%, **p99 +88.3%**. A minority of large winners carries the mean. That
is exactly the $0.30-to-$3 case: real, and dependent on catching the tail. Any
deployment needs wide diversification and sizing that survives long losing
streaks, and every backtest of it will have very fat tails.

### Caveats that must travel with this

1. **Survivorship bias is severe here** and worse than in the overnight work.
   The universe is *today's* top 500 — companies that got bad news and delisted
   are absent. It shows up plainly in the down-day table: at −12 to −20%,
   *normal*-volume drops "bounce" +15.1% over 10 days (n=716) while extreme
   volume gives +3.0%. That ordering is backwards and the levels are absurd;
   it is survivors-only accounting. **The down-day results should not be used
   at all.** Up-day results are less affected, and the *gradient across volume
   buckets* is defensible because all buckets are equally biased — but every
   absolute level is an upper bound. Even the "any mover" baseline of +1.048%
   per 10 days (~26% annualised) is implausible and is mostly this bias.
2. **Volume is a crude proxy.** It cannot distinguish FDA clearance from an
   index rebalance, a block trade, or a capital raise — which plausibly have
   opposite continuations. The user's actual idea (use the announcement
   classifier's type and score) is strictly better and still awaits data.
3. 64 cells were inspected across bands/buckets/horizons/directions. The
   +8-12% monotone pattern is credible because it is a *pattern replicated
   across two horizons and an adjacent band*, not a grid maximum — but no
   selection correction has been applied.

### What would actually move this forward

- **A survivorship-free universe** (point-in-time constituents including
  delisted names) is the single highest-value data upgrade available; it
  contaminates this, the overnight work, and the screener.
- **Real announcement history**, which is accumulating from 2026-08-19. The
  classifier already scores announcements 0-100 — testing continuation by
  announcement *type* is the user's original idea and is likely far sharper
  than volume alone.

## `/scanner` — live movers scanner + momentum paper trades (added 2026-08-22)

**Times on the page are Sydney, not UTC (fixed 2026-08-24).** The API returns everything in UTC (`scanned_at` etc); the page converts at display time via `fmtSydney()` using the browser's `Australia/Sydney` tz database entry, so AEST/AEDT is handled automatically rather than a fixed +10/+11 offset going wrong twice a year. **`/asx` still uses `Australia/Adelaide` for its `fmtTime()` (30-90 min off Sydney depending on DST) — not fixed, flagged for the user, not touched without being asked.**

```
tradingagents/scanner.py      -- the live scan (yfinance daily bars + collector announcements)
tradingagents/momentum_db.py  -- paper-trade log, own table `momentum_trades` in data/swing.db
web/server.py                 -- /scanner, /api/scanner/scan, /api/scanner/trades[/{id}/close]
web/static/scanner.html       -- movers table + paper-trade log
```

**Purpose is forward testing, not another backtest.** The user asked to watch
this live and place trades by hand to build a feel. That is worth more than
another study here for a concrete reason: **a live scan is survivorship-free
by construction**, and every retrospective result in this project is
contaminated because the universe is *today's* top 500 (see
`event_momentum.py`).

### Thresholds are measured, not chosen

From `event_momentum.py`: move >= **+8%** (below that volume carried no
continuation information at all), volume >= **3x** normal (the gradient only
separates from the no-news bucket at 3x+), and rows are **flagged by price
band, never filtered** — `cost_verdict` records whether that band's historical
edge survived its round-trip tick cost, so a 3c rocket still appears and is
plainly marked uneconomic. Anything from +5% shows as "watch".

### Two bugs found while building it — both would have looked like signal

1. **Off-session volume divisor.** Expected volume is scaled by the fraction
   of the session elapsed, with the latest daily bar being partial. Outside
   the session that bar is *complete*, so the divisor must be 1.0. The first
   version scaled by elapsed-fraction unconditionally with a 0.05 floor, which
   made a normal Friday read as **108x normal volume**. `session_state()` now
   returns `(state, effective_fraction)` and only returns a partial fraction
   when the session is genuinely open.
2. **Announcement window too short.** A 24h lookback anchored on "now" drops
   exactly the announcements that explain a gap — they are released after the
   previous close or before the open — and misses the prior session entirely
   when scanning late. `ANNOUNCEMENT_LOOKBACK_HOURS = 36` covers overnight plus
   the full current session. With 24h the join returned zero tickers; with 36h
   it returns ~151.

Known approximations, both documented in-module and surfaced in the UI: the
volume ratio uses a linear session-elapsed scale where real intraday volume is
U-shaped (early ratios read high, late ratios read low), and the Sydney offset
is fixed at +10 (AEST), so it is an hour out during AEDT.

### Paper trades are LOG-BASED, not broker-routed

`momentum_db` records an entry at a user-entered price (pre-filled from the
scan) and marks to market from daily closes. Deliberate for a first cut: the
point is to learn which setups work, and a log cannot half-fill, cannot be
rejected, and cannot leave a stray order on a real book. **`swing_ibkr.py`
already has the paper-order path** (`place_bracket`, `place_day_entry`,
`place_stop`, `fetch_order_statuses`) if realistic fills become the question —
at which point the fill *is* the experiment, since the cost analysis says
spread decides everything below 50c.

**Own table, not `swing_trades`.** Three signal sources now share one paper
account (range model, announcement AI, momentum) and the standing rule is not
to conflate them — mixing them makes per-strategy P&L unrecoverable, which is
the only thing this log exists to produce. Verified after the smoke test that
`swing_trades` was untouched.

### Recording what the trade came from

The dialog captures `source` (scanner / announcement / other),
`ai_score_at_entry` and the user's own `conviction` (1-5) into `setup_json`,
and the trades table shows all three. **Keep these separate** — the
announcement classifier has never been validated for predictive accuracy, so
these trades are the first evidence, and collapsing the AI score into the
user's own read would make it impossible to tell which one (if either) works.
A **"Log a trade manually"** button exists because announcement-led trades
don't originate from a scanner mover row.

**Measured firing rates** (9.8 years, top-500): +5% watch 18.1/day; +8% alone
7.7/day; FULL (+8% and 3x volume) **3.3/day (~16.5/week)**; FULL restricted to
the 50c-$2 band **0.8/day (~4.1/week)**. Announcement side: ~149 classified
per session, ~10/day scoring Buy (>=65), ~3/day scoring >=80.

**Nothing auto-closes.** The page flags a breached stop or an elapsed hold and
leaves the decision to the user, because the exit is precisely what they are
trying to learn. The summary shows median alongside mean because the measured
payoff is lottery-shaped (46% positive, p99 +88%) — a median well below the
mean is the expected shape here, not evidence of failure.

## Announcement classifier auth — Codex, not Grok (switched 2026-08-23)

The classifier ran on `grok_oauth.py` until the user's **xAI credits ran out**.
It now runs on a **ChatGPT/Codex subscription login**.

```
tradingagents/codex_oauth.py   -- Codex-backed ask(), mirrors grok_oauth's surface
tradingagents/grok_oauth.py    -- unchanged, still selectable
scripts/codex_login.py         -- interactive login | --from-hermes | --check
```

**Provider is a one-line switch.** `asx_signals.py` picks via
`ASX_SIGNALS_PROVIDER` (default `codex`, set `grok` to go back). Both modules
expose the same narrow surface — `is_configured()`, `ask()`, `is_quota_error()`,
`is_auth_error()`, plus a quota and an auth exception — and that is the whole
reason the switch is one line. **If you add a third provider, match this
surface** rather than teaching `asx_signals` a new shape.

`_MODEL_LABEL` is stored on every cached signal row, so rows classified by
different models coexist and a later accuracy check can separate them instead
of pooling two models' scores as one. Don't backfill the column.

**Transport differs from Grok and this is why the module isn't a copy-paste**:
Codex speaks the **Responses API** at `chatgpt.com/backend-api/codex` and needs
a `chatgpt-account-id` header derived from a JWT claim inside the access token
— not a bearer key against a normal `/v1` endpoint. `codex_oauth` reuses the
token/header/refresh helpers from `llm_clients/openai_codex_client.py` rather
than reimplementing them, so refresh-token rotation logic lives in one place.

**Credential file**: `~/.tradingagents/codex_auth.json` (override
`TRADINGAGENTS_CODEX_AUTH_FILE`), written in the `credential_pool` shape the
existing Codex client already reads. **Never point it at
`~/.hermes/auth.json`** — the token endpoint **rotates** refresh tokens, so a
refresh through a shared file silently invalidates the other tool's credential.

**`--from-hermes` was tried and failed, 2026-08-23**: Hermes held an
`openai-codex` refresh token from May, but exchanging it returned
`refresh_token_reused` — Hermes had already rotated it. The seeded file was
deleted rather than left in place, because a dead credential makes
`is_configured()` return True while every call fails, which reads as a broken
classifier rather than an unconfigured one. **A fresh interactive login is
required.** The `--from-hermes` path is kept because it costs nothing and would
work against a credential that hasn't been rotated.

**Login flow: use `--manual-start` / `--manual-finish` on this box.** The
loopback flow needs the browser to reach port 1455 *on the VPS*, i.e. an SSH
tunnel. The manual two-step prints the authorize URL, the user signs in, the
redirect fails to load (expected), and the address bar's `?code=...` is pasted
into `--manual-finish`. Same PKCE, state is still verified; only the delivery
of the code differs. **Completed successfully 2026-08-23.**

**`ask()` MUST stream.** The Codex backend rejects a non-streamed Responses
call with `400 {'detail': 'Stream must be set to true'}` -- which reads like a
malformed request rather than a transport requirement, and cost a debugging
round trip. `codex_oauth.ask()` uses `client.responses.stream(...)` and falls
back from `output_text` to walking `output` to accumulated deltas, same as
`llm_clients/openai_codex_client.py`.

**Verified live 2026-08-23**: `--check` returns `OK`, `classify_pending()`
classified 25 announcements with 0 errors, and the collector's own
localhost-exempt webhook (`POST /api/asx/signals/refresh`) works unchanged.
Codex's early spread looks sane -- 12% Buy, 88% Hold, 0% Sell, with buy-backs
and funding news scoring 72-78 and routine filings at 50.

**Classifier state is now in `/api/asx/health`** under `classifier`
(provider / model / configured / login_hint). It belongs in health because
when the credential lapses the announcement feed keeps working and scores
simply stop appearing — which reads as "nothing notable today" rather than
"the classifier is down". That is precisely how the credit exhaustion went
unnoticed.

## Announcement body fetching — the classifier reads the PDF (added 2026-08-23)

```
tradingagents/announcement_body.py  -- fetch + extract + cache PDF text
tradingagents/asx_signals.py        -- classify_one(use_body=True), PROMPT_VERSION
web/server.py                       -- background classify + /api/asx/signals/status
```

### Why — a real miss the user caught

TRE's "First Drawdown Proceeds Received under Gold Stream" scored **78 (Buy)**.
The body says plainly it is the first drawdown of a facility **announced on
13 July**, following satisfaction of conditions precedent — administrative, not
news. The classifier had only ever seen headlines, which are written by the
company and are a lossy, flattering summary. Only the body distinguishes
"$2m received" from "$200m", or a placement at a 5% discount from one at 40%.

Two fixes, both needed:

1. **Prompt (`PROMPT_VERSION`, now `v3-body-text`)** encodes *carrying out
   something already announced is not new information*, however good the
   underlying arrangement — drawdowns, proceeds received, completion,
   settlement, shares under an existing facility. It also states that ASX's
   price-sensitive flag means "worth reading", **not** "bullish"; that flag was
   almost certainly part of what pushed TRE to 78.
2. **Body fetching**, on by default. TRE went **78 → 50**.

Regression-tested on eight known cases before shipping. It did **not**
over-correct: FDA clearance 89, gold intercepts 86, voluntary administration 2,
halt-pending-capital-raising 25. Only execution-of-known-arrangement collapsed
to 50. Keep that test set in mind before editing the prompt again.

### The interstitial gotcha (same one asxbrief hit)

`announcements.url` is **not a PDF**. It is a terms-of-use page on
`www.asx.com.au` carrying the real PDF URL in a hidden `pdfURL` form field, on a
*different* subdomain. Fetch the page, extract the field, fetch that.

### Cache and limits

Cached in **TradingAgents' own signals DB** (`announcement_bodies`, keyed by
fingerprint) — the collector owns `announcements`, so reading it is fine and
writing to it is not. Fetched once ever; re-classifying never re-fetches.
Failures cache as empty so a broken document isn't retried forever, and the
classifier degrades to headline-only rather than skipping the announcement.
Caps: 8MB PDF (one 10.5MB document was skipped and correctly fell back),
12,000 chars kept — **the head**, because ASX announcements front-load material
facts and tail into boilerplate and director bios.

### Three bugs found wiring this up — all silent-failure shaped

1. **`asx_feed.signal_worthy()` didn't SELECT `a.url`**, so `classify_one`
   skipped the fetch for every row: 10 classified, 1 body fetched, everything
   looking healthy. Fixed, and a missing url/fingerprint now **logs a warning**
   instead of degrading silently.
2. **The running service had stale code** while asxbrief's live webhook kept
   classifying on the old prompt with no body — producing rows that looked new
   but weren't. **Restart the service after touching classifier code**; the
   webhook is live during market hours.
3. **asxbrief's webhook timeout is 5s** (`signals_webhook_timeout`), but body
   fetching costs ~6s *per announcement* — it would have timed out on the first
   one, every time. `/api/asx/signals/refresh` now runs in the background
   behind a **single-flight lock** and answers in **0.18s**; the collector fires
   every 5 min and a batch outlasts that, so runs must not pile up. `wait=true`
   still runs synchronously for manual use.

### Load (measured 2026-08-23)

~6.3s per announcement, once. At ~161/day that is ~17 min of work spread across
the day, not a burst. Load average 0.40-0.54, ~300MB free with 790MB cache.
Average body 6,788 chars, so each LLM call is materially larger than before —
**watch Codex usage over a full day**; the lever if it bites is `MAX_TEXT_CHARS`.
`/api/asx/signals/status` (localhost-exempt) reports run state and cache stats.

### Result — a much more discriminating classifier

60 announcements, 58 with body, 0 errors: **22% Buy / 72% Hold / 7% Sell**,
against 12%/88%/0% headline-only under Codex and 6.7%/91%/5.4% under Grok.
Reasons are now substantive rather than headline restatements — "Record earnings
growth, higher dividend", "Beat guidance; strong growth, dividend rise",
"Dilutive placement and SPP announced". All 218 previously-scored rows were
**wiped rather than left mixed** with three scoring regimes.

### Multiple announcements per ticker — net them, never `max()` (added 2026-08-24)

**34% of scored tickers have more than one announcement in a session**, so this
is the normal case, not an edge case. And they are usually **not independent
signals** — ASX convention splits one corporate action across several
documents: a results release, a media release, statutory accounts, an appendix,
a dividend notice, a trading halt.

**The bug this fixed.** `scanner.py` picked `max(ai_score)` across a ticker's
announcements. GNG (2026-08-23) published six documents for one "FY26 results
plus dilutive equity raising" event, scoring 20/34/35/48/50/68. `max()`
displayed **68 (Buy)** and hid the raise completely — a bullish bias that fires
hardest in exactly the case that matters most, a company pairing good results
with a capital raise.

**The fix**: `combine_ticker_day()` scores the SET in one call and writes to a
`ticker_signals` table (ticker, session_date, score, reason, n_announcements).
GNG now reads **Hold 36** — "Strong FY26 performance outweighed near term by
sizable dilutive equity raising."

**Why an LLM call rather than arithmetic.** Any mean/max/min double-counts
whichever facet the company published most documents about. Weighting by
document count is exactly backwards.

**Why it is cheap.** The per-announcement scores already read the PDFs, so the
combine call reasons over their headlines/scores/reasons — never re-reading
bodies. One small call per multi-announcement ticker-session, and single-item
tickers short-circuit with no call at all.

**It does not merely damp everything** — checked before trusting it. Same
session: DTL (4 announcements, all good) → **Buy 86**; REG (4) → **Buy 81**;
NGI (4) → **Hold 49** ("AUM growth offset by sharply weaker earnings"); routine
NTA/buy-back pairs (AMH, DJW, MIR) → 50. It separates "good news, several
documents" from "good and bad news together".

**Two ordering constraints, both easy to get wrong:**

1. `refresh_ticker_signals()` **must run after** `classify_pending()` — a net
   score is meaningless until every announcement in the session is scored. It
   is wired into the background classification run for this reason, not
   scheduled separately.
2. `get_ticker_signals()` defaults to the **most recent 2 sessions**, because
   the scanner joins announcements over a **36-hour** window (news lands after
   the previous close or before the open). A single-session lookup leaves those
   movers with raw announcements on screen and no net score — silently
   reinstating the per-announcement view this replaces.

Scanner rows now carry `ai_is_net`, `ai_reason` and `ai_score_range` so a wide
spread stays visible rather than being hidden behind one number.

**The `/asx` stream groups by company too (added 2026-08-24).** `GET
/api/asx/feed?group=true` returns one row per **ticker AND session date** —
grouping on ticker alone would merge Monday's results with Thursday's placement
into one meaningless row. `asx_signals.group_announcements()` does the work; the
"Group by company" toggle on the page defaults **on**. 300 announcements collapse
to ~151 company rows, 60 of them multi-announcement.

Each grouped row shows the net score plus the **individual range** in
parentheses (GNG: `36 net (20–68)`), and the caret expands to the underlying
announcements with their own scores — collapsing detail, not discarding it. A
net score is only used when its `session_date` matches the group's, otherwise a
ticker that announced on two days would show one day's net against the other
day's announcements. The collapsed headline is the **most material** scored
announcement, not the newest — the newest is often an Appendix filed minutes
after the results it accompanies. The fallback
when no net score exists is the **most material** individual score (furthest
from 50), never the highest.

## `classify_pending`'s limit bug — candidates vs classifications conflated (found 2026-08-24)

**The user asked why EVT wasn't scoring higher despite a good day.** Root
cause: `classify_pending(limit=N)` passed the same `N` straight through to
`signal_worthy(limit=N)`, which fetches the newest N candidates and only THEN
filters out already-scored ones. As the backlog grows, that newest-N window
fills with already-scored items and shrinks toward nothing, while genuinely
unscored OLDER announcements sit below the window and are never even looked
at — not "not yet classified", literally never considered.

**EVT's actual FY26 results release had fallen out of the newest-60 window.**
Two of its five documents *did* get scored — a governance filing (correctly
Hold 50) and its 37-page/13MB slide deck (headline-only fallback, over the PDF
size cap) — so the ticker looked "classified" while the actual news was never
read. **Fixed**: candidates are now fetched from a `CANDIDATE_POOL_SIZE=2000`
pool; `limit` only caps how many get classified, never how many are considered.
Confirmed 402 items were backlogged at the time this was found.

**After the fix, EVT's real results announcement scored 78 (Buy)** — "Strong
earnings growth, positive FY27 outlook, strategic value-unlocking review" —
and the net across all 5 announcements moved **Hold 50 → Buy 74**. The
governance filing and dividend notice were correctly minor; the results
release was the actual signal and it had simply never been read.

**Don't reintroduce this pattern.** Any future "classify N" style function
must fetch its candidate pool independently of its output cap — sizing them
the same is exactly this bug.

## Mover OHLC log + open-vs-prev-close outcomes (added 2026-08-24)

```
tradingagents/mover_log.py  -- log_movers(), finalize_today(), outcomes(); own table
web/server.py                -- GET /api/scanner/outcomes, POST /api/scanner/finalize
web/static/scanner.html      -- "Gap outcomes" card, below Movers
/etc/systemd/system/asx-mover-finalize.timer  -- 16:15 Sydney, Mon-Fri
```

**Why**: the user's exact concern — "a stock could open 20% up and just drift
down all day and that would be no use" — is invisible in the scanner's own
thresholds, which are both measured against the **previous close**. A stock
qualifying as a mover says nothing about whether it then held the move. This
logs prev_close/open/high/low/close for every ticker the scanner ever flags
and reports the three-way split: closed **above open** (held or built on the
gap) vs **above prev close but below open** (the drift case) vs **at/below
prev close** (gave it all back).

### Data model

One row per `(ticker, trade_date)`, upserted on every `scan()` call —
`log_movers()` is called from inside `scan()` itself, not from the page, so a
browser visit or the 2-min auto-refresh populates it; capture doesn't depend
on anyone watching. `open` is written once (the opening auction is fixed for
the day); `high`/`low`/`close` update each time from the daily bar, which
during market hours already reflects the running intraday extremes and last
price — no separate polling loop.

**Logged for every WATCH-level row (move >= +5%), not just FULL setups.** The
question is general ("does a gap hold"), not specific to the news+volume
subset — narrowing to FULL rows would answer a smaller question than asked.

### `finalized` — why mid-session numbers can't answer this

During market hours `close` is really "last traded so far" and could still
move; reporting on it would bias the outcome split toward whatever a partial
session happens to look like. `finalize_today()` re-fetches the true daily bar
and marks rows `finalized=1`; `outcomes()` only reads finalized rows. Runs from
**`asx-mover-finalize.timer`, 16:15 Sydney Mon-Fri** (15 min after the 16:00
close, same pattern as `asx-swing-propose.timer`) — so the true close is
captured even if nobody had the page open. `pending_finalize_count()` surfaces
same-day rows still awaiting this on the page, so a stale/failed timer is
visible rather than silently absent from the report.

### First real result (2026-08-24, 31 movers, same-day)

**27 of 31 (87.1%) held above their open**, averaging +5.04% open→close on top
of the gap; **4 (12.9%) gapped then faded** below their open (avg −0.53%
open→close on a 9.75% average gap — nearly the exact pattern the user
described); **0 gave the whole move back**. One session only — not yet enough
to draw a conclusion, but the split is exactly the one that was asked for and
the pipeline is now running unattended every trading day.

### Bug found and fixed the same day: `finalized` was silently cleared

The UPDATE branch of `log_movers()` set `finalized=0` unconditionally on every
touch, including scans that happen AFTER `finalize_today()` has already
stamped the true close -- and a ticker keeps showing as a mover all day (the
threshold is against `prev_close`, which never changes), so the 2-minute
auto-refresh kept re-logging it and undoing the finalize timer's work within
minutes. Measured: the timer correctly finalized 33 rows at 16:15 Sydney; by
16:19, only 3 were still finalized. **This is why the outcomes card looked
like it was regressing instead of filling in** -- it wasn't slow to populate,
its data was being erased faster than it accumulated. Fixed: `finalized` is
now a one-way stamp for the day, touched only by `finalize_today()`. Verified
after the fix: ran a real scan and all 36 previously-finalized rows held.

### Displayed on `/scanner`, below the Movers table

### Background refresh — scans happen every 10 min whether the page is open or not (added 2026-08-24)

`asx-scanner-refresh.timer` (Mon-Fri, 09:00-17:59 Sydney, every 10 min —
`OnCalendar=Mon..Fri *-*-* 09..17:00/10:00 Australia/Sydney`, same convention
as `asx-swing-propose.timer`/`asx-mover-finalize.timer`) hits
`GET /api/scanner/scan?force=true` on a schedule. Server cache TTL raised to
match (`_SCAN_TTL_SECONDS = 600`), and the page's own `setInterval` polling
was slowed to 10 min to match — polling faster than the TTL would just
re-fetch the identical cached response.

**This closes a real gap, not just a convenience.** `log_movers()` only ever
ran as a side effect of `scan()`, so before this the OHLC log (mover_log,
outcomes report) entirely depended on someone having the page open. Now it
accumulates on schedule regardless of viewers — the same "capture must not
depend on the page being watched" principle `finalize_today()` already
follows for the EOD close.

`/api/scanner/scan` was added to the localhost-exempt path list (the timer's
`curl` call has no auth cookie) — same pattern as `/api/scanner/finalize`,
`/api/asx/signals/refresh`, etc. `/api/scanner/outcomes` was deliberately
**not** exempted; nothing calls it unauthenticated.

### Gap outcomes table is sortable (added 2026-08-24)

Click any header (`sortOutcomes(key)`) — same convention as the `/asx` feed
table (arrow indicator, ascending on ticker/date, numeric-aware compare with
nulls sorted last). Scoped under an `outcomes`-prefixed state
(`outcomesRows`/`outcomesSortKey`/`renderOutcomesTable()`) so it can never
collide with any future sort added to the Movers table above it. The
`outcome` column sorts by outcome rank (above_open > above_prev_close_only >
at_or_below_prev_close > unresolved), not alphabetically on the pill text.

Summary tiles (finalized days, % in each bucket) plus a per-day table with
gap % and open→close % columns and a coloured outcome pill. A note surfaces
`pending_finalize_today` when today's rows are still mid-session, so a 2pm
glance at the page doesn't get misread as a resolved outcome.

## Deploying a change
Static HTML edits need nothing. Anything in `tradingagents/*.py` or
`web/server.py` needs:
```bash
systemctl restart tradingagents
```
This is a live, internet-facing service (real traffic hits it) — a restart is
a ~1-2s blip, acceptable without asking, but don't leave it broken.

Sanity-check after any change:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7777/asx
curl -s http://localhost:7777/api/asx/health
curl -s http://localhost:7777/api/markets/snapshot | head -c 300
```
And for HTML/JS edits, since there's no build step to catch syntax errors:
```bash
node --check <(python3 -c "import re; print(re.search(r'<script>(.*)</script>', open('web/static/asx.html').read(), re.S).group(1))")
```

## Data sources and their quirks

**Announcements** (`asx_feed.py`) — reads `/opt/asxbrief/data/asx.db` read-only
(`ASXBRIEF_DB_PATH` env overrides the path). `recent_announcements()` joins
live against asxbrief's `universe` table so the "top 500 only" filter and
`universe_rank` always reflect the *current* universe, not whatever it was
when a row was collected. `signal_worthy()` is the subset worth an LLM call —
**every announcement from an in-universe company**, not just ones ASX flags
price-sensitive (see AI signals below for why that changed) — see
asxbrief-collector skill for the collector side of this.

**Keyword watch** (added 2026-08-21) — `recent_announcements()` also exposes
`keyword_flags` (list, e.g. `["State Street"]`), sourced from asxbrief's
PDF-body text extraction, not headline matching (headlines for the triggering
notice type — substantial holdings — never contain the holder name; see
asxbrief-collector skill for the interstitial-page/hidden-field gotcha behind
extracting the real PDF URL). Only populated for announcement types asxbrief
actually fetches a PDF for; NULL/empty elsewhere, so it's not a reliable
"checked, no match" signal at this layer. `/asx` renders it as a purple
`🔍 <keyword>` badge per row and has a `kind=keywords` filter button
("🔍 Keyword Watch") that passes through to `asx_feed.py`'s `kind` param.
First live run found 8 real matches (RWC, CAR, ACL, KAR, ING, IRE, PPT, DXS)
out of 1,208 announcements checked.

**Market snapshot** (`markets.py::get_snapshot`) — `yfinance`, 5-min cache.
Tickers include `AUDEUR=X`, `USDEUR=X`, `BTC-USD` (all resolve fine).
**SPI 200 futures has no free Yahoo ticker** (every guess 404s, confirmed
2026-08-21) — instead scraped from Barchart's continuous-contract page
(`https://www.barchart.com/futures/quotes/AP*0`, auto-resolves front month,
`lastPrice` embedded directly in the HTML, no API key). Reported as
**futures last vs the previous ASX 200 (`^AXJO`) close**, not vs the future's
own prior settlement — that's the number that means something (implied
overnight move ahead of the next ASX session), matching how this is
conventionally reported ("SPI futures pointing to a weaker open"). If the
Barchart scrape ever breaks, re-inspect that page live before assuming the
regex (`_SPI200_LAST_RE`) is still right — it's matching on an embedded JS
init blob, not a documented API (already broke once, 2026-08-21, same day it
was built: Barchart started appending a status-suffix letter to the value,
e.g. `"8,988.0s"` — regex now tolerates and discards one trailing letter).

**SPI 200 is refreshed once a day, not on the dashboard's poll cycle**
(changed 2026-08-21) — the user only checks it once each morning to gauge the
overnight move ahead of the ASX open, so scraping Barchart every 5 minutes
(the general snapshot's TTL) was pure waste for a number that doesn't need to
be fresher than that. `asx-spi200-refresh.timer` (`OnCalendar=*-*-*
08:05:00 Australia/Sydney`, DST-aware, ~55min ahead of the user's ~9am check
-- moved from 08:00 sharp on 2026-08-21 since that's not necessarily the
overnight session's actual close print) hits `POST
/api/markets/spi200/refresh` (localhost-only, same `_LOCAL_OR_AUTH_PATHS`
pattern as the signals-refresh endpoint). That endpoint is still the same
Barchart scrape as ever -- **not** IBKR, even though IB Gateway now runs on
this box for the phase-2 bar backfill; nothing about SPI200 touches it, and
switching to a real IBKR SPI 200 contract quote is a separate, not-yet-done
option if the Barchart scrape ever gets less reliable. It scrapes once and
persists `{last, fetched_at}` to
`/opt/tradingagents/data/spi200_cache.json`. `get_snapshot()`'s
`_spi200_last()` just reads that file; it only falls back to a live scrape if
the file is missing or older than 20h (self-healing — e.g. right after
deploy, or if the timer ever misses a day), so the value is never simply
absent. AXJO's own previous-close (the other half of the SPI200 change_pct)
still comes from the regular 5-min `yfinance` snapshot — that's already cheap
and only changes once a trading day itself.

**"Expected Open (SPI overnight move)" tile** (added 2026-08-21, method fixed
same day) — the point figure retail traders quote each morning (a forum
poster the user follows: "-28" on 2026-08-21) is the SPI 200 **futures'
own** overnight change, vs its own previous settlement — NOT `spi_last -
axjo_prev` (the SPI200 tile's own `change_pct` basis). First attempt used the
latter and produced `-96`, wildly off, because SPI futures trade at a
persistent basis to the XJO cash index (cost-of-carry, not part of any
overnight move) — subtracting the cash close folds that basis into the
number. Fixed by reading Barchart's own `priceChange` field straight out of
the same scraped page (`_SPI200_CHANGE_RE`, e.g. `"priceChange":"+12.0"`) —
its change vs its own last settlement, already correct, no extra request
needed. Persisted alongside `last` in `spi200_cache.json` as `change_pts`.
`items` entries can carry `is_point_diff: true`; the frontend
(`loadSnapshot()` in `asx.html`) branches on that flag to render `±N pts`
instead of `±N.NN%` and skips the percentage line entirely. Separately
confirmed live 2026-08-21 that Yahoo Finance does **not** carry a usable ASX
24 SPI 200 futures quote at all: tried every `AP<2-digit-year><month-code>.AX`
combination (H/M/U/Z, 2017-2026) and the only one that resolves is
`AP17H.AX`, an expired March-2017 contract frozen at its final settlement
price — not a live feed. Barchart remains the only viable source.

**Yield curve** (`markets.py::get_yield_curve`) — FRED's `fredgraph.csv`
per-maturity series (`DGS1MO` … `DGS30`), no API key, trimmed to ~2 years via
`cosd=` so each fetch doesn't pull the full history back to 1962. 6-hour
cache (FRED only publishes once/day anyway). Also returns the same curve from
~1 year ago for the "compare" overlay — the whole point of a yield curve is
its *shape* and how that's shifted, not any single maturity's level.

**AI signals** (`asx_signals.py` + `grok_oauth.py`) — classifies *every*
announcement from a top-500-universe company (~161/day as of 2026-08-21).
**Score is a single 0-100 bullish/bearish scale (0=Sell, 50=neutral, 100=Buy),
not an independent conviction axis** — the Signal label is *derived* from the
score band (`_signal_for_score()`: <=35 Sell, >=65 Buy, else Hold), never
asked of the model separately. The first version had the LLM emit signal and
score independently, which produced "Hold, 90" (read as "very confident
nothing's happening" when the user expected "high score = bullish" on one
scale) — corrected 2026-08-21. If this ever needs re-tuning, the whole cached
`signals` table should be wiped and reclassified (see `~/.tradingagents/asx_signals.db`),
since old rows carry whatever the prior semantics were and mixing scales
silently is worse than a one-time reclassification cost (~161 calls, trivial
$).
**This was originally scoped to price-sensitive/halt only** (~64/day) but
that missed too much: debt issuance, buy-backs, and director's dealings from
large companies aren't flagged price-sensitive by ASX but the user wanted
them graded too — widened 2026-08-21 after they noticed EVN/CBA/ALK/BXB
filings going ungraded. If asked to narrow it back for cost/noise reasons,
that's a real option (see `signal_worthy()` in `asx_feed.py`), but don't
narrow it unilaterally — this was a deliberate correction, not the original
design intent. Cached by asxbrief's `fingerprint`, so a given announcement is
classified exactly once. **Primary trigger (added 2026-08-21): the asxbrief
collector POSTs `/api/asx/signals/refresh?limit=60` itself immediately after
any poll that finds new announcements** (its `[integrations]
signals_webhook_url` config — see asxbrief-collector skill), so a new
announcement typically gets a signal within seconds, not up to 10 minutes
later. `asx-signals-refresh.timer` (every 10 min) still runs too, now purely
as a backstop in case the webhook call fails (e.g. this service was mid-restart
when asxbrief polled). The page's "⚡ Classify now" button hits the same
endpoint manually. If new announcements are ever going ungraded for more than
a few seconds, check the webhook first (`journalctl -u asxbrief | grep
"signals webhook"`), not just the timer.

Auth is **not** an API key — it's a SuperGrok/X OAuth credential
(`~/.grok-cli/auth.json`, refreshed automatically every ~6h by
`grok_oauth.py`). If it ever needs re-doing (credential revoked, moved to a
new box), the one-time interactive login is `scripts/grok_login.py` — needs a
human + browser, cannot be done headlessly. Full protocol details (token
formats, the 403-quota-vs-403-auth-error distinction, model pricing) are in
the reference memory `reference-grok-oauth-handoff` — read that before
touching `grok_oauth.py`, don't rediscover it from scratch.

**Known separate/pre-existing gap, not part of this feature**: `factory.py`
in `tradingagents/llm_clients/` references an `xai_grok_client.py` /
`hermes_claude_client.py` that don't exist — the `xai-grok` provider (default
for the main "Run Analysis" feature) is currently broken. `grok_oauth.py` is
a deliberately separate, minimal path built to avoid touching that shared
factory. Fixing the factory-level client is out of scope for the dashboard
unless asked.

## Systemd units involved
- `tradingagents.service` — the web app itself (port 7777)
- `asxbrief.service` — the announcement collector (separate project/repo)
- `asxbrief-universe-refresh.timer` — monthly, rebuilds the top-500 universe
- `asx-signals-refresh.timer` — every 10 min, classification **backstop**
  (primary trigger is asxbrief's post-poll webhook, see AI signals above)
- `asx-spi200-refresh.timer` — daily 08:05 Australia/Sydney, SPI 200 scrape
- `asx-swing-propose.timer` — weekdays 08:20 Australia/Sydney, swing proposals
- `asx-swing-sync.timer` — every 15 min, paper-IBKR fill-status sync
- `ibgateway.service` — headless paper IB Gateway (127.0.0.1:4002),
  `ReadOnlyApi=no` since 2026-08-21 (needed for Swing to place real orders)

## Page conventions to follow when adding panels
- GitHub-dark palette from `performance.html`, not `home.html`'s purple one
- Inline SVG for charts (see `renderFeedTable`/`loadYieldCurve` for the
  pattern: `niceTicks()` for axis ticks, `.chart-tooltip` div positioned on
  `mousemove`) — no Chart.js or other CDN dependency, matches how every other
  chart in this app is built
- Sortable tables: `sortBy(key)` + a `SORT_EXTRACTORS` map + `.sortable`/
  `.sorted` th classes — see the announcement feed table for the reference
  implementation
- Nav link added to *every* static page's header, not just this one (home,
  index, analysis, reports, performance, sync) — grep for `href="/asx"` if
  adding a new top-level page, to find all the places a nav link needs adding
