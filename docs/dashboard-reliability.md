# Dashboard reliability changes — 2026-09-08

Announcement scores require usable extracted document text in both production
and experimental scoring. Missing documents stay unscored, with up to four
fetch attempts and exponential backoff starting at 15 minutes. Scores record
the SHA-256 of the text supplied to the model and whether it was truncated.
The feed distinguishes awaiting, unavailable, read and excerpt states.
Extraction still has a 12,000-character limit and does not perform OCR.

Legacy production scores without document provenance are preserved but hidden
from current signals and excluded from netting. Recent eligible announcements
can be reassessed by the existing classifier jobs. Prior scores used in context
must also be document-backed and available before the announcement timestamp.
A ticker summary remains unscored while a material document is missing.

Scanner sessions use Australia/Sydney daylight saving. Intraday relative
volume compares completed hourly windows with matching hours in up to 20
previous sessions (minimum 10), allowing 20 minutes for delayed quotes. The
ratio is blank before a comparable window exists. Outside trading, the
baseline uses up to 50 previous daily sessions. This remains delayed data;
session boundaries use weekdays, without an exchange holiday calendar.

Each fresh mover observation records its price, score, score time, model and
prompt version. Subsequent scans and annotation backfills cannot overwrite
that initial score. Older mover rows remain retrospective observations.

`document_signal_outcomes` evaluates the first recorded document-backed ticker
score from the first actual opening bar after scoring, excluding incomplete
daily bars. Classifier IC uses only this table. Existing `signal_outcomes`
rows remain separate retrospective research data; unfinished horizons continue
to mature without replacing their scores. Old A/B results remain historical,
and new A/B scores record document hashes. Retrospective prompt comparisons
are not evidence of returns available after a real-time score.

Swing approval claims each proposal once and persists broker IDs before
submission. Both strategies use staged brackets with linked GTC exits;
range-model entry orders remain DAY orders. Target refresh modifies the
existing price. Reconciliation records cumulative fills across replacement
orders, confirms cancellations before closure, retains uncertain exposure as
active, and restricts connections to verified paper accounts. The web service
checks active orders every five seconds between broker calls. Partial-entry
cancellation and resizing may span reconciliation passes. Broker tests use
mocks; no test trades were placed.

Authentication requires `TRADINGAGENTS_WEB_PASSWORD` and a protected random
`TRADINGAGENTS_SESSION_SECRET` of at least 32 characters. There are no fallback
credentials. Cookies use Secure on HTTPS automatically; deployments may set
`TRADINGAGENTS_COOKIE_SECURE=true` explicitly. Rotating the secret invalidates
existing sessions. Static source files are not served.
