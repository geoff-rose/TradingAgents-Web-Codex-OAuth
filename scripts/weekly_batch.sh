#!/usr/bin/env bash
# Weekly ASX300 batch runner — cron entries:
#   0 21 * * 5  /home/geoff/codex/TradingAgents/scripts/weekly_batch.sh >> /home/geoff/.tradingagents/logs/weekly_batch.log 2>&1
#   0 21 * * 0  /home/geoff/codex/TradingAgents/scripts/weekly_batch.sh "$(date -d 'last friday' +\%Y-\%m-\%d)" >> /home/geoff/.tradingagents/logs/weekly_batch.log 2>&1
#
# Refreshes the SuperGrok OAuth token, then runs the full ASX300 analysis.
# batch_asx200.py skips tickers that already have a report for the given
# date, so the Sunday retry only re-attempts whatever failed on Friday
# (e.g. an xAI spending-limit block) — pass the same date explicitly so
# the retry doesn't get treated as a brand-new batch.

set -euo pipefail

cd /home/geoff/codex/TradingAgents

DATE="${1:-$(date +%Y-%m-%d)}"
echo "========================================"
echo "  Weekly batch — ${DATE}"
echo "  Started: $(date)"
echo "========================================"

# ── 1. Refresh the Grok token ─────────────────────────────────────────────────
echo ""
echo "Refreshing SuperGrok token..."
REFRESH_RESULT=$(/home/geoff/.local/bin/uv run python3 -c "
import sys, json, base64, time
sys.path.insert(0, '.')
from tradingagents.llm_clients.xai_grok_client import _refresh_grok_token, _read_grok_token, _token_expiry

refreshed = _refresh_grok_token()
token = _read_grok_token()
if not token:
    print('FAIL: no token after refresh')
    sys.exit(1)

exp = _token_expiry()
if exp and exp < time.time():
    print('FAIL: token still expired after refresh attempt')
    sys.exit(1)

import datetime
exp_str = datetime.datetime.fromtimestamp(exp).strftime('%Y-%m-%d %H:%M') if exp else 'unknown'
print(f'OK: token valid until {exp_str}')
" 2>&1)

echo "${REFRESH_RESULT}"

if echo "${REFRESH_RESULT}" | grep -q "^FAIL"; then
    echo ""
    echo "ERROR: Could not obtain a valid SuperGrok token."
    echo "       Run 'uv run python scripts/grok_login.py' to re-authenticate."
    exit 1
fi

# ── 2. Run the batch ──────────────────────────────────────────────────────────
echo ""
echo "Starting ASX300 batch analysis..."
echo ""

/home/geoff/.local/bin/uv run python scripts/batch_asx200.py \
    --date "${DATE}" \
    --workers 2 \
    --provider xai-grok \
    --deep-model grok-4-fast-non-reasoning \
    --quick-model grok-4-fast-non-reasoning \
    --depth 1

echo ""
echo "Batch complete: $(date)"
