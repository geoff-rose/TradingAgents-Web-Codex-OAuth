#!/usr/bin/env bash
# Sync generated reports from this machine (backend) to the web server (frontend).
# Run once a week after the batch completes.
#
# Usage: ./scripts/sync_reports.sh

set -euo pipefail

REMOTE_HOST="23.95.245.174"
REMOTE_USER="root"
PEM_FILE="/home/geoff/codex/racknerd2gb.pem"
LOCAL_LOGS="$HOME/.tradingagents/logs/"
REMOTE_LOGS="$REMOTE_USER@$REMOTE_HOST:/root/.tradingagents/logs/"

echo "========================================"
echo "  TradingAgents — Report Sync"
echo "========================================"
echo "  From: $LOCAL_LOGS"
echo "  To:   $REMOTE_LOGS"
echo ""

START=$(date +%s)

rsync \
  --archive \
  --checksum \
  --human-readable \
  --stats \
  --exclude="*.tmp" \
  -e "ssh -i $PEM_FILE -o StrictHostKeyChecking=no" \
  "$LOCAL_LOGS" \
  "$REMOTE_LOGS"

END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "Done in ${ELAPSED}s — reports available at http://$REMOTE_HOST:7777/reports"
