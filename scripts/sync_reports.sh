#!/usr/bin/env bash
# Sync generated reports from this machine (backend) to the web server (frontend).
# Run once a week after the batch completes.
#
# Usage: ./scripts/sync_reports.sh

set -euo pipefail

: "${TRADINGAGENTS_SYNC_HOST:?Set TRADINGAGENTS_SYNC_HOST, e.g. user@your-server.example.com}"
: "${TRADINGAGENTS_SYNC_PEM:?Set TRADINGAGENTS_SYNC_PEM to the path of your SSH private key}"
REMOTE_HOST="${TRADINGAGENTS_SYNC_HOST#*@}"
PEM_FILE="$TRADINGAGENTS_SYNC_PEM"
LOCAL_LOGS="$HOME/.tradingagents/logs/"
REMOTE_LOGS="$TRADINGAGENTS_SYNC_HOST:${TRADINGAGENTS_SYNC_REMOTE_PATH:-/root/.tradingagents/logs/}"

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
  --exclude="*.bak" \
  -e "ssh -i $PEM_FILE -o StrictHostKeyChecking=no" \
  "$LOCAL_LOGS" \
  "$REMOTE_LOGS"

END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "Done in ${ELAPSED}s — reports available at http://$REMOTE_HOST:7777/reports"
