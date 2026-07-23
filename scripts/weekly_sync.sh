#!/usr/bin/env bash
# Syncs all local reports to prod and triggers DB ingest on the remote server.
# Runs Sunday 10pm via cron.

set -euo pipefail

: "${TRADINGAGENTS_SYNC_HOST:?Set TRADINGAGENTS_SYNC_HOST, e.g. user@your-server.example.com}"
: "${TRADINGAGENTS_SYNC_PEM:?Set TRADINGAGENTS_SYNC_PEM to the path of your SSH private key}"
PEM="$TRADINGAGENTS_SYNC_PEM"
REMOTE="$TRADINGAGENTS_SYNC_HOST"
LOGS_DIR="$HOME/.tradingagents/logs/"

echo "========================================"
echo "  Weekly sync — $(date)"
echo "========================================"

echo ""
echo "Rsyncing reports to prod..."
REMOTE_PATH="${TRADINGAGENTS_SYNC_REMOTE_PATH:-/root/.tradingagents/logs/}"
rsync --archive --checksum --human-readable \
    --stats --exclude="*.tmp" --exclude="*.bak" \
    -e "ssh -i ${PEM} -o StrictHostKeyChecking=no" \
    "${LOGS_DIR}" "${REMOTE}:${REMOTE_PATH}"

echo ""
echo "Triggering DB ingest on prod..."
RESULT=$(ssh -i "${PEM}" -o StrictHostKeyChecking=no "${REMOTE}" \
    "curl -s -X POST http://localhost:7777/api/performance/ingest")
echo "Ingest result: ${RESULT}"

echo ""
echo "Done: $(date)"
