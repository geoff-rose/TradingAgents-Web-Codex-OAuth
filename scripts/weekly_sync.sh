#!/usr/bin/env bash
# Syncs all local reports to prod and triggers DB ingest on the remote server.
# Runs Sunday 10pm via cron.

set -euo pipefail

PEM="/home/geoff/codex/racknerd2gb.pem"
REMOTE="root@23.95.245.174"
LOGS_DIR="/home/geoff/.tradingagents/logs/"

echo "========================================"
echo "  Weekly sync — $(date)"
echo "========================================"

echo ""
echo "Rsyncing reports to prod..."
rsync --archive --checksum --human-readable \
    --stats --exclude="*.tmp" --exclude="*.bak" \
    -e "ssh -i ${PEM} -o StrictHostKeyChecking=no" \
    "${LOGS_DIR}" "${REMOTE}:/root/.tradingagents/logs/"

echo ""
echo "Triggering DB ingest on prod..."
RESULT=$(ssh -i "${PEM}" -o StrictHostKeyChecking=no "${REMOTE}" \
    "curl -s -X POST http://localhost:7777/api/performance/ingest")
echo "Ingest result: ${RESULT}"

echo ""
echo "Done: $(date)"
