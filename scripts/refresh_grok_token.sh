#!/usr/bin/env bash
# Refreshes the SuperGrok OAuth access token using the stored refresh token.
# Run every 8 hours via cron to keep the token perpetually valid.
#
# Cron entry (already installed):
#   0 */8 * * *  /home/geoff/codex/TradingAgents/scripts/refresh_grok_token.sh

cd /home/geoff/codex/TradingAgents

/home/geoff/.local/bin/uv run python3 -c "
import sys, time, datetime
sys.path.insert(0, '.')
from tradingagents.llm_clients.xai_grok_client import _refresh_grok_token, _token_expiry

new_token = _refresh_grok_token()
exp = _token_expiry()

if new_token and exp and exp > time.time():
    exp_str = datetime.datetime.fromtimestamp(exp).strftime('%Y-%m-%d %H:%M')
    print(f'[{datetime.datetime.now().strftime(\"%Y-%m-%d %H:%M\")}] Token refreshed OK — valid until {exp_str}')
else:
    print(f'[{datetime.datetime.now().strftime(\"%Y-%m-%d %H:%M\")}] ERROR: Token refresh failed — manual re-login required (run grok_login.py)')
    sys.exit(1)
" 2>&1
