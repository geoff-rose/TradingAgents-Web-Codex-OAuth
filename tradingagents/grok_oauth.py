"""Grok API access via SuperGrok/X consumer OAuth (the `grok-cli` login flow),
for callers that don't want to go through the full `xai-grok` LangChain client
(which the factory in llm_clients/ references but does not yet implement --
see llm_clients/factory.py's `xai_grok_client` import).

There is no API key here -- auth is an OAuth access token that expires every
~6 hours and is refreshed from a stored refresh token. The one-time login
(`scripts/grok_login.py`) needs a human and a browser; everything here is
unattended.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx

AUTH_FILE = Path(os.environ.get("GROK_CLI_AUTH_FILE", "~/.grok-cli/auth.json")).expanduser()
CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
TOKEN_URL = "https://auth.x.ai/oauth2/token"
BASE_URL = "https://api.x.ai/v1"
KEYCHAIN_KEY = f"https://auth.x.ai::{CLIENT_ID}"

# xAI answers 403 for two unrelated conditions; the wrong response to each is
# the opposite of the other, so match on the message body, not the status code.
_QUOTA_MARKERS = ("spending-limit", "personal-team-blocked", "out of credits", "quota", "rate_limit")
_AUTH_MARKERS = ("token_expired", "bad-credentials", "unauthenticated", "invalid_token")

_lock = threading.Lock()


class GrokQuotaExceeded(RuntimeError):
    """Out of credits / needs a subscription -- retrying will not help."""


class GrokAuthError(RuntimeError):
    """Token expired/invalid and could not be refreshed -- needs a fresh login."""


def is_quota_error(message: str) -> bool:
    return any(m in message.lower() for m in _QUOTA_MARKERS)


def is_auth_error(message: str) -> bool:
    return not is_quota_error(message) and any(m in message.lower() for m in _AUTH_MARKERS)


def is_configured() -> bool:
    return AUTH_FILE.exists()


def _load() -> Optional[tuple[dict, str, dict]]:
    """Return (token_entry, format, whole_document) or None."""
    if not AUTH_FILE.exists():
        return None
    data = json.loads(AUTH_FILE.read_text())
    if "tokens" in data:
        return data["tokens"], "tokens", data
    if KEYCHAIN_KEY in data:
        return data[KEYCHAIN_KEY], "keychain", data
    return None


def _read_token() -> Optional[str]:
    got = _load()
    if not got:
        return None
    entry, fmt, _ = got
    return entry.get("key") if fmt == "keychain" else entry.get("access_token")


def _token_expiry() -> Optional[float]:
    """Seconds-since-epoch from the JWT `exp` claim -- never trust the on-disk
    `expires_at`/`last_refresh` fields, which record write time, not expiry."""
    tok = _read_token()
    if not tok or "." not in tok:
        return None
    try:
        payload = tok.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload)).get("exp")
    except Exception:
        return None


def _refresh() -> Optional[str]:
    got = _load()
    if not got:
        return None
    entry, fmt, data = got
    old_refresh = entry.get("refresh_token")
    if not old_refresh:
        return None

    r = httpx.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "refresh_token", "refresh_token": old_refresh, "client_id": CLIENT_ID},
        timeout=20.0,
    )
    r.raise_for_status()
    tok = r.json()
    new_access = tok.get("access_token")
    if not new_access:
        return None

    # The endpoint may rotate the refresh token; dropping a rotated one bricks
    # the credential and forces a fresh browser login, so always persist
    # whatever comes back, falling back to the old value if absent.
    if fmt == "keychain":
        data[KEYCHAIN_KEY]["key"] = new_access
        data[KEYCHAIN_KEY]["refresh_token"] = tok.get("refresh_token", old_refresh)
    else:
        data["tokens"]["access_token"] = new_access
        data["tokens"]["refresh_token"] = tok.get("refresh_token", old_refresh)
    AUTH_FILE.write_text(json.dumps(data, indent=2))
    AUTH_FILE.chmod(0o600)
    return new_access


def ensure_fresh_token() -> str:
    """Return a valid access token, refreshing proactively under a lock if
    fewer than 5 minutes remain (or expiry can't be read)."""
    if not AUTH_FILE.exists():
        raise GrokAuthError(
            f"no credential at {AUTH_FILE} -- run scripts/grok_login.py once, "
            "interactively, to sign in"
        )

    exp = _token_expiry()
    if exp is not None and exp - time.time() > 300:
        return _read_token()

    with _lock:
        exp = _token_expiry()
        if exp is not None and exp - time.time() > 300:
            return _read_token()
        new_token = _refresh()
        if new_token:
            return new_token
        raise GrokAuthError("token refresh failed and no valid token remains -- re-run grok_login.py")


def ask(prompt: str, *, system: str = "", model: str = "grok-4.3", timeout: float = 30.0) -> str:
    """One-shot plain-text completion. Chat Completions (not Responses) is
    fine here since we don't need the server-side web_search tool."""
    from openai import OpenAI

    token = ensure_fresh_token()
    client = OpenAI(api_key=token, base_url=BASE_URL, timeout=timeout)

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = client.chat.completions.create(model=model, messages=messages)
    except Exception as e:
        msg = str(e)
        if is_quota_error(msg):
            raise GrokQuotaExceeded(msg) from e
        if is_auth_error(msg):
            # One retry after a forced refresh -- covers the case where the
            # proactive 5-minute check above missed a token that expired
            # between calls.
            token = ensure_fresh_token()
            client = OpenAI(api_key=token, base_url=BASE_URL, timeout=timeout)
            resp = client.chat.completions.create(model=model, messages=messages)
        else:
            raise
    return resp.choices[0].message.content or ""
