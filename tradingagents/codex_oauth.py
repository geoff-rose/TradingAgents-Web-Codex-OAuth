"""Codex (ChatGPT subscription) access for the ASX announcement classifier.

**Why this exists**: the classifier ran on `grok_oauth.py` until the user's xAI
credits ran out (2026-08-23). This is the same narrow interface backed by a
ChatGPT/Codex subscription login instead, so `asx_signals.py` can switch
providers without knowing anything about either auth flow.

**Deliberately mirrors `grok_oauth.py`'s public surface** -- `is_configured()`,
`ask()`, `is_quota_error()`, `is_auth_error()`, and a quota/auth exception pair.
Keeping the two modules interface-compatible is what makes the provider a
one-line choice in `asx_signals`; if you add a third provider, match this
surface rather than teaching the caller a new shape.

**Auth storage**: `~/.tradingagents/codex_auth.json` (override with
`TRADINGAGENTS_CODEX_AUTH_FILE`). This is TradingAgents' own copy and is
never Hermes's `~/.hermes/auth.json` -- the token-refresh endpoint rotates
refresh tokens, so refreshing a shared file would silently invalidate the
other tool's credential. `scripts/codex_login.py --from-hermes` seeds a copy
once; after that the two are independent.

**Transport differs from Grok**: the Codex backend speaks the Responses API at
`chatgpt.com/backend-api/codex` and requires a `chatgpt-account-id` header
derived from a JWT claim inside the access token -- not a plain bearer key
against a normal `/v1` endpoint. The token plumbing is reused from
`llm_clients/openai_codex_client.py` rather than reimplemented, so there is one
place where the header and refresh rules live.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

AUTH_FILE = Path(os.environ.get(
    "TRADINGAGENTS_CODEX_AUTH_FILE", "~/.tradingagents/codex_auth.json",
)).expanduser()

BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_MODEL = os.environ.get("TRADINGAGENTS_CODEX_MODEL", "gpt-5.4")

# The Codex backend surfaces plan/rate limits and auth failures through
# message text rather than distinct status codes, same situation as xAI.
_QUOTA_MARKERS = (
    "rate limit", "rate_limit", "quota", "usage limit", "insufficient_quota",
    "exceeded your current", "plan limit", "too many requests",
)
_AUTH_MARKERS = (
    "token_expired", "invalid_token", "unauthenticated", "unauthorized",
    "invalid_grant", "authentication",
)

_lock = threading.Lock()


class CodexQuotaExceeded(RuntimeError):
    """Plan/rate limit hit -- retrying immediately will not help."""


class CodexAuthError(RuntimeError):
    """Token expired/invalid and could not be refreshed -- needs a fresh login."""


def is_quota_error(message: str) -> bool:
    return any(m in message.lower() for m in _QUOTA_MARKERS)


def is_auth_error(message: str) -> bool:
    return not is_quota_error(message) and any(m in message.lower() for m in _AUTH_MARKERS)


def is_configured() -> bool:
    return AUTH_FILE.exists() or bool(os.environ.get("TRADINGAGENTS_CODEX_ACCESS_TOKEN"))


def _token_helpers():
    """Import the token read/refresh/header helpers from the existing Codex
    client. They are underscore-private there, but duplicating JWT decoding,
    account-id extraction and refresh-token rotation would create two places
    for the same subtle logic to drift -- worse than reaching for the private
    names. If that module is ever refactored, promote these instead of copying.
    """
    from tradingagents.llm_clients.openai_codex_client import (
        _codex_headers, _read_hermes_codex_token, _refresh_codex_token,
    )
    return _read_hermes_codex_token, _refresh_codex_token, _codex_headers


def ensure_fresh_token() -> str:
    read_token, refresh_token, _ = _token_helpers()
    token = read_token()
    if token:
        return token
    with _lock:
        token = read_token()
        if token:
            return token
        token = refresh_token()
        if token:
            return token
    raise CodexAuthError(
        f"no usable Codex credential at {AUTH_FILE} -- run "
        "`uv run python scripts/codex_login.py` once, interactively, to sign in"
    )


def ask(prompt: str, *, system: str = "", model: str = DEFAULT_MODEL,
        timeout: float = 60.0) -> str:
    """One-shot plain-text completion, interface-compatible with
    `grok_oauth.ask()`.

    Uses the Responses API (not Chat Completions) because that is what the
    Codex backend exposes; `store=False` keeps these one-shot classifications
    out of the account's conversation history.
    """
    from openai import OpenAI

    read_token, refresh_token, headers_for = _token_helpers()
    token = ensure_fresh_token()

    def _call(tok: str) -> str:
        client = OpenAI(api_key=tok, base_url=BASE_URL,
                        default_headers=headers_for(tok), timeout=timeout)
        request = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "store": False,
        }
        if system:
            request["instructions"] = system

        # **Streaming is mandatory here, not a preference.** The Codex backend
        # rejects a non-streamed Responses call outright with
        # `400 {'detail': 'Stream must be set to true'}` -- which is easy to
        # misread as a malformed request. `responses.stream()` is also what
        # `llm_clients/openai_codex_client.py` uses, for the same reason.
        deltas: list[str] = []
        with client.responses.stream(**request) as stream:
            for event in stream:
                if "output_text.delta" in getattr(event, "type", ""):
                    delta = getattr(event, "delta", "")
                    if delta:
                        deltas.append(delta)
            final = stream.get_final_response()

        text = getattr(final, "output_text", None)
        if text:
            return text
        parts: list[str] = []
        for item in getattr(final, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) in {"output_text", "text"}:
                    parts.append(getattr(part, "text", "") or "")
        return "".join(parts) or "".join(deltas)

    try:
        return _call(token)
    except Exception as exc:
        msg = str(exc)
        if is_quota_error(msg):
            raise CodexQuotaExceeded(msg) from exc
        if is_auth_error(msg):
            new_token = refresh_token()
            if not new_token:
                raise CodexAuthError(
                    f"token refresh failed -- re-run scripts/codex_login.py ({msg})"
                ) from exc
            return _call(new_token)
        raise
