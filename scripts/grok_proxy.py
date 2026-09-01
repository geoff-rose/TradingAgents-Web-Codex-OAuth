#!/usr/bin/env python3
"""
SuperGrok OAuth → OpenAI-compatible proxy.

Presents an OpenAI-format API on localhost:8765 so Home Assistant (or any
OpenAI-compatible client) can use SuperGrok without needing an API key.
Reads tokens from ~/.grok-cli/auth.json and auto-refreshes when they expire.

Endpoints mirrored:
  GET  /v1/models               → list available Grok models
  POST /v1/chat/completions     → chat (streaming + non-streaming)

Run:
  /opt/grok-proxy/.venv/bin/uvicorn grok_proxy:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("grok-proxy")

app = FastAPI(title="Grok OAuth Proxy")

_XAI_BASE      = "https://api.x.ai/v1"
_TOKEN_URL     = "https://auth.x.ai/oauth2/token"
_CLIENT_ID     = "b1a00492-073a-47ea-816f-4c329264a828"
_KEYCHAIN_KEY  = f"https://auth.x.ai::{_CLIENT_ID}"
_AUTH_FILE     = Path.home() / ".grok-cli" / "auth.json"

_DEFAULT_MODEL = "grok-4.20-non-reasoning"

# ── Token management ──────────────────────────────────────────────────────────

def _load_auth() -> tuple[dict, str]:
    """Return (entry_dict, format) where format is 'keychain' or 'tokens'."""
    data = json.loads(_AUTH_FILE.read_text())
    if _KEYCHAIN_KEY in data:
        return data[_KEYCHAIN_KEY], "keychain"
    if "tokens" in data:
        return data["tokens"], "tokens"
    raise RuntimeError("Unrecognised auth.json format")


def _read_token() -> str | None:
    try:
        entry, fmt = _load_auth()
        return entry.get("key") if fmt == "keychain" else entry.get("access_token")
    except Exception:
        return None


def _token_expired() -> bool:
    try:
        token = _read_token()
        if not token:
            return True
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp", 0)
        return exp < time.time() + 60  # refresh 60 s before expiry
    except Exception:
        return True


def _refresh_token() -> bool:
    try:
        entry, fmt = _load_auth()
        refresh = entry.get("refresh_token")
        if not refresh:
            return False
        resp = httpx.post(
            _TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": refresh,
                  "client_id": _CLIENT_ID},
            timeout=15,
        )
        if resp.status_code != 200:
            log.warning("Token refresh failed: %s %s", resp.status_code, resp.text[:200])
            return False
        new_tokens = resp.json()
        data = json.loads(_AUTH_FILE.read_text())
        if fmt == "keychain":
            data[_KEYCHAIN_KEY]["key"] = new_tokens["access_token"]
            if "refresh_token" in new_tokens:
                data[_KEYCHAIN_KEY]["refresh_token"] = new_tokens["refresh_token"]
        else:
            data["tokens"]["access_token"] = new_tokens["access_token"]
            if "refresh_token" in new_tokens:
                data["tokens"]["refresh_token"] = new_tokens["refresh_token"]
        _AUTH_FILE.write_text(json.dumps(data, indent=2))
        log.info("Token refreshed successfully")
        return True
    except Exception as exc:
        log.warning("Token refresh error: %s", exc)
        return False


def get_token() -> str:
    if _token_expired():
        log.info("Token expired, attempting refresh…")
        _refresh_token()
    token = _read_token()
    if not token:
        raise HTTPException(status_code=503, detail="No Grok token available — run grok_login.py")
    return token


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "grok-4.20-non-reasoning", "object": "model", "owned_by": "xai"},
            {"id": "grok-4.20-reasoning",     "object": "model", "owned_by": "xai"},
            {"id": "grok-4.20",               "object": "model", "owned_by": "xai"},
            {"id": "grok-4-0709",             "object": "model", "owned_by": "xai"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body: dict[str, Any] = await request.json()
    token = get_token()

    # Default model if HA doesn't specify one
    body.setdefault("model", _DEFAULT_MODEL)
    stream = body.get("stream", False)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    if stream:
        return StreamingResponse(
            _stream_upstream(body, headers),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{_XAI_BASE}/chat/completions", json=body, headers=headers)

    if resp.status_code != 200:
        log.warning("xAI error %s: %s", resp.status_code, resp.text[:300])
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return JSONResponse(content=resp.json())


async def _stream_upstream(body: dict, headers: dict) -> AsyncIterator[bytes]:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST", f"{_XAI_BASE}/chat/completions", json=body, headers=headers
        ) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                yield f"data: {json.dumps({'error': err.decode()})}\n\n".encode()
                return
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n".encode()


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    expired = _token_expired()
    return {"status": "ok", "token_valid": not expired}
