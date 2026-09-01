#!/usr/bin/env python3
"""One-time SuperGrok/X OAuth login for the ASX announcement signal classifier.

Run this once, interactively, on the machine that will run the classifier.
It needs a human and a browser -- an agent cannot complete it headlessly.

Usage:
    uv run python scripts/grok_login.py

Over plain SSH (no port auto-forwarding), first tunnel the callback port:
    ssh -L 56121:127.0.0.1:56121 <host>
then run this script in that session and open the printed URL locally.

Writes the credential to ~/.grok-cli/auth.json (mode 0600), or wherever
GROK_CLI_AUTH_FILE points. tradingagents/grok_oauth.py reads it from there.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import httpx

AUTH_FILE = Path(os.environ.get("GROK_CLI_AUTH_FILE", "~/.grok-cli/auth.json")).expanduser()
CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
AUTHORIZE_URL = "https://auth.x.ai/oauth2/authorize"
TOKEN_URL = "https://auth.x.ai/oauth2/token"
BASE_URL = "https://api.x.ai/v1"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
CALLBACK_PORT = 56121
REDIRECT_URI = f"http://127.0.0.1:{CALLBACK_PORT}/callback"


def _pkce():
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def _authorize_url(challenge, state, nonce):
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode({
        "response_type": "code", "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI, "scope": SCOPE,
        "code_challenge": challenge, "code_challenge_method": "S256",
        "state": state, "nonce": nonce,
        "plan": "generic", "referrer": "hermes-agent",
    })


def _wait_for_callback():
    result, done = {}, threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if not self.path.startswith("/callback"):
                self.send_response(404); self.end_headers(); return
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            code = qs.get("code", [None])[0]
            err = qs.get("error", [None])[0]
            body = (f"<h1>Authorisation failed</h1><p>{err}</p>".encode() if err
                    else b"<h1>Authorised. You can close this tab.</h1>" if code
                    else b"<h1>No code received.</h1>")
            self.send_response(400 if (err or not code) else 200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if err:
                result["error"] = err; done.set()
            elif code:
                result["code"] = code; done.set()

    server = http.server.HTTPServer(("127.0.0.1", CALLBACK_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    done.wait()
    server.shutdown()
    if "error" in result:
        raise RuntimeError(f"OAuth error: {result['error']}")
    return result["code"]


def main():
    verifier, challenge = _pkce()
    url = _authorize_url(challenge, secrets.token_urlsafe(16), secrets.token_urlsafe(16))

    print(f"\nListening on http://127.0.0.1:{CALLBACK_PORT}/callback")
    print("Headless box? First run:  ssh -L 56121:127.0.0.1:56121 <host>")
    print("\nOpen this URL and sign in with your X / SuperGrok account:\n")
    print(f"  {url}\n")
    print("Waiting for authorisation...")

    code = _wait_for_callback()
    print("Callback received. Exchanging code for tokens...")

    resp = httpx.post(TOKEN_URL,
                       headers={"Content-Type": "application/x-www-form-urlencoded",
                                "Accept": "application/json"},
                       data={"grant_type": "authorization_code", "code": code,
                             "redirect_uri": REDIRECT_URI, "client_id": CLIENT_ID,
                             "code_verifier": verifier},
                       timeout=20.0)
    resp.raise_for_status()
    tokens = resp.json()

    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps({
        "provider": "xai-oauth",
        "auth_mode": "loopback",
        "base_url": BASE_URL,
        "redirect_uri": REDIRECT_URI,
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "last_auth_error": None,
        "tokens": {
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "id_token": tokens.get("id_token"),
            "expires_in": tokens.get("expires_in"),
            "token_type": tokens.get("token_type", "Bearer"),
        },
        "discovery": {"authorization_endpoint": AUTHORIZE_URL, "token_endpoint": TOKEN_URL},
    }, indent=2))
    AUTH_FILE.chmod(0o600)
    print(f"\nTokens saved to {AUTH_FILE}")


if __name__ == "__main__":
    main()
