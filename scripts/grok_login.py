#!/usr/bin/env python3
"""SuperGrok OAuth login helper.

Performs the PKCE authorization-code flow against auth.x.ai and writes
tokens to ~/.grok-cli/auth.json in the same format grok-cli uses.

Starts a local HTTP listener on port 56121 to catch the OAuth callback.
When run via VS Code Remote SSH, VS Code auto-forwards that port so your
Windows browser can reach http://localhost:56121/callback.

Usage:
    python3 scripts/grok_login.py

Requirements: only the standard library + httpx (already in .venv).
"""

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

AUTH_FILE = Path(os.environ.get("GROK_CLI_AUTH_FILE", "~/.grok-cli/auth.json")).expanduser()

CLIENT_ID    = "b1a00492-073a-47ea-816f-4c329264a828"
AUTHORIZE_URL = "https://auth.x.ai/oauth2/authorize"
TOKEN_URL    = "https://auth.x.ai/oauth2/token"
BASE_URL     = "https://api.x.ai/v1"
SCOPE        = "openid profile email offline_access grok-cli:access api:access"
CALLBACK_PORT = 56121
REDIRECT_URI  = f"http://127.0.0.1:{CALLBACK_PORT}/callback"


def _generate_pkce():
    verifier  = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    digest    = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _build_authorize_url(challenge: str, state: str, nonce: str) -> str:
    params = {
        "response_type":         "code",
        "client_id":             CLIENT_ID,
        "redirect_uri":          REDIRECT_URI,
        "scope":                 SCOPE,
        "code_challenge":        challenge,
        "code_challenge_method": "S256",
        "state":                 state,
        "nonce":                 nonce,
        "plan":                  "generic",
        "referrer":              "hermes-agent",
    }
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)


def _exchange_code(code: str, verifier: str) -> dict:
    import httpx

    resp = httpx.post(
        TOKEN_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept":       "application/json",
        },
        data={
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  REDIRECT_URI,
            "client_id":     CLIENT_ID,
            "code_verifier": verifier,
        },
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


def _wait_for_callback(state: str) -> str:
    """Spin up a local HTTP server and block until the OAuth callback arrives.

    Returns the authorization code.
    """
    result: dict = {}
    ready  = threading.Event()
    done   = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # suppress access log

        def do_GET(self):
            if not self.path.startswith("/callback"):
                self.send_response(404)
                self.end_headers()
                return

            qs   = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            code = qs.get("code", [None])[0]
            err  = qs.get("error", [None])[0]

            if err:
                body = f"<h1>Authorisation failed</h1><p>{err}</p>".encode()
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                result["error"] = err
                done.set()
                return

            if not code:
                body = b"<h1>No code received.</h1>"
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = b"<h1>Authorised! You can close this tab.</h1>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            result["code"] = code
            done.set()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "https://auth.x.ai")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.end_headers()

    server = http.server.HTTPServer(("127.0.0.1", CALLBACK_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Block until callback arrives (or user ctrl-c)
    done.wait()
    server.shutdown()

    if "error" in result:
        raise RuntimeError(f"OAuth error: {result['error']}")
    return result["code"]


def main():
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(16)
    nonce = secrets.token_urlsafe(16)

    url = _build_authorize_url(challenge, state, nonce)

    print("\n=== SuperGrok OAuth Login ===\n")
    print(f"Listening for callback on http://127.0.0.1:{CALLBACK_PORT}/callback")
    print("(VS Code Remote SSH will auto-forward this port to your Windows browser)\n")
    print("Open this URL in your browser (sign in with X / SuperGrok account):\n")
    print(f"  {url}\n")
    print("Waiting for authorisation…")

    try:
        code = _wait_for_callback(state)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    print("Callback received. Exchanging code for tokens…")
    try:
        tokens = _exchange_code(code, verifier)
    except Exception as exc:
        print(f"ERROR: token exchange failed: {exc}")
        raise SystemExit(1)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    auth_state = {
        "provider":        "xai-oauth",
        "auth_mode":       "loopback",
        "base_url":        BASE_URL,
        "redirect_uri":    REDIRECT_URI,
        "last_refresh":    now,
        "last_auth_error": None,
        "tokens": {
            "access_token":  tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "id_token":      tokens.get("id_token"),
            "expires_in":    tokens.get("expires_in"),
            "token_type":    tokens.get("token_type", "Bearer"),
        },
        "discovery": {
            "authorization_endpoint": AUTHORIZE_URL,
            "token_endpoint":         TOKEN_URL,
        },
    }

    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps(auth_state, indent=2))
    AUTH_FILE.chmod(0o600)

    print(f"\nTokens saved to {AUTH_FILE}")
    print("SuperGrok OAuth is ready — select 'SuperGrok OAuth' in the TradingAgents UI.")


if __name__ == "__main__":
    main()
