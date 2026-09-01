#!/usr/bin/env python3
"""One-time ChatGPT/Codex OAuth login for the ASX announcement classifier.

Replaces the SuperGrok login (`scripts/grok_login.py`) for this purpose --
switched 2026-08-23 when the user's xAI credits ran out. The classifier picks
its provider via `ASX_SIGNALS_PROVIDER` (default `codex`), so both logins can
coexist and either can back the classifier.

Two ways in:

    # 1. Reuse the openai-codex credential Hermes already holds (no browser)
    uv run python scripts/codex_login.py --from-hermes

    # 2. Fresh interactive sign-in (needs a human and a browser)
    uv run python scripts/codex_login.py

Over plain SSH, tunnel the callback port first:
    ssh -L 1455:127.0.0.1:1455 <host>
then run this in that session and open the printed URL on your own machine.

Writes to ~/.tradingagents/codex_auth.json (mode 0600), or wherever
TRADINGAGENTS_CODEX_AUTH_FILE points. `tradingagents/codex_oauth.py` reads it.

**Why --from-hermes copies rather than shares**: the token endpoint rotates
refresh tokens, so a refresh through a shared file would invalidate the other
tool's credential. Copying keeps them independent from that point on. Note the
copy still uses the SAME refresh token initially, so the first refresh on
either side may invalidate the other -- if Hermes matters to you, do a fresh
interactive login instead.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import httpx

AUTH_FILE = Path(os.environ.get(
    "TRADINGAGENTS_CODEX_AUTH_FILE", "~/.tradingagents/codex_auth.json",
)).expanduser()
HERMES_AUTH = Path("~/.hermes/auth.json").expanduser()

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
BASE_URL = "https://chatgpt.com/backend-api/codex"
SCOPE = "openid profile email offline_access"
CALLBACK_PORT = 1455
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/auth/callback"


def _write(tokens: dict, source: str) -> None:
    """Written in the credential_pool shape the existing Codex client reads
    (`llm_clients/openai_codex_client.py` looks for a pool entry with a
    refresh_token, then falls back to providers.openai-codex.tokens)."""
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    AUTH_FILE.write_text(json.dumps({
        "version": 1,
        "active_provider": "openai-codex",
        "updated_at": now,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": tokens.get("access_token"),
                    "refresh_token": tokens.get("refresh_token"),
                    "id_token": tokens.get("id_token"),
                    "account_id": tokens.get("account_id"),
                },
                "last_refresh": now,
                "auth_mode": "oauth",
            }
        },
        "credential_pool": {
            "openai-codex": [{
                "id": "ta-001",
                "label": "tradingagents",
                "auth_type": "oauth",
                "priority": 0,
                "source": source,
                "access_token": tokens.get("access_token"),
                "refresh_token": tokens.get("refresh_token"),
                "base_url": BASE_URL,
                "last_refresh": now,
                "request_count": 0,
            }]
        },
    }, indent=2))
    AUTH_FILE.chmod(0o600)
    print(f"\nCredential written to {AUTH_FILE} (mode 0600)")


def seed_from_hermes() -> int:
    if not HERMES_AUTH.exists():
        print(f"No Hermes credential at {HERMES_AUTH}", file=sys.stderr)
        return 1
    data = json.loads(HERMES_AUTH.read_text())
    pool = data.get("credential_pool", {}).get("openai-codex", []) or []
    entry = next((e for e in pool if e.get("refresh_token")), None)
    tokens = data.get("providers", {}).get("openai-codex", {}).get("tokens", {}) or {}
    src = entry or tokens
    if not src.get("refresh_token"):
        print("Hermes has no openai-codex refresh_token to copy.", file=sys.stderr)
        return 1
    _write({
        "access_token": src.get("access_token"),
        "refresh_token": src.get("refresh_token"),
        "id_token": tokens.get("id_token"),
        "account_id": tokens.get("account_id"),
    }, source="hermes-copy")
    print("Seeded from Hermes. Verify with:  uv run python scripts/codex_login.py --check")
    print("NOTE: both files currently hold the same refresh token; the first refresh")
    print("on either side may invalidate the other. Do a fresh login if Hermes matters.")
    return 0


def check() -> int:
    """Prove the credential actually works by making one real call."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tradingagents import codex_oauth
    if not codex_oauth.is_configured():
        print(f"Not configured: no credential at {codex_oauth.AUTH_FILE}", file=sys.stderr)
        return 1
    try:
        out = codex_oauth.ask("Reply with exactly: OK", system="You are a terse test probe.")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"OK -- model replied: {out.strip()[:120]!r}")
    return 0


PKCE_FILE = AUTH_FILE.parent / ".codex_pkce.json"


def manual_start() -> int:
    """Print the authorize URL and stash the PKCE verifier for the second step.

    **Why a two-step manual flow exists**: the loopback flow needs the browser
    to reach a listener on THIS host's port 1455, which on a headless VPS means
    an SSH tunnel. Splitting it removes that requirement -- the redirect simply
    fails to load in the browser, but the address bar still carries
    `?code=...`, which is all the exchange needs. Same PKCE, same security
    properties; only the delivery of the code differs.
    """
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    PKCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PKCE_FILE.write_text(json.dumps({"verifier": verifier, "state": state}))
    PKCE_FILE.chmod(0o600)
    url = AUTHORIZE_URL + "?" + urllib.parse.urlencode({
        "response_type": "code", "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI, "scope": SCOPE,
        "code_challenge": challenge, "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    })
    print("\n1. Open this URL and sign in with the ChatGPT account you want to use:\n")
    print(url)
    print("\n2. The browser will end on a page that fails to load "
          f"(localhost:{CALLBACK_PORT}). That is expected.")
    print("   Copy the FULL address from the address bar - it contains ?code=...")
    print("\n3. Finish with:")
    print("     uv run python scripts/codex_login.py --manual-finish '<paste url or code>'")
    return 0


def manual_finish(code_or_url: str) -> int:
    if not PKCE_FILE.exists():
        print("No pending login. Run --manual-start first.", file=sys.stderr)
        return 1
    saved = json.loads(PKCE_FILE.read_text())

    code = code_or_url.strip()
    if "://" in code or code.startswith("?") or "code=" in code:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(code).query or code.lstrip("?"))
        got_state = (qs.get("state") or [None])[0]
        code = (qs.get("code") or [None])[0]
        if not code:
            print("No ?code= found in that URL.", file=sys.stderr)
            return 1
        # State is checked when present -- it is the CSRF guard for this flow,
        # and skipping it because "the user pasted it themselves" would defeat
        # the point of having generated one.
        if got_state and got_state != saved.get("state"):
            print("State mismatch - this code came from a different login attempt. "
                  "Re-run --manual-start.", file=sys.stderr)
            return 1

    resp = httpx.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"},
        data={"grant_type": "authorization_code", "code": code,
              "redirect_uri": REDIRECT_URI, "client_id": CLIENT_ID,
              "code_verifier": saved["verifier"]},
        timeout=30.0,
    )
    if resp.status_code >= 400:
        print(f"Token exchange failed ({resp.status_code}): {resp.text[:400]}", file=sys.stderr)
        print("Authorization codes are single-use and short-lived - if this took more than a "
              "couple of minutes, run --manual-start again.", file=sys.stderr)
        return 1
    tokens = resp.json()
    _write(tokens, source="manual-login")
    PKCE_FILE.unlink(missing_ok=True)
    print("Now verify with:  uv run python scripts/codex_login.py --check")
    return 0


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def _wait_for_callback() -> str:
    result, done = {}, threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if not parsed.path.startswith("/auth/callback"):
                self.send_response(404); self.end_headers(); return
            qs = urllib.parse.parse_qs(parsed.query)
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


def interactive() -> int:
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    url = AUTHORIZE_URL + "?" + urllib.parse.urlencode({
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    })

    print(f"\nListening on http://localhost:{CALLBACK_PORT}/auth/callback")
    print(f"Headless box? First run:  ssh -L {CALLBACK_PORT}:127.0.0.1:{CALLBACK_PORT} <host>")
    print("\nOpen this URL and sign in with the ChatGPT account whose plan you want to use:\n")
    print(f"  {url}\n")
    print("Waiting for authorisation...")

    code = _wait_for_callback()
    print("Callback received. Exchanging code for tokens...")

    resp = httpx.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"},
        data={"grant_type": "authorization_code", "code": code,
              "redirect_uri": REDIRECT_URI, "client_id": CLIENT_ID,
              "code_verifier": verifier},
        timeout=30.0,
    )
    if resp.status_code >= 400:
        print(f"Token exchange failed ({resp.status_code}): {resp.text[:400]}", file=sys.stderr)
        return 1
    tokens = resp.json()
    _write(tokens, source="loopback-login")
    print("Now verify with:  uv run python scripts/codex_login.py --check")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-hermes", action="store_true",
                    help="copy the openai-codex credential Hermes already holds (no browser)")
    ap.add_argument("--check", action="store_true",
                    help="make one real call to prove the stored credential works")
    ap.add_argument("--manual-start", action="store_true",
                    help="print an authorize URL; no SSH tunnel needed (headless-friendly)")
    ap.add_argument("--manual-finish", metavar="CODE_OR_URL",
                    help="finish a --manual-start login with the pasted code or redirect URL")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.manual_start:
        return manual_start()
    if args.manual_finish:
        return manual_finish(args.manual_finish)
    if args.from_hermes:
        return seed_from_hermes()
    return interactive()


if __name__ == "__main__":
    raise SystemExit(main())
