"""
Swiggy MCP Authentication
=====================================
Handles OAuth 2.1 + PKCE login against Swiggy's MCP server.
Swiggy requires this for every external caller (same flow Claude
Desktop / Cursor use under the hood). No refresh token in v1 —
the access token is good for 5 days, then you re-run login().

Zero cost: everything below runs on http://localhost against
Swiggy's staging endpoint with no approval needed for prototyping.
You only need /access approval when you're ready to go live.

First run:
    python -m app.services.swiggy_auth
This opens a browser, you log in with phone + OTP, and the token
gets saved to .swiggy_token.json (added to .gitignore).
"""

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser

import httpx

SWIGGY_SERVER_BASE = os.getenv("SWIGGY_MCP_SERVER", "https://mcp.swiggy.com/food")
REDIRECT_URI = "http://localhost:8765/callback"
TOKEN_FILE = ".swiggy_token.json"


def _discover_oauth_endpoints() -> dict:
    """
    MCP servers advertise their OAuth endpoints via RFC 8414 metadata.
    We fetch this instead of hardcoding URLs so it stays correct even
    if Swiggy changes their auth server internals.
    """
    metadata_url = f"{SWIGGY_SERVER_BASE}/.well-known/oauth-authorization-server"
    resp = httpx.get(metadata_url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _generate_pkce_pair() -> tuple:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Catches the OAuth redirect on localhost and grabs the auth code."""
    auth_code = None
    state_received = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        _CallbackHandler.auth_code = params.get("code", [None])[0]
        _CallbackHandler.state_received = params.get("state", [None])[0]

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body><h2>Swiggy login complete.</h2>"
                          b"You can close this tab and return to your app.</body></html>")

    def log_message(self, format, *args):
        pass  # silence default request logging


def login(scopes: str = "mcp:tools mcp:resources") -> dict:
    """
    Run the full OAuth 2.1 PKCE flow. Opens a browser window for
    phone + OTP login, then exchanges the returned code for a token.
    Returns the token dict and also saves it to TOKEN_FILE.
    """
    endpoints = _discover_oauth_endpoints()
    authorize_url = endpoints["authorization_endpoint"]
    token_url = endpoints["token_endpoint"]

    verifier, challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(16)

    auth_params = {
        "response_type": "code",
        "client_id": os.getenv("SWIGGY_CLIENT_ID", "auto"),  # DCR: most frameworks auto-register
        "redirect_uri": REDIRECT_URI,
        "scope": scopes,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    full_url = f"{authorize_url}?{urllib.parse.urlencode(auth_params)}"

    # Start a tiny local server to catch the redirect
    server = http.server.HTTPServer(("localhost", 8765), _CallbackHandler)
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    print("Opening browser for Swiggy login (phone + OTP)...")
    webbrowser.open(full_url)

    # Wait for the callback (timeout after 2 minutes)
    waited = 0
    while _CallbackHandler.auth_code is None and waited < 120:
        time.sleep(0.5)
        waited += 0.5
    server_thread.join(timeout=1)

    if _CallbackHandler.auth_code is None:
        raise TimeoutError("Login timed out — no callback received within 2 minutes.")

    if _CallbackHandler.state_received != state:
        raise ValueError("OAuth state mismatch — possible CSRF, aborting.")

    # Exchange code for token
    token_resp = httpx.post(
        token_url,
        data={
            "grant_type": "authorization_code",
            "code": _CallbackHandler.auth_code,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
            "client_id": os.getenv("SWIGGY_CLIENT_ID", "auto"),
        },
        timeout=10,
    )
    token_resp.raise_for_status()
    token = token_resp.json()
    token["obtained_at"] = time.time()

    with open(TOKEN_FILE, "w") as f:
        json.dump(token, f, indent=2)

    print("Login successful. Token saved to", TOKEN_FILE)
    return token


def get_valid_token() -> str:
    """
    Returns a usable bearer token, prompting a fresh login if none
    exists or if the saved one has expired (5-day lifetime, no refresh
    grant in v1 — see Swiggy's auth docs).
    """
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            token = json.load(f)
        age_seconds = time.time() - token.get("obtained_at", 0)
        expires_in = token.get("expires_in", 5 * 24 * 3600)
        if age_seconds < expires_in - 300:  # 5 min safety margin
            return token["access_token"]
        print("Saved Swiggy token expired — re-authenticating.")

    token = login()
    return token["access_token"]


if __name__ == "__main__":
    get_valid_token()
