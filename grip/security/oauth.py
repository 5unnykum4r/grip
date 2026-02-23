"""OAuth 2.0 Authorization Code flow with PKCE for MCP servers.

Starts a temporary local HTTP server on localhost:{redirect_port},
opens the browser to the authorization URL, waits for the callback
with the authorization code, exchanges it for tokens, and returns
the result.

Usage:
    flow = OAuthFlow(oauth_config, server_name="todoist")
    token = await flow.execute()
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import html as html_mod
import secrets
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from loguru import logger

from grip.config.schema import OAuthConfig
from grip.security.token_store import StoredToken

_SUCCESS_HTML = """<!DOCTYPE html>
<html>
<head><title>Login Successful</title></head>
<body style="font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #f8f9fa;">
<div style="text-align: center; padding: 2rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
<h1 style="color: #22c55e; margin-bottom: 0.5rem;">Login Successful!</h1>
<p style="color: #64748b;">You can close this tab and return to grip.</p>
</div>
</body>
</html>"""

_ERROR_HTML = """<!DOCTYPE html>
<html>
<head><title>Login Failed</title></head>
<body style="font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #f8f9fa;">
<div style="text-align: center; padding: 2rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
<h1 style="color: #ef4444; margin-bottom: 0.5rem;">Login Failed</h1>
<p style="color: #64748b;">{error}</p>
</div>
</body>
</html>"""


class OAuthFlowError(Exception):
    """Raised when the OAuth flow fails at any stage."""


class OAuthFlow:
    """Executes a browser-based OAuth 2.0 authorization code flow with PKCE."""

    def __init__(
        self,
        oauth_config: OAuthConfig,
        server_name: str,
        timeout: int = 120,
    ) -> None:
        self._config = oauth_config
        self._server_name = server_name
        self._timeout = timeout
        self._code_verifier = secrets.token_urlsafe(64)
        self._state = secrets.token_urlsafe(32)

    @property
    def _code_challenge(self) -> str:
        """Generate S256 PKCE code challenge from verifier."""
        digest = hashlib.sha256(self._code_verifier.encode("ascii")).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self._config.redirect_port}/callback"

    def build_auth_url(self) -> str:
        """Build the full authorization URL with PKCE and state parameters."""
        params: dict[str, str] = {
            "client_id": self._config.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "state": self._state,
            "code_challenge": self._code_challenge,
            "code_challenge_method": "S256",
        }
        if self._config.scopes:
            params["scope"] = " ".join(self._config.scopes)
        return f"{self._config.auth_url}?{urlencode(params)}"

    async def execute(self) -> StoredToken:
        """Run the full OAuth flow: open browser, wait for callback, exchange code.

        Raises OAuthFlowError on timeout, state mismatch, or token exchange failure.
        """
        import webbrowser

        code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        server = await self._start_callback_server(code_future)
        try:
            auth_url = self.build_auth_url()
            logger.info("Opening browser for '{}' OAuth login", self._server_name)
            webbrowser.open(auth_url)

            auth_code = await asyncio.wait_for(code_future, timeout=self._timeout)
            return await self._exchange_code(auth_code)
        except TimeoutError as exc:
            raise OAuthFlowError(
                f"OAuth login timed out after {self._timeout}s for '{self._server_name}'"
            ) from exc
        finally:
            server.close()
            await server.wait_closed()

    async def refresh(self, refresh_token: str) -> StoredToken:
        """Use a refresh token to obtain a new access token."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self._config.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self._config.client_id,
                },
            )
            if response.status_code != 200:
                raise OAuthFlowError(
                    f"Token refresh failed for '{self._server_name}': "
                    f"{response.status_code} {response.text[:200]}"
                )
            return self._parse_token_response(response.json())

    async def _start_callback_server(
        self, code_future: asyncio.Future[str]
    ) -> asyncio.Server:
        """Start a minimal HTTP server on localhost to receive the OAuth callback."""

        async def handle_connection(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                request_line = await asyncio.wait_for(reader.readline(), timeout=10)
                request_str = request_line.decode("utf-8", errors="replace")

                # Read remaining headers (discard them)
                while True:
                    line = await asyncio.wait_for(reader.readline(), timeout=5)
                    if line in (b"\r\n", b"\n", b""):
                        break

                parsed = urlparse(request_str.split(" ")[1] if " " in request_str else "")
                params = parse_qs(parsed.query)

                error = params.get("error", [None])[0]
                if error:
                    html = _ERROR_HTML.format(error=html_mod.escape(error))
                    self._send_http_response(writer, 400, html)
                    if not code_future.done():
                        code_future.set_exception(
                            OAuthFlowError(f"OAuth provider returned error: {error}")
                        )
                    return

                state = params.get("state", [None])[0]
                code = params.get("code", [None])[0]

                if state != self._state:
                    html = _ERROR_HTML.format(error="State mismatch - possible CSRF attack.")
                    self._send_http_response(writer, 400, html)
                    if not code_future.done():
                        code_future.set_exception(
                            OAuthFlowError("OAuth state mismatch")
                        )
                    return

                if not code:
                    html = _ERROR_HTML.format(error="No authorization code received.")
                    self._send_http_response(writer, 400, html)
                    if not code_future.done():
                        code_future.set_exception(
                            OAuthFlowError("No authorization code in callback")
                        )
                    return

                self._send_http_response(writer, 200, _SUCCESS_HTML)
                if not code_future.done():
                    code_future.set_result(code)

            except Exception as exc:
                logger.debug("OAuth callback handler error: {}", exc)
                if not code_future.done():
                    code_future.set_exception(
                        OAuthFlowError(f"Callback handler error: {exc}")
                    )
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

        server = await asyncio.start_server(
            handle_connection,
            host="127.0.0.1",
            port=self._config.redirect_port,
        )
        logger.debug(
            "OAuth callback server listening on localhost:{}",
            self._config.redirect_port,
        )
        return server

    @staticmethod
    def _send_http_response(
        writer: asyncio.StreamWriter, status: int, html: str
    ) -> None:
        """Write a minimal HTTP response."""
        status_text = "OK" if status == 200 else "Bad Request"
        response = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(html.encode())}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{html}"
        )
        writer.write(response.encode("utf-8"))

    async def _exchange_code(self, auth_code: str) -> StoredToken:
        """Exchange the authorization code for access and refresh tokens."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self._config.token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": auth_code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self._config.client_id,
                    "code_verifier": self._code_verifier,
                },
            )
            if response.status_code != 200:
                raise OAuthFlowError(
                    f"Token exchange failed for '{self._server_name}': "
                    f"{response.status_code} {response.text[:200]}"
                )
            return self._parse_token_response(response.json())

    def _parse_token_response(self, data: dict[str, Any]) -> StoredToken:
        """Parse the OAuth token response into a StoredToken."""
        expires_in = data.get("expires_in", 0)
        expires_at = time.time() + expires_in if expires_in else 0.0

        return StoredToken(
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            expires_at=expires_at,
            token_type=data.get("token_type", "Bearer"),
            scopes=data.get("scope", "").split() if data.get("scope") else [],
        )
