"""MCP OAuth authentication adapter.

Bridges the MCP library's built-in OAuthClientProvider with grip's
file-based token storage. Handles dynamic client registration, PKCE
authorization code flow, and token refresh automatically.

When an MCP server responds with 401, OAuthClientProvider:
  1. Discovers protected resource metadata
  2. Discovers OAuth server metadata
  3. Performs dynamic client registration (if no stored client info)
  4. Opens browser for authorization (redirect_handler)
  5. Waits for OAuth callback (callback_handler)
  6. Exchanges code for tokens and stores them

Usage:
    auth = create_mcp_oauth_auth("supabase", "https://mcp.supabase.com/mcp")
    async with streamablehttp_client(url, auth=auth) as streams:
        ...
"""

from __future__ import annotations

import asyncio
import contextlib
import html as html_mod
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from loguru import logger

_DEFAULT_CALLBACK_PORT = 18801

_CALLBACK_SUCCESS_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Login Successful</title></head>
<body style="font-family:system-ui,sans-serif;display:flex;justify-content:center;\
align-items:center;height:100vh;margin:0;background:#f8f9fa">
<div style="text-align:center;padding:2rem;background:#fff;border-radius:12px;\
box-shadow:0 2px 8px rgba(0,0,0,.1)">
<h1 style="color:#22c55e;margin-bottom:.5rem">Login Successful!</h1>
<p style="color:#64748b">You can close this tab and return to grip.</p>
</div></body></html>"""

_CALLBACK_ERROR_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Login Failed</title></head>
<body style="font-family:system-ui,sans-serif;display:flex;justify-content:center;\
align-items:center;height:100vh;margin:0;background:#f8f9fa">
<div style="text-align:center;padding:2rem;background:#fff;border-radius:12px;\
box-shadow:0 2px 8px rgba(0,0,0,.1)">
<h1 style="color:#ef4444;margin-bottom:.5rem">Login Failed</h1>
<p style="color:#64748b">{error}</p>
</div></body></html>"""


class MCPTokenStorage:
    """Implements the MCP library's TokenStorage protocol using file-backed JSON.

    Stores OAuth tokens in ~/.grip/mcp_tokens.json and dynamic client
    registration data in ~/.grip/mcp_clients.json. Both files use
    restrictive permissions (0o600) and atomic writes.
    """

    def __init__(self, server_name: str, base_dir: Path | None = None) -> None:
        self._server_name = server_name
        self._base_dir = base_dir or Path("~/.grip").expanduser()
        self._tokens_path = self._base_dir / "mcp_tokens.json"
        self._clients_path = self._base_dir / "mcp_clients.json"

    async def get_tokens(self) -> Any:
        """Load stored OAuthToken for this server, or None if not found."""
        from mcp.shared.auth import OAuthToken

        data = self._read_json(self._tokens_path)
        raw = data.get(self._server_name)
        if raw is None:
            return None
        try:
            return OAuthToken(**raw)
        except Exception:
            logger.debug("Failed to parse stored MCP token for '{}'", self._server_name)
            return None

    async def set_tokens(self, tokens: Any) -> None:
        """Persist OAuthToken for this server."""
        data = self._read_json(self._tokens_path)
        data[self._server_name] = tokens.model_dump(mode="json")
        self._write_json(self._tokens_path, data)
        logger.debug("Stored MCP OAuth token for '{}'", self._server_name)

    async def get_client_info(self) -> Any:
        """Load stored OAuthClientInformationFull for this server, or None."""
        from mcp.shared.auth import OAuthClientInformationFull

        data = self._read_json(self._clients_path)
        raw = data.get(self._server_name)
        if raw is None:
            return None
        try:
            return OAuthClientInformationFull(**raw)
        except Exception:
            logger.debug("Failed to parse stored MCP client info for '{}'", self._server_name)
            return None

    async def set_client_info(self, client_info: Any) -> None:
        """Persist OAuthClientInformationFull for this server."""
        data = self._read_json(self._clients_path)
        data[self._server_name] = client_info.model_dump(mode="json")
        self._write_json(self._clients_path, data)
        logger.debug("Stored MCP client registration for '{}'", self._server_name)

    def has_stored_token(self) -> bool:
        """Return True if a stored access token exists for this server."""
        data = self._read_json(self._tokens_path)
        return data.get(self._server_name) is not None

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read {}: {}", path, exc)
            return {}

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(path)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)


async def _open_browser(auth_url: str) -> None:
    """Open the authorization URL in the user's default browser."""
    import webbrowser

    logger.info("Opening browser for MCP OAuth login...")
    webbrowser.open(auth_url)


async def _wait_for_oauth_callback(
    port: int = _DEFAULT_CALLBACK_PORT,
    expected_state: str | None = None,
) -> tuple[str, str | None]:
    """Start a local HTTP server on localhost:{port} and wait for the OAuth redirect.

    Args:
        port: Port for the local callback server.
        expected_state: If provided, the callback will reject responses whose
            ``state`` parameter does not match (CSRF protection).

    Returns (authorization_code, state_or_none) extracted from the callback URL.
    """
    result_future: asyncio.Future[tuple[str, str | None]] = asyncio.get_running_loop().create_future()

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=10)
            request_str = request_line.decode("utf-8", errors="replace")

            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if line in (b"\r\n", b"\n", b""):
                    break

            parts = request_str.split(" ")
            parsed = urlparse(parts[1] if len(parts) > 1 else "")
            params = parse_qs(parsed.query)

            error = params.get("error", [None])[0]
            if error:
                _send_response(writer, 400, _CALLBACK_ERROR_HTML.format(error=html_mod.escape(error)))
                if not result_future.done():
                    result_future.set_exception(RuntimeError(f"OAuth error: {error}"))
                return

            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]

            if expected_state is not None and state != expected_state:
                _send_response(writer, 400, _CALLBACK_ERROR_HTML.format(error="State mismatch"))
                if not result_future.done():
                    result_future.set_exception(RuntimeError("OAuth state mismatch (possible CSRF)"))
                return

            if not code:
                _send_response(writer, 400, _CALLBACK_ERROR_HTML.format(error="No code received"))
                if not result_future.done():
                    result_future.set_exception(RuntimeError("No authorization code in callback"))
                return

            _send_response(writer, 200, _CALLBACK_SUCCESS_HTML)
            if not result_future.done():
                result_future.set_result((code, state))

        except Exception as exc:
            if not result_future.done():
                result_future.set_exception(exc)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    server = await asyncio.start_server(_handle, "127.0.0.1", port)
    logger.debug("MCP OAuth callback server listening on localhost:{}", port)

    try:
        return await asyncio.wait_for(result_future, timeout=300)
    finally:
        server.close()
        await server.wait_closed()


def _send_response(writer: asyncio.StreamWriter, status: int, html: str) -> None:
    status_text = "OK" if status == 200 else "Bad Request"
    raw = (
        f"HTTP/1.1 {status} {status_text}\r\n"
        f"Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(html.encode())}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
        f"{html}"
    )
    writer.write(raw.encode("utf-8"))


async def discover_mcp_oauth_metadata(
    server_name: str,
    server_url: str,
    redirect_uri: str,
) -> tuple[Any, Any]:
    """Perform MCP OAuth discovery and dynamic client registration.

    Follows the MCP OAuth flow: hit the server to get a 401, discover the
    protected resource metadata (RFC 9728), discover the OAuth authorization
    server metadata (RFC 8414), and register a dynamic client if one isn't
    already stored.

    Returns (OAuthMetadata, OAuthClientInformationFull).
    """
    from mcp.client.auth.utils import (
        build_oauth_authorization_server_metadata_discovery_urls,
        build_protected_resource_metadata_discovery_urls,
        create_client_registration_request,
        extract_resource_metadata_from_www_auth,
        extract_scope_from_www_auth,
        get_client_metadata_scopes,
        handle_auth_metadata_response,
        handle_protected_resource_response,
        handle_registration_response,
    )
    from mcp.shared.auth import OAuthClientMetadata

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: Hit server URL expecting a 401 with WWW-Authenticate
        initial_response = await client.get(server_url)
        if initial_response.status_code != 401:
            raise RuntimeError(
                f"Expected 401 from {server_url}, got {initial_response.status_code}"
            )

        # Step 2: Extract resource_metadata URL from WWW-Authenticate header
        www_auth_url = extract_resource_metadata_from_www_auth(initial_response)
        www_auth_scope = extract_scope_from_www_auth(initial_response)

        # Step 3: Discover Protected Resource Metadata (RFC 9728)
        prm = None
        prm_urls = build_protected_resource_metadata_discovery_urls(www_auth_url, server_url)
        for url in prm_urls:
            prm_response = await client.get(url)
            prm = await handle_protected_resource_response(prm_response)
            if prm is not None:
                break

        # Step 4: Extract auth server URL from PRM
        auth_server_url = str(prm.authorization_servers[0]) if prm and prm.authorization_servers else None

        # Step 5: Discover OAuth Authorization Server Metadata (RFC 8414)
        oauth_metadata = None
        metadata_urls = build_oauth_authorization_server_metadata_discovery_urls(
            auth_server_url, server_url
        )
        for url in metadata_urls:
            metadata_response = await client.get(url)
            is_valid, metadata = await handle_auth_metadata_response(metadata_response)
            if is_valid and metadata is not None:
                oauth_metadata = metadata
                break

        if oauth_metadata is None:
            raise RuntimeError(
                f"Could not discover OAuth metadata for {server_url}"
            )

        # Step 6: Select scopes
        scopes = get_client_metadata_scopes(www_auth_scope, prm, oauth_metadata)

        # Step 7: Check for existing client info or register a new one
        storage = MCPTokenStorage(server_name)
        client_info = await storage.get_client_info()
        if client_info is None:
            scope_list = scopes.split() if scopes else []
            client_metadata = OAuthClientMetadata(
                redirect_uris=[redirect_uri],
                client_name="grip",
                token_endpoint_auth_method="client_secret_post",
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope=" ".join(scope_list) if scope_list else None,
            )
            auth_base_url = auth_server_url or server_url
            registration_request = create_client_registration_request(
                oauth_metadata, client_metadata, auth_base_url
            )
            registration_response = await client.send(registration_request)
            client_info = await handle_registration_response(registration_response)
            await storage.set_client_info(client_info)
            logger.debug("Registered new MCP OAuth client for '{}'", server_name)
        else:
            logger.debug("Reusing stored MCP OAuth client for '{}'", server_name)

    return (oauth_metadata, client_info)


def create_mcp_oauth_auth(
    server_name: str,
    server_url: str,
    callback_port: int = _DEFAULT_CALLBACK_PORT,
) -> Any:
    """Build an OAuthClientProvider for automatic MCP server authentication.

    Returns the httpx.Auth provider, or None if mcp.client.auth is unavailable.
    The provider only activates on 401 responses — servers that don't need
    OAuth will work normally.
    """
    try:
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata
    except ImportError:
        logger.debug("mcp.client.auth not available, skipping OAuth provider")
        return None

    storage = MCPTokenStorage(server_name)
    client_metadata = OAuthClientMetadata(
        redirect_uris=[f"http://localhost:{callback_port}/callback"],
        client_name="grip",
        token_endpoint_auth_method="client_secret_post",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )

    async def callback_handler() -> tuple[str, str | None]:
        return await _wait_for_oauth_callback(port=callback_port)

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=_open_browser,
        callback_handler=callback_handler,
    )
