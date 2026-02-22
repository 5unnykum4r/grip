"""Sanitized error handlers for the grip REST API.

All handlers log full details server-side but return only generic
messages to clients, preventing leakage of file paths, stack traces,
config values, or internal class names.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


def register_error_handlers(app: FastAPI) -> None:
    """Attach exception handlers that sanitize all error responses."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return 422 with field-level errors but no internal path info."""
        sanitized_errors = []
        for error in exc.errors():
            sanitized_errors.append(
                {
                    "field": " -> ".join(str(loc) for loc in error.get("loc", []) if loc != "body"),
                    "message": error.get("msg", "Invalid value"),
                    "type": error.get("type", "value_error"),
                }
            )
        logger.warning(
            "Validation error on {} {}: {}",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation error", "errors": sanitized_errors},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all: log the real error, return a generic 500 message."""
        logger.exception(
            "Unhandled error on {} {}: {}",
            request.method,
            request.url.path,
            exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
