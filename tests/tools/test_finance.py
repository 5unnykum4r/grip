"""Tests for finance tools — dry-run and error handling tests always run.

Tests that require a live yfinance install are skipped when yfinance is absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from grip.tools.base import ToolContext

_yfinance_available = importlib.util.find_spec("yfinance") is not None
skip_if_no_yfinance = pytest.mark.skipif(not _yfinance_available, reason="yfinance not installed")


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path)


@pytest.fixture
def dry_ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_path=tmp_path, extra={"dry_run": True})


# ── Tool metadata ──


def test_stock_quote_properties():
    from grip.tools.finance import StockQuoteTool

    tool = StockQuoteTool()
    assert tool.name == "stock_quote"
    assert "symbols" in tool.parameters["properties"]
    assert tool.parameters["required"] == ["symbols"]


def test_stock_history_properties():
    from grip.tools.finance import StockHistoryTool

    tool = StockHistoryTool()
    assert tool.name == "stock_history"
    assert "symbol" in tool.parameters["required"]
    assert "period" in tool.parameters["properties"]


def test_company_info_properties():
    from grip.tools.finance import CompanyInfoTool

    tool = CompanyInfoTool()
    assert tool.name == "company_info"
    assert "symbol" in tool.parameters["required"]


# ── Dry-run mode ──


async def test_stock_quote_dry_run(dry_ctx: ToolContext):
    from grip.tools.finance import StockQuoteTool

    result = await StockQuoteTool().execute({"symbols": "AAPL"}, dry_ctx)
    assert "[DRY RUN]" in result
    assert "AAPL" in result


async def test_stock_history_dry_run(dry_ctx: ToolContext):
    from grip.tools.finance import StockHistoryTool

    result = await StockHistoryTool().execute({"symbol": "MSFT", "period": "1mo"}, dry_ctx)
    assert "[DRY RUN]" in result
    assert "MSFT" in result


async def test_company_info_dry_run(dry_ctx: ToolContext):
    from grip.tools.finance import CompanyInfoTool

    result = await CompanyInfoTool().execute({"symbol": "GOOGL"}, dry_ctx)
    assert "[DRY RUN]" in result
    assert "GOOGL" in result


# ── Import error handling ──


async def test_stock_quote_missing_yfinance(ctx: ToolContext):
    from grip.tools.finance import StockQuoteTool

    with patch("grip.tools.finance._import_yfinance", side_effect=ImportError("not installed")):
        result = await StockQuoteTool().execute({"symbols": "AAPL"}, ctx)
    assert "Error" in result
    assert "not installed" in result


async def test_stock_quote_empty_symbols(ctx: ToolContext):
    from grip.tools.finance import StockQuoteTool

    with patch("grip.tools.finance._import_yfinance"):
        result = await StockQuoteTool().execute({"symbols": ""}, ctx)
    assert "Error" in result


# ── Factory function ──


def test_create_finance_tools_no_yfinance():
    with patch("importlib.util.find_spec", return_value=None):
        from grip.tools.finance import create_finance_tools

        tools = create_finance_tools()
    assert tools == []


@skip_if_no_yfinance
def test_create_finance_tools_with_yfinance():
    from grip.tools.finance import create_finance_tools

    tools = create_finance_tools()
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"stock_quote", "stock_history", "company_info"}


# ── Validation ──


async def test_stock_history_invalid_period(ctx: ToolContext):
    from grip.tools.finance import StockHistoryTool

    with patch("grip.tools.finance._import_yfinance"):
        result = await StockHistoryTool().execute({"symbol": "AAPL", "period": "INVALID"}, ctx)
    assert "Error: Invalid period" in result


async def test_stock_history_invalid_interval(ctx: ToolContext):
    from grip.tools.finance import StockHistoryTool

    with patch("grip.tools.finance._import_yfinance"):
        result = await StockHistoryTool().execute(
            {"symbol": "AAPL", "period": "1mo", "interval": "BAD"}, ctx
        )
    assert "Error: Invalid interval" in result
