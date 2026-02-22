"""Finance tools: real-time stock quotes, historical data, and company info.

Requires the optional 'finance' dependency group:
    uv pip install 'grip[finance]'

All tools return error strings gracefully when yfinance is not installed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from grip.tools.base import Tool, ToolContext


def _import_yfinance():
    """Import yfinance or raise ImportError with install instructions."""
    try:
        import yfinance as yf

        return yf
    except ImportError:
        raise ImportError(  # noqa: B904
            "yfinance is not installed. Run: uv pip install 'grip[finance]'"
        )


class StockQuoteTool(Tool):
    """Fetch current price, change, and volume for one or more tickers."""

    @property
    def category(self) -> str:
        return "finance"

    @property
    def name(self) -> str:
        return "stock_quote"

    @property
    def description(self) -> str:
        return "Get real-time stock quote (price, range, volume, P/E). Accepts comma-separated tickers."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "string",
                    "description": (
                        "Ticker symbol(s), e.g. 'AAPL' or 'AAPL,MSFT,GOOGL'. "
                        "Use '-USD' suffix for crypto (BTC-USD, ETH-USD)."
                    ),
                },
            },
            "required": ["symbols"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        if ctx.extra.get("dry_run"):
            return f"[DRY RUN] Would fetch stock quote for: {params['symbols']}"

        try:
            yf = _import_yfinance()
        except ImportError as exc:
            return f"Error: {exc}"

        raw = params.get("symbols", "")
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        if not symbols:
            return "Error: No ticker symbols provided."

        lines: list[str] = []
        for symbol in symbols:
            try:
                info = await asyncio.to_thread(lambda s=symbol: yf.Ticker(s).info)
                if not info or (
                    info.get("regularMarketPrice") is None and info.get("currentPrice") is None
                ):
                    lines.append(f"{symbol}: No data found. Verify the ticker symbol.")
                    continue

                price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
                change = info.get("regularMarketChange", "N/A")
                change_pct = info.get("regularMarketChangePercent")
                day_low = info.get("dayLow", "N/A")
                day_high = info.get("dayHigh", "N/A")
                week52_low = info.get("fiftyTwoWeekLow", "N/A")
                week52_high = info.get("fiftyTwoWeekHigh", "N/A")
                volume = info.get("volume", "N/A")
                market_cap = info.get("marketCap")
                pe_ratio = info.get("trailingPE", "N/A")

                change_str = (
                    f"{change} ({change_pct:.2f}%)"
                    if isinstance(change_pct, float)
                    else f"{change}"
                )
                vol_str = f"{volume:,}" if isinstance(volume, int) else str(volume)
                cap_str = (
                    f"${market_cap / 1e9:.2f}B"
                    if isinstance(market_cap, (int, float)) and market_cap > 0
                    else "N/A"
                )

                lines.append(f"--- {symbol} ---")
                lines.append(f"  Price:      ${price}")
                lines.append(f"  Change:     {change_str}")
                lines.append(f"  Day Range:  ${day_low} - ${day_high}")
                lines.append(f"  52W Range:  ${week52_low} - ${week52_high}")
                lines.append(f"  Volume:     {vol_str}")
                lines.append(f"  Market Cap: {cap_str}")
                lines.append(f"  P/E Ratio:  {pe_ratio}")
            except Exception as exc:
                logger.warning("stock_quote failed for {}: {}", symbol, exc)
                lines.append(f"{symbol}: Error fetching data: {exc}")

        return "\n".join(lines) if lines else "No data returned."


class StockHistoryTool(Tool):
    """Fetch historical OHLCV data for a ticker."""

    @property
    def category(self) -> str:
        return "finance"

    @property
    def name(self) -> str:
        return "stock_history"

    @property
    def description(self) -> str:
        return "Get historical OHLCV data for a stock or crypto over a specified period."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. 'AAPL' or 'BTC-USD'.",
                },
                "period": {
                    "type": "string",
                    "description": (
                        "Lookback period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. "
                        "Defaults to '1mo'."
                    ),
                },
                "interval": {
                    "type": "string",
                    "description": (
                        "Data granularity: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo. Defaults to '1d'."
                    ),
                },
                "rows": {
                    "type": "integer",
                    "description": "Number of most recent rows to return. Defaults to 10.",
                },
            },
            "required": ["symbol"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        if ctx.extra.get("dry_run"):
            return f"[DRY RUN] Would fetch {params.get('period', '1mo')} history for: {params['symbol']}"

        try:
            yf = _import_yfinance()
        except ImportError as exc:
            return f"Error: {exc}"

        symbol = params["symbol"].strip().upper()
        period = params.get("period", "1mo")
        interval = params.get("interval", "1d")
        rows = int(params.get("rows", 10))

        valid_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
        valid_intervals = {
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        }

        if period not in valid_periods:
            return f"Error: Invalid period '{period}'. Valid: {', '.join(sorted(valid_periods))}"
        if interval not in valid_intervals:
            return (
                f"Error: Invalid interval '{interval}'. Valid: {', '.join(sorted(valid_intervals))}"
            )

        try:
            ticker = yf.Ticker(symbol)
            hist = await asyncio.to_thread(ticker.history, period=period, interval=interval)

            if hist.empty:
                return (
                    f"No historical data for {symbol} (period={period}, interval={interval}). "
                    "Market may be closed or ticker may be invalid."
                )

            subset = hist[["Open", "High", "Low", "Close", "Volume"]].tail(rows)
            header = (
                f"Historical data for {symbol} "
                f"(period={period}, interval={interval}) — last {len(subset)} rows:\n"
            )
            return header + subset.to_string()
        except Exception as exc:
            logger.warning("stock_history failed for {}: {}", symbol, exc)
            return f"Error fetching history for {symbol}: {exc}"


class CompanyInfoTool(Tool):
    """Fetch company fundamentals and description for a ticker."""

    @property
    def category(self) -> str:
        return "finance"

    @property
    def name(self) -> str:
        return "company_info"

    @property
    def description(self) -> str:
        return (
            "Get company fundamentals: sector, market cap, revenue, dividend, and analyst target."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. 'MSFT' or 'NVDA'.",
                },
            },
            "required": ["symbol"],
        }

    async def execute(self, params: dict[str, Any], ctx: ToolContext) -> str:
        if ctx.extra.get("dry_run"):
            return f"[DRY RUN] Would fetch company info for: {params['symbol']}"

        try:
            yf = _import_yfinance()
        except ImportError as exc:
            return f"Error: {exc}"

        symbol = params["symbol"].strip().upper()

        try:
            info = await asyncio.to_thread(lambda: yf.Ticker(symbol).info)

            if not info or not info.get("longName"):
                return f"No company info found for {symbol}. Verify the ticker symbol."

            name = info.get("longName", symbol)
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            country = info.get("country", "N/A")
            employees = info.get("fullTimeEmployees")
            market_cap = info.get("marketCap")
            revenue = info.get("totalRevenue")
            gross_margins = info.get("grossMargins")
            dividend_yield = info.get("dividendYield")
            recommendation = info.get("recommendationKey", "N/A")
            target_mean = info.get("targetMeanPrice", "N/A")
            description = info.get("longBusinessSummary", "N/A")

            if isinstance(description, str) and len(description) > 500:
                description = description[:500] + "..."

            emp_str = f"{employees:,}" if isinstance(employees, int) else "N/A"
            cap_str = (
                f"${market_cap / 1e9:.2f}B"
                if isinstance(market_cap, (int, float)) and market_cap > 0
                else "N/A"
            )
            rev_str = (
                f"${revenue / 1e9:.2f}B"
                if isinstance(revenue, (int, float)) and revenue > 0
                else "N/A"
            )
            margin_str = f"{gross_margins:.1%}" if isinstance(gross_margins, float) else "N/A"
            div_str = f"{dividend_yield:.2%}" if isinstance(dividend_yield, float) else "N/A"

            lines = [
                f"=== {name} ({symbol}) ===",
                f"Sector:          {sector}",
                f"Industry:        {industry}",
                f"Country:         {country}",
                f"Employees:       {emp_str}",
                f"Market Cap:      {cap_str}",
                f"Revenue:         {rev_str}",
                f"Gross Margins:   {margin_str}",
                f"Dividend Yield:  {div_str}",
                f"Analyst Rating:  {recommendation.upper() if isinstance(recommendation, str) else 'N/A'}",
                f"Price Target:    ${target_mean}",
                f"\nDescription:\n{description}",
            ]
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("company_info failed for {}: {}", symbol, exc)
            return f"Error fetching company info for {symbol}: {exc}"


def create_finance_tools() -> list[Tool]:
    """Return finance tool instances if yfinance is available, otherwise empty list."""
    try:
        import importlib.util

        if importlib.util.find_spec("yfinance") is None:
            logger.debug("yfinance not installed — finance tools skipped")
            return []
    except Exception:
        return []

    return [
        StockQuoteTool(),
        StockHistoryTool(),
        CompanyInfoTool(),
    ]
