"""Standalone MCP server exposing the weather and mandi price tools to any MCP client
(Claude Desktop, Claude Code, etc.). Run directly: python mcp_server.py

This reuses app/weather.py and app/crop_price.py as-is; the FastAPI app's own
LangGraph workflow keeps calling those functions directly and does not go through
this server."""

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from app.crop_price import fetch_crop_price, format_price
from app.weather import fetch_weather_forecast, format_forecast

load_dotenv()

mcp = FastMCP("agri-assistant-tools")


@mcp.tool()
async def get_weather_forecast(location: str) -> str:
    """Get a 5-day weather forecast for a city, district, or place name."""
    forecast = await fetch_weather_forecast(location)
    return format_forecast(forecast) if forecast else f"No weather data found for '{location}'."


@mcp.tool()
async def get_crop_price(crop: str, location: str = "") -> str:
    """Get the latest mandi (market) price for a crop, optionally scoped to a
    district or state. Requires DATA_GOV_IN_API_KEY to be configured."""
    price = await fetch_crop_price(crop, location or None)
    if not price:
        where = f" in '{location}'" if location else ""
        return f"No price data found for '{crop}'{where}."
    return format_price(price)


if __name__ == "__main__":
    mcp.run(transport="stdio")
