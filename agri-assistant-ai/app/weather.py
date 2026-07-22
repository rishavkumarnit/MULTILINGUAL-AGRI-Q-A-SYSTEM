"""Location-based weather forecast lookup via the free Open-Meteo APIs (no API key)."""

from dataclasses import dataclass

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO_DESCRIPTIONS = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow fall", 73: "moderate snow fall", 75: "heavy snow fall", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


@dataclass
class DailyForecast:
    date: str
    description: str
    temp_max: float
    temp_min: float
    precipitation_mm: float
    precipitation_probability: int


@dataclass
class WeatherForecast:
    location_label: str
    days: list[DailyForecast]


async def fetch_weather_forecast(location: str) -> WeatherForecast | None:
    """Geocode the location and fetch a 5-day forecast. Returns None on any failure
    or unresolved location, matching the fail-soft pattern used by the other tools."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        place = await _geocode(client, location)
        if not place:
            return None
        return await _forecast(client, place)


async def _geocode(client: httpx.AsyncClient, location: str) -> dict | None:
    try:
        response = await client.get(GEOCODING_URL, params={"name": location, "count": 1, "language": "en", "format": "json"})
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    results = response.json().get("results") or []
    return results[0] if results else None


async def _forecast(client: httpx.AsyncClient, place: dict) -> WeatherForecast | None:
    params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": 5,
    }
    try:
        response = await client.get(FORECAST_URL, params=params)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    daily = response.json().get("daily")
    if not daily:
        return None

    label = place["name"]
    if place.get("country"):
        label = f"{label}, {place['country']}"

    days = [
        DailyForecast(
            date=daily["time"][i],
            description=WMO_DESCRIPTIONS.get(daily["weather_code"][i], "unknown conditions"),
            temp_max=daily["temperature_2m_max"][i],
            temp_min=daily["temperature_2m_min"][i],
            precipitation_mm=daily["precipitation_sum"][i],
            precipitation_probability=daily["precipitation_probability_max"][i],
        )
        for i in range(len(daily["time"]))
    ]
    return WeatherForecast(location_label=label, days=days)
