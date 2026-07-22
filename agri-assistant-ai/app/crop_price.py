"""Live crop price lookup via the data.gov.in Agmarknet market-prices API.

Uses the system curl binary rather than httpx: api.data.gov.in's server hangs
indefinitely on Python's OpenSSL-based TLS handshake (reproduced with both httpx
and stdlib urllib) while curl.exe's Schannel TLS stack connects in well under a
second. curl.exe ships with Windows (System32) and Git for Windows, so it is
reliably available without adding a dependency.

Runs curl via subprocess.run() in a thread executor rather than
asyncio.create_subprocess_exec(): uvicorn forces WindowsSelectorEventLoopPolicy
whenever --reload is active (needed for its own reload supervisor), and asyncio's
native subprocess transport isn't implemented on that loop on Windows. A thread-pool
blocking call works under any event loop.
"""

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode

AGMARKNET_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
AGMARKNET_BASE_URL = f"https://api.data.gov.in/resource/{AGMARKNET_RESOURCE_ID}"


@dataclass
class CropPrice:
    commodity: str
    state: str
    market: str
    modal_price: float
    min_price: float
    max_price: float
    arrival_date: str


async def fetch_crop_price(crop: str, location: str | None) -> CropPrice | None:
    """Look up the most recent modal price for a crop, optionally scoped to a location.

    Tries the location as a district first, then as a state, then falls back to a
    nationwide lookup, since farmers typically state a city/district rather than a
    state name. Returns None on any failure or empty result set instead of raising,
    matching the fail-soft pattern used by the semantic_search/rag/weather tools.
    """
    api_key = os.getenv("DATA_GOV_IN_API_KEY")
    if not api_key:
        return None

    commodity = crop.strip().title()
    attempts: list[dict[str, str]] = []
    if location:
        location_title = location.strip().title()
        attempts.append({"district": location_title})
        attempts.append({"state": location_title})
    attempts.append({})

    for extra_filters in attempts:
        records = await _query(api_key, commodity, extra_filters)
        if records:
            return _pick_latest(records, commodity)
    return None


async def _query(api_key: str, commodity: str, extra_filters: dict[str, str]) -> list[dict]:
    params = {"api-key": api_key, "format": "json", "limit": "50", "filters[commodity]": commodity}
    for field, value in extra_filters.items():
        params[f"filters[{field}]"] = value
    url = f"{AGMARKNET_BASE_URL}?{urlencode(params)}"

    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: subprocess.run(["curl", "-sS", "-m", "15", url], capture_output=True, timeout=20)
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    return data.get("records", [])


def _pick_latest(records: list[dict], commodity: str) -> CropPrice | None:
    def parse_date(record: dict) -> datetime:
        try:
            return datetime.strptime(record["arrival_date"], "%d/%m/%Y")
        except (KeyError, ValueError):
            return datetime.min

    latest = max(records, key=parse_date)
    try:
        return CropPrice(
            commodity=commodity,
            state=latest["state"],
            market=latest["market"],
            modal_price=float(latest["modal_price"]),
            min_price=float(latest["min_price"]),
            max_price=float(latest["max_price"]),
            arrival_date=latest["arrival_date"],
        )
    except (KeyError, ValueError):
        return None


def format_price(price: CropPrice) -> str:
    return (
        f"The latest modal price for {price.commodity} at {price.market}, {price.state} "
        f"(as of {price.arrival_date}) is Rs {price.modal_price:.0f} per quintal "
        f"(range Rs {price.min_price:.0f}-{price.max_price:.0f})."
    )
