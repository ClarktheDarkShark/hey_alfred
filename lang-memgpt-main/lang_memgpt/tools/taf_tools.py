from typing import Any, List, Optional
import aiohttp
import xml.etree.ElementTree as ET  # For XML parsing
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


@tool
async def get_taf_data(
    stations: List[str],  # Explicitly matches docstring
    hours_before_now: int = 1,  # Matches default argument in docstring
    config: Optional[RunnableConfig] = None  # Matches docstring
) -> Optional[dict[str, Any]]:
    """
    Fetch TAF (Terminal Aerodrome Forecast) data for multiple airports.

    Args:
        stations (List[str]): List of ICAO codes for airports (e.g., ["KJFK", "KDCA"]).
        hours_before_now (int): Time range in hours for retrieving TAF data. Default is 1 hour.
        config (Optional[RunnableConfig]): Optional runtime configuration.

    Returns:
        Optional[dict[str, Any]]: TAF data or error messages for each station.

    Example Usage:
        get_taf_data(["KJFK", "KDCA"])
    """

    # Aviation Weather API endpoint
    base_url = "https://aviationweather.gov/api/data/dataserver"

    # Base parameters for TAF data retrieval
    params = {
        "requestType": "retrieve",
        "dataSource": "tafs",
        "format": "xml",  # Output format
        "hoursBeforeNow": hours_before_now,  # Dynamic time range
    }

    # Prepare results dictionary
    results = {}

    # Start asynchronous HTTP session
    async with aiohttp.ClientSession() as session:
        for station in stations:
            # Set station-specific parameter
            params["stationString"] = station

            try:
                # Make the API request
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        # Extract XML text (expand parsing if required later)
                        results[station] = await response.text()
                    else:
                        # Handle HTTP errors
                        results[station] = f"Error: HTTP {response.status}"
            except Exception as e:
                # Handle unexpected exceptions (e.g., network errors)
                results[station] = f"Error: {str(e)}"

    # Return the compiled results
    # print(f'---TAF---', flush=True)
    # print(f'results:\n\n{results}', flush=True)
    return results
