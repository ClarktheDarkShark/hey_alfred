from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from typing import Any, List, Optional
import re


@tool
async def unit_converter(
    query: str, config: Optional[RunnableConfig] = None
) -> str:
    """
    Converts units like weight, temperature, distance, and aviation fuel (JP-5).

    Args:
        query (str): Input query (e.g., 'convert 500 miles to km' or '16k lbs of fuel to gallons').

    Returns:
        str: Converted value or error message.

    Example Usage:
        unit_converter('convert 15k lbs to kg')
    """
    try:
        # Conversion factors for supported unit conversions
        conversions = {
            # Length
            "meters_to_feet": 3.28084,
            "feet_to_meters": 0.3048,
            "miles_to_km": 1.60934,
            "km_to_miles": 0.621371,
            "nautical_to_statute": 1.15078,  # NM to statute miles
            "statute_to_nautical": 0.868976,  # Statute miles to NM
            # Weight
            "kg_to_pounds": 2.20462,
            "pounds_to_kg": 0.453592,
            # Volume
            "liters_to_gallons": 0.264172,
            "gallons_to_liters": 3.78541,
            # Temperature
            "celsius_to_fahrenheit": lambda x: (x * 9 / 5) + 32,
            "fahrenheit_to_celsius": lambda x: (x - 32) * 5 / 9,
            # JP-5 Fuel (6.8 lbs/gal)
            "pounds_to_gallons": 1 / 6.8,
            "gallons_to_pounds": 6.8,
        }

        # Aliases for unit names
        unit_aliases = {
            "miles": "miles", "mi": "miles",
            "nautical miles": "nautical", "nm": "nautical",
            "statute miles": "statute", "statute": "statute",
            "kilometers": "km", "km": "km",
            "meters": "meters", "m": "meters",
            "feet": "feet", "ft": "feet",
            "pounds": "pounds", "lbs": "pounds",
            "kg": "kg", "kilograms": "kg",
            "celsius": "celsius", "c": "celsius",
            "fahrenheit": "fahrenheit", "f": "fahrenheit",
            "liters": "liters", "l": "liters",
            "gallons": "gallons", "gal": "gallons",
        }

        # Remove filler words and normalize the query
        filler_words = ["of", "for", "the", "fuel", "jet fuel"]
        query = re.sub(r'\b(?:' + '|'.join(filler_words) +
                       r')\b', '', query.lower()).strip()

        # Parse the query using regex
        match = re.match(
            r"(convert|change|what is)?\s*([\d,.kK]+)\s*([a-zA-Z\s]+)\s*(to|in)\s*([a-zA-Z\s]+)", query)

        # Error if input does not match expected format
        if not match:
            return "Error: Could not parse input. Example: 'convert 15k lbs to kg'."

        # Extract values and units from query
        value = float(match.group(2).replace("k", "000").replace(",", ""))
        from_unit = match.group(3).strip().lower()
        to_unit = match.group(5).strip().lower()

        # Map aliases to standard unit names
        from_unit = unit_aliases.get(from_unit, from_unit)
        to_unit = unit_aliases.get(to_unit, to_unit)

        # Perform the conversion if valid
        key = f"{from_unit}_to_{to_unit}"
        if key in conversions:
            conversion = conversions[key]
            # Handle callable conversions (temperature)
            result = conversion(value) if callable(
                conversion) else value * conversion
            return f"{value:,} {from_unit} = {result:,.2f} {to_unit}"
        else:
            # Unsupported conversion
            return f"Error: Unsupported conversion from {from_unit} to {to_unit}."

    except Exception as e:
        # General error handling
        return f"Error: {str(e)}"
