from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from typing import Any, List, Optional, Annotated
import datetime
from datetime import timedelta


@tool
async def date_time_tool(
    date_str: str,
    days_to_add: int = 0,
    *,  # Enforces keyword-only arguments after this point
    # Ensures compatibility with LangChain framework
    config: Annotated[RunnableConfig, Any]
) -> str:
    """
    Adds or subtracts days from a given date.

    Args:
        date_str (str): The input date in YYYY-MM-DD format.
        days_to_add (int): Number of days to add (negative for subtraction).

    Returns:
        str: The resulting date in YYYY-MM-DD format, or an error message.
    """
    try:
        # Parse the input date string
        date = datetime.strptime(date_str, "%Y-%m-%d")

        # Perform the date adjustment
        new_date = date + timedelta(days=days_to_add)

        # Return the new date in string format
        return new_date.strftime("%Y-%m-%d")

    except ValueError as e:
        # Return a friendly error message if the date format is invalid
        return f"Error: Invalid date format or value. Details - {str(e)}"
