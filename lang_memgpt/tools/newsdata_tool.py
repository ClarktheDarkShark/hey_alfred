import os
import aiohttp
from dotenv import load_dotenv
from langchain.tools import tool

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('NEWSDATA_API_KEY')


@tool
async def fetch_latest_news(query, language='en', category=None, country=None):
    """
    Fetches the latest news articles based on query parameters.

    Args:
        query (str): Keywords to search for in articles.
        language (str): Language filter (default is 'en' for English).
        category (str): Optional category filter (e.g., 'business', 'technology').
        country (str): Optional country filter (e.g., 'us' for the United States).

    Returns:
        dict: JSON response containing news articles or error details.
    """
    base_url = "https://newsdata.io/api/1/latest"
    params = {
        'apikey': API_KEY,
        'q': query,
        'language': language,
        'category': category,
        'country': country
    }

    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientResponseError as e:
        return {"error": f"HTTP Error: {e.status} - {e.message}"}
    except aiohttp.ClientError as e:
        return {"error": f"Request Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}
