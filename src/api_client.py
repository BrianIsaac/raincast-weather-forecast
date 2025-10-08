import requests
import logging
from typing import Dict, Any, Optional

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class WeatherAPIClient:
    BASE_URL = "https://api-open.data.gov.sg/v2/real-time/api/"

    SUPPORTED_PARAMETERS = {
        "rainfall",
        "air-temperature",
        "relative-humidity",
        "wind-speed",
        "wind-direction",
    }

    def __init__(self):
        pass  # No API key needed for public endpoints currently

    def fetch_weather_data(
        self,
        parameter: str,
        date: Optional[str] = None,
        pagination_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch weather data for a given parameter (e.g., 'rainfall', 'air-temperature'), date, and optional pagination token.

        :param parameter: Weather parameter ('rainfall', 'air-temperature', etc.)
        :param date: Date string in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS' format (SGT). Returns latest if None.
        :param pagination_token: Token to retrieve subsequent pages (if paginated).
        :return: JSON response as Python dict
        """
        if parameter not in self.SUPPORTED_PARAMETERS:
            logging.error(f"Parameter '{parameter}' not supported.")
            return {}

        url = self.BASE_URL + parameter
        params = {}
        if date:
            params["date"] = date
        if pagination_token:
            params["paginationToken"] = pagination_token

        logging.info(f"Requesting {parameter} data from {url} with params: {params}")

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Successfully fetched data for {parameter}")
            return data

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logging.error(f"Other error occurred: {err}")

        return {}
