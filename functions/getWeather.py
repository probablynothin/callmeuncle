import os

import requests

def get_weather(location: str) -> int:
    """Returns the current weather in the given location in Celsius, or -1 if an error occurs.

    Args:
        location: City or State or Country Name.
    """
    api_key = os.getenv("WEATHER_API_KEY")  # Replace with your actual API key
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        weather_data = response.json()
        temperature = weather_data["current"]["temp_c"]

        print(f"The current temperature in {location} is {temperature}°C")
        return temperature

    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return -1


get_weather_tool = dict(
    name="get_weather",
    description=(
        "Returns the current weather in the given location in Celsius, returns -1 if an error occurs."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "location": {
                "type": "STRING",
                "description": "City or State or Country Name.",
            },
        },
        "required": ["location"],
    },
)
