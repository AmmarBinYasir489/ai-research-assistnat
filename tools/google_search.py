import os
from typing import Any

import requests


class SearchError(RuntimeError):
    pass


def google_search(query: str, max_results: int = 6) -> list[dict[str, str]]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key or api_key == "your_serpapi_key_here":
        raise SearchError("Missing SERPAPI_API_KEY. Add it to your .env file.")

    response = requests.get(
        "https://serpapi.com/search",
        params={
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": max_results,
        },
        timeout=30,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    if "error" in data:
        raise SearchError(str(data["error"]))

    results = []
    for item in data.get("organic_results", [])[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )

    return results
