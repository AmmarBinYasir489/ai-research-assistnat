import os
from typing import Any

import requests


class BraveSearchError(RuntimeError):
    pass


def brave_search(query: str, max_results: int = 10) -> list[dict[str, str]]:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key or api_key == "your_brave_api_key_here":
        raise BraveSearchError("Missing BRAVE_API_KEY. Add it to your .env file.")

    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
        params={
            "q": query,
            "count": max_results,
        },
        timeout=30,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    results = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "published": item.get("age", ""),
                "published_at": "",
            }
        )

    if not results:
        raise BraveSearchError("No Brave Search results found.")

    return results
