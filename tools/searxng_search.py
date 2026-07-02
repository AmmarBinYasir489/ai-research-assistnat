import os
from typing import Any

import requests


class SearxngSearchError(RuntimeError):
    pass


def get_instances() -> list[str]:
    default_instances = [
        "https://search.inetol.net",
        "https://searx.tiekoetter.com",
        "https://opnxng.com",
        "https://northboot.xyz",
    ]

    instances = os.getenv("SEARXNG_INSTANCES")
    if instances:
        configured = [instance.strip().rstrip("/") for instance in instances.split(",") if instance.strip()]
        return configured + [instance for instance in default_instances if instance not in configured]

    single_instance = os.getenv("SEARXNG_INSTANCE")
    if single_instance:
        configured = single_instance.rstrip("/")
        return [configured] + [instance for instance in default_instances if instance != configured]

    return default_instances


def searxng_search(query: str, max_results: int = 6) -> list[dict[str, str]]:
    errors = []

    for instance in get_instances():
        try:
            response = requests.get(
                f"{instance}/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": "en",
                },
                headers={"User-Agent": "research-assistant-agent/0.1"},
                timeout=30,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except (requests.RequestException, ValueError) as error:
            errors.append(f"{instance}: {error}")
            continue

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )

        if results:
            return results

        errors.append(f"{instance}: no results")

    details = "; ".join(errors)
    raise SearxngSearchError(f"No SearXNG instance returned results. {details}")
