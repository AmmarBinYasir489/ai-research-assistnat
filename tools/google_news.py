from html import unescape
import re
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests


class GoogleNewsError(RuntimeError):
    pass


def clean_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", value)
    return " ".join(unescape(without_tags).split())


def google_news_search(query: str, max_results: int = 15) -> list[dict[str, str]]:
    encoded_query = quote_plus(query)
    url = (
        "https://news.google.com/rss/search"
        f"?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    )

    response = requests.get(
        url,
        headers={"User-Agent": "research-assistant-agent/0.1"},
        timeout=30,
    )
    response.raise_for_status()

    root = ElementTree.fromstring(response.text)
    items: list[Any] = root.findall("./channel/item")

    results = []
    for item in items[:max_results]:
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        published = item.findtext("pubDate", default="")
        published_at = ""
        if published:
            try:
                published_at = parsedate_to_datetime(published).isoformat()
            except (TypeError, ValueError):
                published_at = ""
        source = item.findtext("source", default="")
        description = clean_html(item.findtext("description", default=""))

        results.append(
            {
                "title": title,
                "url": link,
                "snippet": f"{source} | {published} | {description}",
                "source": source,
                "published": published,
                "published_at": published_at,
            }
        )

    if not results:
        raise GoogleNewsError("No Google News RSS results found.")

    return results
