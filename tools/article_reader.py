from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests


class ArticleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._capture_tag: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._skip_depth += 1
            return

        if self._skip_depth:
            return

        if tag in {"h1", "h2", "h3", "p", "li"}:
            self._capture_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if tag == self._capture_tag:
            self._parts.append("\n")
            self._capture_tag = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._capture_tag:
            return

        text = " ".join(unescape(data).split())
        if text:
            self._parts.append(text)

    def text(self) -> str:
        lines = []
        for raw_line in "".join(self._parts).splitlines():
            line = " ".join(raw_line.split())
            if len(line) >= 40:
                lines.append(line)
        return "\n".join(lines)


def read_article(url: str, max_chars: int = 1800) -> str | None:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 research-assistant-agent/0.1"},
            timeout=25,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type:
        return None

    parser = ArticleTextParser()
    parser.feed(response.text)
    text = parser.text()
    if not text:
        return None

    return text[:max_chars]


def enrich_with_article_text(
    results: list[dict[str, str]],
    max_articles: int = 3,
) -> list[dict[str, str]]:
    enriched = []

    for index, result in enumerate(results):
        item = result.copy()
        if index < max_articles:
            article_text = read_article(result["url"])
            if article_text:
                item["article_text"] = article_text
                item["article_host"] = urlparse(result["url"]).netloc
        enriched.append(item)

    return enriched
