import re
from datetime import UTC, datetime
from urllib.parse import urlparse


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "who",
    "with",
}

AUTHORITY_HINTS = {
    ".gov": 2.0,
    ".edu": 1.5,
    "reuters.com": 1.5,
    "apnews.com": 1.5,
    "bbc.com": 1.2,
    "nature.com": 1.5,
    "science.org": 1.5,
    "arxiv.org": 1.2,
    "patents.google.com": 2.0,
    "uspto.gov": 2.0,
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)

    return parsed.astimezone(UTC)


def authority_score(url: str) -> float:
    host = urlparse(url).netloc.lower()
    return sum(score for hint, score in AUTHORITY_HINTS.items() if hint in host)


def freshness_score(result: dict[str, str]) -> float:
    published_at = parse_datetime(result.get("published_at", ""))
    if not published_at:
        return 0.0

    age_days = max(0, (datetime.now(UTC) - published_at).days)
    if age_days <= 1:
        return 2.0
    if age_days <= 7:
        return 1.5
    if age_days <= 30:
        return 1.0
    return 0.0


def score_result(query: str, result: dict[str, str], freshness_required: bool) -> float:
    query_terms = tokenize(query)
    if not query_terms:
        return 0.0

    title_terms = tokenize(result.get("title", ""))
    snippet_terms = tokenize(result.get("snippet", ""))
    title_overlap = len(query_terms & title_terms)
    snippet_overlap = len(query_terms & snippet_terms)

    score = (title_overlap * 3.0) + (snippet_overlap * 1.0)
    score += authority_score(result.get("url", ""))
    if freshness_required:
        score += freshness_score(result)

    return score


def rank_results(
    query: str,
    results: list[dict[str, str]],
    freshness_required: bool,
) -> list[dict[str, str]]:
    scored_results = []
    for result in results:
        item = result.copy()
        item["relevance_score"] = f"{score_result(query, item, freshness_required):.2f}"
        scored_results.append(item)

    return sorted(scored_results, key=lambda item: float(item["relevance_score"]), reverse=True)
