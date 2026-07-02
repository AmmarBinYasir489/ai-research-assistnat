import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

from tools.article_reader import enrich_with_article_text
from tools.brave_search import BraveSearchError, brave_search
from tools.google_news import GoogleNewsError, google_news_search
from tools.google_search import SearchError, google_search
from tools.searxng_search import SearxngSearchError, searxng_search


ROOT = Path(__file__).parent
SYSTEM_PROMPT = (ROOT / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
load_dotenv()
SEARCH_RESULT_COUNT = int(os.getenv("SEARCH_RESULT_COUNT", "15"))
FINAL_SOURCE_COUNT = int(os.getenv("FINAL_SOURCE_COUNT", "5"))
MAX_ARTICLE_AGE_DAYS = int(os.getenv("MAX_ARTICLE_AGE_DAYS", "30"))
MAX_ALLOWED_SEARCH_RESULTS = 20
MAX_ALLOWED_FINAL_SOURCES = 8
MAX_ALLOWED_ARTICLE_AGE_DAYS = 365


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def normalize_search_tools(research_mode: str, search_tools: list[str]) -> list[str]:
    preferred_by_mode = {
        "news": ["google_news", "brave"],
        "web": ["brave", "searxng", "serpapi"],
        "hybrid": ["google_news", "brave", "searxng"],
    }
    preferred = preferred_by_mode.get(research_mode, ["brave", "searxng"])
    ordered_tools = []

    for tool in preferred + search_tools:
        if tool not in {"google_news", "brave", "searxng", "serpapi"}:
            continue
        if tool in ordered_tools:
            continue
        ordered_tools.append(tool)

    return ordered_tools


def ask_ollama(prompt: str, json_mode: bool = False, max_tokens: int = 350) -> tuple[str | None, str | None]:
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,
        },
    }
    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=240,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return None, str(error)

    return response.json().get("response"), None


def make_search_query(user_question: str) -> str:
    prompt = f"""
Convert the user's research question into one Google search query.
Return JSON only.

Schema:
{{
  "query": "..."
}}

User question:
{user_question}
"""
    response, _error = ask_ollama(prompt, json_mode=True, max_tokens=80)
    if not response:
        return user_question

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return user_question

    query = parsed.get("query")
    if not isinstance(query, str) or not query.strip():
        return user_question

    return query.strip()


def make_research_plan(user_question: str) -> dict[str, int | str | list[str]]:
    prompt = f"""
You are planning a research task.
Choose the best research mode, query, freshness rule, and source count.

Rules:
- research_mode must be one of: "news", "web", "hybrid".
- search_tools can include: "google_news", "brave", "searxng", "serpapi".
- For "latest", "today", "this week", or news questions, use a small max_age_days.
- For patents, named inventors, patent numbers, assignees, historical facts, or old technical records, use research_mode "web" and set requires_freshness to false.
- Old sources can still be valid evidence when the user asks for a specific patent, document, person, or record.
- For broad trend questions, inspect more articles.
- For "latest trends", "current landscape", or questions needing both recency and background, use research_mode "hybrid".
- final_source_count must be smaller than or equal to search_result_count.
- Return JSON only.

Schema:
{{
  "research_mode": "hybrid",
  "search_tools": ["google_news", "brave"],
  "search_query": "...",
  "search_result_count": 15,
  "final_source_count": 5,
  "max_age_days": 30,
  "requires_freshness": true,
  "reason": "..."
}}

User question:
{user_question}
"""
    response, _error = ask_ollama(prompt, json_mode=True, max_tokens=160)
    if not response:
        return {
            "research_mode": "web",
            "search_tools": ["brave", "searxng"],
            "search_query": user_question,
            "search_result_count": SEARCH_RESULT_COUNT,
            "final_source_count": FINAL_SOURCE_COUNT,
            "max_age_days": MAX_ARTICLE_AGE_DAYS,
            "requires_freshness": "yes",
            "reason": "Fallback plan because Ollama did not return JSON.",
        }

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = {}

    research_mode = parsed.get("research_mode", "web")
    search_tools = parsed.get("search_tools", [])
    search_query = parsed.get("search_query")
    if not isinstance(search_query, str) or not search_query.strip():
        search_query = make_search_query(user_question)

    search_result_count = parsed.get("search_result_count", SEARCH_RESULT_COUNT)
    final_source_count = parsed.get("final_source_count", FINAL_SOURCE_COUNT)
    max_age_days = parsed.get("max_age_days", MAX_ARTICLE_AGE_DAYS)
    requires_freshness = parsed.get("requires_freshness", True)
    reason = parsed.get("reason", "Model-created research plan.")

    if not isinstance(research_mode, str):
        research_mode = "web"
    research_mode = research_mode.lower().strip()
    if research_mode not in {"news", "web", "hybrid"}:
        research_mode = "web"

    if not isinstance(search_tools, list):
        search_tools = []
    search_tools = [tool for tool in search_tools if isinstance(tool, str)]
    search_tools = [tool.lower().strip() for tool in search_tools]
    search_tools = [tool for tool in search_tools if tool in {"google_news", "brave", "searxng", "serpapi"}]
    search_tools = normalize_search_tools(research_mode, search_tools)

    if not isinstance(search_result_count, int):
        search_result_count = SEARCH_RESULT_COUNT
    if not isinstance(final_source_count, int):
        final_source_count = FINAL_SOURCE_COUNT
    if not isinstance(max_age_days, int):
        max_age_days = MAX_ARTICLE_AGE_DAYS
    if not isinstance(requires_freshness, bool):
        requires_freshness = research_mode == "news"
    if not isinstance(reason, str):
        reason = "Model-created research plan."

    search_result_count = clamp(search_result_count, 5, MAX_ALLOWED_SEARCH_RESULTS)
    final_source_count = clamp(final_source_count, 1, MAX_ALLOWED_FINAL_SOURCES)
    final_source_count = min(final_source_count, search_result_count)
    max_age_days = clamp(max_age_days, 1, MAX_ALLOWED_ARTICLE_AGE_DAYS)

    return {
        "research_mode": research_mode,
        "search_tools": search_tools,
        "search_query": search_query.strip(),
        "search_result_count": search_result_count,
        "final_source_count": final_source_count,
        "max_age_days": max_age_days,
        "requires_freshness": "yes" if requires_freshness else "no",
        "reason": reason.strip(),
    }


def format_sources(results: list[dict[str, str]], final_source_count: int) -> str:
    lines = []
    for index, result in enumerate(results[:final_source_count], start=1):
        article_text = result.get("article_text")
        evidence = f"Article excerpt: {article_text}" if article_text else f"Snippet: {result['snippet']}"
        published = result.get("published_at") or result.get("published") or "unknown date"
        lines.append(
            f"[{index}] {result['title']}\n"
            f"Published: {published}\n"
            f"URL: {result['url']}\n"
            f"{evidence}"
        )
    return "\n\n".join(lines)


def format_source_links(results: list[dict[str, str]], final_source_count: int) -> str:
    lines = []
    for index, result in enumerate(results[:final_source_count], start=1):
        published = result.get("published_at") or result.get("published") or "unknown date"
        lines.append(f"[{index}] {result['title']}\nPublished: {published}\n{result['url']}")
    return "\n\n".join(lines)


def parse_result_date(result: dict[str, str]) -> datetime | None:
    published_at = result.get("published_at")
    if not published_at:
        return None

    try:
        value = datetime.fromisoformat(published_at)
    except ValueError:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)

    return value.astimezone(UTC)


def select_latest_results(
    results: list[dict[str, str]],
    max_age_days: int,
    final_source_count: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    dated_results = []
    rejected_results = []

    for result in results:
        published_date = parse_result_date(result)
        if not published_date:
            rejected_results.append(result)
            continue

        if published_date < cutoff:
            rejected_results.append(result)
            continue

        dated_results.append((published_date, result))

    dated_results.sort(key=lambda item: item[0], reverse=True)
    selected = [result for _published_date, result in dated_results[:final_source_count]]
    return selected, rejected_results


def select_results_without_freshness_filter(
    results: list[dict[str, str]],
    final_source_count: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return results[:final_source_count], results[final_source_count:]


def deduplicate_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    unique_results = []

    for result in results:
        url = result.get("url", "").split("?")[0].rstrip("/")
        title = " ".join(result.get("title", "").lower().split())
        key = url or title
        if not key or key in seen:
            continue

        seen.add(key)
        unique_results.append(result)

    return unique_results


def answer_with_sources(user_question: str, results: list[dict[str, str]], final_source_count: int) -> str:
    sources = format_sources(results, final_source_count)
    prompt = f"""
{SYSTEM_PROMPT}

Do not write a Sources section. The program will add source links after your answer.

User question:
{user_question}

Search results:
{sources}
"""
    source_links = format_source_links(results, final_source_count)
    response, error = ask_ollama(prompt, max_tokens=450)
    if response:
        return f"{response.strip()}\n\n### Sources Checked\n{source_links}"

    return (
        "Ollama did not return the final summary.\n"
        f"Reason: {error or 'empty response'}\n\n"
        "Here are the sources I found:\n\n"
        f"{source_links}"
    )


def search_with_tool(tool: str, query: str, max_results: int) -> list[dict[str, str]]:
    if tool == "google_news":
        return google_news_search(query, max_results=max_results)

    if tool == "brave":
        return brave_search(query, max_results=max_results)

    if tool == "searxng":
        return searxng_search(query, max_results=max_results)

    if tool == "serpapi":
        return google_search(query, max_results=max_results)

    raise SearchError(f"Unknown search tool: {tool}")


def search_web(query: str, search_result_count: int, search_tools: list[str]) -> list[dict[str, str]]:
    results = []
    errors = []
    per_tool_count = max(3, search_result_count)

    for tool in search_tools:
        try:
            tool_results = search_with_tool(tool, query, per_tool_count)
        except (BraveSearchError, GoogleNewsError, SearchError, SearxngSearchError, requests.RequestException) as error:
            errors.append(f"{tool}: {error}")
            continue

        for result in tool_results:
            result["search_tool"] = tool
        results.extend(tool_results)

    results = deduplicate_results(results)
    if results:
        return results[:search_result_count]

    error_details = "; ".join(errors) if errors else "no tools were selected"
    raise SearchError(f"All search tools failed: {error_details}")


def main() -> None:
    user_question = input("Research question: ").strip()
    if not user_question:
        print("Please enter a research question.")
        return

    plan = make_research_plan(user_question)
    search_query = str(plan["search_query"])
    research_mode = str(plan["research_mode"])
    search_tools = list(plan["search_tools"])
    search_result_count = int(plan["search_result_count"])
    final_source_count = int(plan["final_source_count"])
    max_age_days = int(plan["max_age_days"])
    requires_freshness = plan["requires_freshness"] == "yes"
    print(f"\nSearch query: {search_query}\n")
    print(
        "Research plan: "
        f"mode {research_mode}, tools {', '.join(search_tools)}, "
        f"inspect {search_result_count} results, keep {final_source_count}, "
        f"{'reject articles older than ' + str(max_age_days) + ' days' if requires_freshness else 'do not reject old dated sources'}.\n"
        f"Reason: {plan['reason']}\n"
    )

    try:
        results = search_web(search_query, search_result_count, search_tools)
    except (BraveSearchError, GoogleNewsError, SearchError, SearxngSearchError, requests.RequestException) as error:
        print(f"Search failed: {error}")
        return

    if requires_freshness:
        results, rejected_results = select_latest_results(results, max_age_days, final_source_count)
    else:
        results, rejected_results = select_results_without_freshness_filter(results, final_source_count)
    print(
        f"Checked up to {search_result_count} results. "
        f"Using {len(results)} results and rejecting {len(rejected_results)} extra/filtered results.\n"
    )
    if not results:
        if requires_freshness:
            print(f"No articles were published within the last {max_age_days} days.")
        else:
            print("No results found.")
        return

    print("Reading selected pages...\n")
    results = enrich_with_article_text(results, max_articles=final_source_count)

    print(answer_with_sources(user_question, results, final_source_count))


if __name__ == "__main__":
    main()
