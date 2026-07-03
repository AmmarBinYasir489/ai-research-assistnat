import json


def parse_json_object(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    return [item for item in value if isinstance(item, str)]


def evaluate_evidence(
    user_question: str,
    sources: str,
    ask_model,
) -> dict[str, object]:
    prompt = f"""
You are evaluating evidence quality for a research assistant.
Do not answer the user question.

Judge whether the provided sources are enough to answer the question.

Return JSON only.

Schema:
{{
  "enough_information": true,
  "confidence": "high",
  "missing_information": [],
  "conflicts": [],
  "recommended_next_query": null,
  "reason": "..."
}}

Rules:
- confidence must be "high", "medium", or "low".
- enough_information should be false if sources are off-topic, too few, too old for the task, or only weak snippets.
- mention conflicts if sources disagree.
- recommended_next_query should be a string when another search would help, otherwise null.

User question:
{user_question}

Sources:
{sources}
"""
    response, error = ask_model(prompt, json_mode=True, max_tokens=250)
    if not response:
        return {
            "enough_information": False,
            "confidence": "low",
            "missing_information": ["Evidence evaluator did not return a response."],
            "conflicts": [],
            "recommended_next_query": None,
            "reason": error or "empty evaluator response",
        }

    parsed = parse_json_object(response)
    confidence = parsed.get("confidence", "low")
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    recommended_next_query = parsed.get("recommended_next_query")
    if not isinstance(recommended_next_query, str):
        recommended_next_query = None

    reason = parsed.get("reason")
    if not isinstance(reason, str):
        reason = "Evaluator returned incomplete JSON."

    return {
        "enough_information": bool(parsed.get("enough_information", False)),
        "confidence": confidence,
        "missing_information": normalize_string_list(parsed.get("missing_information")),
        "conflicts": normalize_string_list(parsed.get("conflicts")),
        "recommended_next_query": recommended_next_query,
        "reason": reason,
    }


def format_evaluation_summary(evaluation: dict[str, object]) -> str:
    enough = "yes" if evaluation.get("enough_information") else "no"
    confidence = evaluation.get("confidence", "low")
    reason = evaluation.get("reason", "No evaluator reason provided.")

    lines = [
        "### Evidence Check",
        f"Enough information: {enough}",
        f"Confidence: {confidence}",
        f"Reason: {reason}",
    ]

    missing = evaluation.get("missing_information", [])
    if missing:
        lines.append("Missing information:")
        lines.extend(f"- {item}" for item in missing)

    conflicts = evaluation.get("conflicts", [])
    if conflicts:
        lines.append("Conflicts:")
        lines.extend(f"- {item}" for item in conflicts)

    next_query = evaluation.get("recommended_next_query")
    if next_query:
        lines.append(f"Suggested next search: {next_query}")

    return "\n".join(lines)
