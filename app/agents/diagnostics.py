from typing import Any


def _list_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _bounded_missing_data(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = " ".join(item.split()).strip()
        if not normalized or normalized in items:
            continue
        items.append(normalized[:200])
        if len(items) >= 3:
            break
    return items


def specialist_result_summary(result: object) -> dict[str, Any]:
    """Return debug metadata without exposing specialist facts or recommendations."""
    if not isinstance(result, dict):
        return {
            "agent": "unknown",
            "status": "error",
            "fact_count": 0,
            "recommendation_count": 0,
            "missing_data": [],
            "source_count": 0,
        }

    agent = result.get("agent")
    status = result.get("status")
    return {
        "agent": agent if isinstance(agent, str) else "unknown",
        "status": status if isinstance(status, str) else "error",
        "fact_count": _list_count(result.get("facts")),
        "recommendation_count": _list_count(result.get("recommendations")),
        "missing_data": _bounded_missing_data(result.get("missing_data")),
        "source_count": _list_count(result.get("sources")),
    }


def specialist_results_summary(results: object) -> list[dict[str, Any]]:
    """Summarize all specialist results for traces and the debug console."""
    if not isinstance(results, list):
        return []
    return [specialist_result_summary(result) for result in results]
