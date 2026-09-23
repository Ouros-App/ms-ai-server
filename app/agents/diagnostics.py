from typing import Any


def _list_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


_MISSING_SLOT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("period", ("period", "janela", "dia", "semana", "mes", "ciclo")),
    ("farm", ("fazenda", "granja", "propriedade")),
    ("confirmation", ("confirm", "sim ou nao", "sim/nao")),
    ("number", ("quantidade", "numero", "aves", "frangos", "capacidade", "valor")),
)


def _bounded_missing_data(value: object) -> list[str]:
    """Expose only coarse slot kinds, never model-generated free text."""
    if not isinstance(value, list):
        return []

    slot_kinds: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = " ".join(item.lower().split()).strip()
        for slot_kind, markers in _MISSING_SLOT_PATTERNS:
            if any(marker in normalized for marker in markers):
                if slot_kind not in slot_kinds:
                    slot_kinds.append(slot_kind)
                break
        if len(slot_kinds) >= 3:
            break
    return slot_kinds


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
