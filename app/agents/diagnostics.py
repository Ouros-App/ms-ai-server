import unicodedata
from typing import Any


def _list_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


_MISSING_SLOT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("period", ("period", "janela", "dia", "semana", "mes", "ciclo")),
    ("farm", ("fazenda", "granja", "propriedade")),
    ("confirmation", ("confirm", "sim ou nao", "sim/nao")),
    ("number", ("quantidade", "numero", "aves", "frangos", "capacidade", "valor")),
)


def _normalize_slot_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    stripped = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(stripped.split()).strip()


def missing_slot_kinds(value: object) -> set[str]:
    """Classify model-provided missing-data text into bounded slot categories."""
    if not isinstance(value, list):
        return set()

    kinds: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = _normalize_slot_text(item)
        for slot_kind, markers in _MISSING_SLOT_PATTERNS:
            if any(marker in normalized for marker in markers):
                kinds.add(slot_kind)
    return kinds


def _bounded_missing_data(value: object) -> list[str]:
    """Expose only coarse slot kinds, never model-generated free text."""
    kinds = missing_slot_kinds(value)
    return [
        slot_kind
        for slot_kind, _markers in _MISSING_SLOT_PATTERNS
        if slot_kind in kinds
    ]


def pending_summary(
    missing_data: object,
    by_route: object,
) -> tuple[list[str], dict[str, list[str]]]:
    """Bound pending-state diagnostics while preserving internal routing state."""
    bounded_by_route: dict[str, list[str]] = {}
    if isinstance(by_route, dict):
        for route, values in list(by_route.items())[:4]:
            if isinstance(route, str):
                bounded_by_route[route] = _bounded_missing_data(values)
    return _bounded_missing_data(missing_data), bounded_by_route


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
