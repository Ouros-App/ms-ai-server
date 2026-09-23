from app.agents.diagnostics import (
    specialist_result_summary,
    specialist_results_summary,
)


def test_specialist_summary_exposes_metadata_not_business_facts() -> None:
    result = {
        "agent": "sustainability",
        "status": "needs_input",
        "facts": ["consumo real da fazenda: 123"],
        "recommendations": ["recomendacao privada"],
        "missing_data": [" periodo de analise ", "periodo de analise"],
        "sources": ["get_consumption_summary"],
        "_personal_data_required": True,
    }

    summary = specialist_result_summary(result)

    assert summary == {
        "agent": "sustainability",
        "status": "needs_input",
        "fact_count": 1,
        "recommendation_count": 1,
        "missing_data": ["periodo de analise"],
        "source_count": 1,
    }
    assert "consumo real" not in str(summary)
    assert "_personal_data_required" not in summary


def test_specialist_summary_handles_malformed_results_safely() -> None:
    assert specialist_result_summary("not-a-dict") == {
        "agent": "unknown",
        "status": "error",
        "fact_count": 0,
        "recommendation_count": 0,
        "missing_data": [],
        "source_count": 0,
    }
    assert specialist_results_summary(None) == []
