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
        "missing_data": ["period"],
        "source_count": 1,
    }
    assert "consumo real" not in str(summary)
    assert "periodo de analise" not in str(summary)
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


def test_specialist_summary_drops_unrecognized_free_text_slots() -> None:
    summary = specialist_result_summary(
        {
            "agent": "ranking",
            "status": "needs_input",
            "facts": [],
            "recommendations": [],
            "missing_data": [
                "Produtor João da Fazenda Alfa consumiu 9123 litros",
                "estado da fazenda",
            ],
            "sources": [],
        }
    )

    assert summary["missing_data"] == ["farm"]
    assert "João" not in str(summary)
    assert "9123" not in str(summary)
