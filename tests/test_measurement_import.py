import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.agents.llms import POWERFUL_LLM
from app.schemas.measurement_import import (
    MeasurementImportExtraction,
    MeasurementImportPreview,
)
from app.services.measurement_import import (
    AttachmentProcessingError,
    normalize_measurement_attachment,
)


class MeasurementImportTest(unittest.IsolatedAsyncioTestCase):
    async def test_csv_normalization_returns_typed_records_and_row_issues(self) -> None:
        structured_model = Mock()
        structured_model.ainvoke = AsyncMock(
            return_value={
                "records": [
                    {
                        "kind": "water",
                        "source_reference": "historico.csv, linha 2",
                        "registration_date": "2026-03-01",
                        "start_hydrometer": "120.5000",
                        "end_hydrometer": "135.8000",
                    }
                ],
                "issues": [
                    {
                        "source_reference": "historico.csv, linha 3",
                        "message": "A unidade do consumo está ambígua.",
                    }
                ],
            }
        )
        with patch(
            "app.services.measurement_import.get_structured_chat_model",
            return_value=structured_model,
        ) as get_model:
            preview = await normalize_measurement_attachment(
                "historico.csv",
                b"data,leitura inicial,leitura final\n2026-03-01,120.5,135.8\n2026-03-02,140,150",
            )

        self.assertIsInstance(preview, MeasurementImportPreview)
        self.assertEqual(preview.filename, "historico.csv")
        self.assertEqual(len(preview.records), 1)
        self.assertEqual(preview.records[0].kind, "water")
        self.assertEqual(str(preview.records[0].start_hydrometer), "120.5000")
        self.assertEqual(preview.issues[0].source_reference, "historico.csv, linha 3")
        self.assertTrue(preview.requires_confirmation)
        get_model.assert_called_once_with(MeasurementImportExtraction, POWERFUL_LLM)
        structured_model.ainvoke.assert_awaited_once()

    async def test_xlsx_normalization_reads_measurement_rows(self) -> None:
        from io import BytesIO

        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["data", "leitura inicial", "leitura final"])
        sheet.append(["2026-03-01", 120.5, 135.8])
        content = BytesIO()
        workbook.save(content)
        structured_model = Mock()
        structured_model.ainvoke = AsyncMock(
            return_value={
                "records": [
                    {
                        "kind": "water",
                        "source_reference": "historico.xlsx, aba Sheet, linha 2",
                        "registration_date": "2026-03-01",
                        "start_hydrometer": "120.5",
                        "end_hydrometer": "135.8",
                    }
                ],
                "issues": [],
            }
        )

        with patch(
            "app.services.measurement_import.get_structured_chat_model",
            return_value=structured_model,
        ):
            preview = await normalize_measurement_attachment(
                "historico.xlsx",
                content.getvalue(),
            )

        self.assertEqual(len(preview.records), 1)
        self.assertEqual(preview.records[0].source_reference, "historico.xlsx, aba Sheet, linha 2")
        self.assertEqual(str(preview.records[0].end_hydrometer), "135.8")

    async def test_rejects_pdf_without_searchable_text(self) -> None:
        from io import BytesIO

        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        content = BytesIO()
        writer.write(content)

        with self.assertRaises(AttachmentProcessingError) as raised:
            await normalize_measurement_attachment("scan.pdf", content.getvalue())

        self.assertEqual(raised.exception.status_code, 422)

    async def test_rejects_csv_above_record_limit(self) -> None:
        content = "date,value\\n" + "".join(
            f"2026-03-01,{row}\\n" for row in range(1, 502)
        )
        with self.assertRaises(AttachmentProcessingError) as raised:
            await normalize_measurement_attachment("large.csv", content.encode())

        self.assertEqual(raised.exception.status_code, 413)

    async def test_rejects_unsupported_file_before_calling_model(self) -> None:
        with (
            patch("app.services.measurement_import.get_structured_chat_model") as get_model,
            self.assertRaises(AttachmentProcessingError) as raised,
        ):
            await normalize_measurement_attachment("history.xls", b"not an xlsx")

        self.assertEqual(raised.exception.status_code, 415)
        get_model.assert_not_called()

    async def test_rejects_csv_with_no_measurement_rows(self) -> None:
        with self.assertRaises(AttachmentProcessingError) as raised:
            await normalize_measurement_attachment("empty.csv", b"date,value\n")

        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
