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


    async def test_energy_precision_over_two_decimal_places_becomes_review_issue(self) -> None:
        structured_model = Mock()
        structured_model.ainvoke = AsyncMock(
            return_value={
                "records": [
                    {
                        "kind": "energy",
                        "source_reference": "historico.csv, linha 2",
                        "registration_date": "2026-03-01",
                        "energy_consumption": "7.12",
                    },
                    {
                        "kind": "energy",
                        "source_reference": "historico.csv, linha 3",
                        "registration_date": "2026-03-02",
                        "energy_consumption": "7.125",
                    },
                ],
                "issues": [],
            }
        )
        with patch(
            "app.services.measurement_import.get_structured_chat_model",
            return_value=structured_model,
        ):
            preview = await normalize_measurement_attachment(
                "historico.csv",
                b"data,consumo\n2026-03-01,7.12\n2026-03-02,7.125",
            )

        self.assertEqual(len(preview.records), 1)
        self.assertEqual(str(preview.records[0].energy_consumption), "7.12")
        self.assertEqual(len(preview.issues), 1)
        self.assertEqual(
            preview.issues[0].source_reference,
            "historico.csv, linha 3",
        )
        self.assertEqual(
            preview.issues[0].message,
            "Data ou valores inválidos; corrija ou remova esta linha.",
        )

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


    async def test_xlsx_limits_use_cells_when_sheet_dimensions_are_missing(self) -> None:
        import re
        import zipfile
        from io import BytesIO

        from openpyxl import Workbook

        from app.services.measurement_import import (
            MAX_IMPORT_COLUMNS,
            MAX_IMPORT_RECORDS,
            _read_xlsx,
        )

        def without_dimensions(workbook: Workbook) -> bytes:
            generated = BytesIO()
            workbook.save(generated)
            stripped = BytesIO()
            with (
                zipfile.ZipFile(BytesIO(generated.getvalue())) as source,
                zipfile.ZipFile(stripped, "w") as destination,
            ):
                for entry in source.infolist():
                    content = source.read(entry.filename)
                    if entry.filename == "xl/worksheets/sheet1.xml":
                        content = re.sub(rb"<dimension[^>]*/>", b"", content)
                    destination.writestr(entry, content)
            return stripped.getvalue()

        rows_workbook = Workbook()
        worksheet = rows_workbook.active
        worksheet.append(["date", "value"])
        for row in range(MAX_IMPORT_RECORDS):
            worksheet.append(["2026-03-01", str(row)])
        overflow_row = MAX_IMPORT_RECORDS + 500
        worksheet.cell(row=overflow_row, column=1, value="2026-03-01")
        worksheet.cell(row=overflow_row, column=2, value="overflow")

        with self.assertRaises(AttachmentProcessingError) as raised_rows:
            _read_xlsx(without_dimensions(rows_workbook), "large.xlsx")

        self.assertEqual(raised_rows.exception.status_code, 413)

        columns_workbook = Workbook()
        columns_workbook.active.cell(
            row=1,
            column=MAX_IMPORT_COLUMNS + 1,
            value="overflow",
        )
        with self.assertRaises(AttachmentProcessingError) as raised_columns:
            _read_xlsx(without_dimensions(columns_workbook), "wide.xlsx")

        self.assertEqual(raised_columns.exception.status_code, 413)

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
        content = "date,value\n" + "".join(
            f"2026-03-01,{row}\n" for row in range(1, 502)
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

    async def test_csv_parser_errors_become_unprocessable_attachment(self) -> None:
        import csv

        previous_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(4)
            with self.assertRaises(AttachmentProcessingError) as raised:
                await normalize_measurement_attachment("bad.csv", b"date,value\n2026,12345\n")
        finally:
            csv.field_size_limit(previous_limit)

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIsInstance(raised.exception.__cause__, csv.Error)

    async def test_rejects_csv_with_no_measurement_rows(self) -> None:
        with self.assertRaises(AttachmentProcessingError) as raised:
            await normalize_measurement_attachment("empty.csv", b"date,value\n")

        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
