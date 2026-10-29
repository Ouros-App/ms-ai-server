from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MeasurementRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["water", "energy"]
    source_reference: str = Field(min_length=1, max_length=120)
    registration_date: date
    start_hydrometer: Decimal | None = None
    end_hydrometer: Decimal | None = None
    energy_consumption: Decimal | None = None

    @model_validator(mode="after")
    def validate_measurement(self):
        if self.registration_date > date.today():
            raise ValueError("A data do registro não pode ser futura.")

        if self.kind == "water":
            if self.energy_consumption is not None:
                raise ValueError("Registro de água não pode conter consumo de energia.")
            if self.start_hydrometer is None or self.end_hydrometer is None:
                raise ValueError("Leituras inicial e final são obrigatórias para água.")
            for value in (self.start_hydrometer, self.end_hydrometer):
                if not value.is_finite() or value <= 0:
                    raise ValueError("As leituras de água devem ser positivas.")
                integer_digits = max(len(value.as_tuple().digits) + value.as_tuple().exponent, 0)
                decimal_digits = max(-value.as_tuple().exponent, 0)
                if integer_digits > 15 or decimal_digits > 4:
                    raise ValueError("A leitura de água excede a precisão aceita pela API.")
            if self.end_hydrometer < self.start_hydrometer:
                raise ValueError("A leitura final não pode ser menor que a inicial.")
        else:
            if self.start_hydrometer is not None or self.end_hydrometer is not None:
                raise ValueError("Registro de energia não pode conter leituras de hidrômetro.")
            if (
                self.energy_consumption is None
                or not self.energy_consumption.is_finite()
                or self.energy_consumption <= 0
            ):
                raise ValueError("O consumo de energia deve ser positivo.")
            energy = self.energy_consumption
            integer_digits = max(len(energy.as_tuple().digits) + energy.as_tuple().exponent, 0)
            decimal_digits = max(-energy.as_tuple().exponent, 0)
            if integer_digits > 15 or decimal_digits > 2:
                raise ValueError("O consumo de energia excede a precisão aceita pela API.")
        return self


class MeasurementImportIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_reference: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=240)


class MeasurementImportPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    records: list[MeasurementRecord] = Field(default_factory=list, max_length=500)
    issues: list[MeasurementImportIssue] = Field(default_factory=list, max_length=500)
    requires_confirmation: Literal[True] = True


class ExtractedMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["water", "energy"]
    source_reference: str = Field(min_length=1, max_length=120)
    registration_date: str
    start_hydrometer: str | None = None
    end_hydrometer: str | None = None
    energy_consumption: str | None = None


class ExtractedImportIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_reference: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=240)


class MeasurementImportExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[ExtractedMeasurement] = Field(default_factory=list, max_length=500)
    issues: list[ExtractedImportIssue] = Field(default_factory=list, max_length=500)
