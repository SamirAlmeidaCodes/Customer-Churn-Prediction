"""Request and response models for the scoring API."""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from churn.config import (
    EXAMPLES_PATH,
    MONTHLY_CHARGES_MAX,
    TENURE_MAX,
    TENURE_MIN,
    TOTAL_CHARGES_MAX,
)

EXAMPLES = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


class Gender(str, Enum):
    female = "Female"
    male = "Male"


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class MultipleLinesOption(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone = "No phone service"


class InternetServiceOption(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    none = "No"


class InternetAddon(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"


class ContractOption(str, Enum):
    month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodOption(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


def _options(enum_cls: type[Enum]) -> list[str]:
    return [member.value for member in enum_cls]


class CustomerRecord(BaseModel):
    """One telco customer. Field names match the training columns."""

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLES["high_risk"]]})

    gender: Gender
    SeniorCitizen: Literal[0, 1] = Field(description="1 if the customer is a senior, else 0")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=TENURE_MIN, le=TENURE_MAX, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: MultipleLinesOption
    InternetService: InternetServiceOption
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: ContractOption
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodOption
    MonthlyCharges: float = Field(ge=0, le=MONTHLY_CHARGES_MAX)
    TotalCharges: float = Field(ge=0, le=TOTAL_CHARGES_MAX)

    @field_validator("SeniorCitizen", mode="before")
    @classmethod
    def parse_senior_citizen(cls, value):
        if isinstance(value, bool):
            raise ValueError("SeniorCitizen must be 0 or 1")
        if isinstance(value, str) and value.strip() in {"0", "1"}:
            return int(value.strip())
        return value


class PredictionResponse(BaseModel):
    """Churn probability, outreach tier, and the 0.50 Churn/Stay label."""

    churn_probability: float = Field(ge=0, le=1)
    risk_tier: Literal["Low", "Medium", "High"]
    prediction: Literal["Churn", "Stay"]


FORM_FIELDS = [
    {"name": "gender", "label": "Gender", "kind": "select", "options": _options(Gender)},
    {
        "name": "SeniorCitizen",
        "label": "Senior citizen",
        "kind": "select",
        "options": ["0", "1"],
    },
    {"name": "Partner", "label": "Partner", "kind": "select", "options": _options(YesNo)},
    {
        "name": "Dependents",
        "label": "Dependents",
        "kind": "select",
        "options": _options(YesNo),
    },
    {
        "name": "tenure",
        "label": "Tenure (months)",
        "kind": "number",
        "min": TENURE_MIN,
        "max": TENURE_MAX,
        "step": "1",
    },
    {
        "name": "PhoneService",
        "label": "Phone service",
        "kind": "select",
        "options": _options(YesNo),
    },
    {
        "name": "MultipleLines",
        "label": "Multiple lines",
        "kind": "select",
        "options": _options(MultipleLinesOption),
    },
    {
        "name": "InternetService",
        "label": "Internet service",
        "kind": "select",
        "options": _options(InternetServiceOption),
    },
    {
        "name": "OnlineSecurity",
        "label": "Online security",
        "kind": "select",
        "options": _options(InternetAddon),
    },
    {
        "name": "OnlineBackup",
        "label": "Online backup",
        "kind": "select",
        "options": _options(InternetAddon),
    },
    {
        "name": "DeviceProtection",
        "label": "Device protection",
        "kind": "select",
        "options": _options(InternetAddon),
    },
    {
        "name": "TechSupport",
        "label": "Tech support",
        "kind": "select",
        "options": _options(InternetAddon),
    },
    {
        "name": "StreamingTV",
        "label": "Streaming TV",
        "kind": "select",
        "options": _options(InternetAddon),
    },
    {
        "name": "StreamingMovies",
        "label": "Streaming movies",
        "kind": "select",
        "options": _options(InternetAddon),
    },
    {
        "name": "Contract",
        "label": "Contract",
        "kind": "select",
        "options": _options(ContractOption),
    },
    {
        "name": "PaperlessBilling",
        "label": "Paperless billing",
        "kind": "select",
        "options": _options(YesNo),
    },
    {
        "name": "PaymentMethod",
        "label": "Payment method",
        "kind": "select",
        "options": _options(PaymentMethodOption),
    },
    {
        "name": "MonthlyCharges",
        "label": "Monthly charges",
        "kind": "number",
        "min": 0,
        "max": MONTHLY_CHARGES_MAX,
        "step": "0.01",
    },
    {
        "name": "TotalCharges",
        "label": "Total charges",
        "kind": "number",
        "min": 0,
        "max": TOTAL_CHARGES_MAX,
        "step": "0.01",
    },
]
