from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: int = Field(ge=17, le=90)

    education_num: int = Field(
        alias="education-num",
        ge=1,
        le=16,
    )

    hours_per_week: int = Field(
        alias="hours-per-week",
        ge=1,
        le=99,
    )

    capital_gain: int = Field(
        alias="capital-gain",
        ge=0,
    )

    capital_loss: int = Field(
        alias="capital-loss",
        ge=0,
    )

    workclass: Literal[
        "Federal-gov",
        "Local-gov",
        "Never-worked",
        "Private",
        "Self-emp-inc",
        "Self-emp-not-inc",
        "State-gov",
        "Unknown",
        "Without-pay",
    ]

    marital_status: Literal[
        "Divorced",
        "Married-AF-spouse",
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Never-married",
        "Separated",
        "Widowed",
    ] = Field(alias="marital-status")

    occupation: Literal[
        "Adm-clerical",
        "Armed-Forces",
        "Craft-repair",
        "Exec-managerial",
        "Farming-fishing",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Other-service",
        "Priv-house-serv",
        "Prof-specialty",
        "Protective-serv",
        "Sales",
        "Tech-support",
        "Transport-moving",
        "Unknown",
    ]

    relationship: Literal[
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Own-child",
        "Unmarried",
        "Wife",
    ]


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_version: str