# src/serving/schemas.py

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):

    ankle: float

    arm_length: float = Field(
        alias="arm-length"
    )

    bicep: float

    calf: float

    chest: float

    forearm: float

    height: float

    hip: float

    leg_length: float = Field(
        alias="leg-length"
    )

    shoulder_breadth: float = Field(
        alias="shoulder-breadth"
    )

    shoulder_to_crotch: float = Field(
        alias="shoulder-to-crotch"
    )

    thigh: float

    waist: float

    wrist: float

    class Config:
        populate_by_name = True