"""Pydantic schemas for the 4-step dermatology pipeline."""

from pydantic import BaseModel, Field


class Observation(BaseModel):
    description: str
    confidence: str  # "High" | "Medium" | "Low"


class DifferentialItem(BaseModel):
    disease: str
    rationale: str = ""


class VisualExtractOutput(BaseModel):
    observations: list[Observation]
    top_differentials: list[DifferentialItem]
    raw_text: str = ""


class ClinicalQuestion(BaseModel):
    question: str
    qrst_category: str  # "Q" | "R" | "S" | "T" | ""
    purpose: str = ""
    discriminates: list[str] = Field(default_factory=list)


class ClinicalPlannerOutput(BaseModel):
    verified_findings: str = ""
    additional_findings: str = ""
    uncertainty_assessment: str = ""
    questions: list[ClinicalQuestion]
    raw_text: str = ""


class DiagnosticReasoning(BaseModel):
    ranked_diagnoses: list[dict]
    overall_reasoning: str = ""
    remaining_uncertainty: str = ""
