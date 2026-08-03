"""Schemas for the skin diagnostic workflow API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerItem(BaseModel):
    question_num: int | None = None
    answer: str = Field(min_length=1, max_length=100)


class BulkAnswerRequest(BaseModel):
    answers: list[AnswerItem] = Field(min_length=1)


class PendingQuestion(BaseModel):
    question: str = ""
    pqrst_category: str = ""
    purpose: str = ""
    discriminates: list[str] = Field(default_factory=list)
    question_num: int | None = None
    total: int | None = None


class SkinDiagnosticResult(BaseModel):
    ranked_diagnoses: list[dict] = Field(default_factory=list)
    reasoning: str = ""
    visual_observations: str = ""
    visual_differentials: list[str] = Field(default_factory=list)
    qa_history: str = ""


class SkinDiagnosticStartResponse(BaseModel):
    run_id: str
    status: str
    current_step: str


class SkinDiagnosticAnswersResponse(BaseModel):
    status: str
    current_step: str


class SkinDiagnosticStatusResponse(BaseModel):
    run_id: str
    status: str
    current_step: str = ""
    progress: int = 0
    pending_questions: list[PendingQuestion] | None = None
    result: SkinDiagnosticResult | dict = Field(default_factory=dict)
    error: str | None = None


class SkinDiagnosticDetailResponse(BaseModel):
    run_id: str
    status: str
    current_step: str = ""
    image_url: str = ""
    anamnesis: str = ""
    step_history: list[str] = Field(default_factory=list)
    pending_questions: list[PendingQuestion] | None = None
    result: SkinDiagnosticResult | None = None
    error: str | None = None

