from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SkinImageAnalyzeResponse(BaseModel):
    binary_id: str
    media_id: str
    diagnostic_report_id: str
    modality: str
    analysis_text: str
    image_url: str
    content_type: str | None = None
    created_at: datetime


class SkinImageSummary(BaseModel):
    diagnostic_report_id: str
    media_id: str | None = None
    binary_id: str | None = None
    modality: str | None = None
    conclusion: str
    image_url: str | None = None
    created_at: str | None = None


class SkinImageListResponse(BaseModel):
    items: list[SkinImageSummary]


class SkinImageDetailResponse(BaseModel):
    diagnostic_report_id: str
    media_id: str | None = None
    binary_id: str | None = None
    modality: str | None = None
    conclusion: str
    image_url: str | None = None
    created_at: str | None = None


class SkinImageSearchFilters(BaseModel):
    patient_id: str | None = None
    modality: str | None = None
    date_range: str | None = None
    specific_date: str | None = None
    specific_year: str | None = None
    time: str | None = None
    time_range: str | None = None
    last_N_minutes: int | None = None
    sort: Literal["asc", "desc"] = "desc"
    count: int | None = Field(default=5, ge=1)


class ResolvedSkinImageSearchFilters(BaseModel):
    patient_id: str
    modality: str | None = None
    from_datetime: datetime | None = None
    to_datetime: datetime | None = None
    sort: Literal["asc", "desc"] = "desc"
    count: int | None = Field(default=5, ge=1)


class SkinImageSearchResult(BaseModel):
    diagnostic_report_id: str
    media_id: str
    binary_id: str
    patient_id: str
    modality: str | None = None
    created_at: str | None = None
    conclusion: str | None = None
    image_url: str
    content_type: str | None = None
