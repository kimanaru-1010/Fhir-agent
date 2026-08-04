from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SkinImageAnalyzeResponse(BaseModel):
    binary_id: str
    media_id: str
    diagnostic_report_id: str
    modality: str
    analysis_text: str
    image_url: str
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
