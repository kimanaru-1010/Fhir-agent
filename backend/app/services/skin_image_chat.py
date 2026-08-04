from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.message import ChatImageAttachment
from app.skin_images.neo4j_repository import search_patient_skin_images
from app.skin_images.schemas import SkinImageSearchFilters
from app.skin_images.search_filters import resolve_skin_image_filters


_IMAGE_TERMS = ("ảnh", "image", "photo", "photograph", "hình")
_SKIN_TERMS = ("da", "skin", "dermatology", "lesion", "tổn thương")
_TIME_RANGES = {
    "buổi sáng": "morning",
    "sáng": "morning",
    "morning": "morning",
    "buổi trưa": "noon",
    "trưa": "noon",
    "noon": "noon",
    "buổi chiều": "afternoon",
    "chiều": "afternoon",
    "afternoon": "afternoon",
    "buổi tối": "evening",
    "tối": "evening",
    "evening": "evening",
    "ban đêm": "night",
    "đêm": "night",
    "night": "night",
}


@dataclass(frozen=True)
class SkinImageChatResult:
    handled: bool
    response: str = ""
    attachments: list[ChatImageAttachment] | None = None


async def maybe_answer_skin_image_request(content: str) -> SkinImageChatResult:
    if not _looks_like_skin_image_request(content):
        return SkinImageChatResult(handled=False)

    filters = _extract_filters(content)
    if not filters.patient_id:
        return SkinImageChatResult(
            handled=True,
            response="Vui lòng cung cấp mã bệnh nhân cần xem ảnh da.",
            attachments=[],
        )

    try:
        resolved = resolve_skin_image_filters(filters)
    except ValueError as exc:
        return SkinImageChatResult(handled=True, response=str(exc), attachments=[])

    rows = await search_patient_skin_images(resolved)
    attachments = build_skin_image_attachments(rows)
    if not attachments:
        return SkinImageChatResult(
            handled=True,
            response="Không tìm thấy ảnh da phù hợp cho bệnh nhân này trong điều kiện được yêu cầu.",
            attachments=[],
        )

    order_text = "mới nhất đến cũ nhất" if resolved.sort == "desc" else "cũ nhất đến mới nhất"
    return SkinImageChatResult(
        handled=True,
        response=f"Tôi tìm thấy {len(attachments)} ảnh da của bệnh nhân {resolved.patient_id}, sắp xếp từ {order_text}.",
        attachments=attachments,
    )


def build_skin_image_attachments(rows: list[dict]) -> list[ChatImageAttachment]:
    attachments: list[ChatImageAttachment] = []
    for row in rows:
        if not row.get("image_url") or not row.get("binary_id"):
            continue
        attachments.append(
            ChatImageAttachment(
                patient_id=str(row["patient_id"]),
                diagnostic_report_id=str(row["diagnostic_report_id"]),
                media_id=str(row["media_id"]),
                binary_id=str(row["binary_id"]),
                url=str(row["image_url"]),
                content_type=row.get("content_type"),
                created_at=row.get("created_at"),
                title="Ảnh phân tích da",
                description=row.get("conclusion"),
            )
        )
    return attachments


def _looks_like_skin_image_request(content: str) -> bool:
    text = content.lower()
    has_image = any(term in text for term in _IMAGE_TERMS)
    has_skin = any(term in text for term in _SKIN_TERMS)
    wants_view = any(term in text for term in ("lấy", "xem", "cho xem", "find", "show", "retrieve", "list"))
    return has_image and (has_skin or "xc" in text) and wants_view


def _extract_filters(content: str) -> SkinImageSearchFilters:
    text = content.lower()
    patient_id = _extract_patient_id(text)
    count = _extract_count(text)
    sort = "asc" if any(term in text for term in ("cũ nhất", "oldest", "old ")) else "desc"
    if count is None and any(term in text for term in ("gần nhất", "mới nhất", "latest", "newest")):
        count = 1

    return SkinImageSearchFilters(
        patient_id=patient_id,
        modality="XC" if any(term in text for term in ("da", "skin", "dermatology", "lesion", "xc")) else None,
        date_range=_extract_date_range(text),
        specific_date=_extract_specific_date(text),
        specific_year=_extract_specific_year(text),
        time=_extract_time(text),
        time_range=_extract_time_range(text),
        last_N_minutes=_extract_last_minutes(text),
        sort=sort,
        count=count or 5,
    )


def _extract_patient_id(text: str) -> str | None:
    patterns = [
        r"(?:bệnh nhân|patient|patient id|mã bệnh nhân)\s*[:#-]?\s*(\d+)",
        r"\bpatient/(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_count(text: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s*(?:ảnh|image|images|photo|photos)\b", text)
    if not match:
        return None
    return min(max(int(match.group(1)), 1), 10)


def _extract_date_range(text: str) -> str | None:
    mapping = {
        "hôm nay": "today",
        "today": "today",
        "hôm qua": "yesterday",
        "yesterday": "yesterday",
        "tuần này": "this_week",
        "this week": "this_week",
        "tuần trước": "last_week",
        "last week": "last_week",
        "tháng này": "this_month",
        "this month": "this_month",
        "tháng trước": "last_month",
        "last month": "last_month",
        "năm nay": "this_year",
        "this year": "this_year",
        "năm trước": "last_year",
        "last year": "last_year",
        "gần đây": "recent",
        "recent": "recent",
    }
    for term, value in mapping.items():
        if term in text:
            return value
    return None


def _extract_specific_date(text: str) -> str | None:
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", text)
    return match.group(1) if match else None


def _extract_specific_year(text: str) -> str | None:
    match = re.search(r"(?:năm|year)\s*(20\d{2}|19\d{2})", text)
    return match.group(1) if match else None


def _extract_time(text: str) -> str | None:
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    match = re.search(r"(?:lúc|at)\s*([01]?\d|2[0-3])\s*(?:giờ|h)?\b", text)
    return f"{int(match.group(1)):02d}:00" if match else None


def _extract_time_range(text: str) -> str | None:
    for term, value in _TIME_RANGES.items():
        if term in text:
            return value
    return None


def _extract_last_minutes(text: str) -> int | None:
    match = re.search(r"(?:trong|last|within)\s*(\d{1,4})\s*(?:phút|minutes?|mins?)", text)
    return int(match.group(1)) if match else None
