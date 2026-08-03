"""Answer normalization helpers for skin diagnostic interviews."""

from __future__ import annotations

from fastapi import HTTPException


_YES = {"yes", "y", "true", "1", "co", "có", "dung", "đúng", "ok", "roi", "rồi"}
_NO = {"no", "n", "false", "0", "khong", "không", "sai", "chua", "chưa"}


def normalize_yes_no(answer: str, *, question_num: int | None = None) -> str:
    normalized = (answer or "").strip().lower()
    if normalized in _YES:
        return "Yes"
    if normalized in _NO:
        return "No"

    suffix = f" for question {question_num}" if question_num is not None else ""
    raise HTTPException(
        status_code=400,
        detail=f"Invalid Yes/No answer{suffix}. Use yes/no or co/khong.",
    )

