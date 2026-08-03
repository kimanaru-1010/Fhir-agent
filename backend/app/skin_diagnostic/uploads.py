"""Upload handling for skin diagnostic images."""

from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile


UPLOADS_DIR = Path(__file__).resolve().parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


async def save_upload(image: UploadFile, run_id: str) -> tuple[str, str]:
    ext = Path(image.filename or "").suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    path = UPLOADS_DIR / f"{run_id}{ext}"
    content = await image.read()
    path.write_bytes(content)
    return str(path), f"/api/skin-diagnostics/uploads/{run_id}{ext}"

