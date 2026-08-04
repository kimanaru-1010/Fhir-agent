from __future__ import annotations

from urllib.parse import urlparse


def extract_binary_id(value: str | None) -> str:
    """Return the FHIR Binary id from an id, relative reference, or absolute URL."""
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""

    parsed = urlparse(raw_value)
    path = parsed.path if parsed.scheme and parsed.netloc else raw_value
    parts = [part for part in path.strip("/").split("/") if part]
    for index, part in enumerate(parts):
        if part == "Binary" and index + 1 < len(parts):
            return parts[index + 1]
    return parts[-1] if parts else ""


def build_image_api_url(value: str | None) -> str:
    binary_id = extract_binary_id(value)
    return f"/api/skin-images/files/{binary_id}" if binary_id else ""
