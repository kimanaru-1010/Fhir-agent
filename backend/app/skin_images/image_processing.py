from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from fastapi import UploadFile


@dataclass(frozen=True)
class ProcessedImage:
    raw: bytes
    content_type: str
    data_uri: str
    size: int


def _encode_image(raw: bytes, *, max_size: int | None, jpeg_quality: int) -> ProcessedImage:
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    original_format = image.format or "JPEG"

    if max_size:
        ratio = min(max_size / image.width, max_size / image.height, 1.0)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    output = io.BytesIO()
    if original_format.upper() == "PNG":
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(output, format="PNG")
        content_type = "image/png"
    else:
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(output, format="JPEG", quality=jpeg_quality)
        content_type = "image/jpeg"

    image.close()
    normalized = output.getvalue()
    encoded = base64.b64encode(normalized).decode("ascii")
    return ProcessedImage(
        raw=normalized,
        content_type=content_type,
        data_uri=f"data:{content_type};base64,{encoded}",
        size=len(normalized),
    )


async def normalize_uploaded_skin_image(image: UploadFile) -> ProcessedImage:
    raw = await image.read()
    return _encode_image(raw, max_size=None, jpeg_quality=92)


def resize_image_for_modality(image: ProcessedImage) -> ProcessedImage:
    return _encode_image(image.raw, max_size=768, jpeg_quality=75)


def binary_data(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")
