from __future__ import annotations

from app.core.config import settings
from app.skin_diagnostic.llm_client import call_llm
from app.skin_images.image_processing import ProcessedImage


SKIN_IMAGE_SYSTEM_PROMPT = """You are a dermatology clinician.
Analyze the uploaded skin image.
Only describe visible findings. Do not claim symptoms that are not visible.
Return concise clinical text with:
1. Lesion description
2. Differential diagnosis
3. Suggested next information or care considerations
This is decision support, not a definitive diagnosis."""


async def analyze_skin_image(image: ProcessedImage) -> str:
    base_url = settings.skin_vision_base_url or settings.internal_llm_base_url
    model = settings.skin_vision_model or settings.internal_llm_model
    text = await call_llm(
        prompt="Analyze this skin image and provide a concise dermatology assessment.",
        base_url=base_url,
        model=model,
        image_data_uri=image.data_uri,
        system_prompt=SKIN_IMAGE_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=settings.skin_llm_max_tokens,
    )
    return text.strip()
