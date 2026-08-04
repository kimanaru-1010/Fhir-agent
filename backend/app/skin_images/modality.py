from __future__ import annotations

from app.core.config import settings
from app.skin_diagnostic.llm_client import call_llm
from app.skin_images.image_processing import ProcessedImage, resize_image_for_modality


MODALITY_SYSTEM_PROMPT = """You classify medical image modality.
Return exactly one code: DX, CT, MR, US, or XC.
DX is X-ray. CT is computed tomography. MR is MRI. US is ultrasound.
XC is dermatology or ordinary color skin photography.
Return only the code."""


def modality_display(modality: str) -> str:
    return {
        "DX": "Digital Radiography",
        "CT": "Computed Tomography",
        "MR": "Magnetic Resonance",
        "US": "Ultrasound",
        "XC": "Dermatology",
    }.get(modality.upper(), modality.upper())


async def classify_skin_modality(image: ProcessedImage) -> tuple[str, str]:
    base_url = settings.skin_vision_base_url or settings.internal_llm_base_url
    model = settings.skin_vision_model or settings.internal_llm_model
    classifier_image = resize_image_for_modality(image)
    result = await call_llm(
        prompt="Classify this medical image modality. Return one code only.",
        base_url=base_url,
        model=model,
        image_data_uri=classifier_image.data_uri,
        system_prompt=MODALITY_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=10,
    )
    modality = (result or "").strip().upper().split()[0] if result else "XC"
    if modality not in {"DX", "CT", "MR", "US", "XC"}:
        modality = "XC"
    return modality, modality_display(modality)
