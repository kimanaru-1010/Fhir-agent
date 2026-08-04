from __future__ import annotations

from datetime import datetime, timezone


def utc_now_fhir() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_binary_resource(
    *,
    binary_id: str,
    content_type: str,
    data: str,
    size: int,
    created_at: str,
) -> dict:
    return {
        "resourceType": "Binary",
        "id": binary_id,
        "contentType": content_type,
        "data": data,
        "size": size,
        "created": created_at,
    }


def build_media_resource(
    *,
    media_id: str,
    patient_id: str,
    binary_id: str,
    content_type: str,
    modality: str,
    modality_display: str,
    analysis_text: str,
    created_at: str,
) -> dict:
    return {
        "resourceType": "Media",
        "id": media_id,
        "status": "completed",
        "createdDateTime": created_at,
        "subject": {"reference": f"Patient/{patient_id}"},
        "content": {
            "contentType": content_type,
            "url": f"Binary/{binary_id}",
        },
        "modality": {
            "coding": [
                {
                    "system": "http://dicom.nema.org/resources/ontology/DCM",
                    "code": modality,
                    "display": modality_display,
                }
            ],
            "text": modality_display,
        },
        "note": [{"text": analysis_text}],
    }


def build_diagnostic_report_resource(
    *,
    report_id: str,
    patient_id: str,
    media_id: str,
    analysis_text: str,
    created_at: str,
) -> dict:
    return {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": "final",
        "code": {"text": "AI Skin Lesion Analysis"},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": created_at,
        "issued": created_at,
        "media": [{"link": {"reference": f"Media/{media_id}"}}],
        "conclusion": analysis_text,
    }


def build_skin_analysis_bundle(resources: list[dict]) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": resource} for resource in resources],
    }
