from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.graph.client import execute_cypher
from app.skin_images.fhir_builders import build_skin_analysis_bundle
from app.skin_images.references import build_image_api_url, extract_binary_id
from app.skin_images.schemas import ResolvedSkinImageSearchFilters


FHIR_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SUPPORTED_RESOURCE_TYPES = {"Binary", "Media", "DiagnosticReport"}
CYFHIR_CONFIG = {"validation": False, "version": "R4"}
logger = logging.getLogger(__name__)


def _resource_type(resource: dict) -> str:
    value = str(resource.get("resourceType", "")).strip()
    if not FHIR_LABEL_RE.match(value):
        raise ValueError(f"Invalid FHIR resourceType: {value}")
    if value not in SUPPORTED_RESOURCE_TYPES:
        raise ValueError(f"Unsupported skin image resource type: {value}")
    return value


async def patient_exists(patient_id: str) -> bool:
    rows = await execute_cypher(
        """
        MATCH (patient:FHIRResource:Patient {
          resourceType: "Patient",
          id: $patient_id
        })
        RETURN patient.id AS id
        LIMIT 1
        """,
        {"patient_id": patient_id},
        collect=False,
    )
    return bool(rows)


async def save_skin_analysis(
    resources: list[dict],
    *,
    patient_id: str,
    patient_already_validated: bool = False,
) -> dict[str, str]:
    """Persist skin-analysis FHIR JSON through the CyFHIR Neo4j plugin.

    This intentionally does not create graph nodes by hand. It builds valid
    FHIR-like JSON in the caller, then delegates JSON-to-graph conversion and
    reference resolution to CyFHIR's jar procedures.
    """
    if not patient_already_validated and not await patient_exists(patient_id):
        raise ValueError("Linked Patient was not found in Neo4j")

    ids = _collect_resource_ids(resources)
    stats = await _load_bundle_with_cyfhir(resources)
    _validate_bundle_load_result(stats, expected_count=len(resources))
    return {
        "binary_id": ids.get("Binary", ""),
        "media_id": ids.get("Media", ""),
        "diagnostic_report_id": ids.get("DiagnosticReport", ""),
    }


def _collect_resource_ids(resources: list[dict]) -> dict[str, str]:
    ids: dict[str, str] = {}
    for resource in resources:
        resource_type = _resource_type(resource)
        ids[resource_type] = str(resource["id"])
    return ids


async def _load_bundle_with_cyfhir(resources: list[dict]) -> dict[str, Any]:
    rows = await execute_cypher(
        """
        CALL cyfhir.bundle.load($json, $config) YIELD value
        RETURN value
        """,
        {
            "json": json.dumps(build_skin_analysis_bundle(resources), ensure_ascii=False),
            "config": CYFHIR_CONFIG,
        },
        collect=False,
        timeout=120.0,
    )
    if not rows:
        raise RuntimeError("CyFHIR bundle load returned no result")
    value = rows[0].get("value") or {}
    if not isinstance(value, dict):
        raise RuntimeError("CyFHIR bundle load returned invalid result")
    return value


def _stat_int(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _validate_bundle_load_result(stats: dict[str, Any], *, expected_count: int) -> None:
    loaded_resources = _stat_int(stats, "loadedResources")
    skipped_entries = _stat_int(stats, "skippedEntries")
    ambiguous = _stat_int(stats, "referencesAmbiguous")
    pending = _stat_int(stats, "referencesPending")

    logger.info(
        "skin_image.cyfhir loaded=%s resolved=%s pending=%s ambiguous=%s attachments=%s",
        stats.get("loadedResources"),
        stats.get("referencesResolved"),
        stats.get("referencesPending"),
        stats.get("referencesAmbiguous"),
        stats.get("attachmentRelationships"),
    )

    if loaded_resources != expected_count:
        raise RuntimeError(f"CyFHIR loaded {loaded_resources}/{expected_count} resources")
    if skipped_entries:
        raise RuntimeError(f"CyFHIR skipped {skipped_entries} bundle entries")
    if ambiguous:
        raise RuntimeError(f"CyFHIR found {ambiguous} ambiguous references")
    if pending:
        logger.warning("CyFHIR bundle load left %d pending references", pending)


async def list_skin_images(patient_id: str | None = None) -> list[dict[str, Any]]:
    rows = await execute_cypher(
        """
        MATCH (subject:Reference)<-[:subject]-(report:FHIRResource:DiagnosticReport)
        WHERE $patient_id IS NULL OR subject.reference = "Patient/" + $patient_id
        OPTIONAL MATCH (report)-[:code]->(code:FHIR_ELEMENT:code)
        OPTIONAL MATCH (subject)-[:RESOLVES_TO]->(patient:FHIRResource:Patient)
        WITH subject, patient, report, code
        WHERE code.text IS NULL OR code.text = "AI Skin Lesion Analysis"
        MATCH (report)-[:media]->(:FHIR_ELEMENT:media)
          -[:link]->(:Reference)-[:RESOLVES_TO]->(media:FHIRResource:Media)
        OPTIONAL MATCH (media)-[:content]->(content:FHIR_ELEMENT:content)
        OPTIONAL MATCH (content)-[:RESOLVES_TO]->(binary:FHIRResource:Binary)
        OPTIONAL MATCH (media)-[:modality]->(:FHIR_ELEMENT:modality)
          -[:coding]->(coding:FHIR_ELEMENT:Coding)
        RETURN report.id AS diagnostic_report_id,
               coalesce(patient.id, replace(subject.reference, "Patient/", "")) AS patient_id,
               report.conclusion AS conclusion,
               report.issued AS created_at,
               media.id AS media_id,
               binary.id AS binary_id,
               CASE
                 WHEN binary.id IS NULL THEN null
                 ELSE "/api/skin-images/files/" + binary.id
               END AS image_url,
               coding.code AS modality
        ORDER BY report.issued DESC
        """,
        {"patient_id": patient_id},
        collect=False,
    )
    return rows


async def get_skin_image_detail(report_id: str) -> dict[str, Any] | None:
    rows = await execute_cypher(
        """
        MATCH (subject:Reference)<-[:subject]-(report:FHIRResource:DiagnosticReport {id: $report_id})
        OPTIONAL MATCH (report)-[:code]->(code:FHIR_ELEMENT:code)
        OPTIONAL MATCH (subject)-[:RESOLVES_TO]->(patient:FHIRResource:Patient)
        WITH subject, patient, report, code
        WHERE code.text IS NULL OR code.text = "AI Skin Lesion Analysis"
        OPTIONAL MATCH (report)-[:media]->(:FHIR_ELEMENT:media)
          -[:link]->(:Reference)-[:RESOLVES_TO]->(media:FHIRResource:Media)
        OPTIONAL MATCH (media)-[:content]->(content:FHIR_ELEMENT:content)
        OPTIONAL MATCH (content)-[:RESOLVES_TO]->(binary:FHIRResource:Binary)
        OPTIONAL MATCH (media)-[:modality]->(:FHIR_ELEMENT:modality)
          -[:coding]->(coding:FHIR_ELEMENT:Coding)
        RETURN report.id AS diagnostic_report_id,
               coalesce(patient.id, replace(subject.reference, "Patient/", "")) AS patient_id,
               report.conclusion AS conclusion,
               report.issued AS created_at,
               media.id AS media_id,
               binary.id AS binary_id,
               CASE
                 WHEN binary.id IS NULL THEN null
                 ELSE "/api/skin-images/files/" + binary.id
               END AS image_url,
               coding.code AS modality
        LIMIT 1
        """,
        {"report_id": report_id},
        collect=False,
    )
    return rows[0] if rows else None


async def get_binary_for_skin_image(binary_id: str) -> dict[str, Any] | None:
    rows = await execute_cypher(
        """
        MATCH (patient:FHIRResource:Patient)
          <-[:RESOLVES_TO]-(:Reference)
          <-[:subject]-(report:FHIRResource:DiagnosticReport)
          -[:media]->(:FHIR_ELEMENT:media)
          -[:link]->(:Reference)-[:RESOLVES_TO]->(:FHIRResource:Media)
          -[:content]->(content:FHIR_ELEMENT:content)
        OPTIONAL MATCH (report)-[:code]->(code:FHIR_ELEMENT:code)
        WITH patient, content, code
        WHERE code.text IS NULL OR code.text = "AI Skin Lesion Analysis"
        OPTIONAL MATCH (content)-[:RESOLVES_TO]->(resolved:FHIRResource:Binary)
        WITH patient, content, resolved
        WHERE content.url = "Binary/" + $binary_id
           OR resolved.id = $binary_id
        MATCH (binary:FHIRResource:Binary {id: $binary_id})
        RETURN patient.id AS patient_id,
               binary.id AS binary_id,
               binary.data AS data,
               binary.contentType AS content_type
        LIMIT 1
        """,
        {"binary_id": binary_id},
        collect=False,
    )
    return rows[0] if rows else None


async def search_patient_skin_images(
    filters: ResolvedSkinImageSearchFilters,
) -> list[dict[str, Any]]:
    order_clause = "ORDER BY issuedAt ASC" if filters.sort == "asc" else "ORDER BY issuedAt DESC"
    limit_clause = "LIMIT $count" if filters.count is not None else ""
    rows = await execute_cypher(
        f"""
        MATCH (patient:FHIRResource:Patient {{
          resourceType: "Patient",
          id: $patient_id
        }})
        MATCH (report:FHIRResource:DiagnosticReport)
          -[:subject]->(:Reference)
          -[:RESOLVES_TO]->(patient)
        MATCH (report)-[:media]->(:FHIR_ELEMENT:media)
          -[:link]->(:Reference)
          -[:RESOLVES_TO]->(media:FHIRResource:Media)
        MATCH (media)-[:content]->(content:FHIR_ELEMENT:content)
        OPTIONAL MATCH (content)-[:RESOLVES_TO]->(resolved_binary:FHIRResource:Binary)
        OPTIONAL MATCH (binary:FHIRResource:Binary)
        WHERE binary = resolved_binary
           OR content.url = binary.id
           OR content.url = "Binary/" + binary.id
           OR content.url ENDS WITH "/Binary/" + binary.id
        OPTIONAL MATCH (media)-[:modality]->(:FHIR_ELEMENT:modality)
          -[:coding]->(coding:FHIR_ELEMENT:Coding)
        WITH patient, report, media, content, binary, coding,
             datetime(coalesce(report.issued, media.createdDateTime)) AS issuedAt
        WHERE issuedAt IS NOT NULL
          AND ($from_datetime IS NULL OR issuedAt >= datetime($from_datetime))
          AND ($to_datetime IS NULL OR issuedAt <= datetime($to_datetime))
          AND ($modality IS NULL OR coding.code = $modality)
        RETURN patient.id AS patient_id,
               report.id AS diagnostic_report_id,
               report.conclusion AS conclusion,
               coalesce(report.issued, media.createdDateTime) AS created_at,
               media.id AS media_id,
               binary.id AS binary_id,
               coding.code AS modality,
               coalesce(content.contentType, binary.contentType) AS content_type,
               coalesce(binary.id, content.url) AS binary_reference
        {order_clause}
        {limit_clause}
        """,
        {
            "patient_id": filters.patient_id,
            "modality": filters.modality,
            "from_datetime": filters.from_datetime.isoformat().replace("+00:00", "Z") if filters.from_datetime else None,
            "to_datetime": filters.to_datetime.isoformat().replace("+00:00", "Z") if filters.to_datetime else None,
            "count": filters.count,
        },
        collect=False,
    )
    results: list[dict[str, Any]] = []
    for row in rows:
        binary_id = extract_binary_id(row.get("binary_id") or row.get("binary_reference"))
        if not binary_id:
            continue
        results.append(
            {
                "patient_id": str(row.get("patient_id") or ""),
                "diagnostic_report_id": str(row.get("diagnostic_report_id") or ""),
                "media_id": str(row.get("media_id") or ""),
                "binary_id": binary_id,
                "created_at": row.get("created_at"),
                "conclusion": row.get("conclusion"),
                "content_type": row.get("content_type"),
                "url": build_image_api_url(row.get("binary_reference") or binary_id),
            }
        )
    return results
