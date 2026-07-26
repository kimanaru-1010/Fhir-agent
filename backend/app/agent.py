"""Healthcare AI Agent — bounded, targeted FHIR graph exploration."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import settings
from openai import AsyncOpenAI
from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.context_graph_client import execute_cypher, get_schema
from app.memory import save_conversation_memory, search_memories


SYSTEM_PROMPT = """You are an AI clinical intelligence assistant with access to a FHIR-oriented
Neo4j knowledge graph.

GRAPH MODEL
- Root resources use labels :FHIRResource and :<resourceType>.
- Root resources use properties resourceType and id.
- Complex FHIR fields are child nodes connected by relationships named after the field.
- References use:
  (source)-[:fieldName]->(reference:Reference)-[:RESOLVES_TO]->(target:FHIRResource).

GENERAL RULES
- Use graph tools only when the required information is not already present clearly in conversational memory.
- Call tools directly without introductory text.
- Answer only from tool evidence.
- Do not guess missing clinical facts or code meanings.
- Prefer the smallest suitable tool.
- Do not call get_graph_schema before every query.
- Use run_cypher only when the other tools cannot answer the question.

TOOL SELECTION
- search_patient: find Patient resources by name, identifier, or FHIR id.
- search_resource: find one resource by resourceType and FHIR id.
- get_related_resources: find resources that reference another resource.
- get_resources_for_encounter: find clinical resources associated with one Encounter.
- list_resource_fields: inspect direct fields and direct values for one resource.
- list_resource_fields_batch: inspect direct fields and direct values for multiple resources of the same type in one call.
- get_resource_field: read one relevant named field; metadata can be used first when field structure is uncertain.
- get_resource_fields_batch: read one relevant named field from multiple resources; prefer this when the same field is needed across a set.
- expand_field_node: inspect direct children of one complex node when deeper detail is still needed.
- resolve_reference: resolve a Reference to its target resource.
- resolve_coding: resolve Coding.system and Coding.code through CodeSystem.
- get_graph_schema: inspect labels and relationships when needed.
- run_cypher: fallback for read-only queries not covered by another tool.

LIST-FIRST GUIDANCE
- Use list_resource_fields when the structure of one resource is uncertain.
- Use list_resource_fields_batch when several resources of the same type need to be
  inspected together.
- Prefer one batch listing call over repeated single-resource listing calls.
- The listing tools return root properties plus direct fields and their direct
  properties, but do not traverse nested descendants.
- After listing, read only the specific field branch whose direct information is
  insufficient for the user's question.
- Use get_resource_field or get_resource_fields_batch when a known field must be read
  in more detail.
- Avoid calling both single and batch listing tools for the same resources.
- Avoid re-reading fields whose direct values were already returned by a listing tool.
- Expand one specific complex field node only when its direct properties do not contain
  the required fact.

TARGETED GRAPH EXPLORATION
First identify the exact fact required by the user's question.

A practical default order is:
1. Identify candidate resources.
2. List direct fields in one call, using the batch tool when several resources share the same type.
3. Reuse direct values already returned by the listing result.
4. Read a specific field branch only when the direct listing is insufficient.
5. Resolve references or expand a specific branch only when the answer still lacks a required fact.
6. Stop when the requested fact is adequately supported.

Typical examples:
- disease or diagnosis meaning: Condition.code
- clinical state: Condition.clinicalStatus
- verification state: Condition.verificationStatus
- category: Condition.category
- patient owner: subject
- encounter context: encounter
- coded value meaning: Coding.system, Coding.code, Coding.display
- referenced resource details: Reference.reference and RESOLVES_TO

Use list_resource_fields only when the relevant field is not already known.
Use get_resource_field or get_resource_fields_batch when the relevant field is known.
Prefer batch retrieval when multiple resources of the same type require the same field.

FIELD-READING GUIDANCE
- Use field listing as a lightweight planning step when structure is uncertain.
- Prefer the smallest number of field-reading calls that can answer the question.
- Avoid reading the same field repeatedly unless the prior result was incomplete,
  ambiguous, or failed.
- Prefer one batch call over repeated single-resource calls when practical.
- Use direct properties first.
- Do not continue into nested paths merely because has_children is true.
- Continue deeper when the direct value is empty, structural, ambiguous, or missing
  the specific fact required by the user.
- Prefer expand_field_node for one specific node over re-reading an entire branch.
- For CodeableConcept, direct text may be sufficient; inspect coding when system,
  code, or display is needed.
- For Reference, reference/display may be sufficient; resolve only when target
  resource details add value.
- For Quantity, Period, HumanName, Identifier, Coding, and similar complex types,
  prefer direct properties before expansion.

CONTINUE EXPLORING A BRANCH WHEN ALL ARE TRUE
1. The current branch is directly relevant to the user's question.
2. The current result contains a complex FHIR element, Reference, CodeableConcept,
   Coding, nested concept, or another child node that may contain the missing fact.
3. The required fact is not already present in the current properties.
4. The next relationship or node is supported by the current result.
5. No equivalent value has already been obtained from another node.

STOP EXPLORING A BRANCH WHEN ANY IS TRUE
1. The requested fact has been obtained.
2. The branch contains only unrelated metadata, profile, narrative, audit data,
   identifier data, or extension data.
3. The branch has no outgoing children.
4. The next children repeat information already returned.
5. The branch no longer has a plausible connection to the requested fact.
6. A lookup returned no useful child data.
7. The remaining nodes contain only null, empty, or structural values.
8. Resolving the branch would not improve the final answer.

Do not explore every child merely because it exists.
Do not expand sibling branches after one branch has already supplied the required fact,
unless the user asks for all values or all records.

When a node has empty direct properties but its labels indicate a complex FHIR type
such as CodeableConcept, Coding, Reference, HumanName, Identifier, Period, Quantity,
or concept, inspect its relevant children before concluding that the value is absent.

TRAVERSAL BOUNDARY
- get_resource_field, get_resource_fields_batch, list_resource_fields, and
  expand_field_node inspect only the internal FHIR structure of the selected resource.
- These tools must not traverse RESOLVES_TO or DEFINED_BY.
- These tools must not cross into another node labeled FHIRResource.
- The relationship named coding is an internal FHIR child and remains readable.
- Use resolve_reference to cross RESOLVES_TO.
- Use resolve_coding to query CodeSystem concepts through system and code.
- A Patient, Practitioner, Encounter, Observation, or other resource node is already a
  FHIRResource. Never traverse RESOLVES_TO from a FHIRResource; RESOLVES_TO starts
  only from a Reference node.
- Primitive fields may be stored directly as properties on the root resource. A field
  tool may return source=root_property instead of a child node.

REFERENCE RULE
Resolve a Reference only when the referenced resource is needed for the answer.

CODING RULE
- Use display when present and meaningful.
- Otherwise call resolve_coding with system and code.
- Keep the original system and code in the answer.
- Never guess a code meaning.
- Do not explore unrelated CodeSystem concepts.

ALL-MATCHING-RESOURCES RULE
For questions asking for all matching resources:
1. obtain the full list of matching resource ids;
2. retrieve the required field for every returned resource, preferably in one batch;
3. do not answer after inspecting only the first resource;
4. report records whose requested field is absent instead of silently omitting them.

A branch is exhausted when:
- it has no relevant child nodes;
- all relevant child nodes were inspected;
- all returned values are empty or repetitive;
- or the requested fact was already obtained.

Stop the overall tool loop when every relevant branch is exhausted or the requested
facts have been collected for every matching resource.
If retrieved data is incomplete, conflicting, or ambiguous, state that explicitly instead of inferring missing facts.
- Before finalizing, compare the response against all retrieved records relevant to the user’s request.
- Preserve every distinct relevant fact after deduplication, and do not selectively omit items during summarization.
"""


@dataclass
class AgentDeps:
    """Dependencies injected into the agent."""

    session_id: str
    user_id: str


internal_llm_client = AsyncOpenAI(
    base_url=settings.internal_llm_base_url,
    api_key=settings.internal_llm_api_key or "internal",
)

internal_llm_model = OpenAIChatModel(
    settings.internal_llm_model,
    provider=OpenAIProvider(openai_client=internal_llm_client),
)

agent = Agent(
    internal_llm_model,
    system_prompt=SYSTEM_PROMPT,
    deps_type=AgentDeps,
    retries=1,
)



# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

_LOG_LEVEL = os.getenv("FHIR_AGENT_LOG_LEVEL", "DEBUG").upper()
_LOG_FILE = os.getenv(
    "FHIR_AGENT_LOG_FILE",
    "logs/fhir_agent_debug.log",
)
logger = logging.getLogger("fhir_agent")
logger.setLevel(
    getattr(logging, _LOG_LEVEL, logging.DEBUG)
)
logger.propagate = False

# Write all agent logs to one file only.
if not logger.handlers:
    log_path = os.path.abspath(_LOG_FILE)
    log_directory = os.path.dirname(log_path)

    if log_directory:
        os.makedirs(log_directory, exist_ok=True)

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )
    logger.addHandler(file_handler)


def _pretty_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            indent=2,
        )
    except Exception:
        return repr(value)


def _log_payload(title: str, value: Any) -> None:
    text = _pretty_json(value)
    logger.debug("%s\n%s", title, text)


_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_MAX_MODEL_TOOL_RESULT_CHARS = int(
    os.getenv("FHIR_AGENT_MAX_TOOL_RESULT_CHARS", "400000")
)
_BLOCKED_TRAVERSAL_RELATIONSHIPS = ("RESOLVES_TO", "DEFINED_BY")


def _json_response(
    *,
    status: str,
    data: Any,
    count: int | None = None,
    message: str | None = None,
) -> str:
    payload: dict[str, Any] = {"status": status, "data": data}
    if count is not None:
        payload["count"] = count
    if message:
        payload["message"] = message
    return json.dumps(payload, default=str, ensure_ascii=False)


def _bounded_limit(limit: int | str) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = _DEFAULT_LIMIT
    return max(1, min(value, _MAX_LIMIT))


def _parse_ids(resource_ids: str) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for item in resource_ids.split(",")
            if item.strip()
        )
    )


def _is_read_only(query: str) -> bool:
    normalized = " ".join(query.upper().split())
    blocked = (
        " CREATE ",
        " MERGE ",
        " DELETE ",
        " DETACH ",
        " SET ",
        " REMOVE ",
        " DROP ",
        " FOREACH ",
        " LOAD CSV ",
        " CALL DBMS",
        " CALL DB.",
    )
    padded = f" {normalized} "
    return not any(keyword in padded for keyword in blocked)


async def _execute_tool(
    *,
    tool_name: str,
    cypher: str,
    parameters: dict[str, Any] | None = None,
) -> str:
    actual_parameters = parameters or {}

    run_id = _CURRENT_RUN_ID.get()
    handler_name = _CURRENT_HANDLER.get()
    logger.info(
        "TOOL START | run_id=%s | handler=%s | tool=%s",
        run_id, handler_name, tool_name,
    )
    _log_payload(
        f"TOOL INPUT | {tool_name} | parameters",
        actual_parameters,
    )
    # logger.debug(
    #     "TOOL INPUT | %s | cypher\n%s",
    #     tool_name,
    #     cypher.strip(),
    # )

    try:
        rows = await execute_cypher(
            cypher,
            actual_parameters,
            tool_name=tool_name,
        )

        _log_payload(
            f"NEO4J RAW RESULT | {tool_name}",
            rows,
        )

        payload: dict[str, Any] = {
            "status": "ok",
            "count": len(rows),
            "data": rows,
        }

        model_content = json.dumps(
            payload,
            default=str,
            ensure_ascii=False,
        )

        if len(model_content) > _MAX_MODEL_TOOL_RESULT_CHARS:
            preview_rows = rows[:20]
            payload = {
                "status": "truncated",
                "count": len(rows),
                "returned_count": len(preview_rows),
                "data": preview_rows,
                "message": (
                    "The tool result exceeded the model payload limit. "
                    "Use a narrower field, fewer resource ids, or a dedicated tool."
                ),
            }
            model_content = json.dumps(
                payload,
                default=str,
                ensure_ascii=False,
            )

        _log_payload(
            f"MODEL TOOL RESULT OBJECT | {tool_name}",
            payload,
        )
        logger.debug(
            "MODEL TOOL RESULT STRING | %s\n%s",
            tool_name,
            model_content,
        )
        _record_tool_result_chars(len(model_content))
        logger.info(
            "TOOL END | run_id=%s | handler=%s | tool=%s | status=%s | count=%s | chars=%s",
            run_id, handler_name, tool_name, payload["status"],
            payload["count"], len(model_content),
        )

        return model_content

    except Exception as exc:
        payload = {
            "status": "error",
            "count": 0,
            "data": [],
            "message": str(exc),
        }

        model_content = json.dumps(
            payload,
            default=str,
            ensure_ascii=False,
        )

        logger.exception(
            "TOOL ERROR | %s | %s",
            tool_name,
            exc,
        )
        logger.debug(
            "MODEL TOOL RESULT STRING | %s\n%s",
            tool_name,
            model_content,
        )

        return model_content


# ---------------------------------------------------------------------------
# Generic FHIR tools
# ---------------------------------------------------------------------------

@agent.tool
async def search_patient(
    ctx: RunContext[AgentDeps],
    query: str,
    limit: int = 20,
) -> str:
    """Search Patient by FHIR id, HumanName, or Identifier."""
    cypher = """
    MATCH (patient:FHIRResource:Patient)
    OPTIONAL MATCH (patient)-[:name]->(name)
    OPTIONAL MATCH (patient)-[:identifier]->(identifier)
    WITH patient,
         collect(DISTINCT name) AS names,
         collect(DISTINCT identifier) AS identifiers
    WHERE patient.id = $query
       OR any(name_node IN names WHERE
            toLower(coalesce(name_node.text, '')) CONTAINS toLower($query)
            OR toLower(coalesce(name_node.family, '')) CONTAINS toLower($query)
            OR any(given IN coalesce(name_node.given, [])
                   WHERE toLower(given) CONTAINS toLower($query))
          )
       OR any(identifier_node IN identifiers WHERE
            toLower(coalesce(identifier_node.value, ''))
            CONTAINS toLower($query)
          )
    RETURN patient.id AS resource_id,
           patient.resourceType AS resource_type,
           [name_node IN names | properties(name_node)] AS names,
           [identifier_node IN identifiers | properties(identifier_node)] AS identifiers
    ORDER BY patient.id
    LIMIT $limit
    """
    return await _execute_tool(
        tool_name="search_patient",
        cypher=cypher,
        parameters={"query": query, "limit": _bounded_limit(limit)},
    )


@agent.tool
async def search_resource(
    ctx: RunContext[AgentDeps],
    resource_type: str,
    resource_id: str,
) -> str:
    """Find one FHIR resource by resourceType and FHIR id."""
    cypher = """
    MATCH (resource:FHIRResource {
        resourceType: $resource_type,
        id: $resource_id
    })
    RETURN resource.resourceType AS resource_type,
           resource.id AS resource_id,
           properties(resource) AS properties
    LIMIT 20
    """
    return await _execute_tool(
        tool_name="search_resource",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_id": resource_id,
        },
    )


@agent.tool
async def get_related_resources(
    ctx: RunContext[AgentDeps],
    resource_type: str,
    resource_id: str,
    related_type: str = "",
    limit: int = 100,
) -> str:
    """Find FHIR resource ids that reference the selected resource."""
    cypher = """
    MATCH (target:FHIRResource {
        resourceType: $resource_type,
        id: $resource_id
    })
    MATCH (reference:Reference)-[:RESOLVES_TO]->(target)
    MATCH (source:FHIRResource)-[*1..4]->(reference)
    WHERE source <> target
      AND ($related_type = '' OR source.resourceType = $related_type)
    WITH DISTINCT source
    ORDER BY source.resourceType, source.id
    RETURN source.resourceType AS resource_type,
           source.id AS resource_id
    LIMIT $limit
    """
    return await _execute_tool(
        tool_name="get_related_resources",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_id": resource_id,
            "related_type": related_type,
            "limit": _bounded_limit(limit),
        },
    )


@agent.tool
async def get_resources_for_encounter(
    ctx: RunContext[AgentDeps],
    encounter_id: str,
    resource_types: str = "",
    limit: int = 100,
) -> str:
    """Find clinical resources that reference one Encounter."""

    requested_types = _parse_ids(resource_types)

    cypher = """
    MATCH (encounter:FHIRResource:Encounter {
        resourceType: 'Encounter',
        id: $encounter_id
    })

    MATCH (encounter_reference:Reference)-[:RESOLVES_TO]->(encounter)
    MATCH (resource:FHIRResource)-[*1..4]->(encounter_reference)

    WHERE resource <> encounter
      AND (
          size($resource_types) = 0
          OR resource.resourceType IN $resource_types
      )

    WITH DISTINCT resource
    ORDER BY resource.resourceType, resource.id

    RETURN resource.resourceType AS resource_type,
           resource.id AS resource_id
    LIMIT $limit
    """

    return await _execute_tool(
        tool_name="get_resources_for_encounter",
        cypher=cypher,
        parameters={
            "encounter_id": encounter_id,
            "resource_types": requested_types,
            "limit": _bounded_limit(limit),
        },
    )


@agent.tool
async def list_resource_fields(
    ctx: RunContext[AgentDeps],
    resource_type: str,
    resource_id: str,
) -> str:
    """List root properties and direct internal FHIR fields on one resource."""

    cypher = """
    MATCH (resource:FHIRResource {
        resourceType: $resource_type,
        id: $resource_id
    })

    OPTIONAL MATCH (resource)-[relationship]->(child)

    WITH resource,
         collect(
             DISTINCT CASE
                 WHEN child IS NULL
                   OR type(relationship) IN $blocked_relationships
                   OR child:FHIRResource
                 THEN NULL
                 ELSE {
                     field_name: type(relationship),
                     node_id: toString(id(child)),
                     labels: labels(child),
                     direct_properties: properties(child),
                     has_internal_children:
                         size([
                             (child)-[next]->(grandchild)
                             WHERE NOT type(next) IN $blocked_relationships
                               AND NOT grandchild:FHIRResource
                             | 1
                         ]) > 0
                 }
             END
         ) AS raw_fields

    RETURN resource.resourceType AS resource_type,
           resource.id AS resource_id,
           properties(resource) AS root_properties,
           [field IN raw_fields WHERE field IS NOT NULL] AS fields
    """

    return await _execute_tool(
        tool_name="list_resource_fields",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_id": resource_id,
            "blocked_relationships": list(
                _BLOCKED_TRAVERSAL_RELATIONSHIPS
            ),
        },
    )

@agent.tool
async def list_resource_fields_batch(
    ctx: RunContext[AgentDeps],
    resource_type: str,
    resource_ids: list[str],
    limit: int = 100,
    blocked_relationships: list[str] | None = None,
) -> str:
    """List root properties and direct FHIR fields for multiple resources.

    This tool is generic and intentionally shallow:
    - one call handles multiple resources of the same resource type;
    - returns root properties and direct field properties;
    - reports whether each direct field has internal children;
    - does not traverse nested descendants;
    - does not traverse blocked mapping relationships.
    """

    ids = [
        str(resource_id).strip()
        for resource_id in resource_ids
        if str(resource_id).strip()
    ]
    ids = list(dict.fromkeys(ids))

    if not ids:
        return _json_response(
            status="error",
            count=0,
            data=[],
            message="resource_ids must contain at least one id",
        )

    blocked = (
        blocked_relationships
        if blocked_relationships is not None
        else list(_BLOCKED_TRAVERSAL_RELATIONSHIPS)
    )

    cypher = """
    UNWIND $resource_ids AS requested_id

    OPTIONAL MATCH (resource:FHIRResource {
        resourceType: $resource_type,
        id: requested_id
    })

    OPTIONAL MATCH (resource)-[field_relationship]->(field_node)
    WHERE NOT type(field_relationship) IN $blocked_relationships
      AND NOT field_node:FHIRResource

    OPTIONAL MATCH (field_node)-[internal_relationship]->(internal_child)
    WHERE NOT type(internal_relationship) IN $blocked_relationships
      AND NOT internal_child:FHIRResource

    WITH requested_id,
         resource,
         field_relationship,
         field_node,
         count(internal_child) > 0 AS has_internal_children

    ORDER BY requested_id, type(field_relationship)

    WITH requested_id,
         resource,
         collect(
             CASE
                 WHEN field_node IS NULL THEN NULL
                 ELSE {
                     field_name: type(field_relationship),
                     node_id: toString(id(field_node)),
                     labels: labels(field_node),
                     direct_properties: properties(field_node),
                     has_internal_children: has_internal_children
                 }
             END
         ) AS raw_fields

    RETURN requested_id AS resource_id,
           resource IS NOT NULL AS resource_found,
           CASE
               WHEN resource IS NULL THEN $resource_type
               ELSE resource.resourceType
           END AS resource_type,
           CASE
               WHEN resource IS NULL THEN {}
               ELSE properties(resource)
           END AS root_properties,
           [
               field IN raw_fields
               WHERE field IS NOT NULL
           ] AS fields

    ORDER BY resource_id
    LIMIT $limit
    """

    return await _execute_tool(
        tool_name="list_resource_fields_batch",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_ids": ids,
            "limit": _bounded_limit(limit),
            "blocked_relationships": blocked,
        },
    )


@agent.tool
async def get_resource_field(
    ctx: RunContext[AgentDeps],
    resource_type: str,
    resource_id: str,
    field_name: str,
    limit: int = 100,
) -> str:
    """Read one root property or internal FHIR field without crossing resources."""

    cypher = """
    MATCH (resource:FHIRResource {
        resourceType: $resource_type,
        id: $resource_id
    })

    CALL {
        WITH resource

        WITH resource, resource[$field_name] AS root_value
        WHERE root_value IS NOT NULL

        RETURN {
            source: 'root_property',
            node_id: null,
            path: [$field_name],
            labels: labels(resource),
            properties: {value: root_value},
            has_children: false
        } AS value

        UNION

        WITH resource
        MATCH path=
            (resource)-[first]->(field_node)-[*0..2]->(value_node)

        WHERE type(first) = $field_name
          AND all(
              relationship IN relationships(path)
              WHERE NOT type(relationship) IN $blocked_relationships
          )
          AND all(
              path_node IN nodes(path)[1..]
              WHERE NOT path_node:FHIRResource
          )

        RETURN {
            source: 'child_node',
            node_id: toString(id(value_node)),
            path: [
                relationship IN relationships(path) |
                type(relationship)
            ],
            labels: labels(value_node),
            properties: properties(value_node),
            has_children:
                size([
                    (value_node)-[next]->(child)
                    WHERE NOT type(next) IN $blocked_relationships
                      AND NOT child:FHIRResource
                    | 1
                ]) > 0
        } AS value
    }

    RETURN value
    LIMIT $limit
    """

    return await _execute_tool(
        tool_name="get_resource_field",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_id": resource_id,
            "field_name": field_name,
            "limit": _bounded_limit(limit),
            "blocked_relationships": list(
                _BLOCKED_TRAVERSAL_RELATIONSHIPS
            ),
        },
    )

@agent.tool
async def get_resource_fields_batch(
    ctx: RunContext[AgentDeps],
    resource_type: str,
    resource_ids: str,
    field_name: str,
    limit: int = 100,
) -> str:
    """Read one field from multiple resources using the get_resource_field shape."""

    ids = _parse_ids(resource_ids)

    if not ids:
        return _json_response(
            status="error",
            count=0,
            data=[],
            message="resource_ids must contain at least one id",
        )

    cypher = """
    UNWIND $resource_ids AS requested_id

    OPTIONAL MATCH (resource:FHIRResource {
        resourceType: $resource_type,
        id: requested_id
    })

    CALL {
        WITH requested_id, resource

        WITH requested_id,
             resource,
             resource[$field_name] AS root_value
        WHERE resource IS NOT NULL
          AND root_value IS NOT NULL

        RETURN requested_id AS resource_id,
               true AS resource_found,
               true AS field_found,
               {
                   source: 'root_property',
                   node_id: null,
                   path: [$field_name],
                   labels: labels(resource),
                   properties: {value: root_value},
                   has_children: false
               } AS value

        UNION

        WITH requested_id, resource

        MATCH path=
            (resource)-[first]->(field_node)-[*0..2]->(value_node)

        WHERE type(first) = $field_name
          AND all(
              relationship IN relationships(path)
              WHERE NOT type(relationship) IN $blocked_relationships
          )
          AND all(
              path_node IN nodes(path)[1..]
              WHERE NOT path_node:FHIRResource
          )

        RETURN requested_id AS resource_id,
               true AS resource_found,
               true AS field_found,
               {
                   source: 'child_node',
                   node_id: toString(id(value_node)),
                   path: [
                       relationship IN relationships(path) |
                       type(relationship)
                   ],
                   labels: labels(value_node),
                   properties: properties(value_node),
                   has_children:
                       size([
                           (value_node)-[next]->(child)
                           WHERE NOT type(next) IN $blocked_relationships
                             AND NOT child:FHIRResource
                           | 1
                       ]) > 0
               } AS value

        UNION

        WITH requested_id, resource

        OPTIONAL MATCH path=
            (resource)-[first]->(field_node)-[*0..2]->(value_node)

        WHERE path IS NULL
           OR (
               type(first) = $field_name
               AND all(
                   relationship IN relationships(path)
                   WHERE NOT type(relationship) IN $blocked_relationships
               )
               AND all(
                   path_node IN nodes(path)[1..]
                   WHERE NOT path_node:FHIRResource
               )
           )

        WITH requested_id,
             resource,
             resource[$field_name] AS root_value,
             count(value_node) AS child_value_count

        WHERE resource IS NULL
           OR (
               root_value IS NULL
               AND child_value_count = 0
           )

        RETURN requested_id AS resource_id,
               resource IS NOT NULL AS resource_found,
               false AS field_found,
               null AS value
    }

    RETURN resource_id,
           resource_found,
           field_found,
           CASE WHEN value IS NULL THEN null ELSE value.source END AS source,
           CASE WHEN value IS NULL THEN null ELSE value.node_id END AS node_id,
           CASE WHEN value IS NULL THEN [] ELSE value.path END AS path,
           CASE WHEN value IS NULL THEN [] ELSE value.labels END AS labels,
           CASE WHEN value IS NULL THEN {} ELSE value.properties END AS properties,
           CASE WHEN value IS NULL THEN false ELSE value.has_children END AS has_children
    ORDER BY resource_id, path
    LIMIT $limit
    """

    return await _execute_tool(
        tool_name="get_resource_fields_batch",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_ids": ids,
            "field_name": field_name,
            "limit": _bounded_limit(limit),
            "blocked_relationships": list(
                _BLOCKED_TRAVERSAL_RELATIONSHIPS
            ),
        },
    )

@agent.tool
async def expand_field_node(
    ctx: RunContext[AgentDeps],
    node_id: str,
    limit: int = 50,
) -> str:
    """Read direct internal FHIR children of a previously returned element node."""

    cypher = """
    MATCH (node)
    WHERE id(node) = toInteger($node_id)

    OPTIONAL MATCH (node)-[relationship]->(child)

    WITH node,
         collect(
             DISTINCT CASE
                 WHEN child IS NULL
                   OR type(relationship) IN $blocked_relationships
                   OR child:FHIRResource
                 THEN NULL
                 ELSE {
                     node_id: toString(id(child)),
                     relationship: type(relationship),
                     labels: labels(child),
                     properties: properties(child),
                     has_children:
                         size([
                             (child)-[next]->(grandchild)
                             WHERE NOT type(next) IN $blocked_relationships
                               AND NOT grandchild:FHIRResource
                             | 1
                         ]) > 0
                 }
             END
         ) AS raw_children

    RETURN toString(id(node)) AS parent_node_id,
           labels(node) AS parent_labels,
           properties(node) AS parent_properties,
           [
               child IN raw_children
               WHERE child IS NOT NULL
           ][0..$limit] AS children
    """

    return await _execute_tool(
        tool_name="expand_field_node",
        cypher=cypher,
        parameters={
            "node_id": node_id,
            "limit": _bounded_limit(limit),
            "blocked_relationships": list(
                _BLOCKED_TRAVERSAL_RELATIONSHIPS
            ),
        },
    )

@agent.tool
async def resolve_reference(
    ctx: RunContext[AgentDeps],
    reference: str,
) -> str:
    """Resolve a FHIR reference string to a distinct target resource."""

    cypher = """
    MATCH (:Reference {
        reference: $reference
    })-[:RESOLVES_TO]->(target:FHIRResource)

    RETURN DISTINCT
           $reference AS reference,
           target.resourceType AS resource_type,
           target.id AS resource_id,
           properties(target) AS root_properties
    LIMIT 20
    """

    return await _execute_tool(
        tool_name="resolve_reference",
        cypher=cypher,
        parameters={
            "reference": reference,
        },
    )

@agent.tool
async def resolve_coding(
    ctx: RunContext[AgentDeps],
    system: str,
    code: str,
    limit: int = 20,
) -> str:
    """Resolve a FHIR Coding through CodeSystem concepts."""
    cypher = """
    MATCH (code_system:FHIRResource:CodeSystem)
    WHERE code_system.url = $system OR code_system.id = $system
    MATCH path=(code_system)-[:concept*1..10]->(concept)
    WHERE concept.code = $code
    OPTIONAL MATCH (concept)-[:designation]->(designation)
    RETURN code_system.id AS code_system_id,
           code_system.url AS code_system_url,
           code_system.name AS code_system_name,
           concept.code AS code,
           concept.display AS display,
           concept.definition AS definition,
           collect(DISTINCT properties(designation)) AS designations,
           [relationship IN relationships(path) |
                type(relationship)] AS path
    LIMIT $limit
    """
    return await _execute_tool(
        tool_name="resolve_coding",
        cypher=cypher,
        parameters={
            "system": system,
            "code": code,
            "limit": _bounded_limit(limit),
        },
    )


@agent.tool
async def get_graph_schema(ctx: RunContext[AgentDeps]) -> str:
    """Get graph labels and relationship types."""
    logger.info("TOOL START | get_graph_schema")
    try:
        result = await get_schema()
        payload = {"status": "ok", "data": result}
        model_content = json.dumps(payload, default=str, ensure_ascii=False)
        _log_payload("GRAPH SCHEMA RAW RESULT", result)
        logger.debug("MODEL TOOL RESULT STRING | get_graph_schema\n%s", model_content)
        logger.info("TOOL END | get_graph_schema | status=ok")
        return model_content
    except Exception as exc:
        payload = {"status": "error", "data": {}, "message": str(exc)}
        model_content = json.dumps(payload, default=str, ensure_ascii=False)
        logger.exception("TOOL ERROR | get_graph_schema | %s", exc)
        logger.debug("MODEL TOOL RESULT STRING | get_graph_schema\n%s", model_content)
        return model_content


@agent.tool
async def run_cypher(
    ctx: RunContext[AgentDeps],
    query: str,
    parameters: str = "{}",
) -> str:
    """Execute a read-only Cypher query against the graph."""
    if not _is_read_only(query):
        return _json_response(
            status="error",
            count=0,
            data=[],
            message="Only read-only Cypher queries are allowed",
        )
    try:
        params = json.loads(parameters) if parameters else {}
    except json.JSONDecodeError:
        return _json_response(
            status="error",
            count=0,
            data=[],
            message="parameters must be a valid JSON object",
        )
    params.setdefault("domain", settings.domain_id)
    return await _execute_tool(
        tool_name="run_cypher",
        cypher=query,
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Message handling
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Request/run diagnostics
# ---------------------------------------------------------------------------

_CURRENT_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "fhir_agent_run_id", default="-"
)
_CURRENT_HANDLER: contextvars.ContextVar[str] = contextvars.ContextVar(
    "fhir_agent_handler", default="-"
)
_active_runs: dict[str, dict[str, Any]] = {}
_recent_run_starts: list[dict[str, Any]] = []
_DUPLICATE_WINDOW_SECONDS = float(
    os.getenv("FHIR_AGENT_DUPLICATE_WINDOW_SECONDS", "30")
)

def _generate_run_id() -> str:
    return uuid.uuid4().hex[:12]

def _message_fingerprint(message: str) -> str:
    normalized = " ".join(message.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

def _compact_stack() -> str:
    frames = traceback.extract_stack(limit=12)[:-2]
    return " <- ".join(
        f"{f.name}@{os.path.basename(f.filename)}:{f.lineno}" for f in frames[-6:]
    )

def _prune_recent_runs(now: float) -> None:
    cutoff = now - max(_DUPLICATE_WINDOW_SECONDS, 1.0)
    _recent_run_starts[:] = [
        x for x in _recent_run_starts if x["started_monotonic"] >= cutoff
    ]

def _track_run_start(*, run_id: str, handler_name: str, message: str, supplied_session_id: str | None) -> None:
    now = time.monotonic()
    fingerprint = _message_fingerprint(message)
    _prune_recent_runs(now)
    dupes = [x for x in _recent_run_starts if x["message_fingerprint"] == fingerprint]
    task = asyncio.current_task()
    task_id = id(task) if task is not None else None
    info = {
        "run_id": run_id, "handler": handler_name,
        "message_fingerprint": fingerprint,
        "supplied_session_id": supplied_session_id,
        "started_monotonic": now, "process_id": os.getpid(),
        "task_id": task_id, "tool_calls": 0, "tool_result_chars": 0,
    }
    _active_runs[run_id] = info
    _recent_run_starts.append(info.copy())
    logger.warning(
        "REQUEST DIAGNOSTIC | event=handler_entry | run_id=%s | handler=%s "
        "| process_id=%s | task_id=%s | supplied_session_id=%s "
        "| message_fingerprint=%s | active_runs=%s | duplicate_candidates=%s",
        run_id, handler_name, os.getpid(), task_id, supplied_session_id,
        fingerprint, len(_active_runs),
        [{"run_id": x["run_id"], "handler": x["handler"],
          "age_seconds": round(now-x["started_monotonic"],3)} for x in dupes],
    )
    logger.debug("REQUEST CALL STACK | run_id=%s | %s", run_id, _compact_stack())
    if dupes:
        logger.error(
            "DUPLICATE REQUEST SUSPECTED | run_id=%s | message_fingerprint=%s "
            "| reason=same_message_entered_handler_again | candidates=%s",
            run_id, fingerprint, [x["run_id"] for x in dupes],
        )

def _record_tool_result_chars(chars: int) -> None:
    run_id = _CURRENT_RUN_ID.get()
    info = _active_runs.get(run_id)
    if info is not None:
        info["tool_calls"] += 1
        info["tool_result_chars"] += chars

def _log_model_usage(result: Any, run_id: str) -> None:
    try:
        usage_attr = getattr(result, "usage", None)
        usage_value = usage_attr() if callable(usage_attr) else usage_attr
    except Exception as exc:
        usage_value = {"usage_read_error": str(exc)}
    _log_payload(f"MODEL USAGE | run_id={run_id}", usage_value)

def _classify_run_exception(exc: Exception) -> str:
    value = str(exc).lower()
    markers = ("context length", "context_length", "maximum context",
               "max context", "too many tokens", "token limit",
               "prompt is too long", "context window")
    return "context_overflow_or_token_limit" if any(m in value for m in markers) else "non_context_exception"

def _track_run_end(*, run_id: str, outcome: str, exception: Exception | None = None) -> None:
    now = time.monotonic()
    info = _active_runs.pop(run_id, None) or {}
    duration = now - info.get("started_monotonic", now)
    logger.warning(
        "REQUEST DIAGNOSTIC | event=handler_exit | run_id=%s | handler=%s "
        "| outcome=%s | duration_sec=%.3f | tool_calls=%s "
        "| tool_result_chars=%s | active_runs=%s",
        run_id, info.get("handler", _CURRENT_HANDLER.get()), outcome, duration,
        info.get("tool_calls", 0), info.get("tool_result_chars", 0), len(_active_runs),
    )
    if exception is not None:
        logger.error(
            "RUN FAILURE DIAGNOSTIC | run_id=%s | classification=%s "
            "| exception_type=%s | exception=%s",
            run_id, _classify_run_exception(exception), type(exception).__name__, exception,
        )


async def _prepare_run(
    message: str,
    session_id: str | None,
    user_id: str,
    run_id: str = "",
) -> tuple[str, list[Any], str]:
    resolved_session_id = session_id or str(uuid.uuid4())
    trace = f"[{run_id}]" if run_id else ""

    # Search Mem0 for relevant conversational memories
    memories = await search_memories(
        query=message,
        user_id=user_id,
        session_id=resolved_session_id,
        limit=8,
    )

    memory_prompt = ""
    if memories:
        memory_lines = []
        for item in memories:
            mem_text = item.get("memory", "")
            if mem_text:
                memory_lines.append(f"- {mem_text}")
        memory_prompt = "Relevant conversational memories:\n" + "\n".join(memory_lines)
    else:
        memory_prompt = "No relevant conversational memories were found."

    message_history: list[Any] = []

    estimated_history_chars = sum(len(str(item)) for item in message_history)
    logger.info(
        "PREPARE RUN | run_id=%s | session_id=%s | mem0_results=%d "
        "| current_message_chars=%d | estimated_history_chars=%d | active_runs=%d",
        run_id, resolved_session_id, len(memories), len(message),
        estimated_history_chars, len(_active_runs),
    )
    _log_payload(f"MEM0 RESULTS {trace}", memories)
    logger.debug("MEMORY CONTEXT %s\n%s", trace, memory_prompt)

    return resolved_session_id, message_history, memory_prompt


async def handle_message(
    message: str,
    session_id: str | None = None,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """Handle an incoming non-streaming chat message."""
    run_id = _generate_run_id()
    run_token = _CURRENT_RUN_ID.set(run_id)
    handler_token = _CURRENT_HANDLER.set("handle_message")
    _track_run_start(run_id=run_id, handler_name="handle_message", message=message, supplied_session_id=session_id)
    try:
        resolved_session_id, message_history, memory_prompt = await _prepare_run(
            message, session_id, user_id=user_id, run_id=run_id,
        )
        logger.info("MODEL RUN START | run_id=%s | handler=handle_message | session_id=%s", run_id, resolved_session_id)
        effective_message = (
            "CONVERSATIONAL MEMORY\n"
            f"{memory_prompt}\n\n"
            "CURRENT USER REQUEST\n"
            f"{message}"
            if memory_prompt
            else message
        )

        result = await agent.run(
            effective_message,
            deps=AgentDeps(session_id=resolved_session_id, user_id=user_id),
            message_history=[],
            model_settings=ModelSettings(
                temperature=0,
    ),
        )
        _log_model_usage(result, run_id)
        usage_attr = getattr(result, "usage", None)
        u = usage_attr() if callable(usage_attr) else usage_attr
        logger.info("TOKENS | run_id=%s | session=%s | usage=%s", run_id, resolved_session_id, u)
        response_text = result.output or ""
        logger.debug("MODEL FINAL OUTPUT | run_id=%s | session_id=%s\n%s", run_id, resolved_session_id, response_text)
        logger.info("MODEL RUN END | run_id=%s | handler=handle_message | session_id=%s | chars=%s", run_id, resolved_session_id, len(response_text))
        if not response_text.strip():
            response_text = "I could not obtain enough graph evidence to answer the question."
        # Save the completed conversation to Mem0
        await save_conversation_memory(
            user_id=user_id,
            session_id=resolved_session_id,
            user_message=message,
            assistant_message=response_text,
        )
        _track_run_end(run_id=run_id, outcome="success")
        return {
            "response": response_text,
            "session_id": resolved_session_id,
            "graph_data": None,
            "diagnostic_run_id": run_id,
        }
    except Exception as exc:
        _track_run_end(run_id=run_id, outcome="error", exception=exc)
        raise
    finally:
        _CURRENT_HANDLER.reset(handler_token)
        _CURRENT_RUN_ID.reset(run_token)


async def handle_message_stream(
    message: str,
    session_id: str | None = None,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """Run the full agent loop and then emit the final response."""
    from app.context_graph_client import get_collector
    run_id = _generate_run_id()
    run_token = _CURRENT_RUN_ID.set(run_id)
    handler_token = _CURRENT_HANDLER.set("handle_message_stream")
    _track_run_start(run_id=run_id, handler_name="handle_message_stream", message=message, supplied_session_id=session_id)
    try:
        resolved_session_id, message_history, memory_prompt = await _prepare_run(
            message, session_id, user_id=user_id, run_id=run_id,
        )
        collector = get_collector()
        logger.info("MODEL RUN START | run_id=%s | handler=handle_message_stream | session_id=%s", run_id, resolved_session_id)
        effective_message = (
            "CONVERSATIONAL MEMORY\n"
            f"{memory_prompt}\n\n"
            "CURRENT USER REQUEST\n"
            f"{message}"
            if memory_prompt
            else message
        )

        result = await agent.run(
            effective_message,
            deps=AgentDeps(session_id=resolved_session_id, user_id=user_id),
            message_history=[],
        )
        _log_model_usage(result, run_id)
        usage_attr = getattr(result, "usage", None)
        u = usage_attr() if callable(usage_attr) else usage_attr
        logger.info("TOKENS | run_id=%s | session=%s | usage=%s", run_id, resolved_session_id, u)
        response_text = result.output or ""
        if not response_text.strip():
            response_text = "I could not obtain enough graph evidence to answer the question."
        collector.emit_text_delta(response_text)
        # Save the completed conversation to Mem0
        await save_conversation_memory(
            user_id=user_id,
            session_id=resolved_session_id,
            user_message=message,
            assistant_message=response_text,
        )
        collector.emit_done(response_text, resolved_session_id)
        logger.info("MODEL RUN END | run_id=%s | handler=handle_message_stream | session_id=%s | chars=%s", run_id, resolved_session_id, len(response_text))
        _track_run_end(run_id=run_id, outcome="success")
        return {"response": response_text, "session_id": resolved_session_id, "graph_data": None, "diagnostic_run_id": run_id}
    except Exception as exc:
        _track_run_end(run_id=run_id, outcome="error", exception=exc)
        raise
    finally:
        _CURRENT_HANDLER.reset(handler_token)
        _CURRENT_RUN_ID.reset(run_token)