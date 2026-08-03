"""Healthcare AI Agent â€” bounded, targeted FHIR graph exploration."""

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
from typing import Annotated, Any

from app.core.config import settings
from openai import AsyncOpenAI
from pydantic import Field
from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from app.graph.client import execute_cypher, get_schema
from app.services.long_term_memory import save_conversation_memory, search_memories


# SYSTEM_PROMPT = """
# ROLE
# You are a clinical data assistant with access to a FHIR-oriented Neo4j graph.

# OBJECTIVE
# Answer the user's current request accurately and efficiently using only
# supported evidence.

# EVIDENCE
# - Use supplied conversation context when it clearly answers the request.
# - Query the graph when clinical information is missing, uncertain, mutable,
#   or requires verification.
# - Do not invent values, relationships, code meanings, or clinical conclusions.
# - State when evidence is missing, incomplete, conflicting, or ambiguous.

# TASK PLANNING
# Determine whether the request asks for:
# - one fact;
# - a list of records;
# - a comparison;
# - or a complete timeline.

# Identify the minimum resources and fields needed before selecting tools.

# TOOL USE
# - Use specialized tools first. Treat run_cypher as a last-resort fallback.
# - Prefer batch tools for multiple resources of the same type.
# - Always provide every required argument when calling a tool.
# - Use read-only Cypher only.
# - Reuse values already returned.
# - Read direct properties before expanding nested fields.
# - Resolve references or codings only when their details are required.
# - Do not repeat the same call or an equivalent tool call unless the previous
#   result failed or was incomplete.
# - Use get_graph_schema only when graph structure is uncertain.

# TOOL SELECTION
# - To find a Patient, use search_patient.
# - To verify one known FHIRResource, use search_resource.
# - To find resources that reference a target FHIRResource, use get_related_resources.
# - To find resources for an Encounter, use get_resources_for_encounter.
# - To inspect fields, use list_resource_fields, list_resource_fields_batch,
#   get_resource_field, get_resource_fields_batch, or expand_field_node.
# - To cross a Reference RESOLVES_TO edge, use resolve_reference.
# - To resolve Coding.system and Coding.code, use resolve_coding.
# - Use run_cypher only for read-only aggregation, joins, filtering, ordering, or
#   graph patterns that the specialized tools cannot express.

# TOOL ARGUMENT SAFETY
# - For one resource, use single-resource tools with resource_id.
# - For multiple resources of the same type, use batch tools with resource_ids.
# - list_resource_fields_batch expects resource_ids as an array of strings.
# - get_resource_fields_batch expects resource_ids as comma-separated text.
# - Never pass resource_ids to list_resource_fields.
# - Never pass a JSON string when a tool expects an array.

# RUN_CYPHER LIMITS
# - Before run_cypher, decide why specialized tools are insufficient.
# - Do not use run_cypher to replace search_patient, relationship lookup, field
#   reading, Reference resolution, or Coding resolution.
# - If a run_cypher query returns no useful data or a Cypher error, do not keep
#   rewriting similar Cypher unless the correction is specific and necessary.

# COMPLETENESS
# For requests asking for all records, a full journey, or a complete comparison:
# - retrieve the complete matching set of relevant resources;
# - inspect every matching record;
# - report missing requested fields;
# - do not stop after the first match.

# STOP CONDITION
# Stop using tools when:
# - the requested scope is fully supported;
# - no relevant unexplored branch remains;
# - or further retrieval would not improve the answer.

# FHIR INTERPRETATION
# - Preserve exact resource IDs, codes, dates, statuses, quantities, and units
#   when relevant.
# - Use display text when present.
# - Never guess the meaning of an unresolved code.
# - Do not infer payment, diagnosis, or clinical meaning beyond the retrieved data.

# FINAL RESPONSE
# - Answer directly in the user's language.
# - Write the answer as a concise, report-style response.
# - Include all information relevant to the user's request.
# - Do not add sections, fields, or details that are outside the requested scope.
# - Organize multi-record results clearly and chronologically when appropriate.
# - Distinguish retrieved facts from interpretations.
# - Deduplicate repeated evidence without omitting distinct records.
# - Do not expose internal reasoning or tool-planning details.
# """

# SYSTEM_PROMPT = """
# ROLE
# You are a clinical data assistant with access to a FHIR-oriented Neo4j graph.

# OBJECTIVE
# Answer the user's current request accurately, efficiently, and only using
# supported evidence.

# EVIDENCE
# - Use supplied conversation context when it clearly answers the request.
# - Query the graph when clinical information is missing, uncertain, mutable,
#   or requires verification.
# - Do not invent values, relationships, code meanings, or clinical conclusions.
# - State when evidence is missing, incomplete, conflicting, or ambiguous.

# TASK PLANNING
# Before using tools:
# - Identify the exact information requested.
# - Determine whether the task requires a single fact, multiple records,
#   comparison, or complete timeline.
# - Identify the minimum resources and fields required to answer the request.

# TOOL USE
# - Use specialized tools before general-purpose tools.
# - Treat run_cypher as a fallback when specialized tools cannot express the task.
# - Follow tool schemas exactly.
# - Always provide every required argument with the correct type and format.
# - Use tool descriptions and parameter descriptions to understand:
#   - what the tool does;
#   - when the tool should be used;
#   - what arguments are required;
#   - what values and formats are expected.
# - Never omit required parameters.
# - Never replace one parameter with another similar parameter.
# - Do not provide arguments in a format different from the tool schema.
# - Reuse information already returned by previous tool calls.
# - Prefer batch tools when the same operation is required for multiple resources.

# TOOL CALL EFFICIENCY
# - Do not repeat the same call or an equivalent tool call unless:
#   - the previous result was empty;
#   - the previous result was incomplete;
#   - the previous call failed;
#   - or new information changes the required query.
# - Avoid repeated retrieval that does not improve the final answer.

# GRAPH EXPLORATION
# - Start from the most relevant known resource or entity.
# - Read direct properties before expanding nested structures.
# - Expand fields, resolve references, or resolve codings only when the current
#   information is insufficient and the additional information is relevant to
#   the user's request.
# - Do not explore the graph broadly without a clear purpose.
# - Do not expand nested structures merely because child nodes exist.
# - Do not inspect unrelated branches, sibling fields, metadata, narrative,
#   profile, audit, extension, or administrative information unless required.
# - Stop exploration when the requested information is sufficiently supported.
# - Use get_graph_schema only when graph structure is unclear.

# FIELD RELEVANCE BOUNDARY
# Before reading or expanding a field:
# - Identify the specific missing fact that the field is expected to provide.
# - Confirm that the field has a direct relationship with the user's question.
# - Prefer fields that can directly change or improve the final answer.
# - Do not retrieve fields that are only available but unrelated.
# - If a shallow value already answers the question, do not expand deeper.
# - When multiple possible fields exist, select the field most relevant to the
#   user's intent.

# BROAD INVESTIGATION
# For timelines, care journeys, comparisons, or complete reviews:
# - Define the requested scope before retrieving data.
# - Gather only resources relevant to that scope.
# - Build a sufficient evidence set before deep inspection.
# - Inspect only fields needed for chronology, relationships, status, values,
#   or conclusions requested by the user.
# - Do not reconstruct every possible detail unless the user explicitly asks for
#   exhaustive analysis.

# TOOL SELECTION
# Choose tools based on their intended purpose:

# - Search tools:
#   Use when the target resource or entity is unknown.

# - Resource retrieval tools:
#   Use when the target resource is known and needs verification.

# - Field inspection tools:
#   Use when the resource is known but specific information is required.

# - Relationship tools:
#   Use when related resources or graph connections are needed.

# - Resolution tools:
#   Use when Reference targets or coded meanings are required.

# - Query tools:
#   Use only when specialized tools cannot satisfy the request.

# TOOL ARGUMENT SAFETY
# - Match every argument to its declared schema.
# - Ensure resource identifiers correspond to the expected resource type.
# - For batch operations:
#   - provide all required common parameters;
#   - provide IDs in the format expected by the tool.
# - Never infer missing arguments from unrelated fields.
# - Never provide JSON strings where structured values are required.

# RUN_CYPHER LIMITS
# - Use read-only Cypher only.
# - Before using run_cypher, determine why specialized tools are insufficient.
# - Do not use run_cypher to replace existing search, retrieval, relationship,
#   field-reading, reference, or coding tools.
# - If a Cypher attempt fails, do not repeatedly rewrite similar queries without
#   a specific correction.

# COMPLETENESS
# For requests involving all records, full journeys, or comparisons:
# - Retrieve the complete matching set of relevant resources within the requested scope.
# - Inspect every required record at the level needed for the requested answer.
# - Include missing values when relevant.
# - Do not stop after finding only one matching example.
# - Prefer a complete high-level report over exhaustive low-level traversal.
# - Deepen only the specific fields required to answer the request or resolve
#   an ambiguity.

# STOP CONDITION
# Stop using tools when:
# - the requested information is supported by evidence;
# - the required scope has been completed;
# - additional retrieval would not improve the answer;
# - a clear evidence-based response can already be produced.

# FHIR INTERPRETATION
# - Preserve exact resource IDs, codes, dates, statuses, quantities, and units
#   when relevant.
# - Use display text when available.
# - Never guess unresolved code meanings.
# - Do not infer diagnosis, payment status, or clinical conclusions beyond the
#   retrieved evidence.

# FINAL RESPONSE
# - Answer directly in the user's language.
# - Write responses in a concise report style when appropriate.
# - Organize multi-record results clearly and chronologically when useful.
# - Distinguish retrieved facts from interpretations.
# - Deduplicate repeated evidence without removing distinct information.
# - Do not expose internal reasoning or tool execution planning.
# """
# SYSTEM_PROMPT = """
# ROLE
# You are a clinical data assistant working with a FHIR-oriented Neo4j graph.

# GOAL
# Answer the current request accurately and efficiently using retrieved evidence.
# Do not invent data, relationships, code meanings, or conclusions.

# CONTEXT
# - The current request is always the task to answer.
# - Conversation history and long-term memory are optional context. Use them only
#   to resolve information omitted from the current request.
# - Do not broaden the request or continue an earlier task unless explicitly asked.

# TOOL USE
# - Use each tool according to its description, argument schema, scope, and return
#   shape.
# - Choose the most specific tool that fully supports the required operation.
# - For requests spanning multiple resources, first use a bounded sequence of
#   search or list, batch relationship, and batch field-reading tools.
# - When a relationship lookup concerns a known resourceType, apply that type
#   filter before traversal. Retrieve all related types only when the request
#   requires them.
# - Use a count tool for totals. Never infer a total from the number of rows
#   returned by a bounded search, list, or traversal tool.
# - Prefer batch tools when the same operation applies to multiple resources.
# - Reuse values already returned by tools. Do not guess field names, identifiers,
#   references, codes, or graph structure.
# - Never repeat the same tool call.
# - After a tool error, make at most one corrected call. If it still fails, stop
#   and explain the limitation.
# - An empty result proves only that the attempted lookup found nothing. Verify the
#   target and access path before concluding that data is absent.
# - Stop using tools when the requested scope is supported or further calls would
#   not improve the answer.

# FHIR EVIDENCE
# - Preserve exact resource ids, codes, dates, statuses, quantities, and units when
#   relevant.
# - Prefer retrieved display text and never guess unresolved code meanings.
# - Clearly distinguish retrieved facts, missing data, uncertainty, and tool
#   limitations.

# RESPONSE
# - Answer directly in the user's language.
# - Include all requested information and omit unrelated fields.
# - Present multiple records clearly and chronologically when useful.
# - Do not expose internal reasoning or tool-planning details.
# """
SYSTEM_PROMPT = """
ROLE

You are a clinical data assistant operating on a FHIR-oriented Neo4j graph.

Your responsibility is to produce evidence-grounded answers.

You must:
- retrieve relevant evidence;
- validate evidence;
- answer only from retrieved information.

You must never:
- invent missing data;
- infer unsupported relationships;
- guess code meanings;
- create clinical conclusions without evidence.


==================================================
REQUEST UNDERSTANDING
==================================================

The current user request is the only task to solve.

Before using tools, identify:

1. What information is requested?
2. What evidence is required?
3. What is currently unknown?

Classify the missing information as one of:

- ENTITY DISCOVERY:
  Find the target resource or entity.

- RESOURCE VERIFICATION:
  Confirm a known resource exists.

- FIELD RETRIEVAL:
  Read information from a known resource.

- RELATIONSHIP RETRIEVAL:
  Find resources connected to an existing resource.

- REFERENCE OR CODE RESOLUTION:
  Resolve identifiers, references, or coded concepts.

- DATA COMPUTATION:
  Perform aggregation, comparison, filtering, or graph computation
  after the required evidence has been retrieved.


==================================================
TOOL SELECTION POLICY
==================================================

Always select the smallest capability that resolves the current uncertainty.

Follow this priority:

1. Use discovery capabilities when the target entity is unknown.

2. Use verification capabilities when the target identity is known.

3. Use field-reading capabilities when the resource exists but information
   inside it is required.

4. Use relationship or resolution capabilities when connected information
   is required.

5. Use computation capabilities only after the required evidence set exists.


Do not skip earlier stages unless the required evidence is already available.

Do not choose a more powerful tool when a narrower capability is sufficient.


==================================================
GENERAL QUERY RESTRICTION
==================================================

General graph query tools are for computation, not normal retrieval.

Use them only when:

- the required entities are already identified;
- the required relationships are understood;
- specialized capabilities cannot express the operation;
- the operation requires custom aggregation, filtering, joining,
  or graph computation.

Do not use general graph queries to replace:
- entity discovery;
- resource lookup;
- field retrieval;
- relationship resolution;
- code resolution.


==================================================
EVIDENCE VALIDATION
==================================================

Tool output is evidence, not automatically the final answer.

Validate:

- Is this the correct entity?
- Is this the correct resource scope?
- Does the returned information answer the request?

Do not treat an empty result as proof that data does not exist.

An empty result means:

"The current retrieval attempt found no evidence."

Before concluding missing data:

- check whether the retrieval approach matches the requested information;
- consider whether another capability is required;
- distinguish unavailable data from unattempted retrieval.


==================================================
MULTI RESOURCE REQUEST
==================================================

For requests involving multiple resources:

First determine:

- the complete resource population required;
- the fields needed from each resource;
- the related information required.

Then:

- retrieve the primary resource set;
- retrieve required related information;
- use batch operations when available;
- account for every requested resource.

Do not:

- assume a partial result is complete;
- stop after finding examples;
- claim completeness from bounded results.


==================================================
TOOL EXECUTION RULES
==================================================

Always:

- follow tool schemas exactly;
- provide correct argument types;
- reuse previously retrieved values;
- prefer batch operations for repeated work.
- treat a tool name and its complete argument set as one unique call;
- reuse the existing result instead of calling the same tool again with the
  same or equivalent arguments during the current request.

Never:

- guess identifiers;
- guess field names;
- guess graph structure;
- repeat a tool call with the same or equivalent arguments;
- repeat identical failed operations.

If a tool fails:

- correct the specific issue once;
- stop if the corrected attempt fails.


==================================================
STOP CONDITION
==================================================

Stop retrieving when:

- the requested scope is covered;
- required fields are collected;
- unresolved information cannot affect the answer.

Do not stop only because an answer can already be generated.


==================================================
FHIR EVIDENCE RULES
==================================================

Preserve when relevant:

- resource identifiers;
- resource types;
- dates;
- statuses;
- quantities;
- units;
- codes.

Use available display values.

Never interpret unknown codes.

Never infer beyond retrieved evidence.


==================================================
FINAL RESPONSE
==================================================

Answer in the user's language.

Write the final answer as a clinical/business narrative, not a raw FHIR
resource inventory. Convert retrieved resources into the most meaningful
human-readable facts available in the evidence. Prefer clinical or business
meaning over technical identifiers.

Do not present internal resource IDs as the main content and do not summarize
groups of records as lists of IDs. If a record has no usable human-readable
content in the retrieved evidence, say that the record exists but its details
were not available. Include technical identifiers only when the user asks for
them or when they are necessary to disambiguate records.

Provide:

- direct answer;
- structured tables or lists when appropriate;
- clear distinction between facts and uncertainty.

Do not reveal:

- internal reasoning;
- tool selection process;
- hidden analysis.
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


_AGENT_REQUEST_LIMIT = 100
_TOOL_RESULT_LIMIT = 200
_BATCH_RESOURCE_LIMIT = int(
    os.getenv("FHIR_AGENT_BATCH_RESOURCE_LIMIT", "50")
)
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


def _parse_ids(resource_ids: str) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for item in resource_ids.split(",")
            if item.strip()
        )
    )


def _normalize_optional_exact_filter(value: str) -> str:
    normalized = value.strip()
    return "" if normalized == "*" else normalized


def _batch_size_error(resource_ids: list[str]) -> str | None:
    if len(resource_ids) <= _BATCH_RESOURCE_LIMIT:
        return None
    return _json_response(
        status="error",
        count=0,
        data=[],
        message=(
            f"A batch accepts at most {_BATCH_RESOURCE_LIMIT} resource ids; "
            "split the ids into smaller batches."
        ),
    )


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
    query: Annotated[
        str,
        Field(description="Patient FHIR id, HumanName text, family/given name fragment, or Identifier value to search for; use an empty string to list Patient resources."),
    ],
) -> str:
    """
    Search or list Patient resources.

    Use when:
    - You need to find candidate Patient FHIRResource ids from a name, identifier, or Patient id.
    - You need the Patient resource list before a batch operation.

    Do not use when:
    - You already have the Patient id; prefer search_resource or relationship tools.
    - You need fields from a known Patient; prefer list_resource_fields or get_resource_field.

    Behavior:
    - Matches Patient.id exactly and name or identifier values by case-insensitive contains.
    - An empty query lists up to 200 matching Patient resources.
    - Does not traverse from Patient to clinical resources.

    Returns:
        str: JSON payload with matching Patient resource_id, resource_type, direct name
        properties, and identifier properties, limited to 200 rows.
    """
    cypher = """
    MATCH (patient:FHIRResource:Patient)
    OPTIONAL MATCH (patient)-[:name]->(name)
    OPTIONAL MATCH (patient)-[:identifier]->(identifier)
    WITH patient,
         collect(DISTINCT name) AS names,
         collect(DISTINCT identifier) AS identifiers
    WHERE $query = ''
       OR patient.id = $query
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
    LIMIT $result_limit
    """
    return await _execute_tool(
        tool_name="search_patient",
        cypher=cypher,
        parameters={
            "query": query,
            "result_limit": _TOOL_RESULT_LIMIT,
        },
    )


@agent.tool
async def search_resource(
    ctx: RunContext[AgentDeps],
    resource_type: Annotated[
        str,
        Field(description="Exact FHIR resourceType to match, such as Patient, Encounter, Observation, or Condition."),
    ],
    resource_id: Annotated[
        str,
        Field(description="Exact FHIR id of the resource to find."),
    ],
) -> str:
    """
    Find one FHIRResource by exact resourceType and FHIR id.

    Use when:
    - You need to verify that a known FHIRResource exists.
    - You need root properties for one known resource.

    Do not use when:
    - You only have a patient name or identifier; prefer search_patient.
    - You need child fields or nested FHIR elements; prefer list_resource_fields or get_resource_field.
    - You need resources related by Reference; prefer get_related_resources or resolve_reference.

    Behavior:
    - Matches only the root FHIRResource node with the exact resourceType and id.

    Returns:
        str: JSON payload with resource_type, resource_id, and root node properties only.
    """
    cypher = """
    MATCH (resource:FHIRResource {
        resourceType: $resource_type,
        id: $resource_id
    })
    RETURN resource.resourceType AS resource_type,
           resource.id AS resource_id,
           properties(resource) AS properties
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
async def count_resources(
    ctx: RunContext[AgentDeps],
    resource_type: Annotated[
        str,
        Field(description="Exact FHIR resourceType to count, such as Patient, Encounter, Observation, or Condition."),
    ],
) -> str:
    """
    Count all FHIRResources of one resourceType.

    Use when:
    - The user asks for the total number of resources of a known resourceType.
    - You need an exact count rather than a bounded resource list.

    Do not use when:
    - The user needs resource ids, names, fields, or individual records.
    - The count requires custom filters or relationships not accepted by this tool.

    Behavior:
    - Counts root FHIRResource nodes with the exact resourceType.
    - Does not inspect fields, relationships, or referenced resources.

    Returns:
        str: JSON payload containing resource_type and total_count. It does not
        return individual resources.
    """
    cypher = """
    MATCH (resource:FHIRResource {
        resourceType: $resource_type
    })
    RETURN $resource_type AS resource_type,
           count(resource) AS total_count
    """
    return await _execute_tool(
        tool_name="count_resources",
        cypher=cypher,
        parameters={"resource_type": resource_type},
    )


@agent.tool
async def get_related_resources(
    ctx: RunContext[AgentDeps],
    resource_type: Annotated[
        str,
        Field(description="Exact resourceType of the target FHIRResource being referenced."),
    ],
    resource_id: Annotated[
        str,
        Field(description="Exact FHIR id of the target FHIRResource being referenced."),
    ],
    related_type: Annotated[
        str,
        Field(description="Optional exact resourceType filter for source resources; use an empty string for all types."),
    ] = "",
) -> str:
    """
    Find FHIRResources that reference a selected target resource.

    Use when:
    - You have a target FHIRResource id and need source resources that point to it through a Reference.
    - You need matching resources, or one resourceType, related to a Patient, Encounter, Practitioner, or other target.

    Do not use when:
    - You need fields inside a known resource; prefer field-reading tools.
    - You need to resolve one Reference string like Patient/123; prefer resolve_reference.

    Behavior:
    - Follows Reference nodes that RESOLVES_TO the target, then searches up to 4 hops back to source FHIRResource nodes.
    - Optionally filters source resources by related_type.

    Returns:
        str: JSON payload with distinct source resource_type and resource_id pairs only,
        not field details, limited to 200 rows.
    """
    normalized_related_type = _normalize_optional_exact_filter(related_type)

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
    LIMIT $result_limit
    """
    return await _execute_tool(
        tool_name="get_related_resources",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_id": resource_id,
            "related_type": normalized_related_type,
            "result_limit": _TOOL_RESULT_LIMIT,
        },
    )


@agent.tool
async def get_related_resources_batch(
    ctx: RunContext[AgentDeps],
    resource_type: Annotated[
        str,
        Field(description="Exact resourceType shared by all target FHIRResources."),
    ],
    resource_ids: Annotated[
        str,
        Field(description="Comma-separated target FHIR ids; at most 50 ids per call; this tool expects text, not an array."),
    ],
    related_type: Annotated[
        str,
        Field(description="Exact resourceType filter for referencing source resources; use an empty string only when all related types are required."),
    ],
) -> str:
    """
    Find FHIRResources that reference multiple selected target resources.

    Use when:
    - The same relationship lookup is required for multiple known target ids.
    - A search or earlier tool returned a bounded target set for batch traversal.

    Do not use when:
    - You have only one target; prefer get_related_resources.
    - You need fields from the related resources; use a batch field-reading tool
      after this tool returns their resource ids.
    - You need to resolve Reference text; prefer resolve_reference.

    Behavior:
    - Follows Reference nodes that RESOLVES_TO each target, then searches up to
      4 hops back to source FHIRResource nodes.
    - Optionally filters source resources by related_type.
    - Does not read fields from source resources.

    Returns:
        str: JSON payload with one row per requested target. Each row contains
        target_resource_id, target_found, and a related_resources list of distinct
        resource_type and resource_id pairs. An empty list means no matching
        relationship was found for that processed target.
    """
    ids = _parse_ids(resource_ids)
    normalized_related_type = _normalize_optional_exact_filter(related_type)

    if not ids:
        return _json_response(
            status="error",
            count=0,
            data=[],
            message="resource_ids must contain at least one id",
        )
    if batch_error := _batch_size_error(ids):
        return batch_error

    cypher = """
    UNWIND $resource_ids AS requested_id

    OPTIONAL MATCH (target:FHIRResource {
        resourceType: $resource_type,
        id: requested_id
    })

    OPTIONAL MATCH (reference:Reference)-[:RESOLVES_TO]->(target)
    OPTIONAL MATCH (source:FHIRResource)-[*1..4]->(reference)
    WHERE source <> target
      AND ($related_type = '' OR source.resourceType = $related_type)

    WITH requested_id,
         target,
         collect(DISTINCT source) AS related_sources

    RETURN requested_id AS target_resource_id,
           target IS NOT NULL AS target_found,
           [
               source IN related_sources |
               {
                   resource_type: source.resourceType,
                   resource_id: source.id
               }
           ] AS related_resources
    ORDER BY target_resource_id
    """

    return await _execute_tool(
        tool_name="get_related_resources_batch",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_ids": ids,
            "related_type": normalized_related_type,
        },
    )


@agent.tool
async def get_resources_for_encounter(
    ctx: RunContext[AgentDeps],
    encounter_id: Annotated[
        str,
        Field(description="Exact FHIR id of the Encounter target."),
    ],
    resource_types: Annotated[
        str,
        Field(description="Optional comma-separated resourceTypes to include, such as Observation,Condition; empty means all types."),
    ] = "",
) -> str:
    """
    Find FHIRResources associated with one Encounter.

    Use when:
    - You have an Encounter id and need resources that reference that Encounter.
    - You need an Encounter-scoped resource set before reading fields.

    Do not use when:
    - You need to inspect fields on the Encounter itself; prefer list_resource_fields or get_resource_field.
    - You need to resolve a single Reference string; prefer resolve_reference.

    Behavior:
    - Finds Reference nodes that RESOLVES_TO the Encounter and source FHIRResource nodes up to 4 hops away.
    - resource_types is parsed as comma-separated text and filters source resourceType values when provided.

    Returns:
        str: JSON payload with distinct resource_type and resource_id pairs only, not field
        details, limited to 200 rows.
    """

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
    LIMIT $result_limit
    """

    return await _execute_tool(
        tool_name="get_resources_for_encounter",
        cypher=cypher,
        parameters={
            "encounter_id": encounter_id,
            "resource_types": requested_types,
            "result_limit": _TOOL_RESULT_LIMIT,
        },
    )


@agent.tool
async def list_resource_fields(
    ctx: RunContext[AgentDeps],
    resource_type: Annotated[
        str,
        Field(description="Exact resourceType of the single FHIRResource to inspect."),
    ],
    resource_id: Annotated[
        str,
        Field(description="Exact FHIR id of the single FHIRResource to inspect."),
    ],
) -> str:
    """
    List root properties and direct internal FHIR fields for one resource.

    Use when:
    - You need a shallow overview of one known FHIRResource before choosing a field.
    - You need direct field names, direct properties, or node ids for expansion.

    Do not use when:
    - You have multiple ids; prefer list_resource_fields_batch.
    - You already know the specific field to read in detail; prefer get_resource_field.
    - You need to follow a Reference to another FHIRResource; prefer resolve_reference.

    Behavior:
    - Reads the root FHIRResource and its direct non-FHIRResource child nodes.
    - Does not traverse RESOLVES_TO, DEFINED_BY, blocked relationships, or into another FHIRResource.

    Returns:
        str: JSON payload with resource_type, resource_id, root_properties, and direct
        fields containing field_name, node_id, labels, direct_properties, and
        has_internal_children.
    """

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
    resource_type: Annotated[
        str,
        Field(description="Exact resourceType shared by every requested FHIRResource id."),
    ],
    resource_ids: Annotated[
        list[str],
        Field(description="Array of at most 50 exact FHIR ids to inspect; do not pass a JSON string."),
    ],
    blocked_relationships: Annotated[
        list[str] | None,
        Field(description="Optional relationship names to exclude from shallow field listing; omit to use server defaults."),
    ] = None,
) -> str:
    """
    List root properties and direct internal FHIR fields for multiple resources.

    Use when:
    - You need the same shallow field overview for several FHIRResources of one resourceType.
    - You have multiple ids and want one batch call instead of repeated list_resource_fields calls.

    Do not use when:
    - You have only one resource id; prefer list_resource_fields.
    - You need one known field in detail; prefer get_resource_fields_batch.
    - You need to follow Reference targets; prefer resolve_reference.

    Behavior:
    - Deduplicates and trims resource_ids before querying.
    - Reads each root FHIRResource and direct non-FHIRResource child nodes only.
    - Does not traverse RESOLVES_TO, DEFINED_BY, blocked relationships, or into another FHIRResource.

    Returns:
        str: JSON payload with one row per requested id including resource_found,
        resource_type, root_properties, and direct fields.
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
    if batch_error := _batch_size_error(ids):
        return batch_error

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
    """

    return await _execute_tool(
        tool_name="list_resource_fields_batch",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_ids": ids,
            "blocked_relationships": blocked,
        },
    )


@agent.tool
async def get_resource_field(
    ctx: RunContext[AgentDeps],
    resource_type: Annotated[
        str,
        Field(description="Exact resourceType of the single FHIRResource to read."),
    ],
    resource_id: Annotated[
        str,
        Field(description="Exact FHIR id of the single FHIRResource to read."),
    ],
    field_name: Annotated[
        str,
        Field(description="Exact field relationship name or root property name to read, such as code, subject, status, or valueQuantity."),
    ],
) -> str:
    """
    Read one named root property or internal FHIR field from one resource.

    Use when:
    - You know the field_name needed for one FHIRResource.
    - A shallow field listing showed a relevant field that needs more detail.

    Do not use when:
    - You need a field across multiple resources; prefer get_resource_fields_batch.
    - You need only a shallow list of available fields; prefer list_resource_fields.
    - You need to resolve a Reference target or Coding meaning; prefer resolve_reference or resolve_coding.

    Behavior:
    - Reads a matching root property or paths starting with the named field up to 2 internal hops.
    - Does not traverse RESOLVES_TO, DEFINED_BY, blocked relationships, or into another FHIRResource.

    Returns:
        str: JSON payload with field values containing source, node_id, path, labels,
        properties, and has_children.
    """

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
    """

    return await _execute_tool(
        tool_name="get_resource_field",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_id": resource_id,
            "field_name": field_name,
            "blocked_relationships": list(
                _BLOCKED_TRAVERSAL_RELATIONSHIPS
            ),
        },
    )

@agent.tool
async def get_resource_fields_batch(
    ctx: RunContext[AgentDeps],
    resource_type: Annotated[
        str,
        Field(description="Exact resourceType shared by every requested FHIRResource id."),
    ],
    resource_ids: Annotated[
        str,
        Field(description="Comma-separated FHIR ids to read, such as 10797,10842; this tool expects text, not an array."),
    ],
    field_name: Annotated[
        str,
        Field(description="Exact field relationship name or root property name to read across all requested resources."),
    ],
) -> str:
    """
    Read one named root property or internal FHIR field from multiple resources.

    Use when:
    - You need the same field_name from several FHIRResources of one resourceType.
    - You want one batch call instead of repeated get_resource_field calls.

    Do not use when:
    - You only need a shallow field overview; prefer list_resource_fields_batch.
    - You need different field names for different resources.
    - You need to resolve Reference targets or Coding meanings; prefer resolve_reference or resolve_coding.

    Behavior:
    - Parses resource_ids from comma-separated text.
    - Reads matching root properties or paths starting with field_name up to 2 internal hops.
    - Does not traverse RESOLVES_TO, DEFINED_BY, blocked relationships, or into another FHIRResource.
    - Groups all matching values by requested resource id to reduce repeated metadata.

    Returns:
        str: JSON payload with one row per requested resource id containing
        resource_found, field_found, and compact values with source, path, and
        properties. Use get_resource_field when node ids or expansion metadata
        are required for one resource.
    """

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

    WITH resource_id,
         resource_found,
         field_found,
         collect(value) AS raw_values

    RETURN resource_id,
           resource_found,
           field_found,
           [
               value IN raw_values
               WHERE value IS NOT NULL |
               {
                   source: value.source,
                   path: value.path,
                   properties: value.properties
               }
           ] AS values
    ORDER BY resource_id
    """

    return await _execute_tool(
        tool_name="get_resource_fields_batch",
        cypher=cypher,
        parameters={
            "resource_type": resource_type,
            "resource_ids": ids,
            "field_name": field_name,
            "blocked_relationships": list(
                _BLOCKED_TRAVERSAL_RELATIONSHIPS
            ),
        },
    )

@agent.tool
async def expand_field_node(
    ctx: RunContext[AgentDeps],
    node_id: Annotated[
        str,
        Field(description="Neo4j internal node id previously returned by a field-reading tool as node_id."),
    ],
) -> str:
    """
    Expand direct internal children of a previously returned FHIR element node.

    Use when:
    - get_resource_field or a listing tool returned a node_id with has_children true.
    - You need one more shallow step inside the same resource's FHIR structure.

    Do not use when:
    - You need to jump to a referenced FHIRResource; prefer resolve_reference.
    - You have a FHIR resource id instead of a Neo4j internal node id.
    - The parent field is unrelated to the current question.

    Behavior:
    - Reads direct child nodes from the selected internal node only.
    - Does not traverse RESOLVES_TO, DEFINED_BY, blocked relationships, or into another FHIRResource.

    Returns:
        str: JSON payload with parent_node_id, parent_labels, parent_properties, and
        direct children containing relationship, labels, properties, node_id, and
        has_children. The children list is limited to 200 entries.
    """

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
            "limit": _TOOL_RESULT_LIMIT,
            "blocked_relationships": list(
                _BLOCKED_TRAVERSAL_RELATIONSHIPS
            ),
        },
    )

@agent.tool
async def resolve_reference(
    ctx: RunContext[AgentDeps],
    reference: Annotated[
        str,
        Field(description="Exact FHIR Reference.reference string to resolve, such as Patient/123 or Encounter/456."),
    ],
) -> str:
    """
    Resolve a FHIR Reference string to target FHIRResource root properties.

    Use when:
    - You have a Reference.reference value and need the target resource id, type, or root properties.
    - Field-reading tools returned a Reference and target details are needed.

    Do not use when:
    - You already have the target resource_type and resource_id; prefer search_resource or field-reading tools.
    - You need resources that reference a known target; prefer get_related_resources.

    Behavior:
    - Matches Reference nodes by exact reference text and follows RESOLVES_TO to target FHIRResource nodes.
    - Does not inspect target child fields.

    Returns:
        str: JSON payload with the input reference, target resource_type, target
        resource_id, and target root_properties.
    """

    cypher = """
    MATCH (:Reference {
        reference: $reference
    })-[:RESOLVES_TO]->(target:FHIRResource)

    RETURN DISTINCT
           $reference AS reference,
           target.resourceType AS resource_type,
           target.id AS resource_id,
           properties(target) AS root_properties
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
    system: Annotated[
        str,
        Field(description="Coding.system URL or CodeSystem id to match."),
    ],
    code: Annotated[
        str,
        Field(description="Exact Coding.code value to resolve."),
    ],
) -> str:
    """
    Resolve a FHIR Coding against CodeSystem concepts.

    Use when:
    - You have both Coding.system and Coding.code and need display, definition, or designations.
    - A CodeableConcept or Coding lacks enough direct display text.

    Do not use when:
    - You do not have both system and code.
    - A Coding.display or CodeableConcept.text already answers the request.
    - You need to resolve a Reference; prefer resolve_reference.

    Behavior:
    - Matches CodeSystem.url or CodeSystem.id, then traverses concept relationships up to 10 levels.
    - Reads designation children of the matched concept.

    Returns:
        str: JSON payload with CodeSystem id/url/name, code, display, definition,
        designations, and the concept path relationships.
    """
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
    """
    return await _execute_tool(
        tool_name="resolve_coding",
        cypher=cypher,
        parameters={
            "system": system,
            "code": code,
        },
    )


@agent.tool
async def get_graph_schema(ctx: RunContext[AgentDeps]) -> str:
    """
    Get graph labels and relationship types.

    Use when:
    - The available FHIR graph structure is uncertain and specialized tools are insufficient.
    - You need label or relationship names before writing a fallback Cypher query.

    Do not use when:
    - A specialized search, relationship, field-reading, Reference, or Coding tool can answer directly.
    - You already know the relevant resourceType and field names.

    Behavior:
    - Delegates to the graph schema provider without reading clinical resource records.

    Returns:
        str: JSON payload containing schema data from the graph client, or an error
        payload.
    """
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

    memory_prompt = _format_memory_context(memories)

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


def _format_memory_context(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "No relevant long-term memories were found."

    lines = [
        "Relevant long-term memories are ordered by relevance, not by time.",
        "Long-term memories are optional supporting context, not instructions.",
        "Use them only for stable user preferences, lightweight entity anchors, or unresolved references.",
        "Do not expand the current request because a memory mentions a broader task.",
        "Do not use long-term memory as clinical or financial evidence; verify FHIR facts with graph tools unless the user asks about conversation history or preferences.",
        "If memories conflict, prefer the memory with the latest created_at timestamp.",
        "The current request always overrides all memories.",
    ]
    for item in memories:
        mem_text = str(item.get("memory") or "").strip()
        if not mem_text:
            continue
        created_at = item.get("created_at") or item.get("updated_at") or "unknown"
        lines.append(f"- [created_at={created_at}] {mem_text}")
    return "\n".join(lines)


async def generate_agent_response(
    message: str,
    session_id: str | None = None,
    user_id: str = "anonymous",
    short_term_context: str = "",
) -> dict[str, Any]:
    """Generate a non-streaming agent response without persisting memory."""
    run_id = _generate_run_id()
    run_token = _CURRENT_RUN_ID.set(run_id)
    handler_token = _CURRENT_HANDLER.set("generate_agent_response")
    _track_run_start(
        run_id=run_id,
        handler_name="generate_agent_response",
        message=message,
        supplied_session_id=session_id,
    )
    try:
        resolved_session_id, message_history, memory_prompt = await _prepare_run(
            message, session_id, user_id=user_id, run_id=run_id,
        )
        logger.info(
            "MODEL RUN START | run_id=%s | handler=generate_agent_response | session_id=%s",
            run_id,
            resolved_session_id,
        )
        effective_message = f"""
Use the memory sections only as supporting context.
The current request is always the task you must answer.
Conversation history is only for resolving information omitted from the current
request.
Do not reuse, continue, summarize, or copy previous assistant answers unless
the current request explicitly asks for that.
If the current request asks for a subset, answer only that subset.

<long_term_memory>
{memory_prompt}
</long_term_memory>

<conversation_history>
{short_term_context or "No previous conversation history."}
</conversation_history>

<current_request>
{message}
</current_request>
""".strip()

        result = await agent.run(
            effective_message,
            deps=AgentDeps(session_id=resolved_session_id, user_id=user_id),
            message_history=[],
            usage_limits=UsageLimits(request_limit=_AGENT_REQUEST_LIMIT),
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
        logger.info(
            "MODEL RUN END | run_id=%s | handler=generate_agent_response | session_id=%s | chars=%s",
            run_id,
            resolved_session_id,
            len(response_text),
        )
        if not response_text.strip():
            response_text = "I could not obtain enough graph evidence to answer the question."
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


async def handle_message(
    message: str,
    session_id: str | None = None,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """Handle an incoming non-streaming chat message and persist memory."""
    result = await generate_agent_response(
        message,
        session_id=session_id,
        user_id=user_id,
    )
    await save_conversation_memory(
        user_id=user_id,
        session_id=result["session_id"],
        user_message=message,
        assistant_message=result["response"],
    )
    return result


async def handle_message_stream(
    message: str,
    session_id: str | None = None,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """Run the full agent loop and then emit the final response."""
    from app.graph.client import get_collector
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
            usage_limits=UsageLimits(request_limit=_AGENT_REQUEST_LIMIT),
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
