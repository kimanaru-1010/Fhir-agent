"""Pipeline utility functions."""

import unicodedata


def normalize_disease_name(name: str) -> str:
    """Lowercase + strip Vietnamese accents, for duplicate/hierarchy checks
    on differential disease names."""
    text = (name or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.strip()


def merge_additional_differentials(existing: list, additional_items: list) -> list:
    """Merge a planner's `additional_differentials` output (list of
    {"disease": ..., "rationale": ...} dicts, or plain strings) into an
    existing differential list, skipping entries that already match
    (case/accent-insensitive) something already in the list.

    Returns the merged list (existing items first, new items appended in
    the order they were suggested).
    """
    existing_normalized = {normalize_disease_name(d) for d in existing}
    merged = list(existing)
    for item in additional_items or []:
        disease = (item.get("disease", "") if isinstance(item, dict) else str(item)).strip()
        if not disease:
            continue
        norm = normalize_disease_name(disease)
        if norm and norm not in existing_normalized:
            merged.append(disease)
            existing_normalized.add(norm)
    return merged


def drop_generic_when_specific_present(diff_list: list) -> list:
    """Deterministic backstop for the LLM-based dedup skill
    (deduplicate_differentials.md): if the list contains both a
    generic/umbrella disease name and one or more longer names that start
    with it as a whole-word prefix (e.g. "Viêm da tiếp xúc" alongside
    "Viêm da tiếp xúc kích ứng"), drop the generic entry and keep only the
    more specific subtype name(s).

    This does not rely on the LLM reliably applying the rule on its own —
    it's a plain string-prefix check on normalized (lowercase, no accent)
    names, so it's applied every time regardless of whether the LLM dedup
    step ran, failed, or was skipped.
    """
    if not diff_list or len(diff_list) <= 1:
        return diff_list or []

    normalized = [normalize_disease_name(d) for d in diff_list]
    to_drop = set()
    for i, norm_a in enumerate(normalized):
        if not norm_a:
            continue
        for j, norm_b in enumerate(normalized):
            if i == j:
                continue
            if norm_b.startswith(norm_a + " "):
                to_drop.add(i)
                break

    return [d for i, d in enumerate(diff_list) if i not in to_drop]


def format_visual_for_planner(observations_text: str) -> str:
    """Format visual observations for the Clinical Planner prompt."""
    if not observations_text or observations_text == "Không có quan sát nào.":
        return "(Không có quan sát nào từ hình ảnh)"
    return observations_text


def format_visual_for_diagnose(observations_text: str) -> str:
    """Format visual observations for the Diagnostic Reasoning prompt."""
    if not observations_text or observations_text == "Không có quan sát nào.":
        return "(Không có quan sát nào từ hình ảnh)"
    return observations_text
