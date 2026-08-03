"""Clinical Planner nodes — Round 1 and Round 2."""

import json
import unicodedata

from rich.console import Console

from pipeline.prompts import _PLANNER_INSTRUCTION, _DEDUPLICATE_DIFFERENTIALS
from pipeline.utils import format_visual_for_planner
from utils.json_parser import extract_json
from pipeline.fallback_questions import _fill_to_5
from utils.question_verifier import verify_questions_with_llm
from utils.exam_only_signs import format_exam_only_hints

console = Console(force_terminal=True)


def _normalize_disease(name: str) -> str:
    """Lowercase + strip Vietnamese accents, for duplicate detection when
    merging planner-suggested differentials into the existing list."""
    text = (name or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.strip()


def deduplicate_differentials_with_llm(diff_list: list) -> list:
    """Call LLM to deduplicate and standardize differential disease names using skill prompt."""
    if not diff_list or len(diff_list) <= 1:
        return diff_list or []

    from models.shared_client import call_llm
    from config.settings import REASONING_MODEL_URL, REASONING_MODEL_NAME, REASONING_MODEL_TIMEOUT

    formatted_input = json.dumps(diff_list, ensure_ascii=False, indent=2)
    prompt = _DEDUPLICATE_DIFFERENTIALS.format(differentials_list=formatted_input)

    try:
        output = call_llm(
            prompt=prompt,
            base_url=REASONING_MODEL_URL,
            model=REASONING_MODEL_NAME,
            timeout=REASONING_MODEL_TIMEOUT,
        )
        parsed = extract_json(output)
        if isinstance(parsed, dict) and "deduplicated_differentials" in parsed:
            result = parsed["deduplicated_differentials"]
            if isinstance(result, list) and len(result) > 0:
                console.print(f"[cyan]  + LLM deduplicated differentials: {len(diff_list)} -> {len(result)}[/]")
                return [str(d).strip() for d in result if str(d).strip()]
    except Exception as e:
        console.print(f"[yellow]LLM differential dedup failed: {e}, falling back to Python dedup[/]")

    # Fallback to basic python set dedup
    seen = set()
    deduped = []
    for d in diff_list:
        norm = _normalize_disease(str(d))
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(str(d).strip())
    return deduped


def _call_planner(state: dict, prompt: str, previous_questions: list[dict] | None = None) -> dict:
    """Shared helper: call clinical planner LLM, parse JSON, extract questions."""
    from models.shared_client import call_llm
    from config.settings import REASONING_MODEL_URL, REASONING_MODEL_NAME, REASONING_MODEL_TIMEOUT

    output = call_llm(
        image_path=state.get("image_path"),
        prompt=prompt,
        base_url=REASONING_MODEL_URL,
        model=REASONING_MODEL_NAME,
        timeout=REASONING_MODEL_TIMEOUT,
    )

    parsed = extract_json(output)
    if not isinstance(parsed, dict):
        console.print("[red]X: JSON parse failed[/]")
        console.print(f"[dim]Output preview: {output[:300]}...[/]")
        return {"updated_visual_findings": output[:500], "round1_questions": []}

    # Extract verified findings
    verified = parsed.get("verified_findings", "")
    additional = parsed.get("additional_findings", "")
    if verified or additional:
        parts = [verified] if verified else []
        if additional:
            parts.append(additional)
        updated = "DA XAC NHAN:\n" + verified + ("\n\nBO SUNG:\n" + additional if additional else "")
    else:
        updated = output[:500]

    # Extract questions
    questions = []
    for q in parsed.get("questions", []):
        if isinstance(q, dict):
            questions.append({
                "question": q.get("question", ""),
                "pqrst_category": q.get("pqrst_category", q.get("qrst_category", "")),
                "purpose": q.get("purpose", ""),
                "discriminates": q.get("discriminates", []),
            })
    questions = questions[:5]

    # TASK 2 — merge any anamnesis/history-based additional differentials
    # the planner suggested into the existing candidate list (which started
    # as the vision-only model's list, before it had seen the anamnesis).
    existing_differentials = list(state.get("visual_differentials", []) or [])
    existing_normalized = {_normalize_disease(d) for d in existing_differentials}
    merged_differentials = list(existing_differentials)
    for item in parsed.get("additional_differentials", []):
        disease = (item.get("disease", "") if isinstance(item, dict) else str(item)).strip()
        if not disease:
            continue
        norm = _normalize_disease(disease)
        if norm and norm not in existing_normalized:
            merged_differentials.append(disease)
            existing_normalized.add(norm)
            rationale = item.get("rationale", "") if isinstance(item, dict) else ""
            console.print(f"[cyan]  + Planner added differential: {disease}[/] [dim]({rationale[:80]})[/]")

    # Run LLM dedup to merge duplicate / synonymous disease entries
    merged_differentials = deduplicate_differentials_with_llm(merged_differentials)

    # Verify and rephrase for strict Yes/No, then guarantee exactly 5
    if questions:
        questions = verify_questions_with_llm(
            questions,
            image_path=state.get("image_path"),
            anamnesis=state.get("anamnesis", ""),
            previous_questions=previous_questions,
        )
    # Guarantee exactly 5 questions per round (no more, no less), steering
    # any fallback padding away from previous_questions' text.
    questions = _fill_to_5(questions, previous_questions=previous_questions)

    console.print()

    return {
        "updated_visual_findings": updated,
        "round1_questions": questions,
        "round2_questions": questions,  # placeholder
        "visual_differentials": merged_differentials,
    }


def _format_diff_list(diff_list: list) -> str:
    return "\n".join(f"  {i+1}. {d}" for i, d in enumerate(diff_list)) if diff_list else "(Không có)"


def clinical_planner_round1(state: dict) -> dict:
    raw_diffs = state.get("visual_differentials", [])
    clean_diffs = deduplicate_differentials_with_llm(raw_diffs)
    state["visual_differentials"] = clean_diffs

    visual_summary = format_visual_for_planner(state.get("visual_observations", ""))
    diff_text = _format_diff_list(clean_diffs)
    anamnesis_text = state.get("anamnesis", "") or "(Không có)"

    prompt = _PLANNER_INSTRUCTION.format(
        complaint=anamnesis_text,
        visual_findings=visual_summary,
        visual_differentials=diff_text,
        exam_only_hints=format_exam_only_hints(clean_diffs),
    )

    result = _call_planner(state, prompt)
    return {
        "updated_visual_findings": result["updated_visual_findings"],
        "round1_questions": result["round1_questions"],
        "visual_differentials": result.get("visual_differentials", clean_diffs),
    }


def clinical_planner_round2(state: dict) -> dict:
    qa_history = state.get("qa_history", "")
    round1_questions = state.get("round1_questions", [])
    visual_summary = format_visual_for_planner(state.get("visual_observations", ""))
    diff_text = _format_diff_list(state.get("visual_differentials", []))
    anamnesis_text = state.get("anamnesis", "") or "(Không có)"

    diffs = state.get("visual_differentials", [])
    prompt = _PLANNER_INSTRUCTION.format(
        complaint=anamnesis_text,
        visual_findings=visual_summary,
        visual_differentials=diff_text,
        exam_only_hints=format_exam_only_hints(diffs),
    )
    prompt += (
        "\n\n--- ROUND 2 CONTEXT (EXACTLY 5 NEW PQRST QUESTIONS) ---\n"
        f"Previous Q&A history:\n{qa_history}\n\n"
        "IMPORTANT: Generate EXACTLY 5 NEW questions that are DIFFERENT from the questions "
        "already asked in Round 1. Do NOT repeat any question or ask about the same clinical aspect.\n"
        "Follow the PQRST framework (P=Provocation, Q=Quality, R=Region, S=Severity, T=Time).\n"
        "Aim to cover different PQRST categories."
    )

    result = _call_planner(state, prompt, previous_questions=round1_questions)

    # Extract questions from result and verify with dedup
    questions = result.get("round2_questions", [])
    if questions:
        questions = verify_questions_with_llm(
            questions,
            image_path=state.get("image_path"),
            anamnesis=state.get("anamnesis", ""),
            previous_questions=round1_questions,
        )
    questions = _fill_to_5(questions, previous_questions=round1_questions)

    return {
        "round2_questions": questions,
        "visual_differentials": result.get("visual_differentials", state.get("visual_differentials", [])),
    }