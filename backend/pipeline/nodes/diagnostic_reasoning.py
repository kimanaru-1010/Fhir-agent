"""Diagnostic Reasoning node — final ranked diagnoses."""

from rich.console import Console

from pipeline.prompts import _DIAGNOSE_INSTRUCTION, _DIAGNOSE_SYSTEM
from pipeline.utils import format_visual_for_diagnose
from utils.json_parser import extract_json
from utils.exam_only_signs import format_exam_only_hints

console = Console(force_terminal=True)


def diagnostic_reasoning(state: dict) -> dict:
    visual_summary = format_visual_for_diagnose(state.get("visual_observations", ""))
    diff_list = state.get("visual_differentials", [])
    diff_text = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(diff_list)) if diff_list else "(Không có)"
    anamnesis_text = state.get("anamnesis", "") or "(Không có)"
    qa_history = state.get("qa_history", "(Không có)")
    updated_findings = state.get("updated_visual_findings", "")

    prompt = _DIAGNOSE_INSTRUCTION.format(
        complaint=anamnesis_text,
        image_path=state["image_path"],
        updated_visual_findings=updated_findings if updated_findings else visual_summary,
        visual_differentials=diff_text,
        exam_only_hints=format_exam_only_hints(diff_list),
        qa_history=qa_history,
    )

    from models.shared_client import call_llm
    from config.settings import REASONING_MODEL_URL, REASONING_MODEL_NAME, REASONING_MODEL_TIMEOUT

    system_prompt = _DIAGNOSE_SYSTEM if _DIAGNOSE_SYSTEM else None
    output = call_llm(
        image_path=state.get("image_path"),
        prompt=prompt,
        system_prompt=system_prompt,
        base_url=REASONING_MODEL_URL,
        model=REASONING_MODEL_NAME,
        timeout=REASONING_MODEL_TIMEOUT,
    )

    parsed = extract_json(output)
    if isinstance(parsed, dict):
        ranked = parsed.get("ranked_diagnoses", [])
        reasoning = parsed.get("overall_reasoning", "Không có biện luận.")
    else:
        ranked = []
        reasoning = output[:500]
        console.print("[yellow]Diagnostic reasoning: JSON parse failed[/]")

    console.print()
    return {
        "ranked_diagnoses": ranked,
        "reasoning": reasoning,
    }
