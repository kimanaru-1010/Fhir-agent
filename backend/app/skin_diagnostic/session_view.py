"""Response shaping helpers for skin diagnostic runs."""

from __future__ import annotations


PIPELINE_STEPS = [
    "visual_extract",
    "knowledge_base",
    "clinical_planner_round1",
    "user_interview_round1",
    "clinical_planner_round2",
    "user_interview_round2",
    "diagnostic_reasoning",
]


def get_step_progress(step: str) -> int:
    if not step:
        return 0
    try:
        return PIPELINE_STEPS.index(step) + 1
    except ValueError:
        return 0


def build_result(state: dict) -> dict:
    if not state:
        return {}
    return {
        "ranked_diagnoses": state.get("ranked_diagnoses", []),
        "reasoning": state.get("reasoning", ""),
        "visual_observations": state.get("visual_observations", ""),
        "visual_differentials": state.get("visual_differentials", []),
        "qa_history": state.get("qa_history", ""),
    }


def get_pending_questions(run) -> list[dict] | None:
    if run.status != "interrupt":
        return None
    if run.pending_questions:
        return run.pending_questions
    if run.pending_question:
        return [run.pending_question]
    return None
