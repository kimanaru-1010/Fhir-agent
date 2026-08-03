"""User Interview Agent — Collects YES/NO answers from the Clinical Planner's questions."""

from typing import List, Tuple
from langgraph.types import interrupt
from rich.console import Console

from tools.decision_tree_agent import (
    PQRST_LABELS,
    format_clinical_summary,
)

console = Console(force_terminal=True)


def run_user_interview(
    questions: list[dict],
    anamnesis: str = "",
    start_q_num: int = 1,
    batch_num: int = 1,
    total_questions: int = 10,
) -> List[Tuple[dict, str]]:
    """Run the user interview loop for a batch of questions.

    Args:
        questions: List of question dicts from Clinical Planner.
                   Each dict: {question, pqrst_category, purpose, discriminates}.
        anamnesis: Initial complaint text.
        start_q_num: Starting question index (e.g., 1 for Round 1, 6 for Round 2).
        batch_num: Batch label number (1 or 2).
        total_questions: Total expected questions across all rounds (default 10).

    Returns:
        List of (question_dict, answer_string) tuples collected in this round.
    """
    qa_pairs: List[Tuple[dict, str]] = []

    total_q = start_q_num - 1
    for q_obj in questions:
        total_q += 1
        pqrst = q_obj.get("pqrst_category", "")
        pqrst_label = PQRST_LABELS.get(pqrst, pqrst) if pqrst else ""
        question_text = q_obj.get("question", "")
        discriminates = q_obj.get("discriminates", [])
        purpose = q_obj.get("purpose", "")

        interrupt_value = {
            "question": question_text,
            "pqrst": pqrst,
            "pqrst_label": pqrst_label,
            "discriminates": discriminates,
            "purpose": purpose,
            "batch": batch_num,
            "question_num": total_q,
            "total_in_batch": total_questions,
        }

        # interrupt() blocks here until main.py collects the answer
        answer = interrupt(interrupt_value)
        qa_pairs.append((q_obj, answer))

    return qa_pairs
