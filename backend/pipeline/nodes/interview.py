"""User Interview nodes — Round 1 and Round 2."""

from rich.console import Console

from tools.decision_tree_agent import format_clinical_summary
from tools.interview_agent import run_user_interview

console = Console(force_terminal=True)


def user_interview_round1(state: dict) -> dict:
    questions = state.get("round1_questions", [])
    anamnesis = state.get("anamnesis", "")

    if not questions:
        console.print("[yellow]Không có câu hỏi từ Clinical Planner.[/]")
        return {"round1_qa_pairs": [], "qa_history": "(Không có câu hỏi)"}

    qa_pairs = run_user_interview(
        questions=questions,
        anamnesis=anamnesis,
        start_q_num=1,
        batch_num=1,
        total_questions=10,
    )
    qa_history = format_clinical_summary(qa_pairs)
    return {
        "round1_qa_pairs": qa_pairs,
        "qa_history": qa_history,
    }


def user_interview_round2(state: dict) -> dict:
    questions = state.get("round2_questions", [])
    anamnesis = state.get("anamnesis", "")
    round1_qa_pairs = state.get("round1_qa_pairs", [])

    if not questions:
        console.print("[yellow]Không có câu hỏi bổ sung từ Clinical Planner.[/]")
        return {
            "round2_qa_pairs": [],
            "qa_history": state.get("qa_history", "(Không có câu hỏi)"),
        }

    round2_qa_pairs = run_user_interview(
        questions=questions,
        anamnesis=anamnesis,
        start_q_num=len(round1_qa_pairs) + 1,
        batch_num=2,
        total_questions=len(round1_qa_pairs) + len(questions),
    )

    all_qa_pairs = round1_qa_pairs + round2_qa_pairs
    qa_history = format_clinical_summary(all_qa_pairs)
    return {
        "round2_qa_pairs": round2_qa_pairs,
        "qa_history": qa_history,
    }
