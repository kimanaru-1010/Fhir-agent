"""Runs the diagnostic pipeline for one session, step by step.

This does NOT reuse the LangGraph-style nodes in `pipeline/nodes/*.py` for
the interview steps — it re-implements them procedurally so it can pause
("interrupt") mid-run and poll the session store for the patient's answer,
which is how the HTTP API (request/response, no long-lived connection) has
to work.

Round 1 and Round 2 of the clinical interview follow the exact same shape
(ask the planner for questions -> show them all -> wait for each answer),
so that shape lives once in `_ask_planner()` / `_run_interview_round()` and
both rounds just call it with different inputs.
"""

import asyncio

from app.skin_diagnostic.session_store import get_store


def _format_diff_list(diff_list: list) -> str:
    return "\n".join(f"  {i + 1}. {d}" for i, d in enumerate(diff_list)) if diff_list else "(Không có)"


def _merge_new_differentials(state: dict, new_diffs: list, round_label: str) -> None:
    """Fold planner-suggested `additional_differentials` into state, then re-dedup."""
    from pipeline.nodes.planner import deduplicate_differentials_with_llm
    from pipeline.utils import (
        drop_generic_when_specific_present,
        merge_additional_differentials,
        normalize_disease_name,
    )

    if not new_diffs:
        return

    before_norm = {normalize_disease_name(d) for d in state["visual_differentials"]}
    state["visual_differentials"] = merge_additional_differentials(state["visual_differentials"], new_diffs)
    for item in new_diffs:
        disease = (item.get("disease", "") if isinstance(item, dict) else str(item)).strip()
        if disease and normalize_disease_name(disease) not in before_norm:
            rationale = item.get("rationale", "") if isinstance(item, dict) else ""
            print(f"  + Planner ({round_label}) added differential: {disease} ({rationale[:80]})")
    state["visual_differentials"] = deduplicate_differentials_with_llm(state["visual_differentials"])
    state["visual_differentials"] = drop_generic_when_specific_present(state["visual_differentials"])


async def _run_vision_step(image_path: str) -> tuple[str, list[str]]:
    """Step 1: extract visual observations + candidate differentials from the image."""
    from utils.json_parser import extract_json
    from utils.quality_gate import run_visual_with_quality_gate

    raw = run_visual_with_quality_gate(image_path)
    parsed = extract_json(raw)

    obs_lines = []
    if isinstance(parsed, dict):
        for obs in parsed.get("observations", []):
            if isinstance(obs, dict):
                if obs.get("description"):
                    obs_lines.append(f"  - {obs['description']}")
            elif isinstance(obs, str):
                obs_lines.append(f"  - {obs}")
    observations = "\n".join(obs_lines) if obs_lines else "Không có quan sát"

    differentials = [
        item.get("disease", "") if isinstance(item, dict) else str(item)
        for item in (parsed.get("top_differentials", []) if isinstance(parsed, dict) else [])
        if item
    ]
    return observations, [d for d in differentials if d]


def _augment_with_knowledge_base(state: dict, anamnesis: str) -> None:
    """Step 1b: supplement differentials with embedding/vector-search matches
    from knowledge_base/diseases/*.json, then dedup the combined list."""
    from pipeline.nodes.planner import deduplicate_differentials_with_llm
    from pipeline.utils import drop_generic_when_specific_present
    from utils.knowledge_base import match_kb_candidates

    query_text = "\n".join(part for part in (anamnesis, state["visual_observations"]) if part)
    for match in match_kb_candidates(query_text=query_text):
        state["visual_differentials"].append(match["disease"])

    state["visual_differentials"] = deduplicate_differentials_with_llm(state["visual_differentials"])
    state["visual_differentials"] = drop_generic_when_specific_present(state["visual_differentials"])


def _ask_planner(
    *, image_path: str, complaint: str, state: dict, round_label: str,
    qa_history: str | None = None, previous_questions: list | None = None,
) -> list[dict]:
    """Ask the clinical planner for the next round of questions.

    Also verifies/normalizes questions and updates `state` in place with any
    new differentials or confirmed visual findings the planner surfaced.
    """
    from config.settings import REASONING_MODEL_NAME, REASONING_MODEL_TIMEOUT, REASONING_MODEL_URL
    from models.shared_client import call_llm
    from pipeline.fallback_questions import _fill_to_5
    from pipeline.prompts import _PLANNER_INSTRUCTION
    from pipeline.utils import format_visual_for_planner
    from utils.exam_only_signs import format_exam_only_hints
    from utils.json_parser import extract_json
    from utils.question_verifier import verify_questions_with_llm

    visual = format_visual_for_planner(state["visual_observations"])
    diff_text = _format_diff_list(state["visual_differentials"])
    exam_hints = format_exam_only_hints(state["visual_differentials"])

    prompt = _PLANNER_INSTRUCTION.format(
        complaint=complaint, visual_findings=visual, visual_differentials=diff_text, exam_only_hints=exam_hints,
    )
    if qa_history is not None:
        prompt += f"\n\nPrevious Q&A:\n{qa_history}\n\nGenerate 5 NEW questions different from Round 1."

    output = call_llm(image_path=image_path, prompt=prompt, base_url=REASONING_MODEL_URL, model=REASONING_MODEL_NAME, timeout=REASONING_MODEL_TIMEOUT)
    parsed = extract_json(output)
    if not isinstance(parsed, dict):
        return []

    if round_label == "Round 1":
        verified = parsed.get("verified_findings", "")
        additional = parsed.get("additional_findings", "")
        if verified or additional:
            state["updated_visual_findings"] = "DA XAC NHAN:\n" + verified + ("\n\nBO SUNG:\n" + additional if additional else "")

    _merge_new_differentials(state, parsed.get("additional_differentials", []), round_label)

    questions = [
        {
            "question": q.get("question", ""),
            "pqrst_category": q.get("pqrst_category", q.get("qrst_category", "")),
            "purpose": q.get("purpose", ""),
            "discriminates": q.get("discriminates", []),
        }
        for q in parsed.get("questions", []) if isinstance(q, dict)
    ]
    if questions:
        questions = verify_questions_with_llm(
            questions, image_path=image_path, anamnesis=state["anamnesis"], previous_questions=previous_questions,
        )
    return _fill_to_5(questions, previous_questions=previous_questions)


async def _run_interview_round(store, session_id: str, step_name: str, questions: list[dict], answered_so_far: int) -> list[tuple]:
    """Show all of this round's questions at once, then poll for each answer
    in order. Returns the list of (question, answer) pairs once complete."""
    if not questions:
        return []

    exposed_questions = [
        {
            "question": q.get("question", ""),
            "pqrst_category": q.get("pqrst_category", ""),
            "purpose": q.get("purpose", ""),
            "discriminates": q.get("discriminates", []),
            "question_num": answered_so_far + idx + 1,
            "total": 10,
        }
        for idx, q in enumerate(questions)
    ]
    await store.update(session_id, status="interrupt", current_step=step_name, pending_questions=exposed_questions, pending_answers=[], pending_answer=None)

    qa_pairs = []
    for idx, q_obj in enumerate(questions):
        expected_q_num = answered_so_far + idx + 1
        while True:
            await asyncio.sleep(0.5)
            session = await store.get(session_id)
            if not session:
                continue

            if session.pending_answers:
                match_idx = next((i for i, pa in enumerate(session.pending_answers) if pa.get("question_num") == expected_q_num), None)
                if match_idx is None and session.pending_answers[0].get("question_num") is None:
                    match_idx = 0
                if match_idx is not None:
                    pending = session.pending_answers.pop(match_idx)
                    qa_pairs.append((q_obj, pending["answer"]))
                    break

            if session.pending_answer:
                qa_pairs.append((q_obj, session.pending_answer))
                session.pending_answer = None
                session.pending_question = None
                break

    await store.update(session_id, pending_questions=None, pending_answers=[])
    return qa_pairs


async def run_pipeline_background(session_id: str, image_path: str, anamnesis: str) -> None:
    """Run the full pipeline sequentially, updating the session after each step."""
    store = await get_store()

    try:
        from config.settings import REASONING_MODEL_NAME, REASONING_MODEL_TIMEOUT, REASONING_MODEL_URL
        from models.shared_client import call_llm
        from pipeline.prompts import _DIAGNOSE_INSTRUCTION, _DIAGNOSE_SYSTEM
        from pipeline.utils import drop_generic_when_specific_present, format_visual_for_diagnose, normalize_disease_name
        from tools.decision_tree_agent import format_clinical_summary
        from utils.exam_only_signs import format_exam_only_hints
        from utils.json_parser import extract_json

        state: dict = {
            "image_path": image_path,
            "anamnesis": anamnesis,
            "visual_observations": "",
            "visual_differentials": [],
            "updated_visual_findings": "",
            "round1_questions": [],
            "round2_questions": [],
            "round1_qa_pairs": [],
            "round2_qa_pairs": [],
            "qa_history": "",
            "ranked_diagnoses": [],
            "reasoning": "",
        }
        complaint = anamnesis or "(Không có)"

        # Step 1: Vision + knowledge-base augmented differentials
        await store.update(session_id, status="running", current_step="visual_extract")
        state["visual_observations"], state["visual_differentials"] = await _run_vision_step(image_path)
        _augment_with_knowledge_base(state, anamnesis)

        # Step 2: Round 1 — planner questions, then interview
        await store.update(session_id, status="running", current_step="clinical_planner_round1")
        state["round1_questions"] = _ask_planner(image_path=image_path, complaint=complaint, state=state, round_label="Round 1")

        await store.update(session_id, status="running", current_step="user_interview_round1")
        state["round1_qa_pairs"] = await _run_interview_round(store, session_id, "user_interview_round1", state["round1_questions"], answered_so_far=0)
        state["qa_history"] = format_clinical_summary(state["round1_qa_pairs"]) if state["round1_qa_pairs"] else "(Không có câu hỏi)"

        # Step 3: Round 2 — planner questions (aware of Round 1 answers), then interview
        await store.update(session_id, status="running", current_step="clinical_planner_round2")
        state["round2_questions"] = _ask_planner(
            image_path=image_path, complaint=complaint, state=state, round_label="Round 2",
            qa_history=state["qa_history"], previous_questions=state["round1_questions"],
        )

        await store.update(session_id, status="running", current_step="user_interview_round2")
        state["round2_qa_pairs"] = await _run_interview_round(
            store, session_id, "user_interview_round2", state["round2_questions"], answered_so_far=len(state["round1_qa_pairs"]),
        )
        if state["round2_qa_pairs"]:
            state["qa_history"] = format_clinical_summary(state["round1_qa_pairs"] + state["round2_qa_pairs"])

        # Step 4: Diagnostic reasoning
        await store.update(session_id, status="running", current_step="diagnostic_reasoning")
        diag_visual = format_visual_for_diagnose(state["visual_observations"])
        diag_prompt = _DIAGNOSE_INSTRUCTION.format(
            complaint=complaint,
            image_path=image_path,
            updated_visual_findings=state["updated_visual_findings"] or diag_visual,
            visual_differentials=_format_diff_list(state["visual_differentials"]),
            exam_only_hints=format_exam_only_hints(state["visual_differentials"]),
            qa_history=state["qa_history"],
        )
        diag_output = call_llm(
            image_path=image_path, prompt=diag_prompt, system_prompt=(_DIAGNOSE_SYSTEM or None),
            base_url=REASONING_MODEL_URL, model=REASONING_MODEL_NAME, timeout=REASONING_MODEL_TIMEOUT,
        )
        diag_parsed = extract_json(diag_output)
        if isinstance(diag_parsed, dict):
            state["ranked_diagnoses"] = diag_parsed.get("ranked_diagnoses", [])
            state["reasoning"] = diag_parsed.get("overall_reasoning", "Không có biện luận.")

            # The diagnose instruction allows ranking a disease NOT on the
            # candidate list when there is strong, specific evidence for it.
            # Fold any such disease back into visual_differentials so the
            # displayed candidate list stays consistent with the diagnosis.
            existing_norm = {normalize_disease_name(d) for d in state["visual_differentials"]}
            for diag in state["ranked_diagnoses"]:
                disease = (diag.get("disease", "") if isinstance(diag, dict) else str(diag)).strip()
                norm = normalize_disease_name(disease) if disease else ""
                if norm and norm not in existing_norm:
                    state["visual_differentials"].append(disease)
                    existing_norm.add(norm)
                    print(f"  + Diagnostic reasoning added new differential: {disease}")
            state["visual_differentials"] = drop_generic_when_specific_present(state["visual_differentials"])
        else:
            state["reasoning"] = diag_output[:500]

        await store.update(session_id, status="completed", current_step="diagnostic_reasoning", state=state)

    except Exception as e:
        await store.update(session_id, status="error", error=str(e))
    # The uploaded image is intentionally NOT deleted here — it lives under
    # data/uploads/ and stays available so the conversation can be reloaded
    # later. It's only removed when the session itself is deleted or
    # cleaned up by TTL/capacity (see routers/sessions.py delete_session).


async def resume_pipeline(session_id: str) -> None:
    """Resume a paused pipeline by ensuring `pending_answer` is set from the
    queued `pending_answers`, so the waiting interview loop can pick it up."""
    store = await get_store()
    session = await store.get(session_id)
    if not session or session.pending_answer or not session.pending_questions:
        return
    if not session.pending_answers:
        return

    pending = session.pending_answers.pop(0)
    session.pending_answer = pending["answer"]
    for pending_question in session.pending_questions:
        if pending_question.get("question_num") == pending.get("question_num"):
            session.pending_question = pending_question
            break
