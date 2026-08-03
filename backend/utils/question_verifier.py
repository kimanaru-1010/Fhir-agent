"""Question Verifier Utility — Verifies and rephrases planner questions into strict Yes/No questions using LLM."""

import json
import unicodedata
from pathlib import Path
from rich.console import Console

from utils.json_parser import extract_json

console = Console(force_terminal=True)

_SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Skill not found: {path}")
    text = path.read_text(encoding="utf-8")
    parts = text.split("---\n", 2)
    if len(parts) >= 3:
        return parts[2].strip()
    return text.strip()


VERIFY_YESNO_PROMPT = _load_skill("verify_yesno")


def _strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics for duplicate detection."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.strip()


def _is_duplicate(new_q: str, previous_questions: list[str], threshold: float = 0.85) -> bool:
    """Check if a new question is a semantic duplicate of any previous question."""
    new_stripped = _strip_accents(new_q)
    for prev in previous_questions:
        prev_stripped = _strip_accents(prev)
        # Exact match after stripping
        if new_stripped == prev_stripped:
            return True
        # Check if one contains the other (longer match is more likely duplicate)
        if len(new_stripped) > 10 and prev_stripped in new_stripped:
            return True
        if len(prev_stripped) > 10 and new_stripped in prev_stripped:
            return True
    return False


def verify_questions_with_llm(
    questions: list[dict],
    image_path: str = None,
    anamnesis: str = "",
    previous_questions: list[dict] = None,
) -> list[dict]:
    """Verify and rephrase questions to ensure strict Yes/No compliance.

    Also detects and replaces questions already answered by the patient's chief complaint,
    and removes semantic duplicates against previously asked questions (Round 2 dedup).

    Args:
        questions: List of question dicts from Clinical Planner.
        image_path: Optional path to the skin image (if needed for vision context).
        anamnesis: Patient's chief complaint text.
        previous_questions: Questions already asked in a prior round (for dedup).

    Returns:
        Verified list of question dicts with corrected Yes/No text.
    """
    if not questions:
        return []

    console.print("[dim]Checking questions for strict Yes/No compliance...[/]")

    # Prepare previous questions string for prompt
    previous_texts = [q.get("question", "") for q in (previous_questions or []) if isinstance(q, dict) and q.get("question")]

    # --- Code-based pre-filter (cheap, model-independent) ---
    # Drop any incoming question that is already an exact/near-duplicate of a
    # prior-round question BEFORE asking the LLM to rephrase/verify it. This
    # doesn't depend on the reasoning model correctly following rule 5 of the
    # verify_yesno skill, which a 27B model won't always do reliably.
    if previous_texts:
        pre_filtered = []
        dropped = 0
        for q in questions:
            q_text = q.get("question", "") if isinstance(q, dict) else ""
            if q_text and _is_duplicate(q_text, previous_texts):
                dropped += 1
                continue
            pre_filtered.append(q)
        if dropped:
            console.print(f"[yellow]  - Dropped {dropped} question(s) duplicate with a previous round (code-based check).[/]")
        questions = pre_filtered
        if not questions:
            return []
    if previous_texts:
        prev_qs_formatted = "\n".join(f"- {txt}" for txt in previous_texts)
    else:
        prev_qs_formatted = "(Không có câu hỏi ở các vòng trước)"

    questions_json = json.dumps(questions, ensure_ascii=False, indent=2)
    prompt = VERIFY_YESNO_PROMPT.format(
        questions_json=questions_json,
        anamnesis=anamnesis.strip() if anamnesis else "(Không có than phiền ban đầu)",
        previous_questions_text=prev_qs_formatted,
    )

    from models.shared_client import call_llm
    from config.settings import (
        REASONING_MODEL_URL,
        REASONING_MODEL_NAME,
        REASONING_MODEL_TIMEOUT,
    )

    try:
        output = call_llm(
            image_path=image_path,
            prompt=prompt,
            base_url=REASONING_MODEL_URL,
            model=REASONING_MODEL_NAME,
            timeout=REASONING_MODEL_TIMEOUT,
        )

        parsed = extract_json(output)
        if isinstance(parsed, dict) and "questions" in parsed:
            verified_questions = parsed["questions"]
        elif isinstance(parsed, list):
            verified_questions = parsed
        else:
            console.print("[yellow]⚠️ Yes/No Verification: JSON parse failed. Retaining original questions.[/]")
            return questions

        result = []
        for i, q in enumerate(verified_questions):
            if i >= len(questions):
                break
            if isinstance(q, dict) and q.get("question"):
                # Preserve original metadata if LLM omitted it
                orig = questions[i] if i < len(questions) else {}
                result.append({
                    "question": q.get("question", orig.get("question", "")),
                    "pqrst_category": q.get("pqrst_category", orig.get("pqrst_category", "")),
                    "purpose": q.get("purpose", orig.get("purpose", "")),
                    "discriminates": q.get("discriminates", orig.get("discriminates", [])),
                })

        # --- Code-based post-filter (safety net) ---
        # The LLM rephrase step can itself introduce a duplicate (e.g. it
        # "fixes" wording and happens to converge back onto a previous
        # question). Drop any such survivors rather than trusting the LLM
        # alone; _fill_to_5() downstream (round-aware) will pad back up to 5.
        if previous_texts and result:
            before = len(result)
            result = [
                q for q in result
                if not (q.get("question") and _is_duplicate(q["question"], previous_texts))
            ]
            if len(result) < before:
                console.print(f"[yellow]  - Removed {before - len(result)} duplicate question(s) after LLM verification (code-based check).[/]")

        if len(result) == len(questions):
            console.print("[green]  ✓ Yes/No verification completed successfully.[/]")
            return result
        elif result:
            console.print(f"[yellow]  ✓ Verified {len(result)}/{len(questions)} questions.[/]")
            return result[:len(questions)]
        else:
            return questions

    except Exception as e:
        console.print(f"[yellow]⚠️ Question verification error: {e}. Falling back to original questions.[/]")
        return questions
