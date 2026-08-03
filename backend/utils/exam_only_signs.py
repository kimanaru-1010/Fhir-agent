"""Generic exam-only sign detector for the knowledge base.

Problem this solves: `visual_extract.md` correctly forbids the vision model
from inferring things that can't be seen in a photo (sensation, palpation
findings, systemic symptoms — see its "THREE CATEGORIES OF CLINICAL SIGNS").
But `knowledge_base/diseases.json` freely mixes visual and exam-only phrases
inside the same `symptom_keywords` list for every disease (e.g. Bệnh phong
thể BT's defining feature — mất cảm giác, dây thần kinh to — is exam-only).
When a disease's *distinguishing* feature is exam-only, nothing upstream
ever tells the Clinical Planner to specifically ask about it, so it rarely
makes it into interview questions and the disease under-ranks.

This module is deliberately NOT specific to any one disease. It applies a
generic substring heuristic to whatever `symptom_keywords` phrases exist for
the CURRENT case's candidate differentials (any subset of the 103 KB
entries, or future entries added later) and surfaces only the exam-only
phrases for diseases that are actually in play right now. Fixing/extending
this for a new disease means editing knowledge_base/diseases.json content
(or the trigger list below if a genuinely new *kind* of exam-only cue shows
up) — never editing skills/*.md prompts to name that disease.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

_KB_FILE = Path(__file__).parent.parent / "knowledge_base" / "diseases.json"

# Substrings that mark a symptom_keywords phrase as obtainable ONLY through
# provoked physical examination, a named clinical sign, sensory testing, or
# systemic findings — never through visual inspection of a static photo.
# Deliberately narrow: knowledge_base.py's own docstring says KB keywords are
# "intentionally restricted to visual and tactile observations" so the KB is
# designed to lean tactile-but-photographable (e.g. "sờ thấy vảy dày" mostly
# describes texture that's also visible as scale in the image) — a bare "sờ"
# trigger would flag ~90% of the KB and drown out the real signal. Also
# excludes generic itch/pain/burning wording: those are already covered by
# the standard PQRST "Q" question every round asks regardless of disease.
_EXAM_ONLY_TRIGGERS = [
    "mất cảm giác", "giảm cảm giác", "kém nhạy cảm",
    "dấu hiệu auspitz", "dấu hiệu nikolsky", "dấu hiệu dimple",
    "dây thần kinh",
    "sờ hạch", "nổi hạch", "hạch to",
    "sốt", "toàn thân", "mệt mỏi", "khó thở", "đau khớp",
    "dễ chảy máu", "dễ bầm tím", "dễ rách",
]

_TRIGGER_RE = re.compile("|".join(re.escape(t) for t in _EXAM_ONLY_TRIGGERS), re.IGNORECASE)


def is_exam_only_phrase(phrase: str) -> bool:
    """True if `phrase` describes something confirmable only via touch,
    provocation, a named clinical sign, or systemic findings — not visible
    in a photo. Generic substring heuristic; not tied to any one disease."""
    if not phrase:
        return False
    return bool(_TRIGGER_RE.search(phrase.lower()))


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.strip()


def _load_kb() -> list[dict]:
    if not _KB_FILE.exists():
        return []
    try:
        data = json.loads(_KB_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def get_exam_only_signs_for(disease_names: list[str]) -> dict[str, list[str]]:
    """For each name in `disease_names` that matches a knowledge_base/diseases.json
    entry, return the subset of its symptom_keywords classified as exam-only.
    Diseases with no KB entry, or whose keywords are all visually observable,
    are simply absent from the result — this scales to any candidate list
    from any case, not a fixed disease.
    """
    if not disease_names:
        return {}
    wanted = {_normalize(d) for d in disease_names}
    result: dict[str, list[str]] = {}
    for entry in _load_kb():
        name = entry.get("disease", "")
        if not name or _normalize(name) not in wanted:
            continue
        exam_only = [kw for kw in entry.get("symptom_keywords", []) if is_exam_only_phrase(kw)]
        if exam_only:
            result[name] = exam_only
    return result


def format_exam_only_hints(disease_names: list[str]) -> str:
    """Human-readable block to inject into the planner / diagnose prompts —
    see {exam_only_hints} in skills/planner_instruction.md and
    skills/diagnose_instruction.md."""
    hints = get_exam_only_signs_for(disease_names)
    if not hints:
        return "(Không có đặc điểm nào chỉ xác định được qua hỏi bệnh/khám, theo cơ sở tri thức, cho các bệnh đang xét)"
    lines = []
    for disease, signs in hints.items():
        lines.append(f"- {disease}: {'; '.join(signs)}")
    return "\n".join(lines)
