"""Fallback questions to guarantee exactly 5 per round with strict PQRST coverage.

Each PQRST category has TWO variants so that when a round falls back to
these defaults, Round 2 does not have to reuse the exact same question
text that Round 1 may already have used as a fallback. `_fill_to_5` picks
whichever variant does not collide with `previous_questions`.
"""

import unicodedata

FALLBACK_QUESTIONS_BY_CATEGORY = {
    "P": [
        {
            "pqrst_category": "P",
            "question": "Tình trạng này có ngứa hoặc đau nhiều hơn khi chạm vào hoặc rửa không?",
            "purpose": "Xác định yếu tố khởi phát và mức độ kích ứng",
            "discriminates": [],
        },
        {
            "pqrst_category": "P",
            "question": "Tổn thương có trở nên rõ hơn sau khi tiếp xúc với ánh nắng không?",
            "purpose": "Xác định yếu tố khởi phát liên quan ánh sáng",
            "discriminates": [],
        },
    ],
    "Q": [
        {
            "pqrst_category": "Q",
            "question": "Bạn có cảm thấy ngứa hoặc rát tại tổn thương không?",
            "purpose": "Đánh giá tính chất cảm giác của tổn thương",
            "discriminates": [],
        },
        {
            "pqrst_category": "Q",
            "question": "Bạn có cảm thấy vùng da tổn thương kém nhạy cảm hơn (tê, giảm cảm giác nóng/lạnh/đau) so với vùng da xung quanh không?",
            "purpose": "Đánh giá mất cảm giác tại chỗ — dấu hiệu định hướng bệnh phong",
            "discriminates": [],
        },
    ],
    "R": [
        {
            "pqrst_category": "R",
            "question": "Tổn thương có lan rộng ra vùng da xung quanh không?",
            "purpose": "Đánh giá hướng lan của tổn thương",
            "discriminates": [],
        },
        {
            "pqrst_category": "R",
            "question": "Tổn thương có xuất hiện ở nhiều vị trí khác nhau trên cơ thể không?",
            "purpose": "Đánh giá phân bố tổn thương (khu trú hay lan tỏa)",
            "discriminates": [],
        },
    ],
    "S": [
        {
            "pqrst_category": "S",
            "question": "Tổn thương có gây khó chịu nhiều, ảnh hưởng đến sinh hoạt hàng ngày không?",
            "purpose": "Đánh giá mức độ nghiêm trọng của tổn thương",
            "discriminates": [],
        },
        {
            "pqrst_category": "S",
            "question": "Tổn thương có ảnh hưởng đến giấc ngủ của bạn không?",
            "purpose": "Đánh giá mức độ nghiêm trọng qua ảnh hưởng giấc ngủ",
            "discriminates": [],
        },
    ],
    "T": [
        {
            "pqrst_category": "T",
            "question": "Tổn thương này có xuất hiện lần đầu tiên không?",
            "purpose": "Xác định thời điểm khởi phát lần đầu",
            "discriminates": [],
        },
        {
            "pqrst_category": "T",
            "question": "Tổn thương có xuất hiện rồi tự biến mất và tái phát nhiều lần không?",
            "purpose": "Xác định kiểu diễn tiến (tái phát hay liên tục)",
            "discriminates": [],
        },
    ],
}

# Flat list kept for backward compatibility (e.g. anything iterating "all fallbacks").
FALLBACK_QUESTIONS = [variant for variants in FALLBACK_QUESTIONS_BY_CATEGORY.values() for variant in variants]

PQRST_ORDER = ["P", "Q", "R", "S", "T"]


def _strip_accents(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.strip()


def _collides(question_text: str, exclude_texts: set[str]) -> bool:
    """True if question_text matches (exact or containment, accent-insensitive)
    anything in exclude_texts — used to keep fallback questions from repeating
    text that was already asked (or already used as fallback) in a prior round."""
    stripped = _strip_accents(question_text)
    if not stripped:
        return False
    for other in exclude_texts:
        other_stripped = _strip_accents(other)
        if not other_stripped:
            continue
        if stripped == other_stripped:
            return True
        if len(stripped) > 10 and other_stripped in stripped:
            return True
        if len(other_stripped) > 10 and stripped in other_stripped:
            return True
    return False


def _build_exclude_texts(previous_questions: list[dict] | None) -> set[str]:
    if not previous_questions:
        return set()
    return {q.get("question", "") for q in previous_questions if isinstance(q, dict) and q.get("question")}


def _pick_fallback_for_category(cat: str, used_texts: set[str], exclude_texts: set[str]):
    """Pick a fallback variant for `cat` that avoids both same-round dupes
    (`used_texts`) and prior-round dupes (`exclude_texts`). Falls back to the
    first variant not already used in this round if every variant collides
    with a previous round (better a same-topic repeat within one round than
    silently returning fewer than 5 questions)."""
    variants = FALLBACK_QUESTIONS_BY_CATEGORY.get(cat, [])
    for fb in variants:
        if fb["question"] not in used_texts and not _collides(fb["question"], exclude_texts):
            return fb
    for fb in variants:
        if fb["question"] not in used_texts:
            return fb
    return None


def _rebalance_categories(questions: list[dict], previous_questions: list[dict] | None = None) -> list[dict]:
    """Ensure exactly one question per PQRST category (no duplicates, no gaps).

    Runs regardless of question count: an LLM can return exactly 5 questions
    that already violate PQRST coverage (e.g. two "P" questions and zero "T").
    Any category appearing 2+ times (or an unknown/missing category) has its
    extra occurrences swapped for a fallback question from whichever
    category is missing — chosen to avoid colliding with `previous_questions`.
    """
    exclude_texts = _build_exclude_texts(previous_questions)
    used_texts = {q.get("question", "") for q in questions}

    seen_categories = set()
    duplicate_or_invalid_idx = []
    for i, q in enumerate(questions):
        cat = q.get("pqrst_category")
        if cat not in PQRST_ORDER or cat in seen_categories:
            duplicate_or_invalid_idx.append(i)
        else:
            seen_categories.add(cat)

    missing_categories = [c for c in PQRST_ORDER if c not in seen_categories]

    for idx, cat in zip(duplicate_or_invalid_idx, missing_categories):
        old_text = questions[idx].get("question", "")
        fb = _pick_fallback_for_category(cat, used_texts, exclude_texts)
        if fb:
            used_texts.discard(old_text)
            questions[idx] = fb
            used_texts.add(fb["question"])

    return questions


def _fill_to_5(questions: list[dict], previous_questions: list[dict] | None = None) -> list[dict]:
    """Pad questions to exactly 5 with strict PQRST coverage.

    `previous_questions` (e.g. Round 1's final questions, when filling Round 2)
    is used to steer fallback selection away from text already asked or
    already used as a fallback in a prior round, so a JSON-parse failure or
    an over-aggressive dedup pass in one round can't reintroduce the exact
    same fallback question that was already shown to the patient.

    Strategy:
    1. Identify which PQRST categories are missing from LLM questions
    2. Add missing category fallbacks first (ensures 1P+1Q+1R+1S+1T minimum)
    3. If still fewer than 5, fill with remaining fallbacks
    4. Cap at exactly 5
    5. Rebalance: even if already 5, swap any duplicate-category question
       for a missing-category fallback so coverage is always P+Q+R+S+T
    """
    if len(questions) >= 5:
        return _rebalance_categories(questions[:5], previous_questions)

    exclude_texts = _build_exclude_texts(previous_questions)

    # Track which categories are already covered
    covered_categories = {q.get("pqrst_category") for q in questions if q.get("pqrst_category")}
    asked_texts = {q.get("question", "") for q in questions}

    result = list(questions)
    used = set(asked_texts)

    # Step 1: Add missing categories first
    for cat in PQRST_ORDER:
        if cat not in covered_categories and len(result) < 5:
            fb = _pick_fallback_for_category(cat, used, exclude_texts)
            if fb:
                result.append(fb)
                used.add(fb["question"])

    # Step 2: Fill remaining slots from any fallback (any category, still avoiding collisions)
    if len(result) < 5:
        for cat in PQRST_ORDER:
            if len(result) >= 5:
                break
            fb = _pick_fallback_for_category(cat, used, exclude_texts)
            if fb:
                result.append(fb)
                used.add(fb["question"])

    return _rebalance_categories(result[:5], previous_questions)
