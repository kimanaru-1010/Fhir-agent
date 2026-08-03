"""Utility functions for clinical Q&A — PQRST framework helpers.

Kept as a shared module for the interview_agent and main.py to reuse.
Generation logic (question LLM calls, decision tree agent) has been removed.
"""

import re
from typing import List, Tuple

from rich.console import Console

console = Console(force_terminal=True)


# PQRST category labels for display
PQRST_LABELS = {
    "P": "Provocation / Yếu tố khởi phát",
    "Q": "Quality / Tính chất cảm giác",
    "R": "Radiation / Lan rộng & Vị trí",
    "S": "Severity / Mức độ & Diễn tiến",
    "T": "Timing / Thời gian & Chu kỳ",
}

PQRST_COLORS = {
    "P": "magenta",
    "Q": "cyan",
    "R": "blue",
    "S": "red",
    "T": "yellow",
}


def clean_and_tokenize(text: str) -> set[str]:
    """Chuẩn hóa văn bản tiếng Việt và tách từ, loại bỏ từ dừng."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()

    STOP_WORDS = {
        "bạn", "có", "thấy", "vùng", "bị", "tổn", "thương", "không", "ở", "xuất", "hiện", "trên",
        "nhưng", "những", "các", "tình", "trạng", "này", "của", "đối", "với", "cho", "cách", "như", "thế", "nào",
        "vẻ", "hơn", "nhiều", "ít", "gì", "nào", "đó", "này", "hay", "hoặc", "và", "là", "đã", "đang", "sẽ", "được",
        "tại", "nơi", "vào", "ra", "đi", "lại", "đến", "bị", "mới", "rồi", "nhé", "ạ", "dạ"
    }

    return {w for w in words if w not in STOP_WORDS}


def is_valid_yes_no_question(q_text: str) -> bool:
    """Kiểm tra câu hỏi có đúng chuẩn YES/NO không, loại bỏ các câu hỏi mở."""
    if not q_text or not q_text.strip():
        return False
    q_lower = q_text.lower().strip()

    forbidden_words = [
        "bao lâu", "bao nhiêu", "khi nào", "thế nào", "ra sao",
        "ở đâu", "tại sao", "như thế nào", "mấy tháng", "mấy ngày",
        "mấy tuần", "bằng cách nào", "là gì"
    ]
    for word in forbidden_words:
        if word in q_lower:
            return False

    return True


def format_qa_text(qa_pairs: List[Tuple[dict, str]]) -> str:
    """Format Q&A pairs into a text block for LLM prompts."""
    if not qa_pairs:
        return "(Chưa có câu trả lời nào)"
    lines = []
    for i, (q_obj, a) in enumerate(qa_pairs, 1):
        question = q_obj.get("question", "")
        purpose = q_obj.get("purpose", "")
        pqrst = q_obj.get("pqrst_category", "")
        answer = a
        pqrst_label = f"[{pqrst}] " if pqrst else ""
        lines.append(f"Câu {i}: {pqrst_label}{question}")
        lines.append(f"Trả lời: {answer}")
        lines.append("")
    return "\n".join(lines)


def format_clinical_summary(
    qa_pairs: List[Tuple[dict, str]],
    original_candidates: List[str] | None = None,
    early_stop: bool = False,
) -> str:
    """Format Q&A into a clinical summary text for the diagnose prompt."""
    lines = []
    if early_stop:
        lines.append(f"Số câu hỏi đã trả lời: {len(qa_pairs)} (dừng sớm — đã đủ thông tin phân biệt)")
    else:
        lines.append(f"Số câu hỏi đã trả lời: {len(qa_pairs)}/10")
    lines.append("")

    # PQRST coverage summary
    covered_pqrst = {q_obj.get("pqrst_category", q_obj.get("qrst_category", "")) for q_obj, _ in qa_pairs if q_obj.get("pqrst_category") or q_obj.get("qrst_category")}
    if covered_pqrst:
        lines.append("")

    if not qa_pairs:
        lines.append("(Bệnh nhân đã bỏ qua toàn bộ câu hỏi)")
    else:
        for i, (q_obj, a) in enumerate(qa_pairs, 1):
            question = q_obj.get("question", "")
            discriminates = q_obj.get("discriminates", [])
            purpose = q_obj.get("purpose", "")
            pqrst = q_obj.get("pqrst_category", q_obj.get("qrst_category", ""))
            answer = a
            pqrst_label = PQRST_LABELS.get(pqrst, pqrst)
            lines.append(f"Câu hỏi {i} [{pqrst} — {pqrst_label}]: {question}")
            lines.append(f"Trả lời: {answer}")
            lines.append("")

    if original_candidates:
        lines.append(f"DANH SÁCH BỆNH BAN ĐẦU (từ KB RAG): {', '.join(original_candidates)}")
        lines.append("")

    return "\n".join(lines)
