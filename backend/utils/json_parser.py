"""Tiện ích xử lý JSON cho hệ thống."""
import json
import re


def strip_think_block(text: str) -> str:
    """Loại bỏ block <think>...</think> mà các mô hình reasoning hay thêm vào."""
    if not text:
        return text
    # Strip <think>...</think> block (có thể multi-line)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _try_parse_json(s: str) -> dict | list | None:
    """Thử parse JSON string, trả về None nếu thất bại."""
    try:
        return json.loads(s.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def extract_json(text: str) -> dict | list | str:
    """Trích xuất và parse JSON từ phản hồi của mô hình.

    Xử lý các trường hợp:
    - Mô hình trả về <think>...</think> block (reasoning output)
    - JSON bọc trong ```json ... ``` markdown block
    - Nhiều JSON blocks — ưu tiên block có 'differential_diagnosis'
    - JSON thô không có markdown wrapper
    """
    if not text:
        return text

    # Bước 1: Strip <think>...</think> block trước
    cleaned = strip_think_block(text)

    # Bước 2: Thử parse trực tiếp (text đã cleaned)
    result = _try_parse_json(cleaned)
    if result is not None:
        return result

    # Bước 3: Tìm tất cả các ```json ... ``` blocks, ưu tiên block cuối cùng
    # có chứa các key quan trọng (differential_diagnosis, questions, ranked_diagnoses)
    all_blocks = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if all_blocks:
        # Priority keys để tìm: differential_diagnosis (vision), questions (planner), ranked_diagnoses (diagnose)
        priority_keys = ["differential_diagnosis", "questions", "ranked_diagnoses"]
        for key in priority_keys:
            for block in reversed(all_blocks):
                if f'"{key}"' in block:
                    result = _try_parse_json(block)
                    if result is not None:
                        return result
        # Fallback: thử từng block từ cuối lên đầu
        for block in reversed(all_blocks):
            result = _try_parse_json(block)
            if result is not None:
                return result

    # Bước 4: Tìm tất cả các JSON object {...} trong text, ưu tiên block cuối
    # Dùng cách tìm từng cặp ngoặc nhọn đúng cấp
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(cleaned[start:i+1])
                start = None

    # Ưu tiên candidates với priority keys, tìm từ cuối
    priority_keys = ["differential_diagnosis", "questions", "ranked_diagnoses"]
    for key in priority_keys:
        for candidate in reversed(candidates):
            if f'"{key}"' in candidate:
                result = _try_parse_json(candidate)
                if result is not None:
                    return result
    # Fallback: thử từng candidate từ cuối
    for candidate in reversed(candidates):
        result = _try_parse_json(candidate)
        if result is not None:
            return result

    # Không thể parse, trả về chuỗi thô đã bỏ think block
    return cleaned if cleaned else text


def extract_json_with_error(text: str) -> tuple[dict | list | None, str | None]:
    """Parse JSON và trả về kết quả kèm error message (nếu có).

    Dùng cho quality gate và retry feedback — cần biết chính xác lỗi gì.
    Dùng cùng logic với extract_json (strip think block, ưu tiên block cuối).

    Returns:
        (parsed_result, None) nếu thành công.
        (None, error_message) nếu thất bại.
    """
    if not text:
        return None, "Đầu vào trống, không thể parse JSON."

    result = extract_json(text)
    if isinstance(result, (dict, list)):
        return result, None

    return None, "Không thể trích xuất JSON hợp lệ từ phản hồi."


# Các từ khoá xuất hiện trong chain-of-thought/reasoning — KHÔNG phải tên bệnh
_REASONING_KEYWORDS = {
    # Tiếng Anh
    "analyze", "analysis", "image", "description", "location", "color", "colour",
    "shape", "borders", "border", "surface", "size", "quantity", "distribution",
    "special", "sign", "formulate", "hypothesis", "hypotheses", "fit", "mismatch",
    "confidence", "refin", "alternative", "step", "consider", "conclusion",
    "note", "follow", "output", "format", "instruction", "guideline",
    "key_features", "diagnosis", "diagnos", "the image", "the description",
    "and description", "differential", "number", "lesion", "patient",
    # Tiếng Việt
    "đặc điểm", "lý do", "chẩn đoán", "kết luận", "phân tích",
    "bước", "hướng dẫn", "kết quả", "tổng kết", "sự phù hợp",
    "không phù hợp", "độ tin cậy", "giả thuyết", "biện luận",
}


def _is_reasoning_line(text: str) -> bool:
    """Kiểm tra xem một đoạn text có phải là reasoning/meta text (không phải tên bệnh) không."""
    lower = text.lower().strip()

    # Strip prefix số nếu có (ví dụ: '1. Analyze...' -> 'analyze...')
    # Đây là lý do chính dẫn đến false negative: text bắt đầu bằng '1.' nên không match keyword
    stripped = re.sub(r"^\d+[.)\-\s]+", "", lower).strip()

    # Kiểm tra cả original lẫn stripped
    for kw in _REASONING_KEYWORDS:
        if lower.startswith(kw) or lower == kw:
            return True
        if stripped.startswith(kw) or stripped == kw:
            return True

    # Chỉ có số hoặc ký tự đặc biệt
    if re.match(r'^[\d\s\.:,;!?/\\\-]+$', lower):
        return True

    # Từ quá ngắn hoặc quá dài thường không phải tên bệnh
    effective = stripped if stripped else lower
    if len(effective) < 3 or len(effective) > 80:
        return True

    # Chứa quá nhiều từ tiếng anh (câu dài = mô tả không phải tên bệnh)
    words = effective.split()
    if len(words) > 6:
        return True

    return False


def fallback_parse_differential_from_text(text: str) -> dict:
    """Fallback bóc tách danh sách chẩn đoán phân biệt từ text thô nếu JSON parse hoàn toàn thất bại.

    Dùng regex để quét các dòng dạng '1. Tên bệnh', '- Tên bệnh', '**Bệnh:** ...'.
    Lọc bỏ các dòng thuộc chain-of-thought/reasoning (Analyze, Hypothesis, Image...).
    Fallback: nếu không có dòng nào match regex, parse luôn các dòng plain text.
    """
    if not text:
        return {"key_features": "", "differential_diagnosis": []}

    # Strip think block trước
    text = strip_think_block(text)

    # Nếu sau khi strip think block mà có JSON, parse luôn
    parsed = extract_json(text)
    if isinstance(parsed, dict) and "differential_diagnosis" in parsed:
        return parsed

    diagnoses = []
    seen = set()
    lines = text.splitlines()
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        # Pattern matching: 1. Bệnh..., - Bệnh..., * Bệnh...
        match = re.match(
            r"^(?:\d+[\.)\-]|[\-\*\+])\s*(?:\*\*)?(?:Bệnh|Chẩn đoán|Disease)?\s*:?\s*(?:\*\*)?\s*([^:\n\-\(\)]+)",
            line_str, re.IGNORECASE
        )
        if match:
            benh_name = match.group(1).strip("* ").strip()
            if benh_name and not _is_reasoning_line(benh_name) and benh_name not in seen:
                seen.add(benh_name)
                diagnoses.append({
                    "benh": benh_name,
                    "fit_reason": "",
                    "mismatch_reason": "",
                    "confidence": ""
                })

    # Fallback: nếu regex không bắt được gì, parse plain lines
    if not diagnoses:
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            # Bỏ qua dòng quá ngắn (< 2 ký tự) hoặc quá dài (> 60 ký tự = mô tả không phải tên bệnh)
            if len(line_str) < 2 or len(line_str) > 60:
                continue
            # Bỏ qua dòng là reasoning/meta
            if _is_reasoning_line(line_str):
                continue
            # Bỏ qua dòng chứa từ khóa không phải tên bệnh
            lower = line_str.lower()
            if any(kw in lower for kw in ["phân tích", "lý do", "kết luận", "tóm tắt", "note", "note:", "summary", "analyze", "hypothesis"]):
                continue
            if line_str not in seen:
                seen.add(line_str)
                diagnoses.append({
                    "benh": line_str,
                    "fit_reason": "",
                    "mismatch_reason": "",
                    "confidence": ""
                })

    return {
        "key_features": "",
        "differential_diagnosis": diagnoses
    }


def normalize_screening_json(data: dict | list | str) -> dict:
    """Chuẩn hóa dữ liệu screening thành dict chuẩn chứa key 'differential_diagnosis'.

    Xử lý tất cả các trường hợp alias key (chuan_doan_phan_biet, differential_diagnoses, ...),
    trường hợp data là list, hoặc trường hợp data là string/raw text.
    """
    if isinstance(data, str):
        parsed = extract_json(data)
        if isinstance(parsed, (dict, list)):
            data = parsed
        else:
            return fallback_parse_differential_from_text(data)

    if isinstance(data, list):
        data = {"differential_diagnosis": data}

    if not isinstance(data, dict):
        return {"key_features": "", "differential_diagnosis": []}

    # Tìm key differential_diagnosis hoặc các alias
    target_key = "differential_diagnosis"
    aliases = [
        "differential_diagnosis", "differential_diagnoses",
        "chuan_doan_phan_biet", "chẩn_đoán_phân_biệt", "chuan_doan",
        "diagnoses", "differentials", "ds_chuan_doan", "candidates"
    ]

    found_list = None
    if target_key in data and isinstance(data[target_key], list):
        found_list = data[target_key]
    else:
        for alias in aliases:
            if alias in data and isinstance(data[alias], list):
                found_list = data[alias]
                break

    if found_list is None:
        # Kiểm tra xem có field nào chứa list không
        for k, v in data.items():
            if isinstance(v, list) and k != "key_features":
                found_list = v
                break

    if found_list is None:
        found_list = []

    # Chuẩn hóa từng item trong list bệnh
    normalized_items = []
    for item in found_list:
        if isinstance(item, str):
            normalized_items.append({
                "benh": item,
                "fit_reason": "",
                "mismatch_reason": "",
                "confidence": ""
            })
        elif isinstance(item, dict):
            # Tìm tên bệnh từ các alias
            benh_name = item.get("benh") or item.get("tên_bệnh") or item.get("ten_benh") or item.get("disease") or item.get("name") or item.get("diagnosis") or "N/A"
            fit_reason = item.get("fit_reason") or item.get("phu_hop") or item.get("fit") or item.get("ly_do_phu_hop") or ""
            mismatch_reason = item.get("mismatch_reason") or item.get("khong_phu_hop") or item.get("mismatch") or item.get("ly_do_khong_phu_hop") or ""
            confidence = item.get("confidence") or item.get("do_tin_cay") or item.get("confidence_level") or ""
            
            normalized_items.append({
                "benh": str(benh_name),
                "fit_reason": str(fit_reason),
                "mismatch_reason": str(mismatch_reason),
                "confidence": str(confidence)
            })

    return {
        "key_features": data.get("key_features", "") if isinstance(data.get("key_features"), str) else "",
        "differential_diagnosis": normalized_items
    }

