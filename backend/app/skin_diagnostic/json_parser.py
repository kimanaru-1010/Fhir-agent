"""Small JSON extraction helper for model responses."""

from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    if not text:
        return None

    fenced = _FENCE_RE.search(text)
    candidate = fenced.group(1).strip() if fenced else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start_positions = [idx for idx in (candidate.find("{"), candidate.find("[")) if idx >= 0]
    if not start_positions:
        return None
    start = min(start_positions)
    for end in range(len(candidate), start, -1):
        try:
            return json.loads(candidate[start:end])
        except json.JSONDecodeError:
            continue
    return None

