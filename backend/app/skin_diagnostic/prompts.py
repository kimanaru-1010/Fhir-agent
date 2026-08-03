"""Compatibility exports for the upstream skin diagnostic skill prompts."""

from __future__ import annotations

from pathlib import Path

from pipeline.prompts import (
    _DIAGNOSE_INSTRUCTION,
    _DIAGNOSE_SYSTEM,
    _PLANNER_INSTRUCTION,
)
from utils.quality_gate import VISUAL_EXTRACT_PROMPT


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "skills"

VISUAL_PROMPT = VISUAL_EXTRACT_PROMPT
PLANNER_PROMPT = _PLANNER_INSTRUCTION
DIAGNOSE_SYSTEM = _DIAGNOSE_SYSTEM
DIAGNOSE_PROMPT = _DIAGNOSE_INSTRUCTION
