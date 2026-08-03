"""Load skill prompts from the skills directory."""

from pathlib import Path

_SKILLS_DIR = Path(__file__).parent.parent / "skills"


def load_skill(name: str) -> str:
    """Load a skill prompt from the skills directory."""
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Skill not found: {path}")
    text = path.read_text(encoding="utf-8")
    parts = text.split("---\n", 2)
    if len(parts) >= 3:
        return parts[2].strip()
    return text.strip()


_PLANNER_INSTRUCTION = load_skill("planner_instruction")
_DIAGNOSE_SYSTEM = load_skill("diagnose_system")
_DIAGNOSE_INSTRUCTION = load_skill("diagnose_instruction")
_DEDUPLICATE_DIFFERENTIALS = load_skill("deduplicate_differentials")
