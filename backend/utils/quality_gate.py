"""Quality Gate — retry LLM calls on JSON parse failure."""

from pathlib import Path

from utils.json_parser import extract_json

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


VISUAL_EXTRACT_PROMPT = load_skill("visual_extract")


def run_visual_with_quality_gate(image_path: str, max_retries: int = 2) -> str:
    """Call vision LLM with retry on JSON parse failure.

    Checks that the output contains 'observations' and 'top_differentials' keys.
    """
    from models.shared_client import call_llm
    from config.settings import VISION_MODEL_URL, VISION_MODEL_NAME, VISION_MODEL_TIMEOUT

    prompt = VISUAL_EXTRACT_PROMPT

    for attempt in range(max_retries + 1):
        result = call_llm(
            image_path=image_path,
            prompt=prompt,
            base_url=VISION_MODEL_URL,
            model=VISION_MODEL_NAME,
            timeout=VISION_MODEL_TIMEOUT,
        )

        parsed = extract_json(result)
        if isinstance(parsed, dict) and "observations" in parsed and "top_differentials" in parsed:
            return result

        if attempt < max_retries:
            prompt = (
                f"{VISUAL_EXTRACT_PROMPT}\n\n"
                f"LƯU Ý: Kết quả trước đó không hợp lệ. "
                f"Hãy trả về JSON với key 'observations' và 'top_differentials'."
            )

    return result
