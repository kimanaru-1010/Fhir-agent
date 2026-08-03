"""Visual multimodal vision analysis node."""

from rich.console import Console

from utils.json_parser import extract_json
from utils.quality_gate import run_visual_with_quality_gate

console = Console(force_terminal=True)


def visual_extract(state: dict) -> dict:
    raw_output = run_visual_with_quality_gate(state["image_path"])

    parsed = extract_json(raw_output)
    observations_text = ""
    if isinstance(parsed, dict):
        observations = parsed.get("observations", [])
        if observations:
            obs_lines = []
            for obs in observations:
                if isinstance(obs, dict):
                    desc = obs.get("description", "")
                    if desc:
                        obs_lines.append(f"  - {desc}")
                elif isinstance(obs, str):
                    obs_lines.append(f"  - {obs}")
            observations_text = "\n".join(obs_lines) if obs_lines else "Không có quan sát nào."
        else:
            observations_text = "Không có quan sát nào."
    else:
        observations_text = raw_output[:1000]

    # Extract differential disease names
    differentials = []
    if isinstance(parsed, dict):
        diff_list = parsed.get("top_differentials", [])
        for item in diff_list:
            if isinstance(item, dict):
                disease = item.get("disease", "")
            else:
                disease = str(item)
            if disease:
                differentials.append(disease)
    elif isinstance(parsed, list):
        differentials = [str(d).strip() for d in parsed if d]

    console.print()

    return {
        "visual_observations": observations_text,
        "visual_differentials": differentials,
    }
