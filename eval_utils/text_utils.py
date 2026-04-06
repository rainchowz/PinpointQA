import json
from typing import Any


def to_text_answer(value: Any) -> str:
    """Convert a possibly nested model output into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for x in value:
            parts.append(to_text_answer(x))
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(value, dict):
        for key in ("text", "answer", "content", "output", "model_output", "response"):
            if key in value:
                return to_text_answer(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)
