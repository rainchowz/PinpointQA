import re
from typing import Any, Optional

from .normalize import normalize_spaces, stable_lower


def extract_yes_no(text: str) -> Optional[str]:
    """Extract a yes/no label from free text."""
    s = stable_lower(text)
    m = re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else None


def extract_option_letter(text: str) -> Optional[str]:
    """Extract the first A/B/C/D style answer choice."""
    if not text:
        return None
    s = text.strip()
    patterns = [
        r"^\s*([ABCD])\b",
        r"^\s*([ABCD])\s*[.)]",
        r"\b([ABCD])\s*[.)]",
        r"\boption\s*([ABCD])\b",
        r"\bchoice\s*([ABCD])\b",
    ]
    for p in patterns:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def extract_target_from_question(question: str) -> str:
    """Heuristically extract the target object from the question text."""
    if not question:
        return ""
    patterns = [
        r"did the ([^?]+?) appear",
        r"final location of the ([^?.,]+?) in",
        r"final location of the ([^?.,]+?)\b",
        r"location of the ([^?.,]+?) in",
        r"where .*? the ([^?.,]+?)\b",
        r"describe .*? the ([^?.,]+?)\b",
        r"output .*? the ([^?.,]+?)\b",
    ]
    q = question.lower()
    for p in patterns:
        m = re.search(p, q)
        if m:
            return normalize_spaces(m.group(1))
    return ""


def safe_float(x: Any) -> Optional[float]:
    """Best-effort float conversion."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else None
