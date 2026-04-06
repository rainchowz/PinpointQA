import re
from typing import Any, Dict

from .constants import DEFAULT_OBJECT_SYNONYMS, DEFAULT_SURFACE_SYNONYMS


def stable_lower(s: Any) -> str:
    """Safely convert a value to lowercase string."""
    if s is None:
        return ""
    return str(s).strip().lower()


def normalize_spaces(s: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", s).strip()


def _basic_normalize_label(label: Any) -> str:
    """Apply shared string cleanup before object/surface canonicalization."""
    s = stable_lower(label)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    return normalize_spaces(s)


def _light_singularize(label: str) -> str:
    """Apply a conservative singularization step for common plural labels."""
    if label.endswith("es") and len(label) > 4:
        return label[:-2]
    if label.endswith("s") and len(label) > 3:
        return label[:-1]
    return label


def normalize_object_label(label: Any, synonyms: Dict[str, str]) -> str:
    """Normalize object labels with a conservative synonym map."""
    s = _basic_normalize_label(label)
    merged = dict(DEFAULT_OBJECT_SYNONYMS)
    merged.update(synonyms)
    s = merged.get(s, s)
    singular = _light_singularize(s)
    s = merged.get(singular, singular)
    return s


def normalize_surface_label(label: Any, synonyms: Dict[str, str]) -> str:
    """Normalize support-surface labels with a surface-specific synonym map."""
    s = _basic_normalize_label(label)
    merged = dict(DEFAULT_SURFACE_SYNONYMS)
    merged.update(synonyms)
    s = merged.get(s, s)
    singular = _light_singularize(s)
    s = merged.get(singular, singular)
    return s


def normalize_label(label: Any, synonyms: Dict[str, str]) -> str:
    """Backward-compatible wrapper for generic label normalization."""
    return normalize_object_label(label, synonyms)


def normalize_relation(rel: Any) -> str:
    """Normalize relation tokens with conservative alias handling."""
    s = stable_lower(rel).replace(" ", "_").replace("-", "_")
    mapping = {
        "nextto": "next_to",
        "sideabove": "side_above",
        "sidebelow": "side_below",
        "attached": "attached_to",
        "close_to": "near",
        "closeby": "near",
        "nearby": "near",
        "near_by": "near",
        "by": "near",
        "beside": "next_to",
        "adjacent": "next_to",
        "adjacent_to": "next_to",
        "alongside": "next_to",
        "on_top_of": "on",
        "ontopof": "on",
        "underneath": "under",
    }
    return mapping.get(s, s)
