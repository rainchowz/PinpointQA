import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .constants import RELATION_SET
from .extract import safe_float
from .normalize import (
    normalize_object_label,
    normalize_relation,
    normalize_surface_label,
)


def normalize_structured_payload(
    data: Dict[str, Any], synonyms: Dict[str, str]
) -> Dict[str, Any]:
    """Canonicalize a structured localization JSON object."""
    refs = data.get("references", []) or []
    if not isinstance(refs, list):
        refs = []

    norm_refs = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        obj = normalize_object_label(ref.get("object", ""), synonyms)
        rel = normalize_relation(ref.get("relation", ""))
        dist = safe_float(ref.get("distance_cm"))
        if not obj and not rel and dist is None:
            continue
        norm_refs.append(
            {
                "object": obj,
                "relation": rel,
                "distance_cm": dist,
            }
        )

    norm_refs = sorted(
        norm_refs,
        key=lambda x: (
            x["object"],
            x["relation"],
            float("inf") if x["distance_cm"] is None else x["distance_cm"],
        ),
    )

    payload = {
        "target": normalize_object_label(data.get("target", ""), synonyms),
        "support_surface": normalize_surface_label(
            data.get("support_surface", data.get("primary_surface", "")), synonyms
        ),
        "references": norm_refs,
    }
    return payload


def parse_json_maybe(text: str) -> Optional[Any]:
    """Parse JSON from raw text, optionally stripping code fences."""
    if not text:
        return None
    raw = text.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def stringify_openai_message_content(content: Any) -> str:
    """Convert OpenAI message content blocks into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, dict):
                        text_value = text_value.get("value")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                        continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(text_attr)
                continue
            if isinstance(text_attr, dict) and isinstance(text_attr.get("value"), str):
                parts.append(text_attr["value"])
                continue
            parts.append(str(item))
        return "\n".join([p for p in parts if p]).strip()
    return str(content)


def normalize_json_like_keys(data: Any) -> Any:
    """Normalize dict keys produced by partially malformed judge outputs."""
    if isinstance(data, dict):
        normalized: Dict[str, Any] = {}
        for key, value in data.items():
            key_str = str(key).strip()
            key_str = key_str.strip('"').strip("'")
            normalized[key_str] = normalize_json_like_keys(value)
        return normalized
    if isinstance(data, list):
        return [normalize_json_like_keys(x) for x in data]
    return data


def coerce_task3_judge_payload(value: Any) -> Optional[Dict[str, Any]]:
    """Best-effort conversion of OpenAI responses into the expected Task 3 payload."""
    if value is None:
        return None
    if isinstance(value, dict):
        normalized = normalize_json_like_keys(value)
        if "total_score" in normalized:
            return normalized
        for nested_key in ("parsed", "output", "response", "result", "content"):
            if nested_key in normalized:
                nested = coerce_task3_judge_payload(normalized[nested_key])
                if nested is not None:
                    return nested
        return normalized
    if isinstance(value, list):
        for item in value:
            nested = coerce_task3_judge_payload(item)
            if nested is not None and "total_score" in nested:
                return nested
        text = stringify_openai_message_content(value)
        parsed = parse_json_maybe(text)
        return coerce_task3_judge_payload(parsed)
    if isinstance(value, str):
        parsed = parse_json_maybe(value)
        if parsed is not None:
            return coerce_task3_judge_payload(parsed)
    return None


def validate_task4_schema(data: Any) -> Tuple[bool, List[str]]:
    """Loosely validate the structured output used by Task 4."""
    errors: List[str] = []
    if not isinstance(data, dict):
        return False, ["not_a_json_object"]

    if "support_surface" not in data and "primary_surface" not in data:
        errors.append("missing_support_surface")
    if "references" not in data:
        errors.append("missing_references")

    refs = data.get("references", [])
    if not isinstance(refs, list):
        errors.append("references_not_list")
        refs = []

    for idx, ref in enumerate(refs):
        if not isinstance(ref, dict):
            errors.append(f"reference_{idx}_not_object")
            continue
        for key in ("object", "relation", "distance_cm"):
            if key not in ref:
                errors.append(f"reference_{idx}_missing_{key}")
        rel = normalize_relation(ref.get("relation", ""))
        if rel and rel not in RELATION_SET:
            errors.append(f"reference_{idx}_invalid_relation")
        if ref.get("distance_cm") is not None and safe_float(ref.get("distance_cm")) is None:
            errors.append(f"reference_{idx}_invalid_distance")

    fatal_prefixes = (
        "not_a_json_object",
        "references_not_list",
    )
    schema_ok = not any(
        err == "missing_support_surface" or err == "missing_references" or err.startswith(fatal_prefixes)
        for err in errors
    )
    return schema_ok, errors
