"""
eval.py

Open-source friendly evaluation script for four spatial reasoning tasks:

Task 1: Object Presence Verification
Task 2: Reference-based Localization (multiple choice)
Task 3: Open-ended Precise Localization (LLM-as-a-judge)
Task 4: Structured Spatial Output (programmatic scoring)

Example:
    python eval.py \
      --gt_dir /path/to/gt_dir \
      --pred_dir /path/to/pred_dir \
      --output_dir /path/to/output_dir \
      --eval_prompt /path/to/eval_prompt.txt \
      --openai_api_key YOUR_OPENAI_API_KEY    
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:  
    OpenAI = None  


QUESTION_TYPE_TO_TASK = {
    "presence": "task1",
    "reference_mcq": "task2",
    "free_form_localization": "task3",
    "structured_localization": "task4",
}

RELATION_SET = {
    "on",
    "under",
    "above",
    "below",
    "side_above",
    "side_below",
    "next_to",
    "near",
    "attached_to",
}

TASK3_ERROR_TAGS = {
    "wrong_main_location",
    "wrong_support_surface",
    "missing_key_reference",
    "wrong_spatial_relation",
    "missing_numeric_distance",
    "wrong_numeric_distance",
    "unclear_expression",
    "hallucinated_reference",
}

DEFAULT_OBJECT_SYNONYMS = {
    "couch": "sofa",
    "night stand": "nightstand",
    "night-stand": "nightstand",
    "trash can": "trashcan",
    "trash bin": "trashcan",
    "garbage bin": "trashcan",
    "garbage can": "trashcan",
    "rubbish bin": "trashcan",
    "cell phone": "phone",
    "cellphone": "phone",
    "mobile phone": "phone",
    "mobile": "phone",
    "smartphone": "phone",
    "drawer unit": "drawer",
    "book shelf": "bookshelf",
    "cupboard": "cabinet",
    "display": "monitor",
}

DEFAULT_SURFACE_SYNONYMS = {
    "night stand": "nightstand",
    "night-stand": "nightstand",
    "bed side table": "nightstand",
    "bedside table": "nightstand",
    "counter top": "countertop",
    "counter-top": "countertop",
    "table top": "table",
    "table-top": "table",
    "sofa seat": "sofa",
}

TASK3_JSON_SCHEMA = {
    "name": "task3_judge_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "total_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "dimension_scores": {
                "type": "object",
                "properties": {
                    "main_location": {"type": "integer", "minimum": 0, "maximum": 3},
                    "reference_objects": {"type": "integer", "minimum": 0, "maximum": 2},
                    "spatial_relations": {"type": "integer", "minimum": 0, "maximum": 2},
                    "distance_cm": {"type": "integer", "minimum": 0, "maximum": 2},
                    "clarity": {"type": "integer", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "main_location",
                    "reference_objects",
                    "spatial_relations",
                    "distance_cm",
                    "clarity",
                ],
                "additionalProperties": False,
            },
            "error_tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted(TASK3_ERROR_TAGS),
                },
            },
            "explanation": {"type": "string"},
        },
        "required": ["total_score", "dimension_scores", "error_tags", "explanation"],
        "additionalProperties": False,
    },
}


TASK3_JUDGE_MODEL = "gpt-5.4"



def load_json(path: Path) -> Any:
    """Load a JSON file with UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    """Load a UTF-8 text file."""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def dump_json(obj: Any, path: Path) -> None:
    """Write JSON with readable formatting."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(record: Dict[str, Any], path: Path) -> None:
    """Append one JSON record to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file if it exists."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def ensure_dir(path: Path) -> None:
    """Create a directory recursively if needed."""
    path.mkdir(parents=True, exist_ok=True)


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


def load_synonyms(path: Optional[Path]) -> Dict[str, str]:
    """Load an optional user synonym map and return custom aliases only."""
    synonyms: Dict[str, str] = {}
    if path is None:
        return synonyms
    user_map = load_json(path)
    if not isinstance(user_map, dict):
        raise ValueError(f"Synonym file must be a JSON object: {path}")
    for k, v in user_map.items():
        synonyms[_basic_normalize_label(k)] = _basic_normalize_label(v)
    return synonyms


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


def task4_distance_tolerances(gt_dist: Optional[float]) -> Tuple[float, float]:
    """Return range-aware (full, half) tolerances in cm for Task 4 distance scoring."""
    if gt_dist is None:
        return 0.0, 0.0
    gd = abs(float(gt_dist))
    if gd <= 10.0:
        return 5.0, 10.0
    if gd <= 30.0:
        return 10.0, 15.0
    return 15.0, 20.0



def best_gt_slot_alignment(
    gt_refs: List[Dict[str, Any]],
    pred_refs: List[Dict[str, Any]],
    full_credit_tol: float,
    half_credit_tol: float,
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Align predicted references to GT slots."""
    if not gt_refs:
        return []

    best_alignment: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = [
        (gt_ref, None) for gt_ref in gt_refs
    ]
    best_score: Tuple[float, float, float] = (-1.0, -1.0, -1.0)

    def _assignment_score(
        alignment: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]
    ) -> Tuple[float, float, float]:
        object_credit = 0.0
        relation_credit = 0.0
        distance_credit = 0.0
        for gt_ref, pred_ref in alignment:
            if pred_ref is None:
                continue
            object_match = gt_ref["object"] == pred_ref["object"]
            if not object_match:
                continue
            object_credit += 1.0
            rel_credit = relation_soft_credit(gt_ref["relation"], pred_ref["relation"])
            relation_credit += rel_credit
            distance_credit += rel_credit * distance_soft_credit(
                gt_ref["distance_cm"],
                pred_ref["distance_cm"],
                full_credit_tol=full_credit_tol,
                half_credit_tol=half_credit_tol,
            )
        return (object_credit, relation_credit, distance_credit)

    def _dfs(
        slot_idx: int,
        used_pred_indices: set,
        current: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]],
    ) -> None:
        nonlocal best_alignment, best_score
        if slot_idx == len(gt_refs):
            score = _assignment_score(current)
            if score > best_score:
                best_score = score
                best_alignment = list(current)
            return

        gt_ref = gt_refs[slot_idx]

        current.append((gt_ref, None))
        _dfs(slot_idx + 1, used_pred_indices, current)
        current.pop()

        for pred_idx, pred_ref in enumerate(pred_refs):
            if pred_idx in used_pred_indices:
                continue
            used_pred_indices.add(pred_idx)
            current.append((gt_ref, pred_ref))
            _dfs(slot_idx + 1, used_pred_indices, current)
            current.pop()
            used_pred_indices.remove(pred_idx)

    _dfs(0, set(), [])
    return best_alignment


def distance_soft_credit(
    gt_dist: Optional[float],
    pred_dist: Optional[float],
    full_credit_tol: float,
    half_credit_tol: float,
) -> float:
    if gt_dist is None or pred_dist is None:
        return 0.0
    dynamic_full_credit_tol, dynamic_half_credit_tol = task4_distance_tolerances(float(gt_dist))
    err = abs(float(gt_dist) - float(pred_dist))
    if err <= dynamic_full_credit_tol:
        return 1.0
    if err <= dynamic_half_credit_tol:
        return 0.5
    return 0.0


TASK4_SOFT_RELATION_PAIRS = {
    frozenset(("next_to", "near")),
    frozenset(("under", "below")),
    frozenset(("on", "attached_to")),
}


def relation_soft_credit(gt_rel: Any, pred_rel: Any) -> float:
    """Assign 0/0.5/1 credit for Task 4 relation matching."""
    g = normalize_relation(gt_rel)
    p = normalize_relation(pred_rel)
    if not g or not p:
        return 0.0
    if g == p:
        return 1.0
    if frozenset((g, p)) in TASK4_SOFT_RELATION_PAIRS:
        return 0.5
    return 0.0


@dataclass
class Task3JudgeConfig:
    """Settings for the Task 3 OpenAI judge."""
    prompt_template: str
    model: str
    temperature: float
    max_retries: int
    retry_sleep: float


class Task3Judge:
    """Thin wrapper around the OpenAI API for Task 3 scoring."""

    def __init__(self, config: Task3JudgeConfig):
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. Install it with: pip install openai"
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set, but Task 3 scoring requires it."
            )
        self.client = OpenAI()
        self.config = config

    def _build_prompt(self, target: str, ground_truth: str, model_output: str) -> str:
        """Fill the prompt template without letting JSON braces break formatting."""
        prompt = self.config.prompt_template
        prompt = prompt.replace("{target}", target)
        prompt = prompt.replace("{ground_truth}", ground_truth)
        prompt = prompt.replace("{model_output}", model_output)
        return prompt

    def score(self, target: str, ground_truth: str, model_output: str) -> Dict[str, Any]:
        """Call the OpenAI judge and return parsed JSON."""
        prompt = self._build_prompt(target, ground_truth, model_output)
        last_error = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": TASK3_JSON_SCHEMA,
                    },
                )

                message = response.choices[0].message
                content = stringify_openai_message_content(getattr(message, "content", None))
                parsed_payload = None

                parsed_attr = getattr(message, "parsed", None)
                if parsed_attr is not None:
                    if hasattr(parsed_attr, "model_dump"):
                        parsed_payload = parsed_attr.model_dump()
                    elif isinstance(parsed_attr, dict):
                        parsed_payload = parsed_attr
                    else:
                        parsed_payload = json.loads(json.dumps(parsed_attr, default=lambda o: getattr(o, "__dict__", str(o))))

                if parsed_payload is None:
                    parsed_payload = parse_json_maybe(content)

                parsed = coerce_task3_judge_payload(parsed_payload)
                if not isinstance(parsed, dict) or "total_score" not in parsed:
                    raw_preview = content[:500] if content else "<empty>"
                    raise ValueError(
                        "Task 3 judge did not return the expected JSON payload. "
                        f"raw_preview={raw_preview!r} parsed_type={type(parsed_payload).__name__}"
                    )

                dims = parsed.get("dimension_scores", {})
                total_score = int(parsed["total_score"])
                if not isinstance(dims, dict):
                    raise ValueError(f"Task 3 dimension_scores is not a dict: {dims!r}")

                original_total_score = total_score
                corrected_total = sum(
                    [
                        int(dims.get("main_location", 0)),
                        int(dims.get("reference_objects", 0)),
                        int(dims.get("spatial_relations", 0)),
                        int(dims.get("distance_cm", 0)),
                        int(dims.get("clarity", 0)),
                    ]
                )
                total_was_corrected = total_score != corrected_total
                if total_was_corrected:
                    parsed["total_score"] = corrected_total
                    total_score = corrected_total

                if any(tag not in TASK3_ERROR_TAGS for tag in parsed.get("error_tags", [])):
                    raise ValueError(f"Task 3 judge returned unsupported error tags: {parsed!r}")

                parsed["_judge_original_total_score"] = int(original_total_score)
                parsed["_judge_total_corrected"] = total_was_corrected
                parsed["_raw_response"] = content
                parsed["_attempt"] = attempt
                return parsed
            except Exception as e:  # pragma: no cover
                last_error = str(e)
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_sleep * attempt)

        raise RuntimeError(f"Task 3 judge failed after retries: {last_error}")

def load_benchmark_pairs(gt_path: Path, pred_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load GT and prediction files, then align items by ID."""
    gt_data = load_json(gt_path)
    pred_data = load_json(pred_path)

    gt_pairs = gt_data.get("qa_pairs", [])
    pred_pairs = pred_data.get("qa_pairs", [])
    if not isinstance(gt_pairs, list) or not isinstance(pred_pairs, list):
        raise ValueError("Both files must contain a list field named 'qa_pairs'.")

    gt_map = {item["id"]: item for item in gt_pairs}
    pred_map = {item["id"]: item for item in pred_pairs}

    common_ids = sorted(set(gt_map) & set(pred_map))
    if not common_ids:
        raise ValueError("No overlapping item IDs found between GT and prediction files.")

    merged = []
    for item_id in common_ids:
        gt_item = gt_map[item_id]
        pred_item = pred_map[item_id]
        qtype = gt_item.get("question_type")
        if qtype != pred_item.get("question_type"):
            raise ValueError(
                f"Question type mismatch for id={item_id}: "
                f"{qtype} vs {pred_item.get('question_type')}"
            )

        merged.append(
            {
                "id": item_id,
                "dataset": gt_item.get("dataset", ""),
                "scene_name": gt_item.get("scene_name", ""),
                "question_type": qtype,
                "task": QUESTION_TYPE_TO_TASK.get(qtype, "unknown"),
                "question": gt_item.get("question", ""),
                "instruction": gt_item.get("instruction", ""),
                "ground_truth": gt_item.get("ground_truth"),
                "prediction": pred_item.get("model_outputs"),
            }
        )

    metadata = {
        "gt_dataset_info": gt_data.get("dataset_info", {}),
        "pred_dataset_info": pred_data.get("dataset_info", {}),
        "num_gt_pairs": len(gt_pairs),
        "num_pred_pairs": len(pred_pairs),
        "num_common_pairs": len(merged),
    }
    return merged, metadata


def build_task3_target_map(items: Sequence[Dict[str, Any]], synonyms: Dict[str, str]) -> Dict[str, str]:
    """
    Build a best-effort map from item ID to target label.

    Priority:
    1. Parse sibling structured GT JSON for the same scene and target group.
    2. Extract from the current question text.
    """
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_scene.setdefault(item["scene_name"], []).append(item)

    target_map: Dict[str, str] = {}
    for scene_items in by_scene.values():
        structured_targets = {}
        for item in scene_items:
            if item["question_type"] == "structured_localization":
                gt_json = parse_json_maybe(to_text_answer(item["ground_truth"]))
                if isinstance(gt_json, dict):
                    target = normalize_object_label(gt_json.get("target", ""), synonyms)
                    if target:
                        structured_targets[item["id"]] = target

        # First pass: direct from question.
        for item in scene_items:
            q_target = normalize_object_label(extract_target_from_question(item["question"]), synonyms)
            if q_target:
                target_map[item["id"]] = q_target

        # Second pass: borrow from nearby structured items in the same scene.
        ordered = sorted(scene_items, key=lambda x: x["id"])
        for i, item in enumerate(ordered):
            if target_map.get(item["id"]):
                continue
            candidates = []
            for j in range(max(0, i - 2), min(len(ordered), i + 3)):
                other = ordered[j]
                if other["question_type"] != "structured_localization":
                    continue
                gt_json = parse_json_maybe(to_text_answer(other["ground_truth"]))
                if isinstance(gt_json, dict):
                    target = normalize_object_label(gt_json.get("target", ""), synonyms)
                    if target:
                        candidates.append(target)
            if candidates:
                target_map[item["id"]] = candidates[0]

    return target_map

def score_task1(item: Dict[str, Any]) -> Dict[str, Any]:
    """Score Task 1 with exact yes/no accuracy."""
    gt_text = to_text_answer(item["ground_truth"])
    pred_text = to_text_answer(item["prediction"])

    gt_label = extract_yes_no(gt_text)
    pred_label = extract_yes_no(pred_text)
    valid = pred_label is not None

    return {
        "status": "ok",
        "task": "task1",
        "id": item["id"],
        "question_type": item["question_type"],
        "ground_truth_raw": gt_text,
        "prediction_raw": pred_text,
        "ground_truth_label": gt_label,
        "prediction_label": pred_label,
        "valid_prediction": valid,
        "correct": int(valid and gt_label == pred_label),
    }


def score_task2(item: Dict[str, Any]) -> Dict[str, Any]:
    """Score Task 2 with exact option-letter accuracy."""
    gt_text = to_text_answer(item["ground_truth"])
    pred_text = to_text_answer(item["prediction"])

    gt_letter = extract_option_letter(gt_text)
    pred_letter = extract_option_letter(pred_text)
    valid = pred_letter is not None

    return {
        "status": "ok",
        "task": "task2",
        "id": item["id"],
        "question_type": item["question_type"],
        "ground_truth_raw": gt_text,
        "prediction_raw": pred_text,
        "ground_truth_letter": gt_letter,
        "prediction_letter": pred_letter,
        "valid_prediction": valid,
        "correct": int(valid and gt_letter == pred_letter),
    }


def score_task3(
    item: Dict[str, Any],
    judge: Task3Judge,
    target_map: Dict[str, str],
) -> Dict[str, Any]:
    """Score Task 3 using the configured LLM judge."""
    gt_text = to_text_answer(item["ground_truth"])
    pred_text = to_text_answer(item["prediction"])
    target = target_map.get(item["id"], "") or extract_target_from_question(item["question"])

    judge_result = judge.score(
        target=target,
        ground_truth=gt_text,
        model_output=pred_text,
    )

    return {
        "status": "ok",
        "task": "task3",
        "id": item["id"],
        "question_type": item["question_type"],
        "target": target,
        "ground_truth_raw": gt_text,
        "prediction_raw": pred_text,
        "judge_total_score": int(judge_result["total_score"]),
        "judge_original_total_score": int(judge_result.get("_judge_original_total_score", judge_result["total_score"])),
        "judge_total_corrected": bool(judge_result.get("_judge_total_corrected", False)),
        "judge_normalized_score": float(judge_result["total_score"]) / 10.0,
        "judge_dimension_scores": judge_result["dimension_scores"],
        "judge_error_tags": judge_result["error_tags"],
        "judge_explanation": judge_result["explanation"],
        "judge_attempt": judge_result.get("_attempt"),
        "judge_raw_response": judge_result.get("_raw_response"),
    }


def score_task4(
    item: Dict[str, Any],
    synonyms: Dict[str, str],
    full_credit_tol: float,
    half_credit_tol: float,
) -> Dict[str, Any]:
    """Score Task 4 with relaxed schema handling and slot-level soft scoring."""
    gt_text = to_text_answer(item["ground_truth"])
    pred_text = to_text_answer(item["prediction"])

    gt_json = parse_json_maybe(gt_text)
    if not isinstance(gt_json, dict):
        raise ValueError(f"Task 4 GT is not valid JSON for id={item['id']}")

    pred_json = parse_json_maybe(pred_text)
    parse_ok = isinstance(pred_json, dict)
    schema_ok = False
    schema_errors: List[str] = []

    gt_payload = normalize_structured_payload(gt_json, synonyms)
    pred_payload = None

    if parse_ok:
        schema_ok, schema_errors = validate_task4_schema(pred_json)
        pred_payload = normalize_structured_payload(pred_json, synonyms)

    surface_score = 0.0
    object_score = 0.0
    relation_score = 0.0
    distance_score = 0.0
    soft_score = 0.0

    if parse_ok and pred_payload is not None:
        surface_score = 1.0 if gt_payload["support_surface"] == pred_payload["support_surface"] else 0.0

        slot_alignment = best_gt_slot_alignment(
            gt_payload["references"],
            pred_payload["references"],
            full_credit_tol=full_credit_tol,
            half_credit_tol=half_credit_tol,
        )
        if gt_payload["references"]:
            object_scores = []
            relation_scores = []
            distance_scores = []
            for g_ref, p_ref in slot_alignment:
                object_match = p_ref is not None and g_ref["object"] == p_ref["object"]
                obj_credit = 1.0 if object_match else 0.0
                rel_credit = relation_soft_credit(g_ref["relation"], p_ref.get("relation", "") if p_ref else "") if object_match else 0.0
                dist_credit = 0.0
                if object_match and rel_credit > 0.0:
                    dist_credit = rel_credit * distance_soft_credit(
                        g_ref["distance_cm"],
                        p_ref["distance_cm"],
                        full_credit_tol=full_credit_tol,
                        half_credit_tol=half_credit_tol,
                    )
                object_scores.append(obj_credit)
                relation_scores.append(rel_credit)
                distance_scores.append(dist_credit)

            denom = len(gt_payload["references"])
            object_score = sum(object_scores) / denom
            relation_score = sum(relation_scores) / denom
            distance_score = sum(distance_scores) / denom

        soft_score = (
            0.4 * surface_score
            + 0.2 * object_score
            + 0.2 * relation_score
            + 0.2 * distance_score
        )

    return {
        "status": "ok",
        "task": "task4",
        "id": item["id"],
        "question_type": item["question_type"],
        "ground_truth_raw": gt_text,
        "prediction_raw": pred_text,
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "schema_errors": schema_errors,
        "surface_score": round(surface_score, 6),
        "object_score": round(object_score, 6),
        "relation_score": round(relation_score, 6),
        "distance_score": round(distance_score, 6),
        "soft_score": round(soft_score, 6),
        "gt_normalized": gt_payload,
        "pred_normalized": pred_payload,
    }



TASK_ORDER = ["task1", "task2", "task3", "task4"]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def task_record_to_score(record: Dict[str, Any]) -> Optional[float]:
    """Convert one successful item-level record into a unified 0-1 task score."""
    if record.get("status") != "ok":
        return None
    task = record.get("task")
    if task in {"task1", "task2"}:
        return float(record.get("correct", 0))
    if task == "task3":
        return float(record.get("judge_normalized_score", 0.0))
    if task == "task4":
        return float(record.get("soft_score", 0.0))
    return None


def collect_batch_pairs(gt_dir: Path, pred_dir: Path) -> List[Dict[str, Any]]:
    """Match prediction files with GT files by scene name."""
    pairs: List[Dict[str, Any]] = []
    for pred_path in sorted(pred_dir.glob("*.json")):
        scene_name = pred_path.stem
        flat_gt = gt_dir / f"{scene_name}.json"
        nested_gt = gt_dir / scene_name / f"{scene_name}.json"
        gt_path = flat_gt if flat_gt.exists() else nested_gt
        pairs.append(
            {
                "scene_name": scene_name,
                "pred": pred_path,
                "gt": gt_path,
                "gt_exists": gt_path.exists(),
            }
        )
    return pairs


def build_compact_summary(
    records: Sequence[Dict[str, Any]],
    expected_scene_names: Sequence[str],
) -> Dict[str, Any]:
    """Build the final compact summary with both micro and macro averages."""
    ok_records = [r for r in records if r.get("status") == "ok"]

    # Micro average（按QA对数量加权 - 原来逻辑）
    task_averages_micro: Dict[str, Optional[float]] = {}
    task_counts: Dict[str, int] = {}
    for task in TASK_ORDER:
        xs = [task_record_to_score(r) for r in ok_records if r.get("task") == task]
        xs = [x for x in xs if x is not None]
        task_counts[task] = len(xs)
        task_averages_micro[task] = round(_mean(xs), 6) if xs else None

    # Macro average（每个scene平等权重 - 新增，更适合论文/榜单）
    task_averages_macro: Dict[str, Optional[float]] = {}
    for task in TASK_ORDER:
        scene_means = []
        for scene_name in expected_scene_names:
            values = [task_record_to_score(r) for r in ok_records
                      if r.get("scene_name") == scene_name and r.get("task") == task]
            values = [x for x in values if x is not None]
            if values:
                scene_means.append(_mean(values))
        task_averages_macro[task] = round(_mean(scene_means), 6) if scene_means else None

    task4_soft_micro_values = [float(r.get("soft_score", 0.0)) for r in ok_records if r.get("task") == "task4"]
    task4_soft_macro_values = []
    for scene_name in expected_scene_names:
        scene_task4 = [r for r in ok_records if r.get("scene_name") == scene_name and r.get("task") == "task4"]
        if scene_task4:
            task4_soft_macro_values.append(_mean([float(r.get("soft_score", 0.0)) for r in scene_task4]))

    # per_scene 部分（显式提供 Task 4 soft score）
    per_scene_map: Dict[str, Dict[str, List[float]]] = {
        scene_name: {task: [] for task in TASK_ORDER} for scene_name in expected_scene_names
    }
    for record in ok_records:
        scene_name = record.get("scene_name", "")
        task = record.get("task")
        score = task_record_to_score(record)
        if not scene_name or task not in TASK_ORDER or score is None:
            continue
        per_scene_map.setdefault(scene_name, {t: [] for t in TASK_ORDER})
        per_scene_map[scene_name][task].append(score)

    per_scene: List[Dict[str, Any]] = []
    for scene_name in sorted(per_scene_map):
        row: Dict[str, Any] = {"scene_name": scene_name}
        for task in TASK_ORDER:
            values = per_scene_map[scene_name][task]
            row[f"{task}_score"] = round(_mean(values), 6) if values else None
        row["task4_soft_score"] = row.get("task4_score")
        per_scene.append(row)

    return {
        "num_scenes": len(expected_scene_names),
        "num_records_total": len(records),
        "num_records_ok": len(ok_records),
        "task_counts": task_counts,
        "task_averages_micro": task_averages_micro,
        "task_averages_macro": task_averages_macro,
        "task4_metrics": {
            "micro": {
                "soft_score": round(_mean(task4_soft_micro_values), 6) if task4_soft_micro_values else None,
            },
            "macro": {
                "soft_score": round(_mean(task4_soft_macro_values), 6) if task4_soft_macro_values else None,
            },
        },
        "per_scene": per_scene,
    }


def print_final_summary(summary: Dict[str, Any]) -> None:
    """Print both micro and macro averages for better understanding."""
    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    micro = summary.get("task_averages_micro", {})
    macro = summary.get("task_averages_macro", {})
    task4_metrics = summary.get("task4_metrics", {})

    for task in TASK_ORDER:
        value = micro.get(task)
        print(f"  {task}: {value:.4f}" if value is not None else f"  {task}: N/A")
    task4_micro = task4_metrics.get("micro", {})
    micro_soft = task4_micro.get("soft_score")
    print(f"  task4_soft_score: {micro_soft:.4f}" if micro_soft is not None else "  task4_soft_score: N/A")

    for task in TASK_ORDER:
        value = macro.get(task)
        print(f"  {task}: {value:.4f}" if value is not None else f"  {task}: N/A")
    task4_macro = task4_metrics.get("macro", {})
    macro_soft = task4_macro.get("soft_score")
    print(f"  task4_soft_score: {macro_soft:.4f}" if macro_soft is not None else "  task4_soft_score: N/A")

    print("\n" + "=" * 60)


def record_to_per_scene_item(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one item-level result into a compact per-scene score entry."""
    item = {
        "id": record.get("id"),
        "task": record.get("task"),
        "question_type": record.get("question_type"),
        "status": record.get("status"),
    }

    if record.get("status") != "ok":
        item["error"] = record.get("error")
        return item

    task = record.get("task")
    if task in {"task1", "task2"}:
        item["score"] = float(record.get("correct", 0))
    elif task == "task3":
        item["score"] = float(record.get("judge_normalized_score", 0.0))
        item["task3_subscores"] = {
            "total_score_10": int(record.get("judge_total_score", 0)),
            "main_location": int(record.get("judge_dimension_scores", {}).get("main_location", 0)),
            "reference_objects": int(record.get("judge_dimension_scores", {}).get("reference_objects", 0)),
            "spatial_relations": int(record.get("judge_dimension_scores", {}).get("spatial_relations", 0)),
            "distance_cm": int(record.get("judge_dimension_scores", {}).get("distance_cm", 0)),
            "clarity": int(record.get("judge_dimension_scores", {}).get("clarity", 0)),
        }
    elif task == "task4":
        item["score"] = float(record.get("soft_score", 0.0))
        item["task4_subscores"] = {
            "surface_score": float(record.get("surface_score", 0.0)),
            "object_score": float(record.get("object_score", 0.0)),
            "relation_score": float(record.get("relation_score", 0.0)),
            "distance_score": float(record.get("distance_score", 0.0)),
            "parse_ok": bool(record.get("parse_ok", False)),
            "schema_ok": bool(record.get("schema_ok", False)),
        }

    return item


def write_per_scene_scores(records: Sequence[Dict[str, Any]], scene_name: str, output_dir: Path) -> None:
    """Write one JSON file per scene with all item-level scores."""
    scene_records = [r for r in records if r.get("scene_name") == scene_name]
    payload = {
        "scene_name": scene_name,
        "items": [record_to_per_scene_item(r) for r in scene_records],
    }
    dump_json(payload, output_dir / f"{scene_name}.json")


def run_batch_evaluation(args: argparse.Namespace) -> None:
    """Evaluate all scenes from a GT directory and a prediction directory."""
    ensure_dir(args.output_dir)
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    per_scene_scores_dir = args.output_dir / "per_scene_scores"
    ensure_dir(per_scene_scores_dir)

    if args.overwrite:
        for path in (results_path, summary_path):
            if path.exists():
                path.unlink()
        for path in per_scene_scores_dir.glob("*.json"):
            path.unlink()

    resolved_key = resolve_openai_api_key(args.openai_api_key)
    if resolved_key:
        os.environ["OPENAI_API_KEY"] = resolved_key

    synonyms = load_synonyms(args.synonyms_json)
    all_records = [] if args.overwrite else read_jsonl(results_path)
    processed_ok_ids = {
        rec["id"]
        for rec in all_records
        if rec.get("status") == "ok" and isinstance(rec.get("id"), str)
    }

    pairs = collect_batch_pairs(args.gt_dir, args.pred_dir)
    expected_scene_names = [pair["scene_name"] for pair in pairs]

    eval_prompt_template = read_text(args.eval_prompt)

    task3_judge = Task3Judge(
        Task3JudgeConfig(
            prompt_template=eval_prompt_template,
            model=TASK3_JUDGE_MODEL,
            temperature=args.judge_temperature,
            max_retries=args.judge_max_retries,
            retry_sleep=args.judge_retry_sleep,
        )
    )

    for scene_index, pair in enumerate(pairs, start=1):
        scene_name = pair["scene_name"]
        print(f"[Scene {scene_index}/{len(pairs)}] {scene_name}")

        if not pair["gt_exists"]:
            record = {
                "status": "error",
                "scene_name": scene_name,
                "task": None,
                "id": f"{scene_name}::__missing_gt__",
                "error": f"GT not found: {pair['gt']}",
            }
            append_jsonl(record, results_path)
            all_records.append(record)
            write_per_scene_scores(all_records, scene_name, per_scene_scores_dir)
            summary = build_compact_summary(all_records, expected_scene_names)
            dump_json(summary, summary_path)
            continue

        items, _ = load_benchmark_pairs(pair["gt"], pair["pred"])
        if args.max_items is not None:
            items = items[: args.max_items]

        target_map = build_task3_target_map(items, synonyms)

        for item in items:
            if item["task"] not in TASK_ORDER:
                continue
            if item["id"] in processed_ok_ids:
                continue

            try:
                if item["task"] == "task1":
                    record = score_task1(item)
                elif item["task"] == "task2":
                    record = score_task2(item)
                elif item["task"] == "task3":
                    record = score_task3(item, task3_judge, target_map)
                elif item["task"] == "task4":
                    record = score_task4(
                        item=item,
                        synonyms=synonyms,
                        full_credit_tol=args.task4_full_credit_tol,
                        half_credit_tol=args.task4_half_credit_tol,
                    )
                else:
                    continue
            except Exception as e:
                record = {
                    "status": "error",
                    "scene_name": item.get("scene_name") or scene_name,
                    "task": item.get("task"),
                    "id": item.get("id"),
                    "question_type": item.get("question_type"),
                    "error": str(e),
                }

            record["scene_name"] = item.get("scene_name") or scene_name
            append_jsonl(record, results_path)
            all_records.append(record)
            if record.get("status") == "ok" and isinstance(record.get("id"), str):
                processed_ok_ids.add(record["id"])

        write_per_scene_scores(all_records, scene_name, per_scene_scores_dir)
        summary = build_compact_summary(all_records, expected_scene_names)
        dump_json(summary, summary_path)

    for scene_name in expected_scene_names:
        write_per_scene_scores(all_records, scene_name, per_scene_scores_dir)

    final_summary = build_compact_summary(all_records, expected_scene_names)
    dump_json(final_summary, summary_path)

    print(f"[Done] Results saved to: {results_path}")
    print(f"[Done] Summary saved to: {summary_path}")
    print(f"[Done] Per-scene scores saved to: {per_scene_scores_dir}")
    print_final_summary(final_summary)
    

def resolve_openai_api_key(cli_key: Optional[str]) -> Optional[str]:
    """Resolve the OpenAI API key from CLI or environment."""
    if cli_key:
        return cli_key.strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    return env_key or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for the four spatial reasoning tasks."
    )
    parser.add_argument("--gt_dir", type=Path, required=True, help="Directory containing GT JSON files.")
    parser.add_argument("--pred_dir", type=Path, required=True, help="Directory containing prediction JSON files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for results.jsonl and summary.json.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing results.jsonl, summary.json, and per_scene_scores before running.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Optional cap per scene for debugging.",
    )
    parser.add_argument(
        "--synonyms_json",
        type=Path,
        default=None,
        help="Optional JSON file mapping label synonyms to canonical forms.",
    )
    parser.add_argument(
        "--eval_prompt",
        type=Path,
        required=True,
        help="Path to the eval prompt txt file.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="Optional OpenAI API key. Overrides environment variable.",
    )
    parser.add_argument(
        "--judge_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the Task 3 judge.",
    )
    parser.add_argument(
        "--judge_max_retries",
        type=int,
        default=5,
        help="Maximum retry attempts for Task 3 judge calls.",
    )
    parser.add_argument(
        "--judge_retry_sleep",
        type=float,
        default=2.0,
        help="Base sleep in seconds between Task 3 judge retries.",
    )
    parser.add_argument(
        "--task4_full_credit_tol",
        type=float,
        default=5.0,
        help="Deprecated in range-aware Task 4 distance mode; kept only for backward compatibility.",
    )
    parser.add_argument(
        "--task4_half_credit_tol",
        type=float,
        default=10.0,
        help="Deprecated in range-aware Task 4 distance mode; kept only for backward compatibility.",
    )

    args = parser.parse_args()
    run_batch_evaluation(args)


if __name__ == "__main__":
    main()