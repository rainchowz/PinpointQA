from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .constants import QUESTION_TYPE_TO_TASK, TASK_ID_TO_NAME
from .extract import extract_target_from_question
from .io_utils import load_json
from .normalize import normalize_object_label
from .structured import parse_json_maybe
from .text_utils import to_text_answer


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

        task_id = QUESTION_TYPE_TO_TASK.get(qtype, "unknown")
        merged.append(
            {
                "id": item_id,
                "dataset": gt_item.get("dataset", ""),
                "scene_name": gt_item.get("scene_name", ""),
                "question_type": qtype,
                "task_id": task_id,
                "task": TASK_ID_TO_NAME.get(task_id, "unknown"),
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
