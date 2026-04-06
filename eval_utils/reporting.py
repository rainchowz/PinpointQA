from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .constants import TASK_ID_TO_NAME, TASK_ORDER
from .io_utils import dump_json


def task_id_from_record(record: Dict[str, Any]) -> str:
    """Resolve the internal task ID from either task_id or task name."""
    task_id = record.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    task = record.get("task")
    if isinstance(task, str):
        for key, value in TASK_ID_TO_NAME.items():
            if task == key or task == value:
                return key
    return ""


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def task_record_to_score(record: Dict[str, Any]) -> Optional[float]:
    """Convert one successful item-level record into a unified 0-1 task score."""
    if record.get("status") != "ok":
        return None
    task_id = task_id_from_record(record)
    if task_id in {"task1", "task2"}:
        return float(record.get("correct", 0))
    if task_id == "task3":
        return float(record.get("judge_normalized_score", 0.0))
    if task_id == "task4":
        return float(record.get("soft_score", 0.0))
    return None


def build_compact_summary(
    records: Sequence[Dict[str, Any]],
    expected_scene_names: Sequence[str],
) -> Dict[str, Any]:
    """Build the final compact summary with both micro and macro averages."""
    ok_records = [r for r in records if r.get("status") == "ok"]

    # Micro average（按QA对数量加权 - 原来逻辑）
    task_averages_micro: Dict[str, Optional[float]] = {}
    task_counts: Dict[str, int] = {}
    for task_id in TASK_ORDER:
        xs = [task_record_to_score(r) for r in ok_records if task_id_from_record(r) == task_id]
        xs = [x for x in xs if x is not None]
        task_name = TASK_ID_TO_NAME[task_id]
        task_counts[task_name] = len(xs)
        task_averages_micro[task_name] = round(_mean(xs), 6) if xs else None

    # Macro average（每个scene平等权重 - 原来逻辑）
    task_averages_macro: Dict[str, Optional[float]] = {}
    for task_id in TASK_ORDER:
        scene_means = []
        for scene_name in expected_scene_names:
            values = [
                task_record_to_score(r)
                for r in ok_records
                if r.get("scene_name") == scene_name and task_id_from_record(r) == task_id
            ]
            values = [x for x in values if x is not None]
            if values:
                scene_means.append(_mean(values))
        task_name = TASK_ID_TO_NAME[task_id]
        task_averages_macro[task_name] = round(_mean(scene_means), 6) if scene_means else None

    task4_soft_micro_values = [
        float(r.get("soft_score", 0.0))
        for r in ok_records
        if task_id_from_record(r) == "task4"
    ]
    task4_soft_macro_values = []
    for scene_name in expected_scene_names:
        scene_task4 = [
            r
            for r in ok_records
            if r.get("scene_name") == scene_name and task_id_from_record(r) == "task4"
        ]
        if scene_task4:
            task4_soft_macro_values.append(_mean([float(r.get("soft_score", 0.0)) for r in scene_task4]))

    # per_scene 部分（显式提供 SSP soft score）
    per_scene_map: Dict[str, Dict[str, List[float]]] = {
        scene_name: {task_id: [] for task_id in TASK_ORDER} for scene_name in expected_scene_names
    }
    for record in ok_records:
        scene_name = record.get("scene_name", "")
        task_id = task_id_from_record(record)
        score = task_record_to_score(record)
        if not scene_name or task_id not in TASK_ORDER or score is None:
            continue
        per_scene_map.setdefault(scene_name, {t: [] for t in TASK_ORDER})
        per_scene_map[scene_name][task_id].append(score)

    per_scene: List[Dict[str, Any]] = []
    for scene_name in sorted(per_scene_map):
        row: Dict[str, Any] = {"scene_name": scene_name}
        for task_id in TASK_ORDER:
            values = per_scene_map[scene_name][task_id]
            task_name = TASK_ID_TO_NAME[task_id]
            row[f"{task_name}_score"] = round(_mean(values), 6) if values else None
        row["SSP_soft_score"] = row.get("SSP_score")
        per_scene.append(row)

    return {
        "num_scenes": len(expected_scene_names),
        "num_records_total": len(records),
        "num_records_ok": len(ok_records),
        "task_counts": task_counts,
        "task_averages_micro": task_averages_micro,
        "task_averages_macro": task_averages_macro,
        "SSP_metrics": {
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
    ssp_metrics = summary.get("SSP_metrics", {})

    for task_id in TASK_ORDER:
        task_name = TASK_ID_TO_NAME[task_id]
        value = micro.get(task_name)
        print(f"  {task_name}: {value:.4f}" if value is not None else f"  {task_name}: N/A")
    ssp_micro = ssp_metrics.get("micro", {})
    micro_soft = ssp_micro.get("soft_score")
    print(f"  SSP_soft_score: {micro_soft:.4f}" if micro_soft is not None else "  SSP_soft_score: N/A")

    for task_id in TASK_ORDER:
        task_name = TASK_ID_TO_NAME[task_id]
        value = macro.get(task_name)
        print(f"  {task_name}: {value:.4f}" if value is not None else f"  {task_name}: N/A")
    ssp_macro = ssp_metrics.get("macro", {})
    macro_soft = ssp_macro.get("soft_score")
    print(f"  SSP_soft_score: {macro_soft:.4f}" if macro_soft is not None else "  SSP_soft_score: N/A")

    print("\n" + "=" * 60)


def record_to_per_scene_item(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one item-level result into a compact per-scene score entry."""
    task_id = task_id_from_record(record)
    item = {
        "id": record.get("id"),
        "task": TASK_ID_TO_NAME.get(task_id, record.get("task")),
        "question_type": record.get("question_type"),
        "status": record.get("status"),
    }

    if record.get("status") != "ok":
        item["error"] = record.get("error")
        return item

    if task_id in {"task1", "task2"}:
        item["score"] = float(record.get("correct", 0))
    elif task_id == "task3":
        item["score"] = float(record.get("judge_normalized_score", 0.0))
        item["task3_subscores"] = {
            "total_score_10": int(record.get("judge_total_score", 0)),
            "main_location": int(record.get("judge_dimension_scores", {}).get("main_location", 0)),
            "reference_objects": int(record.get("judge_dimension_scores", {}).get("reference_objects", 0)),
            "spatial_relations": int(record.get("judge_dimension_scores", {}).get("spatial_relations", 0)),
            "distance_cm": int(record.get("judge_dimension_scores", {}).get("distance_cm", 0)),
            "clarity": int(record.get("judge_dimension_scores", {}).get("clarity", 0)),
        }
    elif task_id == "task4":
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
