"""
convert_test_jsonl_to_gt_dir.py

Convert a flat PinpointQA JSONL file (e.g. test.jsonl) into the
scene-level GT directory format expected by eval.py.

Input:
    One JSONL file with sample-level records.

Output:
    One JSON file per scene:
        output_dir/<scene_id>.json

Each output JSON has the structure:
{
  "dataset_info": {...},
  "qa_pairs": [...]
}
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


TASK_TO_QUESTION_TYPE = {
    "TPV": "presence",
    "NRI": "reference_mcq",
    "FSD": "free_form_localization",
    "SSP": "structured_localization",
    "Target Presence Verification": "presence",
    "Nearest Reference Identification": "reference_mcq",
    "Fine-Grained Spatial Description": "free_form_localization",
    "Structured Spatial Prediction": "structured_localization",
    "presence": "presence",
    "reference_mcq": "reference_mcq",
    "free_form_localization": "free_form_localization",
    "structured_localization": "structured_localization",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return records


def dump_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_question_type(record: Dict[str, Any]) -> str:
    task = str(record.get("task", "")).strip()
    question_type = str(record.get("question_type", "")).strip()

    if task in TASK_TO_QUESTION_TYPE:
        return TASK_TO_QUESTION_TYPE[task]
    if question_type in TASK_TO_QUESTION_TYPE:
        return TASK_TO_QUESTION_TYPE[question_type]

    raise ValueError(
        "Cannot map sample to eval.py question_type. "
        f"id={record.get('id')} task={task!r} question_type={question_type!r}"
    )


def convert_record(record: Dict[str, Any]) -> Dict[str, Any]:
    scene_id = record.get("scene_id")
    if not scene_id:
        raise ValueError(f"Missing scene_id in record: {record.get('id')}")

    if "id" not in record:
        raise ValueError(f"Missing id in record for scene_id={scene_id}")

    converted = {
        "id": record["id"],
        "dataset": record.get("source_dataset", ""),
        "scene_name": scene_id,
        "question_type": normalize_question_type(record),
        "question": record.get("question", ""),
        "instruction": record.get("instruction", ""),
        "ground_truth": record.get("answer"),
    }

    # Keep extra fields if present. eval.py ignores them, but they can help debugging.
    optional_keys = [
        "task",
        "target",
        "choices",
        "split",
        "local_sample_id",
    ]
    for key in optional_keys:
        if key in record:
            converted[key] = record[key]

    return converted


def group_by_scene(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        scene_id = str(record.get("scene_id", "")).strip()
        if not scene_id:
            raise ValueError(f"Missing scene_id in record: {record.get('id')}")
        grouped[scene_id].append(convert_record(record))
    return grouped


def build_scene_payload(scene_id: str, qa_pairs: List[Dict[str, Any]], input_jsonl: Path) -> Dict[str, Any]:
    source_datasets = sorted({str(item.get("dataset", "")).strip() for item in qa_pairs if item.get("dataset")})
    return {
        "dataset_info": {
            "source": "converted_from_jsonl",
            "input_jsonl": str(input_jsonl),
            "scene_name": scene_id,
            "num_pairs": len(qa_pairs),
            "source_datasets": source_datasets,
        },
        "qa_pairs": sorted(qa_pairs, key=lambda x: str(x.get("id", ""))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PinpointQA test.jsonl into the scene-level GT directory format used by eval.py."
    )
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        required=True,
        help="Path to the input JSONL file, e.g. test.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save scene-level GT JSON files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene JSON files in output_dir.",
    )
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    ensure_dir(args.output_dir)

    if args.overwrite:
        for path in args.output_dir.glob("*.json"):
            path.unlink()

    records = read_jsonl(args.input_jsonl)
    grouped = group_by_scene(records)

    for scene_id, qa_pairs in sorted(grouped.items()):
        out_path = args.output_dir / f"{scene_id}.json"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output file already exists: {out_path}. Use --overwrite to replace it."
            )
        payload = build_scene_payload(scene_id, qa_pairs, args.input_jsonl)
        dump_json(payload, out_path)

    print(f"[Done] Converted {len(records)} samples into {len(grouped)} scene files.")
    print(f"[Done] Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
