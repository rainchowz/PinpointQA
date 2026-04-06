"""
Open-source friendly evaluation script for four spatial reasoning tasks:

Task 1: Target Presence Verification (TPV)
Task 2: Nearest Reference Identification (NRI)
Task 3: Fine-Grained Spatial Description (FSD)
Task 4: Structured Spatial Prediction (SSP)

Implementation details live under the ``eval_utils`` package; this file holds the
CLI orchestration (batch loop, argument parsing, and result aggregation).

Example:
    python eval.py \\
      --gt_dir /path/to/gt_dir \\
      --pred_dir /path/to/pred_dir \\
      --output_dir /path/to/output_dir \\
      --eval_prompt /path/to/eval_prompt.txt \\
      --openai_api_key YOUR_OPENAI_API_KEY
"""


import argparse
import os
from pathlib import Path
from typing import Optional

from eval_utils.benchmark import (
    build_task3_target_map,
    collect_batch_pairs,
    load_benchmark_pairs,
)
from eval_utils.constants import TASK3_JUDGE_MODEL, TASK_ORDER
from eval_utils.io_utils import (
    append_jsonl,
    dump_json,
    ensure_dir,
    read_jsonl,
    read_text,
)
from eval_utils.reporting import (
    build_compact_summary,
    print_final_summary,
    write_per_scene_scores,
)
from eval_utils.scoring import score_task1, score_task2, score_task3, score_task4
from eval_utils.synonyms import load_synonyms
from eval_utils.task3_judge import Task3Judge, Task3JudgeConfig


def resolve_openai_api_key(cli_key: Optional[str]) -> Optional[str]:
    """Resolve the OpenAI API key from CLI or environment."""
    if cli_key:
        return cli_key.strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    return env_key or None


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
                "task_id": None,
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
            if item.get("task_id") not in TASK_ORDER:
                continue
            if item["id"] in processed_ok_ids:
                continue

            try:
                if item.get("task_id") == "task1":
                    record = score_task1(item)
                elif item.get("task_id") == "task2":
                    record = score_task2(item)
                elif item.get("task_id") == "task3":
                    record = score_task3(item, task3_judge, target_map)
                elif item.get("task_id") == "task4":
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
                    "task_id": item.get("task_id"),
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
