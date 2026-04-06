"""Per-task scoring functions."""


from typing import Any, Dict, List

from .alignment import best_gt_slot_alignment, distance_soft_credit, relation_soft_credit
from .constants import TASK_ID_TO_NAME
from .extract import extract_option_letter, extract_target_from_question, extract_yes_no
from .structured import (
    normalize_structured_payload,
    parse_json_maybe,
    validate_task4_schema,
)
from .task3_judge import Task3Judge
from .text_utils import to_text_answer


def score_task1(item: Dict[str, Any]) -> Dict[str, Any]:
    """Score Task 1 with exact yes/no accuracy."""
    gt_text = to_text_answer(item["ground_truth"])
    pred_text = to_text_answer(item["prediction"])

    gt_label = extract_yes_no(gt_text)
    pred_label = extract_yes_no(pred_text)
    valid = pred_label is not None

    return {
        "status": "ok",
        "task_id": "task1",
        "task": TASK_ID_TO_NAME["task1"],
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
        "task_id": "task2",
        "task": TASK_ID_TO_NAME["task2"],
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
        "task_id": "task3",
        "task": TASK_ID_TO_NAME["task3"],
        "id": item["id"],
        "question_type": item["question_type"],
        "target": target,
        "ground_truth_raw": gt_text,
        "prediction_raw": pred_text,
        "judge_total_score": int(judge_result["total_score"]),
        "judge_original_total_score": int(
            judge_result.get("_judge_original_total_score", judge_result["total_score"])
        ),
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
                rel_credit = (
                    relation_soft_credit(g_ref["relation"], p_ref.get("relation", "") if p_ref else "")
                    if object_match
                    else 0.0
                )
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
        "task_id": "task4",
        "task": TASK_ID_TO_NAME["task4"],
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
