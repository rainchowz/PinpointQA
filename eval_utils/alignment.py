from typing import Any, Dict, List, Optional, Tuple

from .constants import TASK4_SOFT_RELATION_PAIRS
from .normalize import normalize_relation


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
