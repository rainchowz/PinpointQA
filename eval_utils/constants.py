QUESTION_TYPE_TO_TASK = {
    "presence": "task1",
    "reference_mcq": "task2",
    "free_form_localization": "task3",
    "structured_localization": "task4",
}

TASK_ID_TO_NAME = {
    "task1": "TPV",
    "task2": "NRI",
    "task3": "FSD",
    "task4": "SSP",
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

TASK_ORDER = ["task1", "task2", "task3", "task4"]

TASK4_SOFT_RELATION_PAIRS = {
    frozenset(("next_to", "near")),
    frozenset(("under", "below")),
    frozenset(("on", "attached_to")),
}
