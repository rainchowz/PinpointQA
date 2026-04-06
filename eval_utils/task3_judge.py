import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[misc, assignment]

from .constants import TASK3_ERROR_TAGS, TASK3_JSON_SCHEMA
from .structured import (
    coerce_task3_judge_payload,
    parse_json_maybe,
    stringify_openai_message_content,
)


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
                        parsed_payload = json.loads(
                            json.dumps(parsed_attr, default=lambda o: getattr(o, "__dict__", str(o)))
                        )

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
