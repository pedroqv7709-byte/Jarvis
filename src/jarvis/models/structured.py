"""Strict parsing of a model's text answer into a validated structured object.

Accepted: exactly one JSON object, optionally wrapped in a single markdown code fence. Anything
else (prose, several objects, trailing text, arrays, schema mismatches) is rejected. Free text is
never interpreted as a plan.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from jarvis.models.base import ModelResponseError

_FENCE = re.compile(r"^```(?:json)?[ \t]*\n(?P<body>.*)\n```$", re.DOTALL)
MAX_RESPONSE_CHARS = 100_000


def parse_structured[T: BaseModel](content: str, schema: type[T]) -> T:
    if len(content) > MAX_RESPONSE_CHARS:
        raise ModelResponseError(f"response exceeds {MAX_RESPONSE_CHARS} characters")
    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group("body").strip()
    if not text.startswith("{"):
        raise ModelResponseError("response is not a JSON object")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelResponseError(f"malformed JSON: {exc.msg} at position {exc.pos}") from exc
    if not isinstance(data, dict):
        raise ModelResponseError("response is not a JSON object")
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise ModelResponseError(
            f"response does not match the expected schema ({exc.error_count()} errors)"
        ) from exc
