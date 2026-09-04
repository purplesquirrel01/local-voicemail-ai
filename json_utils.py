from __future__ import annotations

import json
import re
from typing import Any


class AgentOutputError(ValueError):
    """A focused agent returned output that could not be parsed safely."""


def _strip_markdown_fences(raw: str) -> str:
    text = str(raw or "").strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _first_json_object(raw: str) -> str:
    start = raw.find("{")
    if start < 0:
        raise AgentOutputError("agent output did not contain a JSON object")

    in_string = False
    escape = False
    depth = 0
    for index in range(start, len(raw)):
        char = raw[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    raise AgentOutputError("agent output contained an incomplete JSON object")


def parse_json_strict_or_repair(raw: str, expected_fields: list[str]) -> dict[str, Any]:
    """Parse focused-agent JSON while rejecting unsafe malformed output."""

    text = _strip_markdown_fences(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = json.loads(_first_json_object(text))
        except (json.JSONDecodeError, AgentOutputError) as exc:
            raise AgentOutputError(f"agent output was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise AgentOutputError("agent output must be a JSON object")

    result: dict[str, Any] = {}
    for field in expected_fields:
        value = payload.get(field, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            raise AgentOutputError(f"agent output field {field!r} must be an array")
        result[field] = value
    return result
