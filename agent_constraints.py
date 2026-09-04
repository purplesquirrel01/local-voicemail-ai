from __future__ import annotations

from typing import Any


NUMBER_ID_FIELDS = ["callback_ids", "fax_ids", "uncertain_ids"]
NAME_ID_FIELDS = ["name_ids", "name_correction_ids"]
DOB_ID_FIELDS = ["dob_ids"]
ERROR_FIELD = "errors"
NAME_STYLE_AGENTS = {
    "name",
    "subject_name",
    "caller_name_fallback",
    "name_correction",
    "spelling_correction",
    "caller_id_correction",
}


def _candidate_ids(payload: dict[str, Any], bucket: str) -> list[str]:
    ids: list[str] = []
    for item in payload.get(bucket, []):
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("id") or item.get("candidate_id") or "").strip()
        if candidate_id:
            ids.append(candidate_id)
    return ids


def _id_array_schema(candidate_ids: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "string"},
    }
    if candidate_ids:
        schema["items"]["enum"] = candidate_ids
    else:
        schema["maxItems"] = 0
    return schema


def _errors_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
    }


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_agent_constraint_schema(agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build a PHI-free compact output schema for the focused agent request."""
    agent = str(agent_name or "").strip()
    if agent == "numbers":
        number_ids = _candidate_ids(payload, "number_candidates")
        return _object_schema(
            {
                "callback_ids": _id_array_schema(number_ids),
                "fax_ids": _id_array_schema(number_ids),
                "uncertain_ids": _id_array_schema(number_ids),
                ERROR_FIELD: _errors_schema(),
            },
            [*NUMBER_ID_FIELDS, ERROR_FIELD],
        )

    if agent in NAME_STYLE_AGENTS:
        name_ids = _candidate_ids(payload, "name_candidates")
        correction_ids = _candidate_ids(payload, "name_correction_candidates")
        return _object_schema(
            {
                "name_ids": _id_array_schema(name_ids),
                "name_correction_ids": _id_array_schema(correction_ids),
                ERROR_FIELD: _errors_schema(),
            },
            [*NAME_ID_FIELDS, ERROR_FIELD],
        )

    if agent == "dob":
        dob_ids = _candidate_ids(payload, "dob_candidates")
        return _object_schema(
            {
                "dob_ids": _id_array_schema(dob_ids),
                ERROR_FIELD: _errors_schema(),
            },
            [*DOB_ID_FIELDS, ERROR_FIELD],
        )

    if agent == "identity":
        name_ids = _candidate_ids(payload, "name_candidates")
        correction_ids = _candidate_ids(payload, "name_correction_candidates")
        dob_ids = _candidate_ids(payload, "dob_candidates")
        return _object_schema(
            {
                "name_ids": _id_array_schema(name_ids),
                "name_correction_ids": _id_array_schema(correction_ids),
                "dob_ids": _id_array_schema(dob_ids),
                ERROR_FIELD: _errors_schema(),
            },
            ["name_ids", "name_correction_ids", "dob_ids", ERROR_FIELD],
        )

    return _object_schema({ERROR_FIELD: _errors_schema()}, [ERROR_FIELD])
