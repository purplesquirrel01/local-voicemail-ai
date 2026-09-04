from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from typing import Any


FINAL_SCHEMA_KEYS = [
    "patient_names",
    "name_correction_candidates",
    "dob_candidates",
    "callback_numbers",
    "fax_numbers",
    "uncertain_numbers",
    "possible_errors",
]

NUMBER_FIELDS = {"callback_numbers", "fax_numbers", "uncertain_numbers", "possible_errors"}
IDENTITY_FIELDS = {"patient_names", "name_correction_candidates", "dob_candidates", "possible_errors"}
NUMBER_ID_FIELDS = {
    "callback_ids": "callback_numbers",
    "fax_ids": "fax_numbers",
    "uncertain_ids": "uncertain_numbers",
}
IDENTITY_ID_FIELDS = {
    "name_ids": "patient_names",
    "name_correction_ids": "name_correction_candidates",
    "dob_ids": "dob_candidates",
}
NAME_CORRECTION_REASONS = {
    "phonetic_last_first_match",
    "last_name_phonetic_match",
    "weak_phonetic_match",
}


@dataclass(frozen=True)
class NormalizedPhone:
    raw: str
    normalized: str | None
    formatted: str | None
    valid: bool


def _digits_from_text(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def normalize_phone_candidate(value: Any) -> NormalizedPhone:
    raw = str(value or "")
    digits = _digits_from_text(raw)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return NormalizedPhone(raw=raw, normalized=None, formatted=None, valid=False)
    return NormalizedPhone(
        raw=raw,
        normalized=digits,
        formatted=f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",
        valid=True,
    )


def _expand_year(year: int) -> int:
    if year < 100:
        return 1900 + year if year >= 30 else 2000 + year
    return year


def parse_dob(value: Any) -> date | None:
    text = str(value or "").strip()
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$", text)
    if not match:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    year = _expand_year(int(match.group(3)))
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    today = date.today()
    age = today.year - parsed.year - ((today.month, today.day) < (parsed.month, parsed.day))
    if parsed > today or age > 120:
        return None
    return parsed


def format_dob(value: date) -> str:
    return f"{value.month:02d}/{value.day:02d}/{value.year:04d}"


def empty_final_json() -> dict[str, list[Any]]:
    return {key: [] for key in FINAL_SCHEMA_KEYS}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _candidate_id_items(value: Any) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for item in _as_list(value):
        if isinstance(item, str):
            candidate_id = item.strip()
        elif isinstance(item, dict):
            candidate_id = str(item.get("candidate_id") or item.get("id") or "").strip()
        else:
            candidate_id = ""
        if candidate_id:
            items.append({"candidate_id": candidate_id})
    return items


def merge_agent_outputs(numbers_output: dict[str, Any], identity_output: dict[str, Any]) -> dict[str, Any]:
    merged = empty_final_json()
    for key in ("callback_numbers", "fax_numbers", "uncertain_numbers"):
        merged[key] = [item for item in _as_list(numbers_output.get(key))]
    for id_key, final_key in NUMBER_ID_FIELDS.items():
        merged[final_key].extend(_candidate_id_items(numbers_output.get(id_key)))
    for key in ("patient_names", "name_correction_candidates", "dob_candidates"):
        merged[key] = [item for item in _as_list(identity_output.get(key))]
    for id_key, final_key in IDENTITY_ID_FIELDS.items():
        merged[final_key].extend(_candidate_id_items(identity_output.get(id_key)))
    merged["possible_errors"] = [
        *[item for item in _as_list(numbers_output.get("possible_errors"))],
        *[item for item in _as_list(numbers_output.get("errors"))],
        *[item for item in _as_list(identity_output.get("possible_errors"))],
        *[item for item in _as_list(identity_output.get("errors"))],
    ]
    return merged


def merge_split_agent_outputs(
    numbers_output: dict[str, Any],
    name_output: dict[str, Any],
    dob_output: dict[str, Any],
    name_correction_output: dict[str, Any] | None = None,
    caller_id_correction_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    correction_outputs = [output for output in (name_correction_output, caller_id_correction_output) if output]
    merged = empty_final_json()
    for key in ("callback_numbers", "fax_numbers", "uncertain_numbers"):
        merged[key] = [item for item in _as_list(numbers_output.get(key))]
    for id_key, final_key in NUMBER_ID_FIELDS.items():
        merged[final_key].extend(_candidate_id_items(numbers_output.get(id_key)))

    merged["patient_names"] = []
    for output in correction_outputs:
        merged["patient_names"].extend(item for item in _as_list(output.get("patient_names")))
        for item in _candidate_id_items(output.get("name_ids")):
            merged["patient_names"].append(item)
    merged["patient_names"].extend(item for item in _as_list(name_output.get("patient_names")))
    for item in _candidate_id_items(name_output.get("name_ids")):
        merged["patient_names"].append(item)

    merged["name_correction_candidates"] = []
    for output in correction_outputs:
        merged["name_correction_candidates"].extend(
            item for item in _as_list(output.get("name_correction_candidates"))
        )
        for item in _candidate_id_items(output.get("name_correction_ids")):
            merged["name_correction_candidates"].append(item)
    merged["name_correction_candidates"].extend(
        item for item in _as_list(name_output.get("name_correction_candidates"))
    )
    for item in _candidate_id_items(name_output.get("name_correction_ids")):
        merged["name_correction_candidates"].append(item)

    merged["dob_candidates"] = [item for item in _as_list(dob_output.get("dob_candidates"))]
    for item in _candidate_id_items(dob_output.get("dob_ids")):
        merged["dob_candidates"].append(item)

    merged["possible_errors"] = []
    for output in (numbers_output, *correction_outputs, name_output, dob_output):
        merged["possible_errors"].extend(item for item in _as_list(output.get("possible_errors")))
        merged["possible_errors"].extend(item for item in _as_list(output.get("errors")))
    return merged


def _candidate_indexes(
    candidates: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    numbers: dict[str, dict[str, Any]] = {}
    for item in _as_list(candidates.get("number_candidates")):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "")
        if cid:
            numbers[cid] = item
        normalized = str(item.get("normalized") or "")
        if normalized:
            numbers[f"normalized:{normalized}"] = item

    dobs: dict[str, dict[str, Any]] = {}
    for item in _as_list(candidates.get("dob_candidates")):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "")
        if cid:
            dobs[cid] = item
        normalized = str(item.get("normalized") or "")
        if normalized:
            dobs[f"normalized:{normalized}"] = item

    names: dict[str, dict[str, Any]] = {}
    for item in _as_list(candidates.get("name_candidates")) + _as_list(candidates.get("spelled_sequences")):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "")
        if cid:
            names[cid] = item
        for key in ("value", "raw", "letters"):
            value = str(item.get(key) or "").strip().lower()
            if value:
                names[f"{key}:{value}"] = item

    name_corrections: dict[str, dict[str, Any]] = {}
    for item in _as_list(candidates.get("name_correction_candidates")):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "")
        if cid:
            name_corrections[cid] = item
        for key in ("suggested_value", "raw"):
            value = str(item.get(key) or "").strip().lower()
            if value:
                name_corrections[f"{key}:{value}"] = item
    return numbers, dobs, names, name_corrections


def _error(errors: list[Any], code: str, field: str, item: Any) -> None:
    errors.append(
        {
            "code": code,
            "field": field,
            "candidate_id": item.get("candidate_id") if isinstance(item, dict) else "",
        }
    )


def _resolve_candidate(index: dict[str, dict[str, Any]], item: Any, *, normalized_key: str = "normalized") -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    candidate_id = str(item.get("candidate_id") or item.get("id") or "")
    if candidate_id and candidate_id in index:
        return index[candidate_id]
    normalized = str(item.get(normalized_key) or item.get("value") or item.get("raw") or "").strip()
    if normalized:
        return index.get(f"normalized:{normalized}") or index.get(f"value:{normalized.lower()}") or index.get(f"raw:{normalized.lower()}")
    return None


def _evidence_text(candidate: dict[str, Any], item: dict[str, Any]) -> str:
    return str(item.get("evidence_text") or candidate.get("evidence_text") or "").strip()


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").lower())


def _name_evidence_supports_raw(raw: str, evidence: str) -> bool:
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", raw)
    evidence_tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", evidence)
    if not raw_tokens or not evidence_tokens:
        return False
    raw_key = _name_key(" ".join(raw_tokens))
    evidence_key = _name_key(evidence)
    if raw_key and raw_key in evidence_key:
        return True
    token_count = len(raw_tokens)
    if token_count > len(evidence_tokens):
        return False
    for index in range(0, len(evidence_tokens) - token_count + 1):
        window = " ".join(evidence_tokens[index : index + token_count])
        if SequenceMatcher(None, raw_key, _name_key(window)).ratio() >= 0.78:
            return True
    return False


def _final_name_source(source: Any) -> str:
    value = str(source or "").strip()
    if value in {"relationship_subject", "transcript_spelling_corrected", "caller_id_corrected", "broad_name_recall"}:
        return value
    return "transcript"


def _validate_number_items(
    field: str,
    items: list[Any],
    number_index: dict[str, dict[str, Any]],
    errors: list[Any],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            _error(errors, "invalid_agent_item", field, {})
            continue
        candidate = _resolve_candidate(number_index, item)
        if candidate is None:
            _error(errors, "unknown_candidate_id", field, item)
            continue
        evidence = _evidence_text(candidate, item)
        if not evidence:
            _error(errors, "missing_evidence_text", field, item)
            continue
        normalized = normalize_phone_candidate(candidate.get("normalized") or candidate.get("raw"))
        if not normalized.valid or not normalized.normalized:
            _error(errors, "invalid_phone", field, item)
            continue
        if normalized.normalized in seen:
            continue
        cues = candidate.get("nearby_cues") if isinstance(candidate.get("nearby_cues"), list) else []
        accepted.append(
            {
                "raw": str(candidate.get("raw") or normalized.normalized),
                "normalized": normalized.normalized,
                "formatted": normalized.formatted,
                "label_cue": str(item.get("label_cue") or (cues[0] if cues else "") or ""),
                "evidence_text": evidence,
            }
        )
        seen.add(normalized.normalized)
    return accepted


def _validate_dob_items(
    items: list[Any],
    dob_index: dict[str, dict[str, Any]],
    errors: list[Any],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            _error(errors, "invalid_agent_item", "dob_candidates", {})
            continue
        candidate = _resolve_candidate(dob_index, item)
        if candidate is None:
            _error(errors, "unknown_candidate_id", "dob_candidates", item)
            continue
        evidence = _evidence_text(candidate, item)
        if not evidence:
            _error(errors, "missing_evidence_text", "dob_candidates", item)
            continue
        parsed = parse_dob(candidate.get("normalized") or candidate.get("raw"))
        if parsed is None:
            _error(errors, "invalid_dob", "dob_candidates", item)
            continue
        normalized = format_dob(parsed)
        if normalized in seen:
            continue
        accepted.append(
            {
                "raw": str(candidate.get("raw") or normalized),
                "normalized": normalized,
                "evidence_text": evidence,
            }
        )
        seen.add(normalized)
    return accepted


def _validate_name_items(
    items: list[Any],
    name_index: dict[str, dict[str, Any]],
    errors: list[Any],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            _error(errors, "invalid_agent_item", "patient_names", {})
            continue
        candidate = _resolve_candidate(name_index, item, normalized_key="value")
        if candidate is None:
            _error(errors, "unknown_candidate_id", "patient_names", item)
            continue
        evidence = _evidence_text(candidate, item)
        if not evidence:
            _error(errors, "missing_evidence_text", "patient_names", item)
            continue
        value = str(candidate.get("value") or candidate.get("raw") or "").strip()
        raw = str(candidate.get("raw") or value).strip()
        if not value:
            _error(errors, "invalid_name", "patient_names", item)
            continue
        if not _name_evidence_supports_raw(raw, evidence):
            _error(errors, "unsupported_name_evidence", "patient_names", item)
            continue
        value_key = _name_key(value)
        raw_key = _name_key(raw)
        if value_key in seen or raw_key in seen:
            continue
        source = _final_name_source(candidate.get("source"))
        accepted.append(
            {
                "raw": raw,
                "value": value,
                "evidence_text": evidence,
                "source": source,
                "caller_id_used": str(candidate.get("caller_id_used") or "") if source == "caller_id_corrected" else "",
            }
        )
        seen.add(value_key)
        seen.add(raw_key)
    return accepted


def _validate_name_correction_items(
    items: list[Any],
    correction_index: dict[str, dict[str, Any]],
    errors: list[Any],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            _error(errors, "invalid_agent_item", "name_correction_candidates", {})
            continue
        candidate = _resolve_candidate(correction_index, item, normalized_key="suggested_value")
        if candidate is None:
            _error(errors, "unknown_candidate_id", "name_correction_candidates", item)
            continue
        evidence = _evidence_text(candidate, item)
        raw = str(candidate.get("raw") or "").strip()
        suggested = str(candidate.get("suggested_value") or "").strip()
        caller_id_used = str(candidate.get("caller_id_used") or "").strip()
        reason = str(candidate.get("reason") or "").strip()
        if not raw or not suggested or not caller_id_used:
            _error(errors, "invalid_name_correction", "name_correction_candidates", item)
            continue
        if reason not in NAME_CORRECTION_REASONS:
            _error(errors, "invalid_name_correction_reason", "name_correction_candidates", item)
            continue
        if _name_key(raw) == _name_key(suggested):
            continue
        if not evidence or not _name_evidence_supports_raw(raw, evidence):
            _error(errors, "unsupported_name_evidence", "name_correction_candidates", item)
            continue
        key = (_name_key(raw), _name_key(suggested))
        if key in seen:
            continue
        accepted.append(
            {
                "raw": raw,
                "suggested_value": suggested,
                "evidence_text": evidence,
                "caller_id_used": caller_id_used,
                "reason": reason,
            }
        )
        seen.add(key)
    return accepted


def validate_final_json(final: dict[str, Any], candidates: dict[str, Any], transcript: str) -> dict[str, Any]:
    del transcript
    number_index, dob_index, name_index, correction_index = _candidate_indexes(candidates)
    result = empty_final_json()
    errors: list[Any] = [item for item in _as_list(final.get("possible_errors"))]

    result["callback_numbers"] = _validate_number_items(
        "callback_numbers",
        _as_list(final.get("callback_numbers")),
        number_index,
        errors,
    )
    result["fax_numbers"] = _validate_number_items(
        "fax_numbers",
        _as_list(final.get("fax_numbers")),
        number_index,
        errors,
    )
    result["uncertain_numbers"] = _validate_number_items(
        "uncertain_numbers",
        _as_list(final.get("uncertain_numbers")),
        number_index,
        errors,
    )
    result["patient_names"] = _validate_name_items(_as_list(final.get("patient_names")), name_index, errors)
    result["name_correction_candidates"] = _validate_name_correction_items(
        _as_list(final.get("name_correction_candidates")),
        correction_index,
        errors,
    )
    result["dob_candidates"] = _validate_dob_items(_as_list(final.get("dob_candidates")), dob_index, errors)
    result["possible_errors"] = errors
    return {key: result[key] for key in FINAL_SCHEMA_KEYS}
