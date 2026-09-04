#!/usr/bin/env python3
"""
Local voicemail field verification helpers.

This module is intentionally stdlib-only. Network calls and file clipping are
thin wrappers so the deterministic parsing, attribution, and resolver behavior
can be unit-tested without running Gemma, Parakeet, Whisper, or Asterisk.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from typing import Any, Optional


GEMMA_ARRAY_KEYS = (
    "patient_names",
    "name_correction_candidates",
    "dob_candidates",
    "callback_numbers",
    "fax_numbers",
    "uncertain_numbers",
    "possible_errors",
)
COMPACT_GEMMA_ARRAY_KEYS = ("n", "d", "c", "f", "u", "e")
COMPACT_GEMMA_TUPLE_SCHEMAS = {
    "n": ("patient_names", ("raw", "value", "evidence_text", "source", "caller_id_used")),
    "r": ("name_correction_candidates", ("raw", "suggested_value", "evidence_text", "caller_id_used", "reason")),
    "d": ("dob_candidates", ("raw", "normalized", "evidence_text")),
    "c": ("callback_numbers", ("raw", "normalized", "formatted", "label_cue", "evidence_text")),
    "f": ("fax_numbers", ("raw", "normalized", "formatted", "label_cue", "evidence_text")),
}
COMPACT_GEMMA_PASSTHROUGH_KEYS = {
    "u": "uncertain_numbers",
    "e": "possible_errors",
}
MAX_GEMMA_CANDIDATES_PER_FIELD = 10
MAX_GEMMA_EVIDENCE_CHARS = 500
MAX_GEMMA_VALUE_CHARS = 200
GEMMA_EVIDENCE_KEYS = {"evidence_text", "label_cue"}
GEMMA_VALUE_KEYS = {
    "raw",
    "value",
    "suggested_value",
    "normalized",
    "formatted",
    "source",
    "caller_id_used",
    "reason",
    "confidence",
}

PHONE_FIELD_NAMES = {"callback_number", "fax_number"}
PHONE_CLIP_NEIGHBOR_GAP_SECONDS = 0.1

DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "won": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
}
DIGIT_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(DIGIT_WORDS, key=len, reverse=True)) + r")\b",
    re.I,
)
PHONE_CHUNK_RE = re.compile(r"(?<!\d)(?:\+?1[\s().-]*)?(?:\d[\s().-]*){9}\d(?![\s().-]*\d)")
PHONE_LIKE_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*[A-Za-z0-9]{1,8})"
    r"(?![A-Za-z0-9])"
)
EXTENSION_DIGITS_RE = re.compile(
    r"(?:\bextension\b|\bext\b\.?|\bext\.?(?=\d)|\bx\b|\bx(?=\d))[\s:;,#.-]*(?:\d[\s.-]*){1,6}(?=\D|$)",
    re.I,
)

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

DAY_WORDS = {
    1: {"first", "one"},
    2: {"second", "two"},
    3: {"third", "three"},
    4: {"fourth", "four"},
    5: {"fifth", "five"},
    6: {"sixth", "six"},
    7: {"seventh", "seven"},
    8: {"eighth", "eight"},
    9: {"ninth", "nine"},
    10: {"tenth", "ten"},
    11: {"eleventh", "eleven"},
    12: {"twelfth", "twelve"},
    13: {"thirteenth", "thirteen"},
    14: {"fourteenth", "fourteen"},
    15: {"fifteenth", "fifteen"},
    16: {"sixteenth", "sixteen"},
    17: {"seventeenth", "seventeen"},
    18: {"eighteenth", "eighteen"},
    19: {"nineteenth", "nineteen"},
    20: {"twentieth", "twenty"},
    21: {"twenty first", "twenty-first", "twenty one", "twenty-one"},
    22: {"twenty second", "twenty-second", "twenty two", "twenty-two"},
    23: {"twenty third", "twenty-third", "twenty three", "twenty-three"},
    24: {"twenty fourth", "twenty-fourth", "twenty four", "twenty-four"},
    25: {"twenty fifth", "twenty-fifth", "twenty five", "twenty-five"},
    26: {"twenty sixth", "twenty-sixth", "twenty six", "twenty-six"},
    27: {"twenty seventh", "twenty-seventh", "twenty seven", "twenty-seven"},
    28: {"twenty eighth", "twenty-eighth", "twenty eight", "twenty-eight"},
    29: {"twenty ninth", "twenty-ninth", "twenty nine", "twenty-nine"},
    30: {"thirtieth", "thirty"},
    31: {"thirty first", "thirty-first", "thirty one", "thirty-one"},
}

DOB_CUE_RE = re.compile(
    r"\b(?:date\s+of\s+birth|birth\s+date|birthdate|d\.?\s*o\.?\s*b\.?|dob|born|birthday)\b",
    re.IGNORECASE,
)
COMPACT_DOB_PHONE_CONTEXT_RE = re.compile(
    r"\b(?:call(?:back)?|phone|telephone|tel|fax|extension|ext|contact|cell|mobile|number)\b",
    re.IGNORECASE,
)
COMPACT_DOB_DIGIT_RE = re.compile(r"(?<!\d)(\d(?:[\d\s.,/-]*\d){3,7})(?!\d)")
COMPACT_DOB_FILLER_DIGIT_RE = re.compile(
    r"(?<!\d)(\d{3,4}|\d{1,2}[\s/-]+\d{1,2})\s+(?:of|off)\s+(\d{2})(?!\d)",
    re.IGNORECASE,
)
NUMBER_TOKEN_RE = re.compile(r"[A-Za-z]+|\d{1,4}")
COMPACT_DOB_YEAR_PIVOT = 27
COMPACT_DOB_INTRODUCED_NAME_RE = re.compile(
    r"(?i)\b(?:this\s+is|it'?s|my\s+name\s+is|i\s+am|i'm|patient\s+is|patient\s+name\s+is|"
    r"calling\s+for|regarding|for)\s+"
    r"([A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){1,3})"
    r"(?=\s*(?:[,.;:]|\d|\b(?:d\.?\s*o\.?\s*b\.?|date|birth|born)\b))"
)
TRANSCRIPT_SPELLING_CORRECTED_SOURCE = "transcript_spelling_corrected"
EXPLICIT_PATIENT_NAME_SOURCE = "explicit_patient_name"
SELF_IDENTIFICATION_NAME_SOURCE = "self_identification"
RELATIONSHIP_SUBJECT_SOURCE = "relationship_subject"
SUBJECT_REFERENCE_SOURCE = "subject_reference"
EXPLICIT_PATIENT_NAME_RE = re.compile(
    r"(?i)\b(?P<prefix>patient(?:'s|s)?\s+name\s+is|patient\s+is|"
    r"calling\s+on\s+(?:a\s+)?patient\s*,?)\s+"
    r"(?P<name>[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){1,4})"
)
RELATIONSHIP_NAME_RE = re.compile(
    r"(?i)\b(?P<prefix>(?:my|our|his|her|their)\s+"
    r"(?:husband|wife|son|daughter|mother|father|spouse))\s+"
    r"(?P<name>[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){1,4})"
)
PROXY_SUBJECT_NAME_RE = re.compile(
    r"(?i)\b(?P<prefix>(?:calling\s+)?on\s+behalf\s+of)\s+"
    r"(?P<name>[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){0,4})"
)
REVERSE_RELATIONSHIP_NAME_RE = re.compile(
    r"(?i)\b(?P<name>[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){1,4})"
    r"\s*(?:'s|’s)\s+"
    r"(?P<relationship>husband|wife|son|daughter|mother|father|spouse)\b"
)
SELF_IDENTIFICATION_NAME_RE = re.compile(
    r"(?i)\b(?P<prefix>this\s+is|my\s+name\s+is|the\s+name\s+is)\s+"
    r"(?!a\b|an\b|the\b|called\b|calling\b|returning\b|home\b|trying\b|going\b)"
    r"(?P<name>[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){0,4})"
)
NAME_FALLBACK_SPELLING_CONFIRMATION_TAIL_RE = re.compile(
    r"(?i)\s*(?:[,.;:!?]\s*)?(?:that'?s|spelled|spell|spells|spelling)\b.*$"
)
SUBJECT_REFERENCE_NAME_RE = re.compile(
    r"(?i)\b(?P<prefix>(?:calling|call|called|message|following\s+up|reaching\s+out)\s+"
    r"(?:in\s+)?(?:regards?|reference|relation)\s+to|(?:in\s+)?(?:regards?|reference|relation)\s+to|"
    r"regarding|about)\s+"
    r"(?P<name>[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){1,4})"
)
NAME_SPELLING_INTRODUCED_NAME_RE = re.compile(
    r"(?i)\b(?:this\s+is|it'?s|my\s+name\s+is|i\s+am|i'm|patient\s+is|patient\s+name\s+is|"
    r"calling\s+for|regarding|for)\s+"
    r"([A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){1,3})"
    r"(?=\s*[,;:.-])"
)
NAME_SPELLING_TOKEN_RE = re.compile(r"double|triple|[A-Za-z]+|[,;:!?.-]", re.IGNORECASE)
NAME_SPELLING_EVIDENCE_RE = re.compile(
    r"(?i)(?:\b[a-z]\s*-\s*){1,}[a-z]\b|\b[A-Z]{2,}\b|\b(?:double|triple)\s+[A-Za-z]\b"
)

NUMBER_WORD_VALUES = {
    **{word: int(value) for word, value in DIGIT_WORDS.items() if len(word) > 1 or word in {"o"}},
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
}
TENS_WORD_VALUES = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
}
COMPACT_DOB_FILLER_WORDS = {"is", "was", "the", "a", "an"}
COMPACT_DOB_NAME_STOP_WORDS = {
    "a",
    "an",
    "call",
    "calling",
    "doctor",
    "dr",
    "from",
    "message",
    "nurse",
    "please",
    "the",
}
NAME_SPELLING_PREFIX_WORDS = {"spelled", "spell", "spells", "spelling", "first", "last", "name", "is", "as"}
NAME_SPELLING_STOP_ACRONYMS = {
    "A1C",
    "AETNA",
    "AI",
    "BCBS",
    "COVID",
    "CPAP",
    "CPT",
    "CT",
    "DOB",
    "DME",
    "ECG",
    "EKG",
    "ENT",
    "ER",
    "HIPAA",
    "HMO",
    "HR",
    "ICD",
    "ICU",
    "ID",
    "INC",
    "IT",
    "LLC",
    "MD",
    "MRI",
    "MRN",
    "NPI",
    "NP",
    "OT",
    "PA",
    "PET",
    "PO",
    "PPO",
    "PT",
    "RN",
    "RX",
    "SSN",
    "TV",
    "UHC",
}
NAME_SPELLING_MULTIPLIERS = {"double": 2, "triple": 3}


COMPANY_NAME_WORDS = {
    "clinic",
    "company",
    "hospital",
    "health",
    "healthcare",
    "medical",
    "center",
    "centre",
    "department",
    "office",
    "orthopedic",
    "orthopaedic",
    "medical center",
    "pharmacy",
    "lab",
    "laboratory",
    "insurance",
    "wireless",
    "caller",
    "unknown",
    "private",
    "anonymous",
    "restricted",
}


def organization_name_words() -> set[str]:
    configured = {
        value.strip().lower()
        for value in os.environ.get("VOICEMAIL_ORGANIZATION_TERMS", "").split(",")
        if value.strip()
    }
    return COMPANY_NAME_WORDS | configured


class GemmaSchemaError(ValueError):
    """Gemma returned parseable JSON that does not match the required schema."""


class VerificationBudgetExceeded(TimeoutError):
    """The field verification budget has been exhausted."""


@dataclass(frozen=True)
class NormalizedPhone:
    raw: str
    normalized: Optional[str]
    formatted: Optional[str]
    valid: bool
    leading_one_stripped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "normalized": self.normalized,
            "formatted": self.formatted,
            "valid": self.valid,
            "leading_one_stripped": self.leading_one_stripped,
        }


@dataclass
class AttributionResult:
    candidate_id: str
    field_name: str
    evidence_text: str
    mapped: bool
    mapping_method: Optional[str] = None
    word_start: Optional[int] = None
    word_end: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    segment_index_start: Optional[int] = None
    segment_index_end: Optional[int] = None
    clip_start: Optional[float] = None
    clip_end: Optional[float] = None
    matched_text: str = ""
    review_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "field_name": self.field_name,
            "evidence_text": self.evidence_text,
            "mapped": self.mapped,
            "mapping_method": self.mapping_method,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "start": self.start,
            "end": self.end,
            "segment_start": self.segment_start,
            "segment_end": self.segment_end,
            "segment_index_start": self.segment_index_start,
            "segment_index_end": self.segment_index_end,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "matched_text": self.matched_text,
            "review_reasons": list(self.review_reasons),
        }


@dataclass
class ParakeetResult:
    candidate_id: str
    text: str = ""
    normalized_numbers: list[str] = field(default_factory=list)
    formatted_numbers: list[str] = field(default_factory=list)
    raw_output: Any = field(default_factory=dict)
    error: Optional[str] = None

    def valid_unique_numbers(self) -> set[str]:
        numbers: set[str] = set()
        for value in self.normalized_numbers:
            normalized = normalize_phone_candidate(value)
            if normalized.valid and normalized.normalized:
                numbers.add(normalized.normalized)
        return numbers

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "text": self.text,
            "normalized_numbers": list(self.normalized_numbers),
            "formatted_numbers": list(self.formatted_numbers),
            "raw_output": self.raw_output,
            "error": self.error,
        }


@dataclass
class CandidateRecord:
    candidate_id: str
    field_name: str
    gemma: dict[str, Any]
    attribution: AttributionResult
    whisper_numbers: list[str] = field(default_factory=list)
    parakeet: Optional[ParakeetResult] = None
    clip: dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldResolution:
    field_name: str
    final_value: Optional[str]
    normalized_value: Optional[str]
    status: str
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    attribution_json: list[dict[str, Any]] = field(default_factory=list)
    whisper_json: dict[str, Any] = field(default_factory=dict)
    gemma_json: Any = field(default_factory=list)
    parakeet_json: list[dict[str, Any]] = field(default_factory=list)
    clip_json: list[dict[str, Any]] = field(default_factory=list)
    warnings_json: list[str] = field(default_factory=list)

    def as_audit_row(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "final_value": self.final_value,
            "normalized_value": self.normalized_value,
            "status": self.status,
            "needs_review": self.needs_review,
            "review_reasons": list(dict.fromkeys(self.review_reasons)),
            "attribution_json": self.attribution_json,
            "whisper_json": self.whisper_json,
            "gemma_json": self.gemma_json,
            "parakeet_json": self.parakeet_json,
            "clip_json": self.clip_json,
            "warnings_json": self.warnings_json,
        }


def check_budget(deadline: Optional[float]) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise VerificationBudgetExceeded("verification total timeout exceeded")


def remaining_budget(deadline: Optional[float], default: float) -> float:
    if deadline is None:
        return default
    return max(0.1, min(default, deadline - time.monotonic()))


def normalize_key(value: Any) -> str:
    tokens = re.findall(r"[A-Za-z0-9']+", str(value or "").lower())
    normalized: list[str] = []
    for token in tokens:
        if token in DIGIT_WORDS:
            normalized.append(DIGIT_WORDS[token])
        else:
            normalized.append(re.sub(r"[^a-z0-9]+", "", token))
    return "".join(item for item in normalized if item)


def digits_from_text(value: Any) -> str:
    normalized_parts: list[str] = []
    for token in re.findall(r"[A-Za-z]+|\d+", str(value or "").lower()):
        if token.isdigit():
            normalized_parts.append(token)
        elif token in DIGIT_WORDS:
            normalized_parts.append(DIGIT_WORDS[token])
    return "".join(normalized_parts)


def words_to_digits(value: Any) -> str:
    return DIGIT_WORD_RE.sub(lambda match: DIGIT_WORDS[match.group(1).lower()], str(value or ""))


def strip_extension_digits(value: Any) -> str:
    return EXTENSION_DIGITS_RE.sub(" ", str(value or ""))


def normalize_phone_candidate(value: Any) -> NormalizedPhone:
    raw = str(value or "")
    digits = digits_from_text(raw)
    if not digits:
        digits = re.sub(r"\D", "", raw)

    leading_one_stripped = False
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
        leading_one_stripped = True

    if len(digits) != 10:
        return NormalizedPhone(raw=raw, normalized=None, formatted=None, valid=False)

    return NormalizedPhone(
        raw=raw,
        normalized=digits,
        formatted=f"{digits[:3]}-{digits[3:6]}-{digits[6:]}",
        valid=True,
        leading_one_stripped=leading_one_stripped,
    )


def extract_numbers_from_text(text: Any) -> list[NormalizedPhone]:
    value = str(text or "")
    scan_value = strip_extension_digits(words_to_digits(value))
    candidates: list[NormalizedPhone] = []
    seen: set[str] = set()

    spoken_tokens = [
        DIGIT_WORDS[token]
        for token in re.findall(r"[A-Za-z]+", value.lower())
        if token in DIGIT_WORDS
    ]
    if len(spoken_tokens) >= 7:
        spoken = normalize_phone_candidate("".join(spoken_tokens))
        if spoken.valid and spoken.normalized:
            candidates.append(spoken)
            seen.add(spoken.normalized)

    for match in PHONE_CHUNK_RE.finditer(scan_value):
        normalized = normalize_phone_candidate(match.group(0))
        if normalized.valid and normalized.normalized and normalized.normalized not in seen:
            candidates.append(normalized)
            seen.add(normalized.normalized)

    return candidates


def unique_valid_phone_digits(values: list[Any]) -> set[str]:
    numbers: set[str] = set()
    for value in values:
        if isinstance(value, NormalizedPhone):
            normalized = value
        else:
            normalized = normalize_phone_candidate(value)
        if normalized.valid and normalized.normalized:
            numbers.add(normalized.normalized)
    return numbers


def format_phone_digits(digits: Optional[str]) -> Optional[str]:
    if not digits:
        return None
    normalized = normalize_phone_candidate(digits)
    return normalized.formatted


def validate_gemma_candidate_limits(key: str, items: list[Any]) -> None:
    if len(items) > MAX_GEMMA_CANDIDATES_PER_FIELD:
        raise GemmaSchemaError(
            f"Gemma response key {key!r} exceeded candidate limit {MAX_GEMMA_CANDIDATES_PER_FIELD}"
        )
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        for item_key, value in item.items():
            if not isinstance(value, str):
                continue
            if item_key in GEMMA_EVIDENCE_KEYS and len(value) > MAX_GEMMA_EVIDENCE_CHARS:
                raise GemmaSchemaError(
                    f"Gemma response key {key!r}[{index}].{item_key} exceeded {MAX_GEMMA_EVIDENCE_CHARS} chars"
                )
            if item_key in GEMMA_VALUE_KEYS and len(value) > MAX_GEMMA_VALUE_CHARS:
                raise GemmaSchemaError(
                    f"Gemma response key {key!r}[{index}].{item_key} exceeded {MAX_GEMMA_VALUE_CHARS} chars"
                )


def compact_gemma_value_to_string(value: Any) -> str:
    if value is None:
        return ""
    if value is False:
        return ""
    return str(value)


def expand_compact_gemma_response(payload: dict[str, Any]) -> dict[str, Any]:
    if all(key in payload for key in GEMMA_ARRAY_KEYS):
        return payload

    if not any(key in payload for key in COMPACT_GEMMA_ARRAY_KEYS):
        return payload

    canonical: dict[str, Any] = {}
    for compact_key in COMPACT_GEMMA_ARRAY_KEYS:
        if compact_key not in payload:
            raise GemmaSchemaError(f"Gemma compact response missing required key {compact_key!r}")
        items = payload[compact_key]
        if not isinstance(items, list):
            raise GemmaSchemaError(f"Gemma compact response key {compact_key!r} must be an array")

        if compact_key in COMPACT_GEMMA_PASSTHROUGH_KEYS:
            canonical[COMPACT_GEMMA_PASSTHROUGH_KEYS[compact_key]] = list(items)
            continue

        canonical_key, field_names = COMPACT_GEMMA_TUPLE_SCHEMAS[compact_key]
        expanded_items: list[dict[str, str]] = []
        for index, item in enumerate(items):
            if not isinstance(item, (list, tuple)) or len(item) != len(field_names):
                raise GemmaSchemaError(
                    f"Gemma compact response key {compact_key!r}[{index}] must be an array of "
                    f"{len(field_names)} values"
                )
            expanded_items.append(
                {
                    field_name: compact_gemma_value_to_string(item[field_index])
                    for field_index, field_name in enumerate(field_names)
                }
            )
        canonical[canonical_key] = expanded_items

    if "r" in payload:
        items = payload["r"]
        if not isinstance(items, list):
            raise GemmaSchemaError("Gemma compact response key 'r' must be an array")
        canonical_key, field_names = COMPACT_GEMMA_TUPLE_SCHEMAS["r"]
        expanded_items: list[dict[str, str]] = []
        for index, item in enumerate(items):
            if not isinstance(item, (list, tuple)) or len(item) != len(field_names):
                raise GemmaSchemaError(
                    f"Gemma compact response key 'r'[{index}] must be an array of {len(field_names)} values"
                )
            expanded_items.append(
                {
                    field_name: compact_gemma_value_to_string(item[field_index])
                    for field_index, field_name in enumerate(field_names)
                }
            )
        canonical[canonical_key] = expanded_items
    else:
        canonical.setdefault("name_correction_candidates", [])
    return canonical


def parse_gemma_response(value: Any) -> dict[str, Any]:
    payload = value
    if isinstance(payload, dict) and isinstance(payload.get("response"), str):
        payload = payload["response"]

    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("```"):
            raise GemmaSchemaError("Gemma returned markdown instead of strict JSON")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise GemmaSchemaError(f"Gemma returned invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise GemmaSchemaError("Gemma response must be a JSON object")

    payload = expand_compact_gemma_response(payload)
    payload.setdefault("name_correction_candidates", [])

    for key in GEMMA_ARRAY_KEYS:
        if key not in payload:
            raise GemmaSchemaError(f"Gemma response missing required key {key!r}")
        if not isinstance(payload[key], list):
            raise GemmaSchemaError(f"Gemma response key {key!r} must be an array")
        validate_gemma_candidate_limits(key, payload[key])

    return {key: list(payload[key]) for key in GEMMA_ARRAY_KEYS}


def candidate_id(field_name: str, index: int) -> str:
    if field_name == "name":
        return f"name:{index}"
    if field_name == "dob":
        return f"dob:{index}"
    return f"{field_name}:{index}"


def normalize_word_item(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    word = str(item.get("word", item.get("text", "")) or "").strip()
    if not word:
        return None
    try:
        start = float(item.get("start"))
        end = float(item.get("end"))
    except (TypeError, ValueError):
        return None
    if start < 0 or end < start:
        return None
    return {"word": word, "start": start, "end": end}


def normalize_segment_item(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    try:
        start = float(item.get("start"))
        end = float(item.get("end"))
    except (TypeError, ValueError):
        return None
    if start < 0 or end < start:
        return None
    text = str(item.get("text", "") or "")
    return {"text": text, "start": start, "end": end, "words": item.get("words") or []}


def segment_bounds_for_span(
    start: float,
    end: float,
    segments: list[dict[str, Any]],
) -> tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    first_index: Optional[int] = None
    last_index: Optional[int] = None
    for index, segment in enumerate(segments):
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        if seg_end < start or seg_start > end:
            continue
        if first_index is None:
            first_index = index
        last_index = index

    if first_index is None or last_index is None:
        return None, None, None, None
    return (
        float(segments[first_index]["start"]),
        float(segments[last_index]["end"]),
        first_index,
        last_index,
    )


def set_padded_clip_bounds(attribution: AttributionResult, padding_seconds: float = 2.0) -> None:
    if attribution.start is None or attribution.end is None:
        return

    pre_padding = padding_seconds
    post_padding = padding_seconds
    min_duration = 0.0
    lower = attribution.segment_start if attribution.segment_start is not None else 0.0
    upper = attribution.segment_end

    if attribution.field_name in PHONE_FIELD_NAMES:
        # Phone timestamps are often compressed into a short word span even when
        # the spoken digits take several seconds. Prefer a wider audio window so
        # Parakeet hears the complete number instead of a trailing fragment.
        pre_padding = max(pre_padding, 2.0)
        post_padding = max(post_padding, 4.0)
        min_duration = 6.0
        lower = 0.0
        upper = None

    clip_start = max(float(lower), float(attribution.start) - pre_padding)
    clip_end = float(attribution.end) + post_padding
    if clip_end < clip_start:
        clip_end = clip_start

    if min_duration and clip_end - clip_start < min_duration:
        clip_end = clip_start + min_duration

    if upper is not None:
        clip_end = min(float(upper), clip_end)
        if min_duration and clip_end - clip_start < min_duration:
            clip_start = max(float(lower), clip_end - min_duration)

    attribution.clip_start = round(clip_start, 3)
    attribution.clip_end = round(clip_end, 3)


def constrain_phone_clip_bounds_for_neighbors(
    records: list[CandidateRecord],
    gap_seconds: float = PHONE_CLIP_NEIGHBOR_GAP_SECONDS,
) -> None:
    phone_records = [
        record
        for record in records
        if record.field_name in PHONE_FIELD_NAMES
        and record.attribution.mapped
        and record.attribution.start is not None
        and record.attribution.end is not None
        and record.attribution.clip_start is not None
        and record.attribution.clip_end is not None
    ]
    phone_records.sort(key=lambda record: (float(record.attribution.start or 0.0), float(record.attribution.end or 0.0)))

    for index, record in enumerate(phone_records):
        attribution = record.attribution
        clip_start = float(attribution.clip_start or 0.0)
        clip_end = float(attribution.clip_end or 0.0)
        span_start = float(attribution.start or 0.0)
        span_end = float(attribution.end or 0.0)

        if index > 0:
            previous = phone_records[index - 1].attribution
            previous_end = float(previous.end or 0.0)
            lower = previous_end + gap_seconds
            if lower <= span_start:
                clip_start = max(clip_start, lower)

        if index + 1 < len(phone_records):
            following = phone_records[index + 1].attribution
            following_start = float(following.start or 0.0)
            upper = following_start - gap_seconds
            if upper >= span_end:
                clip_end = min(clip_end, upper)

        if clip_end < clip_start:
            clip_start = span_start
            clip_end = span_end

        attribution.clip_start = round(clip_start, 3)
        attribution.clip_end = round(clip_end, 3)


def map_evidence_to_timestamps(
    field_name: str,
    candidate_id_value: str,
    evidence_text: Any,
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    candidate_digits: Optional[str] = None,
) -> AttributionResult:
    """Map Gemma's evidence_text to Whisper timing data.

    Gemma is the authority for field attribution, so this function deliberately
    grounds the evidence phrase itself.  candidate_digits is accepted for API
    compatibility with older callers, but it is not used to create a successful
    mapping; a digit-only match would weaken the field attribution guarantee.
    """
    _ = candidate_digits
    evidence = str(evidence_text or "").strip()
    result = AttributionResult(
        candidate_id=candidate_id_value,
        field_name=field_name,
        evidence_text=evidence,
        mapped=False,
    )
    normalized_words = [normalize_word_item(item) for item in words]
    normalized_words = [item for item in normalized_words if item is not None]
    normalized_segments = [normalize_segment_item(item) for item in segments]
    normalized_segments = [item for item in normalized_segments if item is not None]

    evidence_key = normalize_key(evidence)
    if evidence_key and normalized_words:
        max_window = min(64, max(8, len(re.findall(r"\S+", evidence)) + 12))
        for start_index in range(len(normalized_words)):
            combined = ""
            for end_index in range(start_index, min(len(normalized_words), start_index + max_window)):
                combined += normalize_key(normalized_words[end_index]["word"])
                if not combined:
                    continue
                if combined == evidence_key:
                    start = float(normalized_words[start_index]["start"])
                    end = float(normalized_words[end_index]["end"])
                    seg_start, seg_end, seg_i, seg_j = segment_bounds_for_span(start, end, normalized_segments)
                    result.mapped = True
                    result.mapping_method = "word"
                    result.word_start = start_index
                    result.word_end = end_index
                    result.start = round(start, 3)
                    result.end = round(end, 3)
                    result.segment_start = seg_start
                    result.segment_end = seg_end
                    result.segment_index_start = seg_i
                    result.segment_index_end = seg_j
                    result.matched_text = " ".join(word["word"] for word in normalized_words[start_index : end_index + 1])
                    set_padded_clip_bounds(result)
                    return result
                if len(combined) > len(evidence_key) + 20:
                    break

    for index, segment in enumerate(normalized_segments):
        segment_key = normalize_key(segment["text"])
        if evidence_key and evidence_key in segment_key:
            result.mapped = True
            result.mapping_method = "segment"
            result.start = round(float(segment["start"]), 3)
            result.end = round(float(segment["end"]), 3)
            result.segment_start = result.start
            result.segment_end = result.end
            result.segment_index_start = index
            result.segment_index_end = index
            result.clip_start = result.start
            result.clip_end = result.end
            result.matched_text = str(segment["text"] or "")
            result.review_reasons.append("segment_fallback")
            if field_name in PHONE_FIELD_NAMES and len(extract_numbers_from_text(result.matched_text)) > 1:
                result.review_reasons.append("multiple_phone_values_in_segment")
            return result

    result.review_reasons.append("evidence_not_mapped")
    return result


def safe_clip_filename(candidate_id_value: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id_value).strip("_") or "candidate"
    return f"{safe_id}-{uuid.uuid4().hex[:12]}.wav"


def create_verification_clip(
    audio_path: str,
    attribution: AttributionResult,
    output_dir: str,
    expected_sample_rate: int = 16000,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    if attribution.clip_start is None or attribution.clip_end is None:
        return {
            "candidate_id": attribution.candidate_id,
            "error": "missing_clip_bounds",
        }

    try:
        os.makedirs(output_dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(output_dir, 0o700)
        except OSError:
            # Best-effort hardening. Directory ownership/mount options may prevent chmod.
            pass
    except OSError as exc:
        return {"candidate_id": attribution.candidate_id, "error": f"clip_dir_failed:{str(exc)[:80]}"}

    clip_path = os.path.join(output_dir, safe_clip_filename(attribution.candidate_id))
    duration = max(0.1, attribution.clip_end - attribution.clip_start)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{attribution.clip_start:.3f}",
        "-i",
        audio_path,
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        str(expected_sample_rate),
        clip_path,
    ]

    try:
        subprocess.run(command, check=True, timeout=timeout_seconds)
    except FileNotFoundError:
        return {"candidate_id": attribution.candidate_id, "error": "ffmpeg_missing"}
    except subprocess.TimeoutExpired:
        return {"candidate_id": attribution.candidate_id, "error": "clip_timeout"}
    except subprocess.CalledProcessError as exc:
        return {"candidate_id": attribution.candidate_id, "error": f"ffmpeg_failed:{exc.returncode}"}

    return {
        "candidate_id": attribution.candidate_id,
        "path": clip_path,
        "start": attribution.clip_start,
        "end": attribution.clip_end,
        "sample_rate": expected_sample_rate,
    }


def normalize_parakeet_payload(candidate_id_value: str, payload: Any) -> ParakeetResult:
    if isinstance(payload, str):
        text = payload
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            numbers = extract_numbers_from_text(text)
            return ParakeetResult(
                candidate_id=candidate_id_value,
                text=text,
                normalized_numbers=[item.normalized for item in numbers if item.normalized],
                formatted_numbers=[item.formatted for item in numbers if item.formatted],
                raw_output=text,
            )

    if not isinstance(payload, dict):
        return ParakeetResult(candidate_id=candidate_id_value, raw_output=payload, error="invalid_parakeet_payload")

    text = str(payload.get("text") or "")
    text_numbers = [item for item in extract_numbers_from_text(text) if item.normalized] if text else []
    using_text_numbers = bool(text_numbers)
    if using_text_numbers:
        raw_numbers = [item.normalized for item in text_numbers if item.normalized]
    else:
        raw_numbers = payload.get("normalized_numbers")
        if not isinstance(raw_numbers, list):
            raw_numbers = []

    normalized_numbers: list[str] = []
    formatted_numbers: list[str] = []
    for value in raw_numbers:
        normalized = normalize_phone_candidate(value)
        if normalized.valid and normalized.normalized:
            normalized_numbers.append(normalized.normalized)
            if normalized.formatted:
                formatted_numbers.append(normalized.formatted)

    raw_formatted = payload.get("formatted_numbers")
    if not using_text_numbers and isinstance(raw_formatted, list) and raw_formatted:
        formatted_numbers = [str(item) for item in raw_formatted]

    return ParakeetResult(
        candidate_id=candidate_id_value,
        text=text,
        normalized_numbers=list(dict.fromkeys(normalized_numbers)),
        formatted_numbers=list(dict.fromkeys(formatted_numbers)),
        raw_output=payload,
        error=payload.get("error") if isinstance(payload.get("error"), str) else None,
    )


def call_parakeet_http(
    candidate_id_value: str,
    url: str,
    clip_path: str,
    timeout_seconds: float,
    requests_module: Any,
    headers: Optional[dict[str, str]] = None,
) -> ParakeetResult:
    try:
        with open(clip_path, "rb") as clip:
            response = requests_module.post(
                url,
                files={"file": (os.path.basename(clip_path), clip, "audio/wav")},
                headers=headers or None,
                timeout=(min(5, timeout_seconds), timeout_seconds),
            )
        if response.status_code >= 400:
            return ParakeetResult(candidate_id=candidate_id_value, error=f"http_{response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        return normalize_parakeet_payload(candidate_id_value, payload)
    except Exception as exc:
        return ParakeetResult(candidate_id=candidate_id_value, error=str(exc)[:200])


def call_parakeet_cli(
    candidate_id_value: str,
    command: str,
    clip_path: str,
    timeout_seconds: float,
) -> ParakeetResult:
    if not command.strip():
        return ParakeetResult(candidate_id=candidate_id_value, error="missing_cli_command")
    args = shlex.split(command) + [clip_path]
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return ParakeetResult(candidate_id=candidate_id_value, error=str(exc)[:200])
    if completed.returncode != 0:
        return ParakeetResult(
            candidate_id=candidate_id_value,
            raw_output={"stderr": completed.stderr},
            error=f"cli_exit_{completed.returncode}",
        )
    return normalize_parakeet_payload(candidate_id_value, completed.stdout.strip())


def whisper_numbers_from_records(records: list[CandidateRecord]) -> set[str]:
    numbers: set[str] = set()
    for record in records:
        for number in record.whisper_numbers:
            normalized = normalize_phone_candidate(number)
            if normalized.valid and normalized.normalized:
                numbers.add(normalized.normalized)
    return numbers


def resolution_common_json(records: list[CandidateRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [record.attribution.as_dict() for record in records],
        [record.gemma for record in records],
        [record.parakeet.as_dict() for record in records if record.parakeet is not None],
        [record.clip for record in records if record.clip],
    )


def resolve_legacy_field(field_name: str, legacy_value: Any, fail_open: bool, reason: str) -> FieldResolution:
    if fail_open:
        if field_name in PHONE_FIELD_NAMES:
            normalized = normalize_phone_candidate(legacy_value)
            final = normalized.formatted if normalized.valid else None
            normalized_value = normalized.normalized if normalized.valid else None
        else:
            final = str(legacy_value).strip() if legacy_value else None
            normalized_value = final
        return FieldResolution(
            field_name=field_name,
            final_value=final,
            normalized_value=normalized_value,
            status="legacy_fallback",
            needs_review=True,
            review_reasons=[reason],
            whisper_json={"legacy_value": legacy_value},
        )
    return FieldResolution(
        field_name=field_name,
        final_value=None,
        normalized_value=None,
        status="not_included",
        needs_review=True,
        review_reasons=[reason],
    )


def _record_valid_phone_numbers(values: list[Any]) -> set[str]:
    return unique_valid_phone_digits(values)


def whisper_span_consensus(records: list[CandidateRecord]) -> tuple[Optional[str], bool, set[str]]:
    """Return (selected, agrees, all_numbers) for mapped Whisper evidence spans.

    Consensus is stricter than "there is one number somewhere": every usable
    Gemma-attributed span must contain exactly one valid phone value, and all
    spans must agree on the same normalized 10-digit number.
    """
    all_numbers: set[str] = set()
    per_record: list[set[str]] = []
    for record in records:
        values = _record_valid_phone_numbers(record.whisper_numbers)
        per_record.append(values)
        all_numbers.update(values)

    if not per_record:
        return None, False, all_numbers
    if all(len(values) == 1 for values in per_record) and len(all_numbers) == 1:
        return next(iter(all_numbers)), True, all_numbers
    return None, False, all_numbers


def record_whisper_phone_values(record: CandidateRecord) -> set[str]:
    values: set[str] = set()
    for number in record.whisper_numbers:
        normalized = normalize_phone_candidate(number)
        if normalized.valid and normalized.normalized:
            values.add(normalized.normalized)
    return values


def record_gemma_phone_values(record: CandidateRecord) -> set[str]:
    values: set[str] = set()
    for key in ("normalized", "formatted", "raw"):
        normalized = normalize_phone_candidate(record.gemma.get(key))
        if normalized.valid and normalized.normalized:
            values.add(normalized.normalized)
    return values


def phone_like_values_in_text(text: Any) -> list[str]:
    values: list[str] = []
    for match in PHONE_LIKE_TEXT_RE.finditer(str(text or "")):
        candidate = match.group(0)
        digits = re.sub(r"\D", "", candidate)
        alnum = re.sub(r"[^A-Za-z0-9]", "", candidate)
        if len(alnum) == 11 and alnum.startswith("1"):
            alnum = alnum[1:]
        if len(digits) >= 7 and 7 <= len(alnum) <= 14:
            values.append(candidate)
    return values


def attribution_has_identified_word_span(attribution: AttributionResult, field_name: str) -> bool:
    if attribution.field_name != field_name or attribution.mapping_method != "word":
        return False
    try:
        start_index = int(attribution.word_start)
        end_index = int(attribution.word_end)
    except (TypeError, ValueError):
        return False
    return start_index <= end_index


def fax_parakeet_scope_review_reasons(record: CandidateRecord, parakeet_numbers: set[str]) -> list[str]:
    if record.field_name != "fax_number":
        return []

    reasons: list[str] = []
    if not attribution_has_identified_word_span(record.attribution, "fax_number"):
        reasons.append("parakeet_outside_attributed_span")

    span_text = record.attribution.matched_text or record.attribution.evidence_text or record.gemma.get("evidence_text")
    span_values = phone_like_values_in_text(span_text)
    if len(span_values) > 1:
        reasons.append("multiple_phone_values_in_span")

    candidate_numbers = record_gemma_phone_values(record) | record_whisper_phone_values(record)
    if not span_values and not (record_gemma_phone_values(record) or record_whisper_phone_values(record)):
        reasons.append("parakeet_outside_attributed_span")
    if reasons and candidate_numbers and not parakeet_numbers.issubset(candidate_numbers):
        reasons.append("parakeet_disagreed")

    return list(dict.fromkeys(reasons))


def normalized_legacy_phone_value(value: Any) -> Optional[str]:
    normalized = normalize_phone_candidate(value)
    if normalized.valid and normalized.normalized:
        return normalized.normalized
    return None


def phone_record_blocks_parakeet_override(record: CandidateRecord) -> bool:
    reasons = set(record.attribution.review_reasons)
    return (
        "multiple_phone_values_in_segment" in reasons
        or record.attribution.mapping_method == "word_number"
    )


def resolve_phone_field(
    field_name: str,
    records: list[CandidateRecord],
    legacy_value: Any = None,
    caller_id_value: Any = None,
    gemma_unavailable: bool = False,
    fail_open: bool = True,
) -> FieldResolution:
    if gemma_unavailable:
        return resolve_legacy_field(field_name, legacy_value, fail_open, "gemma_unavailable")

    legacy_number = normalized_legacy_phone_value(legacy_value)
    caller_id_number = normalized_legacy_phone_value(caller_id_value)
    caller_id_entity_agrees = (
        field_name == "callback_number"
        and legacy_number is not None
        and caller_id_number is not None
        and legacy_number == caller_id_number
    )

    if not records:
        if caller_id_entity_agrees:
            return FieldResolution(
                field_name=field_name,
                final_value=format_phone_digits(legacy_number),
                normalized_value=legacy_number,
                status="whisper_caller_id_verified",
                needs_review=False,
                whisper_json={
                    "span_numbers": [],
                    "entity_number": legacy_number,
                    "caller_id_number": caller_id_number,
                    "agreement_source": "entity_caller_id",
                },
            )
        return FieldResolution(
            field_name=field_name,
            final_value=None,
            normalized_value=None,
            status="not_included",
            review_reasons=["no_gemma_candidate"],
        )

    attributions, gemma_json, parakeet_json, clip_json = resolution_common_json(records)
    usable = [record for record in records if record.attribution.mapped]
    reasons: list[str] = []
    for record in records:
        reasons.extend(record.attribution.review_reasons)

    if not usable:
        return FieldResolution(
            field_name=field_name,
            final_value=None,
            normalized_value=None,
            status="not_included",
            needs_review=True,
            review_reasons=list(dict.fromkeys(reasons or ["evidence_not_mapped"])),
            attribution_json=attributions,
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    whisper_sets_by_candidate = [record_whisper_phone_values(record) for record in usable]
    whisper_span_numbers: set[str] = set().union(*whisper_sets_by_candidate) if whisper_sets_by_candidate else set()
    whisper_spans_agree = bool(whisper_sets_by_candidate) and all(
        values and len(values) == 1 and values == whisper_sets_by_candidate[0]
        for values in whisper_sets_by_candidate
    )
    caller_id_span_agrees = (
        field_name == "callback_number"
        and caller_id_number is not None
        and whisper_spans_agree
        and whisper_span_numbers == {caller_id_number}
    )

    parakeet_sets_by_candidate: list[set[str]] = []
    parakeet_complete = True
    parakeet_blocked = False
    any_parakeet_multiple = False
    gemma_fax_numbers: set[str] = set()
    if field_name == "fax_number":
        for record in usable:
            gemma_fax_numbers.update(record_gemma_phone_values(record))

    for record in usable:
        if phone_record_blocks_parakeet_override(record):
            parakeet_complete = False
            parakeet_blocked = True
            if record.attribution.mapping_method == "word_number":
                reasons.append("evidence_not_mapped")
            continue

        if record.parakeet is None:
            parakeet_complete = False
            reasons.append("parakeet_unavailable")
            continue
        if record.parakeet.error:
            parakeet_complete = False
            reasons.append("parakeet_unavailable")
            continue

        valid = record.parakeet.valid_unique_numbers()
        if len(valid) > 1:
            parakeet_complete = False
            any_parakeet_multiple = True
            reasons.append("multiple_parakeet_numbers")
        elif len(valid) == 0:
            parakeet_complete = False
            reasons.append("parakeet_invalid")
        else:
            scope_reasons = fax_parakeet_scope_review_reasons(record, valid)
            if scope_reasons:
                parakeet_complete = False
                parakeet_blocked = True
                reasons.extend(scope_reasons)
                continue
            parakeet_sets_by_candidate.append(valid)

    def fax_gemma_resolution(extra_reasons: list[str]) -> FieldResolution | None:
        if field_name != "fax_number" or not gemma_fax_numbers:
            return None
        review_reasons = list(dict.fromkeys([*reasons, *extra_reasons]))
        parakeet_numbers = set().union(*parakeet_sets_by_candidate) if parakeet_sets_by_candidate else set()
        if len(gemma_fax_numbers) != 1:
            if parakeet_numbers and not parakeet_numbers.issubset(gemma_fax_numbers):
                review_reasons.append("parakeet_disagreed")
            if whisper_span_numbers and not whisper_span_numbers.issubset(gemma_fax_numbers):
                review_reasons.append("whisper_span_disagreed")
            return FieldResolution(
                field_name=field_name,
                final_value=None,
                normalized_value=None,
                status="ambiguous",
                needs_review=True,
                review_reasons=list(dict.fromkeys(review_reasons or ["multiple_field_candidates"])),
                attribution_json=attributions,
                whisper_json={"span_numbers": sorted(whisper_span_numbers)},
                gemma_json=gemma_json,
                parakeet_json=parakeet_json,
                clip_json=clip_json,
            )
        selected = next(iter(gemma_fax_numbers))
        if parakeet_numbers and selected not in parakeet_numbers:
            review_reasons.append("parakeet_disagreed")
        if whisper_span_numbers and selected not in whisper_span_numbers:
            review_reasons.append("whisper_span_disagreed")
        status = "verified" if selected in whisper_span_numbers or legacy_number == selected else "ambiguous"
        return FieldResolution(
            field_name=field_name,
            final_value=format_phone_digits(selected),
            normalized_value=selected,
            status=status,
            needs_review=bool(review_reasons),
            review_reasons=list(dict.fromkeys(review_reasons)),
            attribution_json=attributions,
            whisper_json={"span_numbers": sorted(whisper_span_numbers)},
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    if field_name == "fax_number" and parakeet_blocked and gemma_fax_numbers:
        scoped_fax_resolution = fax_gemma_resolution([])
        if scoped_fax_resolution is not None:
            return scoped_fax_resolution

    if caller_id_entity_agrees or caller_id_span_agrees:
        selected = caller_id_number if caller_id_number is not None else legacy_number
        if selected is not None:
            agreement_source = "entity_caller_id" if caller_id_entity_agrees else "span_caller_id"
            whisper_json = {
                "span_numbers": sorted(whisper_span_numbers),
                "caller_id_number": caller_id_number,
                "agreement_source": agreement_source,
            }
            if legacy_number:
                whisper_json["entity_number"] = legacy_number
            parakeet_numbers = set().union(*parakeet_sets_by_candidate) if parakeet_sets_by_candidate else set()
            review_reasons = list(dict.fromkeys(reasons))
            if parakeet_numbers and selected not in parakeet_numbers:
                review_reasons.append("parakeet_disagreed")
            return FieldResolution(
                field_name=field_name,
                final_value=format_phone_digits(selected),
                normalized_value=selected,
                status="whisper_caller_id_verified",
                needs_review=False,
                review_reasons=list(dict.fromkeys(review_reasons)),
                attribution_json=attributions,
                whisper_json=whisper_json,
                gemma_json=gemma_json,
                parakeet_json=parakeet_json,
                clip_json=clip_json,
            )

    if parakeet_complete and len(parakeet_sets_by_candidate) == len(usable):
        parakeet_numbers: set[str] = set().union(*parakeet_sets_by_candidate)
        if len(parakeet_numbers) == 1:
            selected = next(iter(parakeet_numbers))
            whisper_agrees = whisper_span_numbers == {selected} or legacy_number == selected
            status = "verified" if whisper_agrees else "parakeet_override"
            needs_review = status == "parakeet_override"
            if needs_review:
                reasons.append("parakeet_override")
            if field_name == "fax_number" and gemma_fax_numbers and selected not in gemma_fax_numbers:
                reasons.append("parakeet_disagreed")
            whisper_json = {"span_numbers": sorted(whisper_span_numbers)}
            if legacy_number:
                whisper_json["entity_number"] = legacy_number
                whisper_json["agreement_source"] = "entity" if legacy_number == selected else "span"
            return FieldResolution(
                field_name=field_name,
                final_value=format_phone_digits(selected),
                normalized_value=selected,
                status=status,
                needs_review=needs_review,
                review_reasons=list(dict.fromkeys(reasons)),
                attribution_json=attributions,
                whisper_json=whisper_json,
                gemma_json=gemma_json,
                parakeet_json=parakeet_json,
                clip_json=clip_json,
            )

        reasons.append("multiple_parakeet_numbers")
        fallback = next(iter(whisper_span_numbers)) if whisper_spans_agree else None
        if field_name == "fax_number" and fallback and gemma_fax_numbers and fallback not in gemma_fax_numbers:
            return fax_gemma_resolution(["multiple_parakeet_numbers", "whisper_span_disagreed"])
        return FieldResolution(
            field_name=field_name,
            final_value=format_phone_digits(fallback),
            normalized_value=fallback,
            status="ambiguous",
            needs_review=True,
            review_reasons=list(dict.fromkeys(reasons)),
            attribution_json=attributions,
            whisper_json={"span_numbers": sorted(whisper_span_numbers)},
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    if len(set().union(*parakeet_sets_by_candidate)) > 1 or any_parakeet_multiple:
        reasons.append("multiple_parakeet_numbers")
        fallback = next(iter(whisper_span_numbers)) if whisper_spans_agree else None
        if field_name == "fax_number" and fallback and gemma_fax_numbers and fallback not in gemma_fax_numbers:
            return fax_gemma_resolution(["multiple_parakeet_numbers", "whisper_span_disagreed"])
        return FieldResolution(
            field_name=field_name,
            final_value=format_phone_digits(fallback),
            normalized_value=fallback,
            status="ambiguous",
            needs_review=True,
            review_reasons=list(dict.fromkeys(reasons)),
            attribution_json=attributions,
            whisper_json={"span_numbers": sorted(whisper_span_numbers)},
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    if whisper_spans_agree:
        selected = next(iter(whisper_span_numbers))
        if field_name == "fax_number" and gemma_fax_numbers and selected not in gemma_fax_numbers:
            return fax_gemma_resolution(["whisper_span_disagreed"])
        return FieldResolution(
            field_name=field_name,
            final_value=format_phone_digits(selected),
            normalized_value=selected,
            status="whisper_span_fallback",
            needs_review=bool(reasons),
            review_reasons=list(dict.fromkeys(reasons)),
            attribution_json=attributions,
            whisper_json={"span_numbers": sorted(whisper_span_numbers)},
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    if whisper_span_numbers or parakeet_sets_by_candidate or parakeet_blocked:
        if len(whisper_span_numbers) > 1:
            reasons.append("multiple_field_candidates")
        return FieldResolution(
            field_name=field_name,
            final_value=None,
            normalized_value=None,
            status="ambiguous",
            needs_review=True,
            review_reasons=list(dict.fromkeys(reasons or ["multiple_field_candidates"])),
            attribution_json=attributions,
            whisper_json={"span_numbers": sorted(whisper_span_numbers)},
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    return FieldResolution(
        field_name=field_name,
        final_value=None,
        normalized_value=None,
        status="not_included",
        needs_review=bool(reasons),
        review_reasons=list(dict.fromkeys(reasons)),
        attribution_json=attributions,
        whisper_json={"span_numbers": []},
        gemma_json=gemma_json,
        parakeet_json=parakeet_json,
        clip_json=clip_json,
    )

def clean_name(value: Any) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:-")
    text = re.sub(
        r"(?i)^(this is|my name is|the name is|patient is|calling for|regarding|for|on behalf of)\s+",
        "",
        text,
    ).strip(" ,.;:-")
    if not text:
        return None
    return " ".join(
        part[:1].upper() + (part[1:].lower() if part.isupper() else part[1:])
        for part in text.split()
    )


def clean_caller_id_name(value: Any) -> Optional[str]:
    raw = str(value or "").strip().strip('"')
    if "," in raw and not any(ch.isdigit() for ch in raw):
        parts = [part.strip(" ,.;:-") for part in raw.split(",", 1)]
        if len(parts) == 2 and all(re.search(r"[A-Za-z]", part) for part in parts):
            raw = f"{parts[1]} {parts[0]}"
    return clean_name(raw)


def name_key(value: Any) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").lower())


def person_like_name(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or any(ch.isdigit() for ch in text):
        return False
    lowered = text.lower()
    if lowered in {"unknown", "wireless caller", "no caller id", "private", "restricted"}:
        return False
    words = set(re.findall(r"[a-z]+", lowered))
    vocabulary = organization_name_words()
    if words & vocabulary or lowered in vocabulary:
        return False
    return bool(re.search(r"[A-Za-z]{2,}", text))


ADDRESSEE_EVIDENCE_RE = re.compile(
    r"(?i)^\s*(?:hey|hi|hello|dear|good\s+(?:morning|afternoon|evening))"
    r"\s*,?\s+[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,2}\s*[,!.]?\s*$"
)
UNCERTAIN_ADDRESSEE_EVIDENCE_RE = re.compile(
    r"(?i)^\s*[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,2}\s*,?\s+"
    r"(?:i\s+think\s+that'?s\s+your\s+name|.*\b(?:your\s+name|heard\s+that\s+correctly)\b)"
)
SELF_IDENTIFICATION_RE = re.compile(
    r"(?i)\b(?:this\s+is|it'?s|my\s+name\s+is|the\s+name\s+is|i\s+am|i'm|calling\s+for|patient\s+is|regarding)\b"
)
NAME_NOISE_WORDS = {
    "after",
    "ago",
    "again",
    "appointment",
    "before",
    "call",
    "calling",
    "correctly",
    "doctor",
    "heard",
    "having",
    "impatient",
    "name",
    "number",
    "office",
    "please",
    "probably",
    "problems",
    "safest",
    "week",
    "weeks",
    "today",
    "very",
}
RELATIONSHIP_NAME_TRAILING_STOP_WORDS = {
    "about",
    "and",
    "again",
    "back",
    "because",
    "birth",
    "call",
    "calling",
    "date",
    "dob",
    "for",
    "has",
    "i",
    "is",
    "it",
    "it's",
    "its",
    "need",
    "needs",
    "please",
    "regarding",
    "to",
    "was",
    "who",
}
NAME_FALLBACK_REJECT_WORDS = COMPACT_DOB_NAME_STOP_WORDS | NAME_NOISE_WORDS | {
    "about",
    "affect",
    "affected",
    "affecting",
    "any",
    "as",
    "back",
    "been",
    "being",
    "body",
    "called",
    "drive",
    "getting",
    "going",
    "her",
    "his",
    "home",
    "my",
    "not",
    "now",
    "on",
    "organ",
    "organs",
    "our",
    "other",
    "regard",
    "regarding",
    "regards",
    "reference",
    "relation",
    "right",
    "said",
    "scheduled",
    "soon",
    "street",
    "system",
    "systems",
    "team",
    "their",
    "that",
    "your",
    "with",
}
NAME_FALLBACK_BREAK_WORDS = NAME_FALLBACK_REJECT_WORDS | {
    "are",
    "as",
    "at",
    "but",
    "by",
    "due",
    "gave",
    "get",
    "gets",
    "got",
    "had",
    "has",
    "have",
    "he",
    "i",
    "in",
    "injection",
    "injections",
    "is",
    "just",
    "left",
    "me",
    "message",
    "no",
    "of",
    "on",
    "right",
    "said",
    "scheduled",
    "she",
    "soon",
    "they",
    "to",
    "us",
    "was",
    "we",
    "were",
}
NAME_FALLBACK_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr"}
SUBJECT_REFERENCE_TOPIC_WORDS = {
    "account",
    "appointment",
    "authorization",
    "balance",
    "benefits",
    "billing",
    "claim",
    "ct",
    "eligibility",
    "insurance",
    "lab",
    "medication",
    "mri",
    "pre",
    "prescription",
    "procedure",
    "records",
    "referral",
    "results",
    "scheduling",
    "surgery",
    "xray",
}
SUBJECT_NAME_SOURCES = {RELATIONSHIP_SUBJECT_SOURCE}


def evidence_is_addressee_only(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return bool(ADDRESSEE_EVIDENCE_RE.match(text) or UNCERTAIN_ADDRESSEE_EVIDENCE_RE.match(text))


def name_candidate_score(value: Any, evidence_text: Any) -> int:
    text = str(value or "").strip()
    evidence = str(evidence_text or "")
    if not person_like_name(text):
        return -100
    parts = text.split()
    score = 0
    if SELF_IDENTIFICATION_RE.search(evidence):
        score += 5
    if len(parts) >= 2:
        score += 2
    if any(part[:1].isupper() for part in parts[1:]):
        score += 1
    if set(re.findall(r"[a-z]+", text.lower())) & NAME_NOISE_WORDS:
        return -100
    return score


def trim_trailing_name_stop_words(value: Any) -> Optional[str]:
    without_spelling_tail = NAME_FALLBACK_SPELLING_CONFIRMATION_TAIL_RE.sub("", str(value or ""))
    cleaned = clean_name(without_spelling_tail)
    if not cleaned:
        return None
    parts = cleaned.split()
    while parts and parts[-1].lower().strip("'.-") in RELATIONSHIP_NAME_TRAILING_STOP_WORDS:
        parts.pop()
    if len(parts) < 2:
        return None
    return clean_name(" ".join(parts))


def clean_fallback_person_name(value: Any, minimum_tokens: int = 2) -> Optional[str]:
    cleaned = bounded_fallback_person_name(value, minimum_tokens=minimum_tokens)
    if not cleaned:
        return None
    tokens = re.findall(r"[a-z]+", cleaned.lower())
    if len(tokens) < minimum_tokens:
        return None
    if any(token in NAME_FALLBACK_REJECT_WORDS for token in tokens):
        return None
    if not person_like_name(cleaned):
        return None
    return cleaned


def bounded_fallback_person_name(value: Any, minimum_tokens: int = 2) -> Optional[str]:
    kept: list[str] = []
    for match in re.finditer(r"[A-Za-z][A-Za-z'.-]*", str(value or "")):
        token = match.group(0)
        core = token.strip(" ,;:!?")
        lower = core.strip(".").lower()
        if lower in NAME_FALLBACK_BREAK_WORDS:
            break
        kept.append(core)
        letters = re.sub(r"[^A-Za-z]+", "", core)
        if token.rstrip().endswith((".", "!", "?")) and lower not in NAME_FALLBACK_HONORIFICS and len(letters) > 1:
            break

    without_spelling_tail = NAME_FALLBACK_SPELLING_CONFIRMATION_TAIL_RE.sub("", " ".join(kept))
    cleaned = clean_name(without_spelling_tail)
    if not cleaned:
        return None
    parts = cleaned.split()
    while parts and parts[-1].lower().strip("'.-") in RELATIONSHIP_NAME_TRAILING_STOP_WORDS:
        parts.pop()
    if len(parts) < minimum_tokens:
        return None
    return clean_name(" ".join(parts))


def subject_reference_looks_like_topic(value: Any) -> bool:
    tokens = set(re.findall(r"[a-z]+", str(value or "").lower()))
    return bool(tokens & SUBJECT_REFERENCE_TOPIC_WORDS)


def names_plausibly_match(left: Any, right: Any) -> bool:
    left_clean = clean_name(left)
    right_clean = clean_name(right)
    if not left_clean or not right_clean:
        return False
    left_parts = left_clean.lower().split()
    right_parts = right_clean.lower().split()
    if len(left_parts) >= 2 and len(right_parts) >= 2:
        if left_parts[-1] == right_parts[-1] and left_parts[0][:1] == right_parts[0][:1]:
            return True
    return SequenceMatcher(None, name_key(left_clean), name_key(right_clean)).ratio() >= 0.78


def name_alpha_tokens(value: Any) -> list[str]:
    cleaned = clean_name(value)
    if not cleaned:
        return []
    tokens = re.findall(r"[a-z]+", cleaned.lower())
    if len(tokens) >= 3:
        tokens = [token for index, token in enumerate(tokens) if not (0 < index < len(tokens) - 1 and len(token) == 1)]
    return tokens


def rough_last_name_key(value: str) -> str:
    text = re.sub(r"[^a-z]+", "", str(value or "").lower())
    if not text:
        return ""
    text = (
        text.replace("ph", "f")
        .replace("ck", "k")
        .replace("qu", "k")
        .replace("c", "k")
        .replace("q", "k")
        .replace("v", "f")
        .replace("z", "s")
    )
    return re.sub(r"[aeiouy]+", "", text)


def title_from_spelled_letters(value: str) -> str:
    text = re.sub(r"[^A-Za-z]+", "", str(value or "")).lower()
    return text[:1].upper() + text[1:] if text else ""


def spelling_group_is_plausible(value: str) -> bool:
    text = re.sub(r"[^A-Za-z]+", "", str(value or "")).upper()
    if len(text) < 2 or len(text) > 32:
        return False
    if text in NAME_SPELLING_STOP_ACRONYMS:
        return False
    if len(text) > 2 and not re.search(r"[AEIOUY]", text):
        return False
    lowered = text.lower()
    return lowered not in organization_name_words() and lowered not in NAME_NOISE_WORDS


def parse_spelling_group(tokens: list[re.Match[str]], start_index: int) -> Optional[tuple[str, int, int, int]]:
    index = start_index
    letters: list[str] = []
    group_start: Optional[int] = None
    group_end: Optional[int] = None

    while index < len(tokens):
        token = tokens[index].group(0)
        lower = token.lower()

        if token in {"!", "?", ".", ";", ":"}:
            break
        if token in {",", "-"}:
            if letters and token == ",":
                break
            index += 1
            continue
        if lower in NAME_SPELLING_PREFIX_WORDS and not letters:
            index += 1
            continue
        if lower in NAME_SPELLING_MULTIPLIERS:
            next_index = index + 1
            while next_index < len(tokens) and tokens[next_index].group(0) == "-":
                next_index += 1
            if next_index >= len(tokens):
                break
            next_token = tokens[next_index].group(0)
            if not (len(next_token) == 1 and next_token.isalpha()):
                break
            if group_start is None:
                group_start = tokens[index].start()
            letters.extend(next_token.lower() for _ in range(NAME_SPELLING_MULTIPLIERS[lower]))
            group_end = tokens[next_index].end()
            index = next_index + 1
            continue
        if len(token) == 1 and token.isalpha():
            if group_start is None:
                group_start = tokens[index].start()
            letters.append(token.lower())
            group_end = tokens[index].end()
            index += 1
            continue
        if token.isalpha() and token.isupper() and 2 <= len(token) <= 20:
            if group_start is None:
                group_start = tokens[index].start()
            letters.extend(token.lower())
            group_end = tokens[index].end()
            index += 1
            break
        break

    spelled = "".join(letters)
    if group_start is None or group_end is None or not spelling_group_is_plausible(spelled):
        return None
    return spelled, group_start, group_end, index


def parse_spelling_groups_from_tail(tail: str) -> list[tuple[str, int, int]]:
    tokens = list(NAME_SPELLING_TOKEN_RE.finditer(str(tail or "")))
    groups: list[tuple[str, int, int]] = []
    index = 0
    repeated_first_match = re.match(
        r"\s*[,.;:!?]?\s*[A-Za-z][A-Za-z'.-]*\s+with\s+(?:a|an)\s+[A-Za-z]\s*,?\s*",
        str(tail or ""),
        re.IGNORECASE,
    )
    if repeated_first_match:
        while index < len(tokens) and tokens[index].start() < repeated_first_match.end():
            index += 1
    while index < len(tokens) and len(groups) < 4:
        parsed = parse_spelling_group(tokens, index)
        if not parsed:
            break
        spelled, start, end, next_index = parsed
        groups.append((spelled, start, end))
        index = next_index
    return groups


def corrected_name_from_spelling(raw_name: Any, groups: list[tuple[str, int, int]]) -> Optional[str]:
    raw = clean_name(raw_name)
    if not raw or not groups:
        return None
    raw_parts = raw.split()
    spelled_words = [title_from_spelled_letters(group[0]) for group in groups if group[0]]
    spelled_words = [word for word in spelled_words if word]
    if not spelled_words:
        return None
    if len(spelled_words) >= 2:
        corrected = " ".join(spelled_words)
    elif len(raw_parts) >= 2:
        corrected = " ".join(raw_parts[:-1] + [spelled_words[0]])
    else:
        corrected = spelled_words[0]
    corrected = clean_name(corrected)
    if not corrected or not person_like_name(corrected):
        return None
    if name_key(corrected) == name_key(raw):
        return None
    return corrected


def name_occurrence_is_addressee(transcript: str, start: int, end: int) -> bool:
    prefix_start = max(transcript.rfind(".", 0, start), transcript.rfind("!", 0, start), transcript.rfind("?", 0, start))
    prefix_start = 0 if prefix_start < 0 else prefix_start + 1
    phrase = transcript[prefix_start:end]
    return evidence_is_addressee_only(phrase)


def introduced_spelled_name_occurrences(transcript: str) -> list[tuple[str, int, int]]:
    occurrences: list[tuple[str, int, int]] = []
    for match in NAME_SPELLING_INTRODUCED_NAME_RE.finditer(transcript):
        raw = clean_name(match.group(1))
        if not raw or not person_like_name(raw):
            continue
        tokens = re.findall(r"[A-Za-z]+", raw.lower())
        if len(tokens) < 2 or tokens[0] in COMPACT_DOB_NAME_STOP_WORDS:
            continue
        if any(token in COMPACT_DOB_NAME_STOP_WORDS for token in tokens):
            continue
        occurrences.append((raw, match.start(1), match.end(1)))
    return occurrences


def patient_name_occurrences(transcript: str, patient_names: Optional[list[str]]) -> list[tuple[str, int, int]]:
    occurrences: list[tuple[str, int, int]] = []
    for name in patient_names or []:
        raw = clean_name(name)
        if not raw or not person_like_name(raw):
            continue
        pattern = re.compile(
            r"(?<![A-Za-z])" + r"\s+".join(re.escape(part) for part in raw.split()) + r"(?![A-Za-z])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(transcript):
            occurrences.append((raw, match.start(), match.end()))
    return occurrences


def find_name_part_after(transcript: str, part: str, cursor: int, max_gap: int = 80) -> Optional[re.Match[str]]:
    pattern = re.compile(r"(?<![A-Za-z])" + re.escape(part) + r"(?![A-Za-z])", re.IGNORECASE)
    window_end = min(len(transcript), cursor + max_gap)
    for match in pattern.finditer(transcript, cursor, window_end):
        gap = transcript[cursor : match.start()]
        if re.search(r"[.!?;:]", gap):
            return None
        return match
    return None


def interleaved_spelled_name_candidates(
    transcript: str,
    patient_names: Optional[list[str]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for name in patient_names or []:
        raw_name = clean_name(name)
        if not raw_name or not person_like_name(raw_name):
            continue
        raw_parts = raw_name.split()
        if len(raw_parts) < 2:
            continue
        first_pattern = re.compile(r"(?<![A-Za-z])" + re.escape(raw_parts[0]) + r"(?![A-Za-z])", re.IGNORECASE)
        for first_match in first_pattern.finditer(transcript):
            cursor = first_match.end()
            evidence_start = first_match.start()
            evidence_end = first_match.end()
            corrected_parts = list(raw_parts)
            spelled_count = 0
            failed = False
            for index, part in enumerate(raw_parts):
                if index == 0:
                    word_match = first_match
                else:
                    word_match = find_name_part_after(transcript, part, cursor)
                    if word_match is None:
                        failed = True
                        break
                    cursor = word_match.end()
                    evidence_end = word_match.end()

                tail = transcript[word_match.end() : min(len(transcript), word_match.end() + 80)]
                groups = parse_spelling_groups_from_tail(tail)
                if groups:
                    spelled_word = title_from_spelled_letters(groups[0][0])
                    if spelled_word:
                        corrected_parts[index] = spelled_word
                        spelled_count += 1
                        evidence_end = max(evidence_end, word_match.end() + groups[0][2])
                        cursor = evidence_end

            if failed or spelled_count < 1:
                continue
            corrected = clean_name(" ".join(corrected_parts))
            if not corrected or not person_like_name(corrected) or name_key(corrected) == name_key(raw_name):
                continue
            if name_occurrence_is_addressee(transcript, evidence_start, evidence_end):
                continue
            candidates.append(
                {
                    "raw": raw_name,
                    "value": corrected,
                    "evidence_text": transcript[evidence_start:evidence_end].strip(" ,.;:"),
                    "source": TRANSCRIPT_SPELLING_CORRECTED_SOURCE,
                    "caller_id_used": "",
                    "confidence": "high",
                }
            )
    return candidates


def extract_spelled_name_candidates(
    text: Any,
    patient_names: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    transcript = str(text or "")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for candidate in interleaved_spelled_name_candidates(transcript, patient_names):
        key = (
            name_key(candidate.get("raw")),
            name_key(candidate.get("value")),
            str(candidate.get("evidence_text") or "").lower(),
        )
        if key in seen:
            continue
        candidates.append(candidate)
        seen.add(key)

    occurrences = introduced_spelled_name_occurrences(transcript) + patient_name_occurrences(transcript, patient_names)
    occurrence_seen: set[tuple[str, int, int]] = set()

    for raw_name, start, end in sorted(occurrences, key=lambda item: (item[1], item[2])):
        occurrence_key = (name_key(raw_name), start, end)
        if occurrence_key in occurrence_seen:
            continue
        occurrence_seen.add(occurrence_key)
        if name_occurrence_is_addressee(transcript, start, end):
            continue
        tail = transcript[end : min(len(transcript), end + 120)]
        groups = parse_spelling_groups_from_tail(tail)
        corrected = corrected_name_from_spelling(raw_name, groups)
        if not corrected:
            continue
        evidence_end = end + max(group[2] for group in groups)
        evidence_text = transcript[start:evidence_end].strip(" ,.;:")
        key = (name_key(raw_name), name_key(corrected), evidence_text.lower())
        if key in seen:
            continue
        candidates.append(
            {
                "raw": raw_name,
                "value": corrected,
                "evidence_text": evidence_text,
                "source": TRANSCRIPT_SPELLING_CORRECTED_SOURCE,
                "caller_id_used": "",
                "confidence": "high",
            }
        )
        seen.add(key)
    return candidates


def extract_explicit_patient_name_candidates(text: Any) -> list[dict[str, Any]]:
    transcript = str(text or "")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in EXPLICIT_PATIENT_NAME_RE.finditer(transcript):
        raw_name = clean_fallback_person_name(match.group("name"))
        if not raw_name:
            continue
        evidence = name_fallback_evidence(transcript, match.start("prefix"), match.start("name"), raw_name)
        key = (name_key(raw_name), evidence.lower())
        if key in seen:
            continue
        candidates.append(name_fallback_candidate(raw_name, evidence, EXPLICIT_PATIENT_NAME_SOURCE))
        seen.add(key)
    return candidates


def relationship_prefix_is_caller_identity(transcript: str, prefix_start: int) -> bool:
    context = transcript[max(0, prefix_start - 24) : prefix_start]
    return bool(re.search(r"(?i)\b(?:i\s+am|i'm|this\s+is|my\s+name\s+is)\s*$", context))


def extract_relationship_name_candidates(text: Any) -> list[dict[str, Any]]:
    transcript = str(text or "")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_candidate(raw_name: Optional[str], evidence: str) -> None:
        if not raw_name or not person_like_name(raw_name):
            return
        key = (name_key(raw_name), evidence.lower())
        if key in seen:
            return
        candidates.append(
            {
                "raw": raw_name,
                "value": raw_name,
                "evidence_text": evidence,
                "source": RELATIONSHIP_SUBJECT_SOURCE,
                "caller_id_used": "",
                "confidence": "high",
            }
        )
        seen.add(key)

    for match in RELATIONSHIP_NAME_RE.finditer(transcript):
        if relationship_prefix_is_caller_identity(transcript, match.start("prefix")):
            continue
        raw_name = clean_fallback_person_name(match.group("name"))
        evidence = name_fallback_evidence(transcript, match.start("prefix"), match.start("name"), raw_name or "")
        add_candidate(raw_name, evidence)
    for match in PROXY_SUBJECT_NAME_RE.finditer(transcript):
        raw_name = clean_fallback_person_name(match.group("name"))
        evidence = name_fallback_evidence(transcript, match.start("prefix"), match.start("name"), raw_name or "")
        add_candidate(raw_name, evidence)
    for match in REVERSE_RELATIONSHIP_NAME_RE.finditer(transcript):
        raw_name = clean_fallback_person_name(match.group("name"))
        evidence = transcript[match.start("name") : match.end("relationship")].strip(" ,.;:")
        add_candidate(raw_name, evidence)
    return candidates


def name_fallback_candidate(raw_name: str, evidence: str, source: str) -> dict[str, Any]:
    return {
        "raw": raw_name,
        "value": raw_name,
        "evidence_text": evidence,
        "source": source,
        "caller_id_used": "",
        "confidence": "high",
    }


def name_fallback_evidence(transcript: str, prefix_start: int, name_start: int, raw_name: str) -> str:
    raw_tokens = [
        re.sub(r"[^a-z]+", "", token.lower())
        for token in re.findall(r"[A-Za-z][A-Za-z'.-]*", str(raw_name or ""))
    ]
    raw_tokens = [token for token in raw_tokens if token]
    token_count = len(raw_tokens)
    if token_count <= 0:
        return ""
    token_matches = list(re.finditer(r"[A-Za-z][A-Za-z'.-]*", transcript[name_start:]))
    token_values = [re.sub(r"[^a-z]+", "", match.group(0).lower()) for match in token_matches]
    for start_index in range(0, len(token_values) - token_count + 1):
        if token_values[start_index : start_index + token_count] == raw_tokens:
            evidence_end = name_start + token_matches[start_index + token_count - 1].end()
            return transcript[prefix_start:evidence_end].strip(" ,.;:")
    if len(token_matches) < token_count:
        return transcript[prefix_start : name_start + len(raw_name)].strip(" ,.;:")
    evidence_end = name_start + token_matches[token_count - 1].end()
    return transcript[prefix_start:evidence_end].strip(" ,.;:")


def extract_self_identification_name_candidates(text: Any) -> list[dict[str, Any]]:
    transcript = str(text or "")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in SELF_IDENTIFICATION_NAME_RE.finditer(transcript):
        raw_name = clean_fallback_person_name(match.group("name"), minimum_tokens=1)
        if not raw_name:
            continue
        evidence = name_fallback_evidence(transcript, match.start("prefix"), match.start("name"), raw_name)
        if evidence_is_addressee_only(evidence):
            continue
        key = (name_key(raw_name), evidence.lower())
        if key in seen:
            continue
        candidates.append(name_fallback_candidate(raw_name, evidence, SELF_IDENTIFICATION_NAME_SOURCE))
        seen.add(key)
    return candidates


def extract_subject_reference_name_candidates(text: Any) -> list[dict[str, Any]]:
    transcript = str(text or "")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in SUBJECT_REFERENCE_NAME_RE.finditer(transcript):
        raw_name = clean_fallback_person_name(match.group("name"))
        if not raw_name:
            continue
        if subject_reference_looks_like_topic(raw_name):
            continue
        evidence = name_fallback_evidence(transcript, match.start("prefix"), match.start("name"), raw_name)
        key = (name_key(raw_name), evidence.lower())
        if key in seen:
            continue
        candidates.append(name_fallback_candidate(raw_name, evidence, SUBJECT_REFERENCE_SOURCE))
        seen.add(key)
    return candidates


def spelling_evidence_supports_name(raw_name: Any, corrected_name: Any, evidence_text: Any) -> bool:
    raw = clean_name(raw_name)
    corrected = clean_name(corrected_name)
    if not raw or not corrected or name_key(raw) == name_key(corrected):
        return False
    for candidate in extract_spelled_name_candidates(str(evidence_text or ""), [raw]):
        if name_key(candidate.get("value")) == name_key(corrected):
            return True
    return False


def spelling_evidence_confirms_name(raw_name: Any, value: Any, evidence_text: Any) -> bool:
    raw = clean_name(raw_name)
    corrected = clean_name(value)
    if not raw or not corrected or name_key(raw) != name_key(corrected):
        return False
    return bool(NAME_SPELLING_EVIDENCE_RE.search(str(evidence_text or "")))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def caller_id_last_name_only_correction_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION", True)


def _last_names_strongly_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if rough_last_name_key(left) == rough_last_name_key(right):
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.74


def _last_names_match(left: str, right: str) -> bool:
    return _last_names_strongly_match(left, right) or SequenceMatcher(None, left, right).ratio() >= 0.58


def _first_names_match(left: str, right: str) -> bool:
    return left == right or SequenceMatcher(None, left, right).ratio() >= 0.58


def _caller_first_name_small_suffix_expansion(spoken_first: str, caller_first: str) -> bool:
    left = name_key(spoken_first)
    right = name_key(caller_first)
    if not left or not right or left == right or len(left) < 3:
        return False
    if not right.startswith(left):
        return False
    return 1 <= len(right) - len(left) <= 3


def _caller_id_name_orders(tokens: list[str]) -> list[tuple[str, str]]:
    if len(tokens) < 2:
        return []
    if len(tokens) == 2:
        return [(tokens[0], tokens[1]), (tokens[1], tokens[0])]
    if len(tokens) == 3 and len(tokens[1]) == 1:
        return [(tokens[0], tokens[2])]
    if len(tokens) == 3 and len(tokens[2]) == 1:
        return [(tokens[1], tokens[0])]
    return [(tokens[0], tokens[-1]), (tokens[-1], tokens[0])]


def caller_id_can_correct_spoken_name(
    spoken_name: Any,
    caller_id_name: Any,
    evidence_text: Any,
    corrected_name: Any = None,
) -> bool:
    spoken = name_alpha_tokens(spoken_name)
    caller = name_alpha_tokens(caller_id_name)
    if len(spoken) < 2 or len(caller) < 2:
        return False
    if not person_like_name(caller_id_name):
        return False
    if evidence_is_addressee_only(evidence_text):
        return False
    if not SELF_IDENTIFICATION_RE.search(str(evidence_text or "")):
        return False

    spoken_last = spoken[-1]
    corrected = name_alpha_tokens(corrected_name)
    caller_orders = _caller_id_name_orders(caller)

    for caller_first, caller_last in caller_orders:
        caller_first_expands_spoken = _caller_first_name_small_suffix_expansion(spoken[0], caller_first)
        if spoken[0] == caller_first and _last_names_match(spoken_last, caller_last):
            return True
        if (
            len(corrected) == 2
            and corrected == [caller_first, caller_last]
            and _first_names_match(spoken[0], caller_first)
            and _last_names_match(spoken_last, caller_last)
            and not caller_first_expands_spoken
        ):
            return True

    if not caller_id_last_name_only_correction_enabled():
        return False
    if len(corrected) != len(spoken):
        return False
    if corrected[:-1] != spoken[:-1]:
        return False
    return any(
        corrected[-1] == caller_last and _last_names_strongly_match(spoken_last, caller_last)
        for _caller_first, caller_last in caller_orders
    )


NAME_RESOLUTION_STATUS_PRIORITY = {
    "transcript_spelling_corrected": 600,
    "explicit_patient_name": 550,
    "relationship_subject": 500,
    "caller_id_spelling_corrected": 400,
    "subject_reference": 375,
    "self_identification": 350,
    "gemma_final": 300,
}


def best_name_resolution_candidate(values: list[tuple[str, int, str]]) -> tuple[str, int, str]:
    subject_reference_values = [item for item in values if item[2] == SUBJECT_REFERENCE_SOURCE]
    if subject_reference_values:
        matching_gemma_values = [
            item
            for item in values
            if item[2] == "gemma_final"
            and any(names_plausibly_match(item[0], subject_value) for subject_value, _score, _status in subject_reference_values)
        ]
        if matching_gemma_values:
            return max(matching_gemma_values, key=lambda item: (item[1], len(item[0])))

    return max(
        values,
        key=lambda item: (
            NAME_RESOLUTION_STATUS_PRIORITY.get(item[2], 250),
            item[1],
            len(item[0]),
        ),
    )


def grouped_name_resolution_candidates(
    values: list[tuple[str, int, str]]
) -> list[list[tuple[str, int, str]]]:
    groups: list[list[tuple[str, int, str]]] = []
    for value, score, status in values:
        for group in groups:
            if names_plausibly_match(group[0][0], value):
                group.append((value, score, status))
                break
        else:
            groups.append([(value, score, status)])
    return groups


def resolve_name_field(records: list[CandidateRecord], caller_id_name: Any = None) -> FieldResolution:
    if not records:
        return FieldResolution("name", None, None, "not_included", review_reasons=["no_gemma_candidate"])

    attributions, gemma_json, parakeet_json, clip_json = resolution_common_json(records)
    usable = [record for record in records if record.attribution.mapped]
    reasons: list[str] = []
    for record in records:
        reasons.extend(record.attribution.review_reasons)
    if not usable:
        return FieldResolution(
            "name",
            None,
            None,
            "not_included",
            needs_review=True,
            review_reasons=list(dict.fromkeys(reasons or ["evidence_not_mapped"])),
            attribution_json=attributions,
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    values: list[tuple[str, int, str]] = []
    for record in usable:
        raw = clean_name(record.gemma.get("raw"))
        value = clean_name(record.gemma.get("value") or raw)
        evidence_text = record.gemma.get("evidence_text") or record.attribution.evidence_text
        source = str(record.gemma.get("source") or "").lower()
        candidate_status = "gemma_final"
        score_bonus = 0
        if evidence_is_addressee_only(evidence_text):
            reasons.append("addressee_name_rejected")
            continue
        if source == TRANSCRIPT_SPELLING_CORRECTED_SOURCE:
            if raw and value and (
                spelling_evidence_supports_name(raw, value, evidence_text)
                or spelling_evidence_confirms_name(raw, value, evidence_text)
            ):
                candidate_status = "transcript_spelling_corrected"
                score_bonus += 50
            else:
                reasons.append("name_spelling_evidence_rejected")
                value = raw
        elif source == "caller_id_corrected":
            caller_id_for_record = record.gemma.get("caller_id_used") or caller_id_name
            if raw and value and caller_id_can_correct_spoken_name(raw, caller_id_for_record, evidence_text, value):
                candidate_status = "caller_id_spelling_corrected"
                score_bonus += 30
            else:
                reasons.append("caller_id_correction_disabled")
                value = raw
        elif source == EXPLICIT_PATIENT_NAME_SOURCE:
            candidate_status = EXPLICIT_PATIENT_NAME_SOURCE
            score_bonus += 40
        elif source in SUBJECT_NAME_SOURCES:
            candidate_status = source
            score_bonus += 40
        elif source == SUBJECT_REFERENCE_SOURCE:
            candidate_status = SUBJECT_REFERENCE_SOURCE
        elif source == SELF_IDENTIFICATION_NAME_SOURCE:
            candidate_status = SELF_IDENTIFICATION_NAME_SOURCE
            score_bonus += 20
        if value:
            score = name_candidate_score(value, evidence_text)
            if score < 0:
                reasons.append("name_not_person_like")
                continue
            values.append((value, score + score_bonus, candidate_status))

    if not values:
        return FieldResolution(
            "name",
            None,
            None,
            "not_included",
            needs_review=bool(reasons),
            review_reasons=list(dict.fromkeys(reasons)),
            attribution_json=attributions,
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    subject_reference_values = [item for item in values if item[2] == SUBJECT_REFERENCE_SOURCE]
    if subject_reference_values:
        corroborating_subject_values = [
            item
            for item in values
            if item[2] not in {SUBJECT_REFERENCE_SOURCE, SELF_IDENTIFICATION_NAME_SOURCE}
            and any(names_plausibly_match(item[0], subject_value) for subject_value, _score, _status in subject_reference_values)
        ]
        if not corroborating_subject_values:
            reasons.append("subject_reference_unconfirmed")

    if not values:
        return FieldResolution(
            "name",
            None,
            None,
            "not_included",
            needs_review=bool(reasons),
            review_reasons=list(dict.fromkeys(reasons)),
            attribution_json=attributions,
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    groups = grouped_name_resolution_candidates(values)
    same_status_conflict = any(
        len({name_key(value) for value, _score, candidate_status in values if candidate_status == status}) > 1
        for status in {candidate_status for _value, _score, candidate_status in values}
    )
    if len(groups) > 1 or same_status_conflict:
        reasons.append("multiple_field_candidates")

    final, _score, status = best_name_resolution_candidate(values)
    return FieldResolution(
        "name",
        final,
        name_key(final),
        status,
        needs_review=bool(reasons),
        review_reasons=list(dict.fromkeys(reasons)),
        attribution_json=attributions,
        gemma_json=gemma_json,
        parakeet_json=parakeet_json,
        clip_json=clip_json,
    )


def expand_two_digit_year(year: int) -> int:
    return 1900 + year if year >= COMPACT_DOB_YEAR_PIVOT else 2000 + year


def parse_compact_dob_digits(digits: str) -> Optional[date]:
    compact = re.sub(r"\D", "", str(digits or ""))
    candidates: list[date] = []

    def add(month: int, day: int, year: int) -> None:
        try:
            candidate = date(year, month, day)
        except ValueError:
            return
        if candidate not in candidates:
            candidates.append(candidate)

    if len(compact) == 4:
        add(int(compact[0]), int(compact[1]), expand_two_digit_year(int(compact[2:])))
    elif len(compact) == 5:
        add(int(compact[0]), int(compact[1:3]), expand_two_digit_year(int(compact[3:])))
        add(int(compact[:2]), int(compact[2]), expand_two_digit_year(int(compact[3:])))
    elif len(compact) == 6:
        add(int(compact[:2]), int(compact[2:4]), expand_two_digit_year(int(compact[4:])))
    elif len(compact) == 8:
        add(int(compact[:2]), int(compact[2:4]), int(compact[4:]))

    return candidates[0] if len(candidates) == 1 else None


def parse_number_at(tokens: list[str], index: int) -> list[tuple[int, int]]:
    if index >= len(tokens):
        return []

    token = tokens[index].lower()
    if token.isdigit():
        return [(int(token), index + 1)]
    if token in NUMBER_WORD_VALUES:
        return [(NUMBER_WORD_VALUES[token], index + 1)]
    if token not in TENS_WORD_VALUES:
        return []

    results: list[tuple[int, int]] = []
    if index + 1 < len(tokens):
        next_token = tokens[index + 1].lower()
        if next_token in NUMBER_WORD_VALUES and 0 < NUMBER_WORD_VALUES[next_token] < 10:
            results.append((TENS_WORD_VALUES[token] + NUMBER_WORD_VALUES[next_token], index + 2))
    results.append((TENS_WORD_VALUES[token], index + 1))
    return results


def parse_year_at(tokens: list[str], index: int) -> list[tuple[int, int]]:
    results: list[tuple[int, int]] = []
    for value, end_index in parse_number_at(tokens, index):
        if 0 <= value <= 99:
            results.append((expand_two_digit_year(value), end_index))
        elif 1900 <= value <= 2099:
            results.append((value, end_index))
        if value in {19, 20}:
            for suffix, suffix_end in parse_number_at(tokens, end_index):
                if 0 <= suffix <= 99:
                    results.append((value * 100 + suffix, suffix_end))
    return results


def parse_spoken_compact_dob_tokens(tokens: list[str]) -> Optional[date]:
    lowered = [token.lower() for token in tokens if token.lower() not in COMPACT_DOB_FILLER_WORDS]
    scored_candidates: dict[date, int] = {}
    for start_index in range(min(3, len(lowered))):
        for month, month_end in parse_number_at(lowered, start_index):
            if not 1 <= month <= 12:
                continue
            for day, day_end in parse_number_at(lowered, month_end):
                if not 1 <= day <= 31:
                    continue
                for year, year_end in parse_year_at(lowered, day_end):
                    try:
                        candidate = date(year, month, day)
                    except ValueError:
                        continue
                    score = year_end - start_index
                    if score > scored_candidates.get(candidate, 0):
                        scored_candidates[candidate] = score
    if not scored_candidates:
        return None
    best_score = max(scored_candidates.values())
    best = [candidate for candidate, score in scored_candidates.items() if score == best_score]
    return best[0] if len(best) == 1 else None


def parse_dob(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$", text)
    if match:
        month, day, year = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if year < 100:
            year = expand_two_digit_year(year)
        try:
            return date(year, month, day)
        except ValueError:
            return None
    compact = re.sub(r"\D", "", text)
    if 4 <= len(compact) <= 8:
        parsed = parse_compact_dob_digits(compact)
        if parsed:
            return parsed
    tokens = [match.group(0) for match in NUMBER_TOKEN_RE.finditer(text)]
    return parse_spoken_compact_dob_tokens(tokens)


def format_dob(value: date) -> str:
    return f"{value.month:02d}/{value.day:02d}/{value.year:04d}"


def dob_is_plausible(value: date, today: Optional[date] = None) -> bool:
    today = today or date.today()
    if value > today:
        return False
    age = today.year - value.year - ((today.month, today.day) < (value.month, value.day))
    return 0 <= age <= 120


def evidence_supports_day(evidence_text: Any, day: int) -> bool:
    evidence = str(evidence_text or "").lower()
    if re.search(rf"\b0?{day}(?:st|nd|rd|th)?\b", evidence):
        return True
    normalized_evidence = re.sub(r"[-_/]+", " ", evidence)
    for token in DAY_WORDS.get(day, set()):
        if re.search(r"\b" + re.escape(token).replace(r"\ ", r"\s+") + r"\b", normalized_evidence):
            return True
    return False


def _day_supported_in_evidence(evidence: str, day: int) -> bool:
    day_words = {
        1: {"1", "1st", "one", "first"},
        2: {"2", "2nd", "two", "second"},
        3: {"3", "3rd", "three", "third"},
        4: {"4", "4th", "four", "fourth"},
        5: {"5", "5th", "five", "fifth"},
        6: {"6", "6th", "six", "sixth"},
        7: {"7", "7th", "seven", "seventh"},
        8: {"8", "8th", "eight", "eighth"},
        9: {"9", "9th", "nine", "ninth"},
        10: {"10", "10th", "ten", "tenth"},
        11: {"11", "11th", "eleven", "eleventh"},
        12: {"12", "12th", "twelve", "twelfth"},
        13: {"13", "13th", "thirteen", "thirteenth"},
        14: {"14", "14th", "fourteen", "fourteenth"},
        15: {"15", "15th", "fifteen", "fifteenth"},
        16: {"16", "16th", "sixteen", "sixteenth"},
        17: {"17", "17th", "seventeen", "seventeenth"},
        18: {"18", "18th", "eighteen", "eighteenth"},
        19: {"19", "19th", "nineteen", "nineteenth"},
        20: {"20", "20th", "twenty", "twentieth"},
        21: {"21", "21st", "twenty one", "twenty first"},
        22: {"22", "22nd", "twenty two", "twenty second"},
        23: {"23", "23rd", "twenty three", "twenty third"},
        24: {"24", "24th", "twenty four", "twenty fourth"},
        25: {"25", "25th", "twenty five", "twenty fifth"},
        26: {"26", "26th", "twenty six", "twenty sixth"},
        27: {"27", "27th", "twenty seven", "twenty seventh"},
        28: {"28", "28th", "twenty eight", "twenty eighth"},
        29: {"29", "29th", "twenty nine", "twenty ninth"},
        30: {"30", "30th", "thirty", "thirtieth"},
        31: {"31", "31st", "thirty one", "thirty first"},
    }
    tokens = set(re.findall(r"\b\d{1,2}(?:st|nd|rd|th)?\b|\b[a-z]+(?:\s+[a-z]+)?\b", evidence.lower()))
    compact = normalize_key(evidence)
    for variant in day_words.get(day, set()):
        if variant in tokens or normalize_key(variant) in compact:
            return True
    return False


def evidence_supports_dob(evidence_text: Any, normalized: str, raw: Any = None) -> bool:
    evidence = str(evidence_text or "")
    if raw and normalize_key(raw) and normalize_key(raw) in normalize_key(evidence):
        return True
    normalized_digits = re.sub(r"\D", "", normalized)
    evidence_digits = digits_from_text(evidence)
    if normalized_digits and normalized_digits in evidence_digits:
        return True
    parsed = parse_dob(normalized)
    if parsed:
        words = set(re.findall(r"[a-z]+", evidence.lower()))
        month_names = {name for name, number in MONTHS.items() if number == parsed.month}
        year_supported = str(parsed.year) in evidence or str(parsed.year)[-2:] in re.findall(r"\b\d{2}\b", evidence)
        if words & month_names and year_supported and _day_supported_in_evidence(evidence, parsed.day):
            return True
    return False


def compact_dob_evidence_has_phone_context(record: CandidateRecord) -> bool:
    source = str(record.gemma.get("source") or "").lower()
    if not source.startswith("compact_dob"):
        return False
    evidence = str(record.attribution.matched_text or record.gemma.get("evidence_text") or "")
    if DOB_CUE_RE.search(evidence):
        return False
    return bool(COMPACT_DOB_PHONE_CONTEXT_RE.search(evidence))


def compact_dob_candidate(raw: str, evidence_text: str, source: str) -> Optional[dict[str, Any]]:
    parsed = parse_dob(raw)
    if not parsed or not dob_is_plausible(parsed):
        return None
    return {
        "raw": raw.strip(" ,.;:"),
        "normalized": format_dob(parsed),
        "evidence_text": evidence_text.strip(" ,.;:"),
        "source": source,
    }


def compact_dob_prefix_has_phone_context(prefix: str) -> bool:
    if COMPACT_DOB_PHONE_CONTEXT_RE.search(prefix):
        return True
    return bool(re.search(r"(?:^|\D)\(?\d{3}\)?[\s.-]*$", prefix))


def spoken_compact_candidate_from_window(window: str, offset: int) -> Optional[tuple[str, int, int]]:
    token_matches = list(NUMBER_TOKEN_RE.finditer(window))
    if not token_matches:
        return None

    lowered = [match.group(0).lower() for match in token_matches]
    matches: list[tuple[int, str, int, int, date]] = []
    for start_index in range(len(token_matches)):
        if lowered[start_index] in COMPACT_DOB_FILLER_WORDS:
            continue
        for month, month_end in parse_number_at(lowered, start_index):
            if not 1 <= month <= 12:
                continue
            for day, day_end in parse_number_at(lowered, month_end):
                if not 1 <= day <= 31:
                    continue
                for year, year_end in parse_year_at(lowered, day_end):
                    try:
                        parsed = date(year, month, day)
                    except ValueError:
                        continue
                    if not dob_is_plausible(parsed):
                        continue
                    start = token_matches[start_index].start()
                    end = token_matches[year_end - 1].end()
                    score = year_end - start_index
                    matches.append((score, window[start:end], offset + start, offset + end, parsed))
    if not matches:
        return None
    best_score = max(score for score, *_rest in matches)
    best = [match for match in matches if match[0] == best_score]
    best_dates = {match[4] for match in best}
    if len(best_dates) != 1:
        return None
    _score, raw, start, end, _parsed = best[0]
    return raw, start, end


def filler_compact_candidate_from_window(window: str, offset: int) -> Optional[tuple[str, int, int]]:
    matches: list[tuple[str, int, int, date]] = []
    for match in COMPACT_DOB_FILLER_DIGIT_RE.finditer(window):
        raw = match.group(0)
        parsed = parse_dob(raw)
        if not parsed or not dob_is_plausible(parsed):
            continue
        matches.append((raw, offset + match.start(), offset + match.end(), parsed))
    if not matches:
        return None
    dates = {match[3] for match in matches}
    if len(dates) != 1:
        return None
    raw, start, end, _parsed = matches[0]
    return raw, start, end


def add_compact_candidate(
    candidates: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    raw: str,
    evidence_text: str,
    source: str,
) -> None:
    candidate = compact_dob_candidate(raw, evidence_text, source)
    if not candidate:
        return
    key = (candidate["normalized"], candidate["evidence_text"].lower())
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def introduced_patient_names_for_compact_dob(text: Any) -> list[str]:
    names: list[str] = []
    for match in COMPACT_DOB_INTRODUCED_NAME_RE.finditer(str(text or "")):
        cleaned = clean_name(match.group(1))
        if not cleaned:
            continue
        tokens = re.findall(r"[A-Za-z]+", cleaned.lower())
        if len(tokens) < 2:
            continue
        if tokens[0] in COMPACT_DOB_NAME_STOP_WORDS:
            continue
        if any(token in COMPACT_DOB_NAME_STOP_WORDS for token in tokens):
            continue
        if not person_like_name(cleaned):
            continue
        if cleaned not in names:
            names.append(cleaned)
    return names


def patient_names_before_filler_compact_dob(text: Any) -> list[str]:
    transcript = str(text or "")
    names: list[str] = []
    for match in COMPACT_DOB_FILLER_DIGIT_RE.finditer(transcript):
        prefix = transcript[max(0, match.start() - 80) : match.start()]
        if DOB_CUE_RE.search(prefix):
            continue
        if compact_dob_prefix_has_phone_context(prefix):
            continue
        name_match = re.search(
            r"([A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){1,3})\s*,?\s*$",
            prefix,
        )
        if not name_match:
            continue
        cleaned = clean_name(name_match.group(1))
        if not cleaned or not person_like_name(cleaned):
            continue
        tokens = re.findall(r"[A-Za-z]+", cleaned.lower())
        if tokens[0] in COMPACT_DOB_NAME_STOP_WORDS:
            continue
        if any(token in COMPACT_DOB_NAME_STOP_WORDS for token in tokens):
            continue
        if cleaned not in names:
            names.append(cleaned)
    return names


def extract_compact_dob_candidates(
    text: Any,
    patient_names: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    transcript = str(text or "")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for cue_match in DOB_CUE_RE.finditer(transcript):
        window_start = cue_match.end()
        window_end = min(len(transcript), window_start + 72)
        window = transcript[window_start:window_end]
        for raw_match in COMPACT_DOB_DIGIT_RE.finditer(window):
            raw = raw_match.group(1).strip(" ,.;:")
            if not 4 <= len(re.sub(r"\D", "", raw)) <= 8:
                continue
            evidence = transcript[cue_match.start() : window_start + raw_match.end()]
            add_compact_candidate(candidates, seen, raw, evidence, "compact_dob_cue_fallback")

        filler = filler_compact_candidate_from_window(window, window_start)
        if filler:
            raw, _raw_start, raw_end = filler
            evidence = transcript[cue_match.start() : raw_end]
            add_compact_candidate(candidates, seen, raw, evidence, "compact_dob_cue_fallback")

        spoken = spoken_compact_candidate_from_window(window, window_start)
        if spoken:
            raw, _raw_start, raw_end = spoken
            evidence = transcript[cue_match.start() : raw_end]
            add_compact_candidate(candidates, seen, raw, evidence, "compact_dob_cue_fallback")

    all_patient_names = list(patient_names or [])
    for name in introduced_patient_names_for_compact_dob(transcript):
        if name not in all_patient_names:
            all_patient_names.append(name)
    for name in patient_names_before_filler_compact_dob(transcript):
        if name not in all_patient_names:
            all_patient_names.append(name)

    for name in all_patient_names:
        normalized_name = re.sub(r"\s+", " ", str(name or "").strip())
        if not normalized_name:
            continue
        name_pattern = re.compile(
            r"(?<![A-Za-z])" + r"\s+".join(re.escape(part) for part in normalized_name.split()) + r"(?![A-Za-z])",
            re.IGNORECASE,
        )
        for name_match in name_pattern.finditer(transcript):
            window_start = name_match.end()
            window_end = min(len(transcript), window_start + 72)
            window = transcript[window_start:window_end]
            raw_match = COMPACT_DOB_DIGIT_RE.search(window)
            filler_match = COMPACT_DOB_FILLER_DIGIT_RE.search(window)
            if filler_match and (not raw_match or filler_match.start() < raw_match.start()):
                raw = filler_match.group(0).strip(" ,.;:")
                prefix = window[: filler_match.start()].lower()
                evidence = transcript[name_match.start() : window_start + filler_match.end()]
                if compact_dob_prefix_has_phone_context(prefix):
                    continue
                add_compact_candidate(candidates, seen, raw, evidence, "compact_dob_name_fallback")
                continue
            if not raw_match:
                continue
            prefix = window[: raw_match.start()].lower()
            if compact_dob_prefix_has_phone_context(prefix):
                continue
            raw = raw_match.group(1).strip(" ,.;:")
            if not 4 <= len(re.sub(r"\D", "", raw)) <= 8:
                continue
            evidence = transcript[name_match.start() : window_start + raw_match.end()]
            add_compact_candidate(candidates, seen, raw, evidence, "compact_dob_name_fallback")

    return candidates


def extract_dobs_from_text(text: Any) -> set[str]:
    transcript = str(text or "")
    found: set[str] = set()
    for match in re.finditer(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", transcript):
        parsed = parse_dob(match.group(0))
        if parsed:
            found.add(format_dob(parsed))
    for match in COMPACT_DOB_DIGIT_RE.finditer(transcript):
        raw = match.group(1)
        if not 4 <= len(re.sub(r"\D", "", raw)) <= 8:
            continue
        prefix = transcript[max(0, match.start() - 16) : match.start()]
        if re.search(r"(?i)(?:^|[^A-Za-z])(?:to|too|two)\s*$", prefix):
            parsed_with_leading_two = parse_dob("2" + raw)
            if parsed_with_leading_two:
                found.add(format_dob(parsed_with_leading_two))
                continue
        parsed = parse_dob(raw)
        if parsed:
            found.add(format_dob(parsed))
    for match in re.finditer(r"\b\d{8}\b", transcript):
        parsed = parse_dob(match.group(0))
        if parsed:
            found.add(format_dob(parsed))
    return found


def candidate_is_corrected(candidate: dict[str, Any]) -> bool:
    if candidate.get("correction") is True or candidate.get("is_corrected") is True:
        return True
    return "corrected" in str(candidate.get("confidence") or "").lower()


def plausible_parakeet_dobs(records: list[CandidateRecord], today: Optional[date] = None) -> set[str]:
    dobs: set[str] = set()
    for record in records:
        if not record.parakeet or record.parakeet.error:
            continue
        for value in extract_dobs_from_text(record.parakeet.text):
            parsed = parse_dob(value)
            if parsed and dob_is_plausible(parsed, today):
                dobs.add(format_dob(parsed))
    return dobs


def resolve_dob_field(records: list[CandidateRecord], today: Optional[date] = None) -> FieldResolution:
    if not records:
        return FieldResolution("dob", None, None, "not_included", review_reasons=["no_gemma_candidate"])

    attributions, gemma_json, parakeet_json, clip_json = resolution_common_json(records)
    usable = [record for record in records if record.attribution.mapped]
    reasons: list[str] = []
    for record in records:
        reasons.extend(record.attribution.review_reasons)
    if not usable:
        return FieldResolution(
            "dob",
            None,
            None,
            "not_included",
            needs_review=True,
            review_reasons=list(dict.fromkeys(reasons or ["evidence_not_mapped"])),
            attribution_json=attributions,
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    valid_records: list[tuple[CandidateRecord, str]] = []
    for record in usable:
        normalized = str(record.gemma.get("normalized") or "").strip()
        if compact_dob_evidence_has_phone_context(record):
            reasons.append("dob_phone_context_rejected")
            continue
        parsed = parse_dob(normalized)
        if not parsed or not dob_is_plausible(parsed, today):
            reasons.append("dob_implausible")
            continue
        if not evidence_supports_dob(record.attribution.matched_text or record.gemma.get("evidence_text"), normalized, record.gemma.get("raw")):
            reasons.append("dob_implausible")
            continue
        valid_records.append((record, format_dob(parsed)))

    if not valid_records:
        return FieldResolution(
            "dob",
            None,
            None,
            "not_included",
            needs_review=True,
            review_reasons=list(dict.fromkeys(reasons)),
            attribution_json=attributions,
            gemma_json=gemma_json,
            parakeet_json=parakeet_json,
            clip_json=clip_json,
        )

    dates = {dob for _, dob in valid_records}
    parakeet_dobs = plausible_parakeet_dobs(usable, today)
    if len(parakeet_dobs) > 1:
        reasons.append("multiple_parakeet_dobs")

    if len(dates) > 1:
        corrected = [(record, dob) for record, dob in valid_records if candidate_is_corrected(record.gemma)]
        if len(corrected) == 1:
            final = corrected[0][1]
        else:
            reasons.append("multiple_field_candidates")
            if len(parakeet_dobs) == 1:
                reasons.append("dob_parakeet_audit_disagreement")
            return FieldResolution(
                "dob",
                None,
                None,
                "ambiguous",
                needs_review=True,
                review_reasons=list(dict.fromkeys(reasons)),
                attribution_json=attributions,
                gemma_json=gemma_json,
                parakeet_json=parakeet_json,
                clip_json=clip_json,
            )
    else:
        final = next(iter(dates))

    if len(parakeet_dobs) == 1:
        parakeet_final = next(iter(parakeet_dobs))
        if parakeet_final != final:
            reasons.append("dob_parakeet_audit_disagreement")

    return FieldResolution(
        "dob",
        final,
        re.sub(r"\D", "", final),
        "gemma_final",
        needs_review=bool(reasons),
        review_reasons=list(dict.fromkeys(reasons)),
        attribution_json=attributions,
        gemma_json=gemma_json,
        parakeet_json=parakeet_json,
        clip_json=clip_json,
    )
