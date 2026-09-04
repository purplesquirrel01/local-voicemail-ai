"""DOB verification compatibility surface."""

from __future__ import annotations

from verification import (
    compact_dob_candidate,
    dob_is_plausible,
    evidence_supports_dob,
    evidence_supports_day,
    expand_two_digit_year,
    extract_compact_dob_candidates,
    extract_dobs_from_text,
    format_dob,
    parse_compact_dob_digits,
    parse_dob,
    resolve_dob_field,
)

__all__ = [
    "compact_dob_candidate",
    "dob_is_plausible",
    "evidence_supports_dob",
    "evidence_supports_day",
    "expand_two_digit_year",
    "extract_compact_dob_candidates",
    "extract_dobs_from_text",
    "format_dob",
    "parse_compact_dob_digits",
    "parse_dob",
    "resolve_dob_field",
]
