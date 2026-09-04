"""Phone verification compatibility surface."""

from __future__ import annotations

from verification import (
    digits_from_text,
    extract_numbers_from_text,
    format_phone_digits,
    normalize_phone_candidate,
    phone_record_blocks_parakeet_override,
    record_whisper_phone_values,
    resolve_phone_field,
    unique_valid_phone_digits,
    whisper_numbers_from_records,
    whisper_span_consensus,
)

__all__ = [
    "digits_from_text",
    "extract_numbers_from_text",
    "format_phone_digits",
    "normalize_phone_candidate",
    "phone_record_blocks_parakeet_override",
    "record_whisper_phone_values",
    "resolve_phone_field",
    "unique_valid_phone_digits",
    "whisper_numbers_from_records",
    "whisper_span_consensus",
]
