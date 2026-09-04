"""Transcript correction helpers for verified voicemail fields."""

from __future__ import annotations

import re
from typing import Any, Optional

from verification import format_dob, format_phone_digits, parse_dob


PHONE_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:"
    r"(?:\+?1[\s().-]*)?(?:\d[\s().-]*){9,11}\d"
    r"|"
    r"(?:\+?1[\s.-]*)?"
    r"\(?[A-Za-z0-9]{3}\)?[\s.-]*"
    r"[A-Za-z0-9]{3}[\s.-]*"
    r"[A-Za-z0-9]{4}"
    r")"
    r"(?![A-Za-z0-9])"
)

PHONE_TRANSCRIPT_CORRECTION_STATUSES = {
    "verified",
    "parakeet_override",
    "whisper_caller_id_verified",
    "whisper_span_fallback",
}
DOB_TRANSCRIPT_CORRECTION_STATUSES = {"parakeet_override"}

DOB_TEXT_RE = re.compile(
    r"(?<!\d)(?:\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{2,4}|\d{6,8})(?!\d)"
)


def optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def phone_alnum(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()
    if len(text) == 11 and text.startswith("1"):
        text = text[1:]
    return text


def loose_phone_text_matches(candidate: Any, normalized_digits: Any) -> bool:
    normalized = re.sub(r"\D", "", str(normalized_digits or ""))
    if len(normalized) == 11 and normalized.startswith("1"):
        normalized = normalized[1:]
    if len(normalized) != 10:
        return False

    candidate_text = str(candidate or "")
    candidate_digits = re.sub(r"\D", "", candidate_text)
    if len(candidate_digits) == 11 and candidate_digits.startswith("1"):
        candidate_digits = candidate_digits[1:]
    if candidate_digits == normalized:
        return True
    if candidate_digits == normalized[:3] + normalized:
        return True
    if len(candidate_digits) > len(normalized) and len(candidate_digits) <= len(normalized) + 2:
        extra = candidate_digits[len(normalized) :]
        if candidate_digits.startswith(normalized) and extra and set(extra) == {normalized[-1]}:
            return True
    if len(candidate_digits) >= 7 and (normalized.startswith(candidate_digits) or candidate_digits in normalized):
        return True
    if len(candidate_digits) > len(normalized) and len(candidate_digits) <= 14 and normalized in candidate_digits:
        return True

    candidate_key = phone_alnum(candidate_text)
    if len(candidate_key) != 10:
        return False

    positional_matches = sum(1 for left, right in zip(candidate_key, normalized) if left == right)
    digit_matches = sum(
        1
        for left, right in zip(candidate_key, normalized)
        if left.isdigit() and left == right
    )
    return positional_matches >= 8 or digit_matches >= 8


def repeated_area_code_prefix_matches(
    words: list[dict[str, Any]],
    prefix_index: int,
    span_end: int,
    normalized_digits: str,
) -> bool:
    normalized = re.sub(r"\D", "", str(normalized_digits or ""))
    if len(normalized) == 11 and normalized.startswith("1"):
        normalized = normalized[1:]
    if len(normalized) != 10 or prefix_index < 0:
        return False

    prefix_digits = re.sub(r"\D", "", str(words[prefix_index].get("word", "")))
    if prefix_digits != normalized[:3]:
        return False

    candidate = " ".join(str(words[i].get("word", "")) for i in range(prefix_index, span_end + 1))
    candidate_digits = re.sub(r"\D", "", candidate)
    if len(candidate_digits) == 11 and candidate_digits.startswith("1"):
        candidate_digits = candidate_digits[1:]
    return candidate_digits in {normalized, normalized[:3] + normalized}


def replace_phone_text_in_transcript(transcript: str, normalized_digits: str, formatted_number: str) -> tuple[str, int]:
    replacements = 0

    def replace_match(match: re.Match[str]) -> str:
        nonlocal replacements
        candidate = match.group(0)
        if not loose_phone_text_matches(candidate, normalized_digits):
            return candidate
        replacements += 1
        return formatted_number

    return PHONE_TEXT_RE.sub(replace_match, transcript), replacements


def attributed_word_ranges(
    audit_row: dict[str, Any],
    field_name: Optional[str] = None,
    word_count: Optional[int] = None,
    require_field_name: bool = False,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for attribution in audit_row.get("attribution_json") or []:
        if not isinstance(attribution, dict):
            continue
        attribution_field = attribution.get("field_name")
        if field_name:
            if require_field_name and attribution_field != field_name:
                continue
            if not require_field_name and attribution_field and attribution_field != field_name:
                continue
        try:
            start_index = int(attribution.get("word_start"))
            end_index = int(attribution.get("word_end"))
        except (TypeError, ValueError):
            continue
        if start_index > end_index:
            continue
        if word_count is not None:
            start_index = max(0, start_index)
            end_index = min(word_count - 1, end_index)
            if start_index > end_index:
                continue
        ranges.append((start_index, end_index))
    return ranges


def phone_like_values_in_text(text: Any) -> list[str]:
    values: list[str] = []
    for match in PHONE_TEXT_RE.finditer(str(text or "")):
        candidate = match.group(0)
        digits = re.sub(r"\D", "", candidate)
        alnum = phone_alnum(candidate)
        if len(digits) >= 7 and len(alnum) == 10:
            values.append(candidate)
    return values


def phone_word_range_supports_expected_value(
    words: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    normalized_digits: str,
) -> bool:
    range_text = " ".join(str(words[i].get("word", "")) for i in range(start_index, end_index + 1))
    values = phone_like_values_in_text(range_text)
    if len(values) > 1:
        return False
    if values and not loose_phone_text_matches(values[0], normalized_digits):
        return False
    return True


def corrected_phone_word_span(
    words: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    normalized_digits: str,
) -> Optional[tuple[int, int]]:
    if not words:
        return None

    start_index = max(0, start_index)
    end_index = min(len(words) - 1, end_index)
    if start_index > end_index:
        return None

    best: Optional[tuple[int, int, int]] = None
    max_width = min(5, end_index - start_index + 1)
    for width in range(1, max_width + 1):
        for index in range(start_index, end_index - width + 2):
            candidate = " ".join(str(words[i].get("word", "")) for i in range(index, index + width))
            if not loose_phone_text_matches(candidate, normalized_digits):
                continue
            score = len(re.sub(r"\D", "", candidate))
            current = (index, index + width - 1, score)
            if best is None or width < best[1] - best[0] + 1 or score > best[2]:
                best = current
        if best is not None:
            break

    if best is None:
        return None

    span_start, span_end, _score = best
    if span_start > start_index and repeated_area_code_prefix_matches(
        words,
        span_start - 1,
        span_end,
        normalized_digits,
    ):
        return span_start - 1, span_end

    return span_start, span_end


def transcript_word_char_spans(
    transcript: str,
    words: list[dict[str, Any]],
) -> list[Optional[tuple[int, int]]]:
    spans: list[Optional[tuple[int, int]]] = []
    cursor = 0
    for item in words:
        token = str(item.get("word", "") or "")
        if not token:
            spans.append(None)
            continue
        start = transcript.find(token, cursor)
        if start < 0:
            stripped = token.strip(" \t\r\n")
            start = transcript.find(stripped, cursor) if stripped else -1
            token = stripped
        if start < 0:
            spans.append(None)
            continue
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


def phone_span_replacement_text(original_text: str, formatted_number: str) -> str:
    trailing = re.search(r"([^\w\s]+)$", original_text or "")
    if trailing and not formatted_number.endswith(trailing.group(1)):
        return formatted_number + trailing.group(1)
    return formatted_number


def replace_attributed_phone_text_in_transcript(
    transcript: str,
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    normalized_digits: str,
    formatted_number: str,
) -> tuple[str, int]:
    if not transcript or not words:
        return transcript, 0

    field_name = str(audit_row.get("field_name") or "")
    ranges = attributed_word_ranges(
        audit_row,
        field_name=field_name,
        word_count=len(words),
        require_field_name=True,
    )
    if not ranges:
        return transcript, 0

    char_spans = transcript_word_char_spans(transcript, words)
    replacements: list[tuple[int, int, str]] = []
    for start_index, end_index in ranges:
        if not phone_word_range_supports_expected_value(words, start_index, end_index, normalized_digits):
            continue
        span = corrected_phone_word_span(words, start_index, end_index, normalized_digits)
        if span is None:
            continue

        span_start, span_end = span
        start_span = char_spans[span_start]
        end_span = char_spans[span_end]
        if start_span is None or end_span is None:
            continue

        char_start = start_span[0]
        char_end = end_span[1]
        original_text = transcript[char_start:char_end]
        if not loose_phone_text_matches(original_text, normalized_digits):
            continue
        replacements.append((char_start, char_end, phone_span_replacement_text(original_text, formatted_number)))

    if not replacements:
        return transcript, 0

    replacements.sort(key=lambda item: (item[0], item[1]))
    non_overlapping: list[tuple[int, int, str]] = []
    previous_end = -1
    for char_start, char_end, replacement in replacements:
        if char_start < previous_end:
            continue
        non_overlapping.append((char_start, char_end, replacement))
        previous_end = char_end

    corrected = transcript
    for char_start, char_end, replacement in reversed(non_overlapping):
        corrected = corrected[:char_start] + replacement + corrected[char_end:]
    return corrected, len(non_overlapping)


def phone_row_sort_key(audit_row: dict[str, Any]) -> int:
    field_name = str(audit_row.get("field_name") or "")
    ranges = attributed_word_ranges(audit_row, field_name=field_name, require_field_name=True)
    if not ranges:
        return -1
    return min(start_index for start_index, _end_index in ranges)


def merge_corrected_phone_words(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    normalized_digits: str,
    formatted_number: str,
) -> tuple[list[dict[str, Any]], bool]:
    if not words:
        return words, False

    field_name = str(audit_row.get("field_name") or "")
    ranges = attributed_word_ranges(
        audit_row,
        field_name=field_name,
        word_count=len(words),
        require_field_name=True,
    )
    if not ranges:
        return words, False

    for start_index, end_index in ranges:
        if not phone_word_range_supports_expected_value(words, start_index, end_index, normalized_digits):
            continue
        span = corrected_phone_word_span(words, start_index, end_index, normalized_digits)
        if span is None:
            continue

        span_start, span_end = span
        first = words[span_start]
        last = words[span_end]
        merged = dict(first)
        merged["word"] = formatted_number
        if last.get("end") is not None:
            merged["end"] = last.get("end")
        return words[:span_start] + [merged] + words[span_end + 1 :], True

    return words, False


def dob_values_from_audit_row(audit_row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in audit_row.get("gemma_json") or []:
        if not isinstance(item, dict):
            continue
        for key in ("normalized", "raw", "value", "formatted"):
            parsed = parse_dob(item.get(key))
            if parsed:
                values.add(format_dob(parsed))
    return values


def dob_text_matches(candidate: Any, accepted_dobs: set[str]) -> bool:
    parsed = parse_dob(candidate)
    return bool(parsed and format_dob(parsed) in accepted_dobs)


def dob_span_replacement_text(original_text: str, final_dob: str) -> str:
    trailing = re.search(r"([^\w\s]+)$", original_text or "")
    if trailing and not final_dob.endswith(trailing.group(1)):
        return final_dob + trailing.group(1)
    return final_dob


def replace_attributed_dob_text_in_transcript(
    transcript: str,
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    final_dob: str,
) -> tuple[str, int]:
    if not transcript or not words or not final_dob:
        return transcript, 0

    accepted_dobs = dob_values_from_audit_row(audit_row)
    if not accepted_dobs:
        return transcript, 0

    ranges = attributed_word_ranges(audit_row, field_name="dob", word_count=len(words))
    if not ranges:
        return transcript, 0

    char_spans = transcript_word_char_spans(transcript, words)
    replacements: list[tuple[int, int, str]] = []
    for start_index, end_index in ranges:
        start_span = char_spans[start_index]
        end_span = char_spans[end_index]
        if start_span is None or end_span is None:
            continue

        char_start = start_span[0]
        char_end = end_span[1]
        span_text = transcript[char_start:char_end]
        matches = [
            match
            for match in DOB_TEXT_RE.finditer(span_text)
            if dob_text_matches(match.group(0), accepted_dobs)
        ]
        if len(matches) != 1:
            continue
        match = matches[0]
        replacements.append(
            (
                char_start + match.start(),
                char_start + match.end(),
                dob_span_replacement_text(match.group(0), final_dob),
            )
        )

    if not replacements:
        return transcript, 0

    replacements.sort(key=lambda item: (item[0], item[1]))
    non_overlapping: list[tuple[int, int, str]] = []
    previous_end = -1
    for char_start, char_end, replacement in replacements:
        if char_start < previous_end:
            continue
        non_overlapping.append((char_start, char_end, replacement))
        previous_end = char_end

    corrected = transcript
    for char_start, char_end, replacement in reversed(non_overlapping):
        corrected = corrected[:char_start] + replacement + corrected[char_end:]
    return corrected, len(non_overlapping)


def corrected_dob_word_span(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    accepted_dobs: set[str],
) -> Optional[tuple[int, int]]:
    if not words or not accepted_dobs:
        return None

    ranges = attributed_word_ranges(audit_row, field_name="dob", word_count=len(words))
    for start_index, end_index in ranges:
        max_width = min(3, end_index - start_index + 1)
        for width in range(1, max_width + 1):
            for index in range(start_index, end_index - width + 2):
                candidate = " ".join(str(words[i].get("word", "")) for i in range(index, index + width))
                if dob_text_matches(candidate, accepted_dobs):
                    return index, index + width - 1
    return None


def merge_corrected_dob_words(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    final_dob: str,
) -> tuple[list[dict[str, Any]], bool]:
    accepted_dobs = dob_values_from_audit_row(audit_row)
    span = corrected_dob_word_span(words, audit_row, accepted_dobs)
    if span is None:
        return words, False

    span_start, span_end = span
    original_text = " ".join(str(item.get("word", "")) for item in words[span_start : span_end + 1])
    first = words[span_start]
    last = words[span_end]
    merged = dict(first)
    merged["word"] = dob_span_replacement_text(original_text, final_dob)
    if last.get("end") is not None:
        merged["end"] = last.get("end")
    return words[:span_start] + [merged] + words[span_end + 1 :], True


def replacement_name_values(audit_row: dict[str, Any]) -> list[tuple[str, str]]:
    if audit_row.get("field_name") != "name":
        return []
    status = str(audit_row.get("status") or "")
    if status not in {"caller_id_spelling_corrected", "transcript_spelling_corrected"}:
        return []

    final_value = re.sub(r"\s+", " ", str(audit_row.get("final_value") or "")).strip()
    if not final_value:
        return []

    gemma_items = [item for item in audit_row.get("gemma_json") or [] if isinstance(item, dict)]
    replacements: list[tuple[str, str]] = []
    for item in gemma_items:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip().lower()
        raw = re.sub(r"\s+", " ", str(item.get("raw") or "")).strip()
        value = re.sub(r"\s+", " ", str(item.get("value") or "")).strip()
        caller_id_used = re.sub(r"\s+", " ", str(item.get("caller_id_used") or "")).strip()
        if status == "caller_id_spelling_corrected":
            if source:
                if source != "caller_id_corrected":
                    continue
                corrected_value = value
            elif len(gemma_items) != 1:
                continue
            else:
                corrected_value = caller_id_used or value
            if corrected_value != final_value:
                continue
        elif status == "transcript_spelling_corrected":
            if source:
                if source != "transcript_spelling_corrected":
                    continue
            elif len(gemma_items) != 1:
                continue
            if value != final_value:
                continue
        if raw and raw != final_value:
            replacements.append((raw, final_value))
        elif value and value != final_value:
            replacements.append((value, final_value))

    return list(dict.fromkeys(replacements))


def replace_name_text_in_transcript(transcript: str, raw_name: str, final_name: str) -> tuple[str, int]:
    raw_name = re.sub(r"\s+", " ", raw_name or "").strip()
    final_name = re.sub(r"\s+", " ", final_name or "").strip()
    if not raw_name or not final_name or raw_name == final_name:
        return transcript, 0

    token_pairs = changed_name_token_pairs(raw_name, final_name)
    if token_pairs:
        return replace_changed_name_tokens_in_transcript(transcript, token_pairs)

    pattern = name_text_pattern(raw_name)
    return pattern.subn(final_name, transcript)


def name_text_pattern(value: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z])" + re.escape(value) + r"(?![A-Za-z])", re.IGNORECASE)


def name_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"\s+", str(value or "").strip()) if token]


def normalized_name_token(value: Any) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").lower())


def changed_name_token_pairs(raw_name: str, final_name: str) -> list[tuple[str, str]]:
    raw_tokens = name_tokens(raw_name)
    final_tokens = name_tokens(final_name)
    if len(raw_tokens) != len(final_tokens):
        return []

    pairs: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for raw_token, final_token in zip(raw_tokens, final_tokens):
        raw_key = normalized_name_token(raw_token)
        final_key = normalized_name_token(final_token)
        if not raw_key or not final_key or raw_key == final_key:
            continue
        previous = seen.get(raw_key)
        if previous is not None and normalized_name_token(previous) != final_key:
            return []
        seen[raw_key] = final_token
        pairs.append((raw_token, final_token))
    return list(dict.fromkeys(pairs))


def replace_changed_name_tokens_in_transcript(
    transcript: str,
    token_pairs: list[tuple[str, str]],
) -> tuple[str, int]:
    if not transcript or not token_pairs:
        return transcript, 0

    replacements_by_key = {
        normalized_name_token(raw_token): final_token
        for raw_token, final_token in token_pairs
        if normalized_name_token(raw_token) and final_token
    }
    if not replacements_by_key:
        return transcript, 0

    alternatives = sorted(
        (re.escape(raw_token) for raw_token, _final_token in token_pairs),
        key=len,
        reverse=True,
    )
    pattern = re.compile(r"(?<![A-Za-z])(?:" + "|".join(alternatives) + r")(?![A-Za-z])", re.IGNORECASE)

    def replace_match(match: re.Match[str]) -> str:
        return replacements_by_key.get(normalized_name_token(match.group(0)), match.group(0))

    return pattern.subn(replace_match, transcript)


def word_span_matches_name(words: list[dict[str, Any]], start_index: int, tokens: list[str]) -> bool:
    if not tokens or start_index + len(tokens) > len(words):
        return False
    for offset, token in enumerate(tokens):
        if normalized_name_token(words[start_index + offset].get("word")) != normalized_name_token(token):
            return False
    return True


def corrected_name_word_span(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    raw_name: str,
) -> Optional[tuple[int, int]]:
    tokens = name_tokens(raw_name)
    if not words or not tokens:
        return None

    ranges: list[tuple[int, int]] = []
    for attribution in audit_row.get("attribution_json") or []:
        if not isinstance(attribution, dict):
            continue
        try:
            start_index = int(attribution.get("word_start"))
            end_index = int(attribution.get("word_end"))
        except (TypeError, ValueError):
            continue
        if start_index <= end_index:
            ranges.append((max(0, start_index), min(len(words) - 1, end_index)))

    if not ranges:
        return None

    width = len(tokens)
    for start_index, end_index in ranges:
        for index in range(start_index, end_index - width + 2):
            if word_span_matches_name(words, index, tokens):
                return index, index + width - 1

    return None


def name_word_occurrence_index(
    words: list[dict[str, Any]],
    span_start: int,
    raw_name: str,
) -> Optional[int]:
    tokens = name_tokens(raw_name)
    if not words or not tokens:
        return None

    occurrence_index = 0
    for index in range(0, max(0, span_start) + 1):
        if word_span_matches_name(words, index, tokens):
            if index == span_start:
                return occurrence_index
            occurrence_index += 1
    return None


def replace_attributed_name_text_in_transcript(
    transcript: str,
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    raw_name: str,
    final_name: str,
) -> tuple[str, int]:
    del words, audit_row
    raw_name = re.sub(r"\s+", " ", raw_name or "").strip()
    final_name = re.sub(r"\s+", " ", final_name or "").strip()
    if not raw_name or not final_name or raw_name == final_name:
        return transcript, 0
    return replace_name_text_in_transcript(transcript, raw_name, final_name)


def corrected_name_word_items(
    original_words: list[dict[str, Any]],
    final_name: str,
) -> list[dict[str, Any]]:
    tokens = name_tokens(final_name)
    if not original_words or not tokens:
        return original_words

    trailing = re.search(r"([^\w\s]+)$", str(original_words[-1].get("word", "")))
    if trailing and tokens and not str(tokens[-1]).endswith(trailing.group(1)):
        tokens[-1] = tokens[-1] + trailing.group(1)

    if len(tokens) == len(original_words):
        return [
            {
                **dict(original_words[index]),
                "word": token,
            }
            for index, token in enumerate(tokens)
        ]

    first = original_words[0]
    last = original_words[-1]
    start = optional_float(first.get("start"))
    end = optional_float(last.get("end"))
    if len(tokens) == 1:
        merged = dict(first)
        merged["word"] = tokens[0]
        if last.get("end") is not None:
            merged["end"] = last.get("end")
        return [merged]

    duration = (end - start) if start is not None and end is not None and end >= start else None
    replacement: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        item = dict(first)
        item["word"] = token
        if duration is not None and start is not None:
            item["start"] = start + duration * index / len(tokens)
            item["end"] = start + duration * (index + 1) / len(tokens)
        replacement.append(item)
    return replacement


def merge_corrected_name_words(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    raw_name: str,
    final_name: str,
) -> tuple[list[dict[str, Any]], bool]:
    del audit_row
    token_pairs = changed_name_token_pairs(raw_name, final_name)
    if token_pairs:
        return replace_changed_name_token_words(words, token_pairs)
    return replace_full_name_phrase_words(words, raw_name, final_name)


def replace_changed_name_token_words(
    words: list[dict[str, Any]],
    token_pairs: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], bool]:
    replacements_by_key = {
        normalized_name_token(raw_token): final_token
        for raw_token, final_token in token_pairs
        if normalized_name_token(raw_token) and final_token
    }
    if not words or not replacements_by_key:
        return words, False

    changed = False
    updated: list[dict[str, Any]] = []
    for item in words:
        replacement = corrected_name_word_token(item.get("word"), replacements_by_key)
        if replacement is None:
            updated.append(item)
            continue
        changed_item = dict(item)
        changed_item["word"] = replacement
        updated.append(changed_item)
        changed = True
    return updated if changed else words, changed


def corrected_name_word_token(word: Any, replacements_by_key: dict[str, str]) -> Optional[str]:
    text = str(word or "")
    match = re.match(r"^([^A-Za-z]*)([A-Za-z][A-Za-z'-]*)([^A-Za-z]*)$", text)
    if not match:
        return None
    key = normalized_name_token(match.group(2))
    replacement = replacements_by_key.get(key)
    if not replacement:
        return None
    return f"{match.group(1)}{replacement}{match.group(3)}"


def replace_full_name_phrase_words(
    words: list[dict[str, Any]],
    raw_name: str,
    final_name: str,
) -> tuple[list[dict[str, Any]], bool]:
    tokens = name_tokens(raw_name)
    if not words or not tokens:
        return words, False

    spans: list[tuple[int, int]] = []
    width = len(tokens)
    index = 0
    while index <= len(words) - width:
        if word_span_matches_name(words, index, tokens):
            spans.append((index, index + width - 1))
            index += width
        else:
            index += 1

    if not spans:
        return words, False

    updated = words
    for span_start, span_end in reversed(spans):
        replacement = corrected_name_word_items(updated[span_start : span_end + 1], final_name)
        updated = updated[:span_start] + replacement + updated[span_end + 1 :]
    return updated, True


def apply_verified_phone_corrections_to_transcript(
    transcript: str,
    entities: dict[str, Any],
    audit_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    corrected_transcript = transcript
    corrected_entities = dict(entities or {})
    words = corrected_entities.get("_word_timestamps")
    corrected_words = [dict(item) for item in words] if isinstance(words, list) else []
    corrections: list[dict[str, Any]] = []
    phone_rows: list[dict[str, Any]] = []

    for row in audit_rows:
        if not isinstance(row, dict):
            continue
        if row.get("field_name") == "name":
            for raw_name, final_name in replacement_name_values(row):
                corrected_transcript, replacements = replace_attributed_name_text_in_transcript(
                    corrected_transcript,
                    corrected_words,
                    row,
                    raw_name,
                    final_name,
                )
                words_changed = False
                if corrected_words:
                    corrected_words, words_changed = merge_corrected_name_words(
                        corrected_words,
                        row,
                        raw_name,
                        final_name,
                    )
                if replacements or words_changed:
                    corrections.append(
                        {
                            "field_name": "name",
                            "raw": raw_name,
                            "value": final_name,
                            "status": row.get("status"),
                            "transcript_replacements": replacements,
                            "word_timestamps_updated": words_changed,
                        }
                    )
            continue
        if row.get("field_name") == "dob":
            if row.get("status") not in DOB_TRANSCRIPT_CORRECTION_STATUSES:
                continue
            final_dob = str(row.get("final_value") or "")
            if not final_dob:
                continue
            corrected_transcript, replacements = replace_attributed_dob_text_in_transcript(
                corrected_transcript,
                corrected_words,
                row,
                final_dob,
            )
            words_changed = False
            if corrected_words:
                corrected_words, words_changed = merge_corrected_dob_words(
                    corrected_words,
                    row,
                    final_dob,
                )
            if replacements or words_changed:
                corrections.append(
                    {
                        "field_name": "dob",
                        "value": final_dob,
                        "status": row.get("status"),
                        "transcript_replacements": replacements,
                        "word_timestamps_updated": words_changed,
                    }
                )
            continue
        if row.get("field_name") not in {"callback_number", "fax_number"}:
            continue
        if row.get("status") not in PHONE_TRANSCRIPT_CORRECTION_STATUSES:
            continue
        phone_rows.append(row)

    for row in sorted(phone_rows, key=phone_row_sort_key, reverse=True):
        normalized = str(row.get("normalized_value") or "")
        formatted = str(row.get("final_value") or format_phone_digits(normalized) or "")
        if not normalized or not formatted:
            continue

        corrected_transcript, replacements = replace_attributed_phone_text_in_transcript(
            corrected_transcript,
            corrected_words,
            row,
            normalized,
            formatted,
        )
        words_changed = False
        if corrected_words:
            corrected_words, words_changed = merge_corrected_phone_words(
                corrected_words,
                row,
                normalized,
                formatted,
            )

        if replacements or words_changed:
            corrections.append(
                {
                    "field_name": row.get("field_name"),
                    "number": formatted,
                    "normalized": normalized,
                    "status": row.get("status"),
                    "transcript_replacements": replacements,
                    "word_timestamps_updated": words_changed,
                }
            )

    if corrected_words and corrections:
        corrected_entities["_word_timestamps"] = corrected_words
    if corrections:
        corrected_entities["transcript_corrections"] = corrections

    return corrected_transcript, corrected_entities
