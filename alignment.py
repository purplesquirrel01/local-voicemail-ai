from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from asr_types import SpanCandidate


def normalize_token_for_alignment(token: str) -> str:
    text = str(token or "").lower()
    text = re.sub(r"\bzero\b|\boh\b|\bo\b", "0", text)
    text = re.sub(r"\bone\b|\bwon\b", "1", text)
    text = re.sub(r"\btwo\b|\bto\b|\btoo\b", "2", text)
    text = re.sub(r"\bthree\b", "3", text)
    text = re.sub(r"\bfour\b|\bfor\b", "4", text)
    text = re.sub(r"\bfive\b", "5", text)
    text = re.sub(r"\bsix\b", "6", text)
    text = re.sub(r"\bseven\b", "7", text)
    text = re.sub(r"\beight\b|\bate\b", "8", text)
    text = re.sub(r"\bnine\b", "9", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _token_text(item: dict[str, Any]) -> str:
    return str(item.get("word", item.get("text", "")) or "")


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounds(words: list[dict], start_index: int, end_index: int) -> tuple[float | None, float | None]:
    if not words or start_index >= len(words) or end_index <= start_index:
        return None, None
    start = _optional_float(words[start_index].get("start"))
    end = _optional_float(words[min(len(words), end_index) - 1].get("end"))
    return start, end


def align_word_streams(a_words: list[dict], b_words: list[dict]) -> list[dict]:
    a_tokens = [normalize_token_for_alignment(_token_text(item)) for item in a_words]
    b_tokens = [normalize_token_for_alignment(_token_text(item)) for item in b_words]
    matcher = SequenceMatcher(None, a_tokens, b_tokens, autojunk=False)
    alignment: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                alignment.append({"tag": tag, "a": [i1 + offset, i1 + offset + 1], "b": [j1 + offset, j1 + offset + 1]})
        else:
            alignment.append({"tag": tag, "a": [i1, i2], "b": [j1, j2]})
    return alignment


def _contains_relevant_context(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(call|callback|phone|number|fax|dob|date of birth|birth|birthday|this is|my name|patient|regarding)\b",
            text,
        )
        or re.search(r"\d", text)
    )


def find_disagreement_regions(alignment: list[dict], a_words: list[dict] | None = None, b_words: list[dict] | None = None, file_key: str = "", source_run_id: str = "") -> list[SpanCandidate]:
    a_words = a_words or []
    b_words = b_words or []
    spans: list[SpanCandidate] = []
    for index, item in enumerate(alignment):
        if item.get("tag") == "equal":
            continue
        a_start, a_end = item.get("a", [0, 0])
        b_start, b_end = item.get("b", [0, 0])
        a_text = " ".join(_token_text(word) for word in a_words[a_start:a_end])
        b_text = " ".join(_token_text(word) for word in b_words[b_start:b_end])
        text = f"{a_text} | {b_text}".strip(" |")
        if not _contains_relevant_context(text):
            continue
        start, end = _bounds(a_words, a_start, a_end)
        if start is None or end is None:
            start, end = _bounds(b_words, b_start, b_end)
        spans.append(
            SpanCandidate(
                span_id=f"asr_disagreement:{file_key}:{index}",
                file_key=file_key,
                field_type="asr_disagreement",
                source="asr_disagreement",
                source_run_id=source_run_id,
                start=start,
                end=end,
                word_start=a_start if a_start < a_end else None,
                word_end=(a_end - 1) if a_start < a_end else None,
                text=text,
                reasons=["asr_disagreement"],
            )
        )
    return spans
