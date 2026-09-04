from __future__ import annotations

import os
import re
from datetime import date
from difflib import SequenceMatcher
from typing import Any


DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s.\-]*)?(?:\(?\d{3}\)?[\s.\-]*)\d{3}[\s.\-]*\d{4}(?!\d)"
)
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
DOB_CUE_RE = re.compile(r"\b(?:date\s+of\s+birth|birth\s+date|dob|d\.o\.b\.|born)\b", re.IGNORECASE)
SPELLING_RE = re.compile(r"\b[A-Za-z](?:\s*-\s*[A-Za-z]){2,}\b")
SPELLING_TOKEN_RE = re.compile(r"^[A-Za-z](?:-[A-Za-z]){2,}$")
NAME_TOKEN_PATTERN = r"[A-Za-z][A-Za-z'.-]*"
SELF_NAME_TOKEN_PATTERN = r"(?:Mr\.|Mrs\.|Ms\.|Miss\.|Dr\.|[A-Za-z]\.|[A-Za-z][A-Za-z'-]*)"
NAME_TOKEN_RE = re.compile(NAME_TOKEN_PATTERN)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'.-]*|\d+")
INTERLEAVED_SPELLED_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:this\s+is|my\s+name\s+is|calling\s+for|calling\s+about|"
    r"regarding|in\s+regards\s+to|about|for)\s+)"
    rf"(?P<first>{NAME_TOKEN_PATTERN})\s*,?\s+"
    r"(?P<first_spell>[A-Za-z](?:\s*-\s*[A-Za-z]){2,})"
    r"(?:\s*,?\s+"
    rf"(?P<last>{NAME_TOKEN_PATTERN})\s*,?\s+"
    r"(?P<last_spell>[A-Za-z](?:\s*-\s*[A-Za-z]){2,}))?",
    re.IGNORECASE,
)

SELF_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:this\s+is|my\s+name\s+is|i\s+am|i'm|it's|it\s+is|the\s+name\s+is)\s+)"
    rf"(?P<name>{SELF_NAME_TOKEN_PATTERN}(?:\s+{SELF_NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
IT_IS_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:it's|it\s+is)\s+)"
    rf"(?P<name>{SELF_NAME_TOKEN_PATTERN}(?:\s+{SELF_NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
SUBJECT_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:calling\s+for)\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
THIS_IS_FOR_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:this\s+is\s+for)\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{1,3}})",
    re.IGNORECASE,
)
SUBJECT_POSSESSIVE_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:(?:his|her|their|client(?:'s)?|patient(?:'s)?)\s+name\s+is)\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
FOR_PATIENT_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:this\s+is\s+)?for\s+(?:a\s+)?patient\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
PATIENT_CUE_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:calling\s+)?(?:about|regarding|in\s+regards\s+to)\s+"
    r"(?:a\s+)?(?:patient|client)\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
GENERIC_SUBJECT_CUE_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:calling\s+)?(?:about|regarding|in\s+regards\s+to|on\s+behalf\s+of|"
    r"calling\s+on\s+behalf\s+of)\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{1,3}})",
    re.IGNORECASE,
)
MUTUAL_CLIENT_NAME_RE = re.compile(
    r"(?P<prefix>\bpertaining\s+to\s+(?:our\s+)?mutual\s+client\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
ONE_OF_PROVIDER_PATIENTS_NAME_RE = re.compile(
    r"(?P<prefix>\bone\s+of\s+(?:Dr\.?\s+)?"
    rf"{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,2}}(?:'s|\u2019s)\s+patients?\s*,\s*)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
PATIENT_HERE_NAME_RE = re.compile(
    r"(?P<prefix>\bpatient\s+here\b[^?\n]{0,100}?\b(?:it\s+is|it's|this\s+is)\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
PATIENT_OF_PROVIDER_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:for\s+)?(?:a\s+)?patient\s+of\s+(?:Dr\.?\s+)?"
    rf"{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,2}}\s*,\s*)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{1,2}})",
    re.IGNORECASE,
)
SELF_PATIENT_OF_PROVIDER_RE = re.compile(
    r"(?P<prefix>\b(?:this\s+is|my\s+name\s+is|i\s+am|i'm|it's|it\s+is)\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})"
    r"(?P<subject_cue>[^.!?\n]{0,40}(?:[.!?]\s+|,\s+|\s+)"
    r"(?:i\s+am|i'm)\s+(?:a\s+)?patient\s+of\s+(?:Dr\.?\s+)?"
    rf"{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,2}})",
    re.IGNORECASE,
)
REQUEST_FOR_SUBJECT_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:request|authorization|authorizations|prior\s+authorization|"
    r"pre[-\s]?certification|availability\s+request|appeal|case|packet|questionnaire)\b"
    r"[^.!?\n]{0,180}\bfor\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{1,3}})",
    re.IGNORECASE,
)
REQUEST_ON_SUBJECT_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:request|authorization|authorizations|prior\s+authorization|"
    r"pre[-\s]?certification|availability\s+request|appeal|case|packet|questionnaire)\b"
    r"[^.!?\n]{0,180}\b(?:received\s+)?on\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{1,3}})",
    re.IGNORECASE,
)
BROAD_NAME_RECALL_RE = re.compile(
    r"(?P<prefix>\b(?:"
    r"message\s+(?:is\s+)?(?:for|about|regarding|concerns)|"
    r"call\s+(?:is\s+)?(?:for|about|regarding|concerns)|"
    r"voicemail\s+(?:is\s+)?(?:for|about|regarding|concerns)|"
    r"concerns|regarding|about|in\s+regards\s+to|on\s+behalf\s+of|"
    r"patient(?:'s)?\s+name(?:\s+is)?|patient\s+is|"
    r"member(?:'s)?\s+name(?:\s+is)?|member\s+is|"
    r"client(?:'s)?\s+name(?:\s+is)?|client\s+is"
    r")\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{1,4}})",
    re.IGNORECASE,
)
COMPACT_DOB_SUBJECT_NAME_RE = re.compile(
    r"(?P<prefix>(?:^|(?<=[.!?]\s)))"
    rf"(?P<name>{NAME_TOKEN_PATTERN}\s+{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN})?)"
    r"\s*,\s*(?P<subject_cue>\d{3,4}\s+(?:of|or)\s+\d{2,4})\b"
    r"(?=[^.!?]{0,120}\b(?:had|has|having|need|needs|surgery|revision|procedure|appointment|patient|therapy)\b)",
    re.IGNORECASE,
)
LEADING_NAME_DOB_RE = re.compile(
    r"(?P<prefix>(?:^|(?<=[.!?]\s)))"
    rf"(?P<name>{NAME_TOKEN_PATTERN}\s+{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN})?)"
    r"\s+(?P<subject_cue>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
    r"(?=[^.!?\n]{0,160}\b(?:test\s+results?|results?|schedule|surgery|procedure|appointment)\b)",
    re.IGNORECASE,
)
FACILITY_FOR_DOB_SUBJECT_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:calling\s+)?from\s+[^.!?\n]{0,80}?\bfor\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}\s+{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN})?)"
    r"\s*,?\s+(?P<subject_cue>(?:date\s+of\s+birth|birth\s+date|dob|d\.o\.b\.|born)\s+"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.IGNORECASE,
)
POSSESSIVE_RELATIONSHIP_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:my|our|his|her|their)\s+"
    r"(?:father|mother|dad|mom|husband|wife|son|daughter|parent|child|spouse)\s*,?\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
REVERSE_RELATIONSHIP_NAME_RE = re.compile(
    rf"(?P<prefix>\b)(?P<name>{NAME_TOKEN_PATTERN}\s+{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN})?)"
    r"(?P<subject_cue>\s*(?:'s|\u2019s)\s+"
    r"(?:father|mother|dad|mom|husband|wife|son|daughter|parent|child|spouse)\b)",
    re.IGNORECASE,
)
PRONOUN_APPOSITIVE_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:(?:message|call|voicemail|question|questions)\s+)?"
    r"about\s+(?:him|her|them|patient|the\s+patient)\s*,\s*)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}\s+{NAME_TOKEN_PATTERN})"
    r"(?=\s*,?\s+(?:being|who|that|is|was|had|has|needs?|there|at)\b|[,.;!?])",
    re.IGNORECASE,
)
RELATIONSHIP_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:calling\s+about|calling\s+for|about|regarding|for)\s+"
    r"(?:(?:my|his|her|their)\s+)?"
    r"(?:father|mother|dad|mom|husband|wife|son|daughter|parent|child)\s*,?\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
EXPLICIT_PATIENT_NAME_RE = re.compile(
    r"(?P<prefix>\b(?:patient(?:'s)?\s+name\s+is|patient\s+is|calling\s+on\s+a\s+patient)\s+)"
    rf"(?P<name>{NAME_TOKEN_PATTERN}(?:\s+{NAME_TOKEN_PATTERN}){{0,3}})",
    re.IGNORECASE,
)
PATIENT_NAME_ALTERNATE_SPELLING_RE = re.compile(
    r"(?P<prefix>\bpatient(?:'s)?\s+name\b[,:]?\s*)"
    r"(?P<body>[\s\S]{0,320}?)(?=\b(?:date\s+of\s+birth|birth\s+date|dob|d\.o\.b\.|born)\b|$)",
    re.IGNORECASE,
)

NAME_STOP_WORDS = {
    "about",
    "again",
    "and",
    "at",
    "because",
    "birth",
    "calling",
    "call",
    "date",
    "dob",
    "for",
    "from",
    "i",
    "i'm",
    "im",
    "is",
    "just",
    "me",
    "number",
    "of",
    "or",
    "over",
    "phone",
    "please",
    "regarding",
    "that",
    "that's",
    "to",
    "trying",
    "was",
    "with",
}

NAME_LEADING_DESCRIPTORS = {
    "a",
    "client",
    "miss",
    "mr",
    "mrs",
    "ms",
    "patient",
    "the",
}

GENERIC_NAME_DESCRIPTOR_EXACT = {
    "a request",
    "an availability request",
    "an authorization",
    "an authorization request",
    "availability request",
    "prior authorization",
    "prior authorization request",
    "prior authorizations",
    "office following up",
    "test voicemail",
    "the availability request",
    "the office following up",
    "the request",
    "your request",
}

GENERIC_NAME_DESCRIPTOR_START_WORDS = {
    "a",
    "an",
    "our",
    "that",
    "the",
    "this",
    "your",
}

GENERIC_NAME_DESCRIPTOR_END_WORDS = {
    "appeal",
    "authorization",
    "authorizations",
    "case",
    "packet",
    "questionnaire",
    "referral",
    "request",
    "requests",
}

SELF_NAME_NON_NAME_FIRST_TOKENS = {
    "calling",
    "due",
    "following",
    "going",
    "gonna",
    "sorry",
    "talking",
}

RELATIONSHIP_WORDS = {
    "child",
    "dad",
    "daughter",
    "father",
    "husband",
    "mom",
    "mother",
    "parent",
    "son",
    "spouse",
    "wife",
}

CUE_PATTERNS = [
    ("fax", re.compile(r"\b(?:fax(?:\s+\w+){0,3}\s+to|fax|faxed|faxing)\b", re.IGNORECASE)),
    ("callback", re.compile(r"\b(?:call\s+me\s+back|call\s+back|my\s+number\s+is|number\s+is|reach\s+me|phone\s+number|give\s+me\s+a\s+call|call\s+me)\b", re.IGNORECASE)),
    ("medical_send", re.compile(r"\b(?:send|sent|referral|records?|order|orders|prescription|script|authorization)\b", re.IGNORECASE)),
    ("dob", DOB_CUE_RE),
    ("appointment", re.compile(r"\b(?:appointment|scheduled|schedule|visit)\b", re.IGNORECASE)),
]


def normalize_transcript(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").strip()


def _context(text: str, start: int, end: int, radius: int = 64) -> tuple[str, str, str]:
    before_start = max(0, start - radius)
    after_end = min(len(text), end + radius)
    return (
        text[before_start:start],
        text[end:after_end],
        text[before_start:after_end],
    )


def _sentence_context(text: str, start: int, end: int, limit: int = 260) -> str:
    left = -1
    for index in sorted(
        (match.start() for match in re.finditer(r"[.!?\n]", text[:start])),
        reverse=True,
    ):
        if text[index] == "." and re.search(r"\b(?:dr|mr|mrs|ms)\.$", text[max(0, index - 5) : index + 1], re.I):
            continue
        left = index
        break
    sentence_start = 0 if left < 0 else left + 1
    right_candidates = [text.find(mark, end) for mark in ".!?\n"]
    right_candidates = [index for index in right_candidates if index >= 0]
    sentence_end = min(right_candidates) + 1 if right_candidates else len(text)
    sentence = re.sub(r"\s+", " ", text[sentence_start:sentence_end]).strip()
    if len(sentence) <= limit:
        return sentence
    candidate_text = text[start:end]
    before = text[max(sentence_start, start - 96) : start]
    after = text[end : min(sentence_end, end + 96)]
    return re.sub(r"\s+", " ", f"{before}{candidate_text}{after}").strip()


def _cue_window(text: str, start: int, end: int) -> tuple[list[str], list[str]]:
    _before, _after, window = _context(text, start, end, radius=72)
    cues: list[str] = []
    phrases: list[str] = []
    for cue, pattern in CUE_PATTERNS:
        for match in pattern.finditer(window):
            if cue not in cues:
                cues.append(cue)
            phrase = re.sub(r"\s+", " ", match.group(0)).strip()
            if phrase and phrase.lower() not in {item.lower() for item in phrases}:
                phrases.append(phrase)
    return cues, phrases


def _candidate_base(
    *,
    cid: str,
    raw: str,
    span: tuple[int, int],
    source: str,
    transcript: str,
    confidence_hint: str,
    evidence_text: str | None = None,
) -> dict[str, Any]:
    start, end = span
    before, after, window = _context(transcript, start, end)
    cues, phrases = _cue_window(transcript, start, end)
    return {
        "id": cid,
        "raw": raw,
        "span": [start, end],
        "source": source,
        "context_before": before,
        "context_after": after,
        "window": window,
        "sentence_context": _sentence_context(transcript, start, end),
        "nearby_cues": cues,
        "cue_phrases": phrases,
        "confidence_hint": confidence_hint,
        "evidence_text": evidence_text if evidence_text is not None else raw,
    }


def _phone_digits(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return digits


def _format_phone(digits: str) -> str:
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def _add_number_candidate(
    candidates: list[dict[str, Any]],
    seen: set[str],
    *,
    raw: str,
    span: tuple[int, int],
    source: str,
    transcript: str,
    confidence_hint: str,
) -> None:
    digits = _phone_digits(raw)
    if not digits or digits in seen:
        return
    item = _candidate_base(
        cid=f"number:{len(candidates)}",
        raw=raw,
        span=span,
        source=source,
        transcript=transcript,
        confidence_hint=confidence_hint,
    )
    item["normalized"] = digits
    item["formatted"] = _format_phone(digits)
    candidates.append(item)
    seen.add(digits)


def _extract_numeric_phones(transcript: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in PHONE_RE.finditer(transcript):
        _add_number_candidate(
            candidates,
            seen,
            raw=match.group(0),
            span=(match.start(), match.end()),
            source="numeric_phone",
            transcript=transcript,
            confidence_hint="high",
        )
    return candidates


def _extract_spoken_phones(transcript: str, candidates: list[dict[str, Any]]) -> None:
    seen = {str(item.get("normalized")) for item in candidates}
    tokens = [match for match in WORD_RE.finditer(transcript)]
    index = 0
    while index < len(tokens):
        token = re.sub(r"[^a-z0-9]+", "", tokens[index].group(0).lower())
        if token not in DIGIT_WORDS:
            index += 1
            continue
        start_index = index
        digits: list[str] = []
        while index < len(tokens):
            current = re.sub(r"[^a-z0-9]+", "", tokens[index].group(0).lower())
            if current not in DIGIT_WORDS:
                break
            digits.append(DIGIT_WORDS[current])
            index += 1
        if len(digits) not in {10, 11}:
            continue
        raw_start = tokens[start_index].start()
        raw_end = tokens[index - 1].end()
        nearby_cues, _phrases = _cue_window(transcript, raw_start, raw_end)
        if "callback" not in nearby_cues and "fax" not in nearby_cues:
            continue
        raw_digits = "".join(digits)
        normalized = _phone_digits(raw_digits)
        if not normalized or normalized in seen:
            continue
        item = _candidate_base(
            cid=f"number:{len(candidates)}",
            raw=transcript[raw_start:raw_end],
            span=(raw_start, raw_end),
            source="spoken_digits",
            transcript=transcript,
            confidence_hint="medium",
        )
        item["normalized"] = normalized
        item["formatted"] = _format_phone(normalized)
        candidates.append(item)
        seen.add(normalized)


def _expand_year(year: int) -> int:
    if year < 100:
        return 1900 + year if year >= 30 else 2000 + year
    return year


def _parse_date(raw: str) -> date | None:
    match = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})$", raw.strip())
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


def _format_date(value: date) -> str:
    return f"{value.month:02d}/{value.day:02d}/{value.year:04d}"


def _extract_dobs(transcript: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cue_match in DOB_CUE_RE.finditer(transcript):
        window_start = cue_match.end()
        window_end = min(len(transcript), window_start + 72)
        for date_match in DATE_RE.finditer(transcript, window_start, window_end):
            raw = date_match.group(0)
            parsed = _parse_date(raw)
            if not parsed:
                continue
            normalized = _format_date(parsed)
            key = f"{normalized}:{date_match.start()}"
            if key in seen:
                continue
            item = _candidate_base(
                cid=f"dob:{len(candidates)}",
                raw=raw,
                span=(date_match.start(), date_match.end()),
                source="date_numeric",
                transcript=transcript,
                confidence_hint="high",
                evidence_text=transcript[cue_match.start() : date_match.end()].strip(" ,.;:"),
            )
            item["normalized"] = normalized
            candidates.append(item)
            seen.add(key)
    for match in LEADING_NAME_DOB_RE.finditer(transcript):
        raw = match.group("subject_cue")
        parsed = _parse_date(raw)
        if not parsed:
            continue
        normalized = _format_date(parsed)
        key = f"{normalized}:{match.start('subject_cue')}"
        if key in seen:
            continue
        item = _candidate_base(
            cid=f"dob:{len(candidates)}",
            raw=raw,
            span=(match.start("subject_cue"), match.end("subject_cue")),
            source="date_numeric_adjacent_patient",
            transcript=transcript,
            confidence_hint="medium",
            evidence_text=transcript[match.start("name") : match.end("subject_cue")].strip(" ,.;:"),
        )
        item["normalized"] = normalized
        candidates.append(item)
        seen.add(key)
    return candidates


def _format_name_token(token: str) -> str:
    token = token.strip(" .,'\"-")
    if not token:
        return ""
    if "-" in token:
        return "-".join(part for part in (_format_name_token(part) for part in token.split("-")) if part)
    if "'" in token:
        return "'".join(part for part in (_format_name_token(part) for part in token.split("'")) if part)
    core = token.strip(".")
    if len(core) == 1:
        return f"{core.upper()}."
    return core[:1].upper() + core[1:].lower()


def _title_name(raw: str) -> str:
    raw_tokens = [
        token.strip(" ,'\"-")
        for token in NAME_TOKEN_RE.findall(raw)
        if token.strip(" .,'\"-")
    ]
    while raw_tokens and raw_tokens[0].strip(".").lower() in NAME_LEADING_DESCRIPTORS:
        raw_tokens.pop(0)
    parts = []
    for token in raw_tokens:
        if not token.strip("."):
            continue
        if SPELLING_TOKEN_RE.match(token):
            continue
        lowered = token.strip(".").lower()
        if lowered in NAME_STOP_WORDS:
            break
        formatted = _format_name_token(token)
        if formatted:
            parts.append(formatted)
        if token.endswith(".") and len(token.strip(".-'")) > 1:
            break
    return " ".join(parts).strip()


def _spelled_letters(raw: Any) -> str:
    return "".join(re.findall(r"[A-Za-z]", str(raw or ""))).upper()


def _contains_spelling_token(raw: str) -> bool:
    return any(
        SPELLING_TOKEN_RE.match(token.strip(" .,'\"-"))
        for token in NAME_TOKEN_RE.findall(raw)
    )


def _name_is_generic_descriptor(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    if not normalized:
        return True
    if normalized in GENERIC_NAME_DESCRIPTOR_EXACT:
        return True
    tokens = normalized.split()
    if len(tokens) >= 2 and tokens[0] in GENERIC_NAME_DESCRIPTOR_START_WORDS:
        return tokens[-1] in GENERIC_NAME_DESCRIPTOR_END_WORDS
    if "availability" in tokens and tokens[-1:] == ["request"]:
        return True
    if tokens[:1] == ["prior"] and any(token.startswith("authorization") for token in tokens[1:]):
        return True
    return False


def _self_name_phrase_is_non_name(raw: str) -> bool:
    tokens = [
        token.strip(" .,'\"-").lower()
        for token in NAME_TOKEN_RE.findall(raw)
        if token.strip(" .,'\"-")
    ]
    if not tokens:
        return True
    if tokens[0] in SELF_NAME_NON_NAME_FIRST_TOKENS:
        return True
    if "office" in tokens and any(token in {"following", "follow", "calling"} for token in tokens):
        return True
    return tuple(tokens[:2]) in {
        ("going", "to"),
        ("supposed", "to"),
        ("ready", "to"),
        ("not", "ready"),
    }


def _relationship_prefix_is_caller_identity(transcript: str, prefix_start: int) -> bool:
    context = transcript[max(0, prefix_start - 32) : prefix_start]
    return bool(
        re.search(
            r"\b(?:this\s+is|my\s+name\s+is|i\s+am|i'm|it's|the\s+name\s+is)\s*$",
            context,
            re.IGNORECASE,
        )
    )


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").lower())


def _rough_name_key(value: Any) -> str:
    text = _name_key(value)
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


def _name_similarity(left: Any, right: Any) -> float:
    return SequenceMatcher(None, _name_key(left), _name_key(right)).ratio()


def _last_names_match(left: Any, right: Any) -> bool:
    if _name_key(left) == _name_key(right):
        return True
    if _name_similarity(left, right) >= 0.58:
        return True
    left_rough = _rough_name_key(left)
    right_rough = _rough_name_key(right)
    return bool(left_rough and right_rough and left_rough == right_rough)


def _last_names_strong_match(left: Any, right: Any) -> bool:
    if _name_key(left) == _name_key(right):
        return True
    left_rough = _rough_name_key(left)
    right_rough = _rough_name_key(right)
    if left_rough and right_rough and left_rough == right_rough:
        return True
    return _name_similarity(left, right) >= 0.74


def _first_names_match(left: Any, right: Any) -> bool:
    if _name_key(left) == _name_key(right):
        return True
    return _name_similarity(left, right) >= 0.58


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _last_name_only_caller_id_correction_enabled() -> bool:
    return _env_bool("CANDIDATE_AGENT_CALLER_ID_LAST_NAME_ONLY_CORRECTION", True)


def _broad_name_recall_enabled() -> bool:
    return _env_bool("CANDIDATE_AGENT_BROAD_NAME_RECALL", True)


def _caller_id_display_name(caller_id: Any) -> str:
    text = " ".join(str(caller_id or "").replace("_", " ").split()).strip()
    quoted_name = re.match(r'^\s*"([^"]+)"', text)
    if quoted_name:
        text = quoted_name.group(1).strip()
    else:
        text = re.sub(r"\([^)]*\d[^)]*\)", " ", text)
    quoted = re.match(r'^"?([^"<]+?)"?\s*<[^>]+>$', text)
    if quoted:
        text = quoted.group(1).strip()
    text = PHONE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" \"'<>")
    return text


def _caller_id_name_shaped(caller_id: Any) -> bool:
    text = _caller_id_display_name(caller_id)
    if not text:
        return False
    upper = text.upper()
    digits = re.findall(r"\d", upper)
    letters = re.findall(r"[A-Z]", upper)
    if not letters:
        return False
    if len(digits) >= 3 or (digits and len(digits) >= max(1, len(letters) // 2)):
        return False
    if upper in {
        "ANONYMOUS",
        "BLOCKED",
        "PRIVATE",
        "PRIVATE CALLER",
        "RESTRICTED",
        "UNKNOWN",
        "UNKNOWN CALLER",
        "UNKNOWN NAME",
        "UNAVAILABLE",
        "WIRELESS CALLER",
        "NO CALLER ID",
        "OUT OF AREA",
    }:
        return False
    if re.search(
        r"\b(?:HOSPITAL|CLINIC|MEDICAL|HEALTH|HEALTHCARE|PHARMACY|OFFICE|CENTER|CENTRE|"
        r"LLC|INC|CORP|COMPANY|ASSOCIATES|SYSTEMS|DEPARTMENT|SCHOOL|BANK)\b",
        upper,
    ):
        return False
    tokens = re.findall(r"[A-Z][A-Z'.-]*", upper)
    if not 1 <= len(tokens) <= 3:
        return False
    if len(tokens) == 3 and len(tokens[1].strip(".-'")) == 1:
        return len(tokens[0].strip(".-'")) >= 2 and len(tokens[2].strip(".-'")) >= 2
    if len(tokens) == 3 and len(tokens[2].strip(".-'")) == 1:
        return len(tokens[0].strip(".-'")) >= 2 and len(tokens[1].strip(".-'")) >= 2
    return all(len(token.strip(".-'")) >= 2 for token in tokens)


def _caller_name_orders(caller_id: Any) -> list[tuple[str, str, str]]:
    text = _caller_id_display_name(caller_id)
    if not _caller_id_name_shaped(text):
        return []
    if "," in text:
        last, first = [part.strip(" ,.;:-") for part in text.split(",", 1)]
        if first and last:
            return [(_title_name(first), _title_name(last), "last_first")]
    tokens = [_title_name(token) for token in NAME_TOKEN_RE.findall(text)]
    tokens = [token for token in tokens if token]
    if len(tokens) < 2:
        return []
    if len(tokens) == 2:
        return [(tokens[0], tokens[1], "first_last"), (tokens[1], tokens[0], "last_first")]
    if len(tokens) == 3 and len(tokens[1].strip(".-'")) == 1:
        return [(tokens[0], tokens[2], "first_last")]
    if len(tokens) == 3 and len(tokens[2].strip(".-'")) == 1:
        return [(tokens[1], tokens[0], "last_first")]
    orders = [(tokens[0], tokens[-1], "first_last")]
    orders.append((tokens[-1], tokens[0], "last_first"))
    return orders


def _caller_token_truncated(transcript_token: str, caller_token: str) -> bool:
    left = _name_key(transcript_token)
    right = _name_key(caller_token)
    return bool(left and right and left != right and left.startswith(right))


def _caller_first_name_small_suffix_expansion(transcript_token: str, caller_token: str) -> bool:
    left = _name_key(transcript_token)
    right = _name_key(caller_token)
    if not left or not right or left == right or len(left) < 3:
        return False
    if not right.startswith(left):
        return False
    return 1 <= len(right) - len(left) <= 3


def _add_caller_id_name_candidates(
    name_candidates: list[dict[str, Any]],
    caller_id: str,
) -> list[dict[str, Any]]:
    corrections: list[dict[str, Any]] = []
    if not _caller_id_name_shaped(caller_id):
        return corrections
    caller_orders = _caller_name_orders(caller_id)
    if not caller_orders:
        return corrections

    for candidate in list(name_candidates):
        if str(candidate.get("source") or "") != "self_identification":
            continue
        raw = str(candidate.get("raw") or candidate.get("value") or "").strip()
        raw_tokens = raw.split()
        if not raw_tokens:
            continue
        if any(_name_key(f"{caller_first} {caller_last}") == _name_key(raw) for caller_first, caller_last, _order in caller_orders):
            continue
        evidence = str(candidate.get("evidence_text") or "").strip()
        spelling_corrected_same_raw = any(
            str(other.get("source") or "") == "transcript_spelling_corrected"
            and _name_key(other.get("raw") or other.get("value")) == _name_key(raw)
            for other in name_candidates
        )

        for caller_first, caller_last, order in caller_orders:
            caller_first_truncated = bool(raw_tokens and _caller_token_truncated(raw_tokens[0], caller_first))
            caller_last_truncated = bool(
                len(raw_tokens) >= 2 and _caller_token_truncated(raw_tokens[-1], caller_last)
            )
            if caller_last_truncated:
                continue

            if len(raw_tokens) == 1:
                if (
                    order == "last_first"
                    and _first_names_match(raw_tokens[0], caller_first)
                    and not _caller_first_name_small_suffix_expansion(raw_tokens[0], caller_first)
                ):
                    suggested_first = _title_name(caller_first)
                    if _name_key(suggested_first) != _name_key(raw):
                        corrected = dict(candidate)
                        corrected["id"] = f"name:{len(name_candidates)}"
                        corrected["raw"] = raw
                        corrected["value"] = suggested_first
                        corrected["source"] = "caller_id_corrected"
                        corrected["caller_id_used"] = _caller_id_display_name(caller_id)
                        name_candidates.append(corrected)
                    break
                continue

            raw_first = raw_tokens[0]
            raw_last = raw_tokens[-1]
            first_matches = _first_names_match(raw_first, caller_first)
            last_matches = _last_names_match(raw_last, caller_last)
            strong_last_matches = _last_names_strong_match(raw_last, caller_last)
            caller_first_expands_spoken = _caller_first_name_small_suffix_expansion(raw_first, caller_first)
            full_correction_first_matches = first_matches
            if order == "last_first":
                full_correction_first_matches = _name_key(raw_first) == _name_key(caller_first)
            suggested = _title_name(f"{caller_first} {caller_last}")
            if not suggested or _name_key(suggested) == _name_key(raw):
                continue

            if (
                order in {"first_last", "last_first"}
                and full_correction_first_matches
                and last_matches
                and not caller_first_truncated
                and not caller_first_expands_spoken
            ):
                corrected = dict(candidate)
                corrected["id"] = f"name:{len(name_candidates)}"
                corrected["raw"] = raw
                corrected["value"] = suggested
                corrected["source"] = "caller_id_corrected"
                corrected["caller_id_used"] = _caller_id_display_name(caller_id)
                name_candidates.append(corrected)
                break

            if (
                _last_name_only_caller_id_correction_enabled()
                and (not first_matches or caller_first_truncated)
                and strong_last_matches
                and not spelling_corrected_same_raw
            ):
                suggested_last = _title_name(caller_last)
                suggested_value = _title_name(" ".join([*raw_tokens[:-1], suggested_last]))
                if suggested_value and _name_key(suggested_value) != _name_key(raw):
                    corrected = dict(candidate)
                    corrected["id"] = f"name:{len(name_candidates)}"
                    corrected["raw"] = raw
                    corrected["value"] = suggested_value
                    corrected["source"] = "caller_id_corrected"
                    corrected["caller_id_used"] = _caller_id_display_name(caller_id)
                    name_candidates.append(corrected)
                    break

            if order == "last_first" and first_matches and last_matches and not caller_first_expands_spoken:
                corrected = dict(candidate)
                corrected["id"] = f"name:{len(name_candidates)}"
                corrected["raw"] = raw
                corrected["value"] = suggested
                corrected["source"] = "caller_id_corrected"
                corrected["caller_id_used"] = _caller_id_display_name(caller_id)
                name_candidates.append(corrected)
                break

            if last_matches:
                suggested_last = _title_name(caller_last)
                suggested_value = _title_name(" ".join([*raw_tokens[:-1], suggested_last]))
                if suggested_value and _name_key(suggested_value) != _name_key(raw):
                    corrections.append(
                        {
                            "id": f"name_correction:{len(corrections)}",
                            "raw": raw,
                            "suggested_value": suggested_value,
                            "evidence_text": evidence,
                            "caller_id_used": _caller_id_display_name(caller_id),
                            "reason": "last_name_phonetic_match",
                        }
                    )
                    break
            elif order != "first_last" and len(_name_key(caller_last)) >= 5 and _name_similarity(raw, suggested) >= 0.50:
                corrections.append(
                    {
                        "id": f"name_correction:{len(corrections)}",
                        "raw": raw,
                        "suggested_value": suggested,
                        "evidence_text": evidence,
                        "caller_id_used": _caller_id_display_name(caller_id),
                        "reason": "weak_phonetic_match",
                    }
                )
                break
    return corrections


def _name_is_plausible(name: str, *, minimum_tokens: int = 1) -> bool:
    parts = name.split()
    if len(parts) < minimum_tokens or len(parts) > 3:
        return False
    for index, part in enumerate(parts):
        core = part.strip(".-'")
        if len(core) >= 2:
            continue
        if len(core) == 1 and 0 < index < len(parts) - 1:
            continue
        return False
    return True


def _add_name_candidate(
    candidates: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    *,
    raw_name: str,
    span: tuple[int, int],
    source: str,
    transcript: str,
    cue: str,
    evidence_start: int,
    evidence_end: int,
    minimum_tokens: int = 1,
) -> None:
    value = _title_name(raw_name)
    if _name_is_generic_descriptor(value):
        return
    if not _name_is_plausible(value, minimum_tokens=minimum_tokens):
        return
    key = (value.lower(), source)
    if key in seen:
        return
    item = _candidate_base(
        cid=f"name:{len(candidates)}",
        raw=value,
        span=span,
        source=source,
        transcript=transcript,
        confidence_hint="medium" if source == "self_identification" else "high",
        evidence_text=transcript[evidence_start:evidence_end].strip(" ,.;:"),
    )
    item["value"] = value
    item["caller_id_used"] = ""
    if cue not in item["nearby_cues"]:
        item["nearby_cues"].append(cue)
    candidates.append(item)
    seen.add(key)


def _add_patient_name_alternate_spelling_candidates(
    transcript: str,
    candidates: list[dict[str, Any]],
    seen: set[tuple[str, str]],
) -> None:
    for match in PATIENT_NAME_ALTERNATE_SPELLING_RE.finditer(transcript):
        body = match.group("body") or ""
        spellings = list(SPELLING_RE.finditer(body))
        if len(spellings) < 2:
            continue
        first = spellings[-2]
        last = spellings[-1]
        first_letters = _spelled_letters(first.group(0))
        last_letters = _spelled_letters(last.group(0))
        if len(first_letters) < 2 or len(last_letters) < 2:
            continue
        value = _title_name(f"{first_letters} {last_letters}")
        if not _name_is_plausible(value, minimum_tokens=2):
            continue
        body_start = match.start("body")
        evidence_end = body_start + last.end()
        span = (body_start + first.start(), body_start + last.end())
        _add_name_candidate(
            candidates,
            seen,
            raw_name=value,
            span=span,
            source="relationship_subject",
            transcript=transcript,
            cue="patient_subject",
            evidence_start=match.start("prefix"),
            evidence_end=evidence_end,
            minimum_tokens=2,
        )


def _add_broad_name_recall_candidates(
    transcript: str,
    candidates: list[dict[str, Any]],
    seen: set[tuple[str, str]],
) -> None:
    if not _broad_name_recall_enabled():
        return
    for match in BROAD_NAME_RECALL_RE.finditer(transcript):
        matched_name = match.group("name")
        raw = _title_name(matched_name)
        if not raw:
            continue
        if _name_is_generic_descriptor(raw):
            continue
        name_start = match.start("name")
        name_end = name_start + len(raw)
        already_seen = any(
            _name_key(candidate.get("value") or candidate.get("raw")) == _name_key(raw)
            for candidate in candidates
        )
        if already_seen:
            continue
        _add_name_candidate(
            candidates,
            seen,
            raw_name=raw,
            span=(name_start, name_end),
            source="broad_name_recall",
            transcript=transcript,
            cue="name_recall",
            evidence_start=match.start("prefix"),
            evidence_end=name_end,
            minimum_tokens=2,
        )


def _extract_names(transcript: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    _add_patient_name_alternate_spelling_candidates(transcript, candidates, seen)

    for regex, source, cue, minimum_tokens in (
        (FOR_PATIENT_NAME_RE, "explicit_patient", "patient_subject", 1),
        (EXPLICIT_PATIENT_NAME_RE, "explicit_patient", "patient_subject", 1),
        (PATIENT_CUE_NAME_RE, "relationship_subject", "patient_subject", 1),
        (MUTUAL_CLIENT_NAME_RE, "relationship_subject", "relationship_subject", 1),
        (SUBJECT_POSSESSIVE_NAME_RE, "relationship_subject", "relationship_subject", 1),
        (FACILITY_FOR_DOB_SUBJECT_NAME_RE, "relationship_subject", "patient_subject", 2),
        (PRONOUN_APPOSITIVE_NAME_RE, "relationship_subject", "relationship_subject", 2),
        (PATIENT_HERE_NAME_RE, "relationship_subject", "patient_subject", 1),
        (ONE_OF_PROVIDER_PATIENTS_NAME_RE, "relationship_subject", "relationship_subject", 1),
        (POSSESSIVE_RELATIONSHIP_NAME_RE, "relationship_subject", "relationship_subject", 1),
        (REVERSE_RELATIONSHIP_NAME_RE, "relationship_subject", "relationship_subject", 2),
        (THIS_IS_FOR_NAME_RE, "relationship_subject", "relationship_subject", 2),
        (SUBJECT_NAME_RE, "relationship_subject", "relationship_subject", 1),
        (SELF_PATIENT_OF_PROVIDER_RE, "relationship_subject", "patient_subject", 1),
        (PATIENT_OF_PROVIDER_NAME_RE, "relationship_subject", "relationship_subject", 2),
        (REQUEST_FOR_SUBJECT_NAME_RE, "relationship_subject", "relationship_subject", 2),
        (REQUEST_ON_SUBJECT_NAME_RE, "relationship_subject", "relationship_subject", 2),
        (LEADING_NAME_DOB_RE, "relationship_subject", "relationship_subject", 2),
        (COMPACT_DOB_SUBJECT_NAME_RE, "relationship_subject", "relationship_subject", 2),
        (RELATIONSHIP_NAME_RE, "relationship_subject", "relationship_subject", 1),
        (GENERIC_SUBJECT_CUE_NAME_RE, "relationship_subject", "relationship_subject", 2),
        (IT_IS_NAME_RE, "self_identification", "self_identification", 1),
        (SELF_NAME_RE, "self_identification", "self_identification", 1),
    ):
        for match in regex.finditer(transcript):
            matched_name = match.group("name")
            if source == "self_identification" and _self_name_phrase_is_non_name(matched_name):
                continue
            raw_tokens = [
                token.strip(" .,'\"-").lower()
                for token in NAME_TOKEN_RE.findall(matched_name)
                if token.strip(" .,'\"-")
            ]
            if regex in {SUBJECT_NAME_RE, GENERIC_SUBJECT_CUE_NAME_RE} and len(raw_tokens) >= 2:
                if raw_tokens[0] in {"my", "his", "her", "their", "our"} and raw_tokens[1] in RELATIONSHIP_WORDS:
                    continue
            if regex is GENERIC_SUBJECT_CUE_NAME_RE and raw_tokens and raw_tokens[0] in {"dr", "doctor"}:
                continue
            if regex is POSSESSIVE_RELATIONSHIP_NAME_RE and _relationship_prefix_is_caller_identity(
                transcript,
                match.start("prefix"),
            ):
                continue
            raw = _title_name(matched_name)
            if not raw:
                continue
            name_start = match.start("name")
            name_end = name_start + len(raw)
            if "subject_cue" in match.groupdict() and match.group("subject_cue") is not None:
                evidence_end = match.end("subject_cue")
            else:
                evidence_end = match.end("name") if _contains_spelling_token(matched_name) else name_end
            _add_name_candidate(
                candidates,
                seen,
                raw_name=raw,
                span=(name_start, name_end),
                source=source,
                transcript=transcript,
                cue=cue,
                evidence_start=match.start("prefix"),
                evidence_end=evidence_end,
                minimum_tokens=minimum_tokens,
            )
    _add_broad_name_recall_candidates(transcript, candidates, seen)
    return candidates


def _extract_spelled_sequences(transcript: str, name_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for match in SPELLING_RE.finditer(transcript):
        raw = match.group(0)
        letters = _spelled_letters(raw)
        if len(letters) < 3:
            continue
        item = _candidate_base(
            cid=f"spelling:{len(sequences)}",
            raw=raw,
            span=(match.start(), match.end()),
            source="spelled_letters",
            transcript=transcript,
            confidence_hint="high",
        )
        item["letters"] = letters
        sequences.append(item)

        prefix = transcript[max(0, match.start() - 48) : match.start()]
        name_tokens = NAME_TOKEN_RE.findall(prefix)
        if len(name_tokens) >= 2:
            raw_name = _title_name(" ".join(name_tokens[-2:]))
            if any(
                _name_key(candidate.get("value") or candidate.get("raw")) == _name_key(raw_name)
                for candidate in name_candidates
            ):
                continue
            seen = {(str(candidate.get("value") or candidate.get("raw")).lower(), str(candidate.get("source"))) for candidate in name_candidates}
            _add_name_candidate(
                name_candidates,
                seen,
                raw_name=raw_name,
                span=(match.start() - len(" ".join(name_tokens[-2:])), match.start()),
                source="spelled_sequence_context",
                transcript=transcript,
                cue="spelled_sequence",
                evidence_start=max(0, match.start() - len(prefix)),
                evidence_end=match.end(),
                minimum_tokens=2,
            )
    return sequences


def _add_interleaved_spelling_corrections(
    transcript: str,
    name_candidates: list[dict[str, Any]],
) -> None:
    seen: set[tuple[str, str, str]] = {
        (
            str(candidate.get("raw") or "").strip().lower(),
            str(candidate.get("value") or candidate.get("raw") or "").strip().lower(),
            str(candidate.get("source") or "").strip(),
        )
        for candidate in name_candidates
    }
    for match in INTERLEAVED_SPELLED_NAME_RE.finditer(transcript):
        raw_tokens = [match.group("first")]
        value_tokens = [_title_name(_spelled_letters(match.group("first_spell")))]
        if match.group("last") and match.group("last_spell"):
            raw_tokens.append(match.group("last"))
            value_tokens.append(_title_name(_spelled_letters(match.group("last_spell"))))

        raw = _title_name(" ".join(raw_tokens))
        value = _title_name(" ".join(value_tokens))
        if not _name_is_plausible(raw, minimum_tokens=1) or not _name_is_plausible(value, minimum_tokens=1):
            continue
        key = (raw.lower(), value.lower(), "transcript_spelling_corrected")
        if key in seen:
            continue

        name_start = match.start("first")
        name_end = match.end("last") if match.group("last") else match.end("first")
        item = _candidate_base(
            cid=f"name:{len(name_candidates)}",
            raw=raw,
            span=(name_start, name_end),
            source="transcript_spelling_corrected",
            transcript=transcript,
            confidence_hint="high",
            evidence_text=match.group(0).strip(" ,.;:"),
        )
        item["value"] = value
        item["caller_id_used"] = ""
        if "spelled_sequence" not in item["nearby_cues"]:
            item["nearby_cues"].append("spelled_sequence")
        name_candidates.append(item)
        seen.add(key)


def _apply_spelling_corrections(
    transcript: str,
    name_candidates: list[dict[str, Any]],
    spelled_sequences: list[dict[str, Any]],
) -> None:
    seen: set[tuple[str, str, str]] = {
        (
            str(candidate.get("raw") or "").strip().lower(),
            str(candidate.get("value") or candidate.get("raw") or "").strip().lower(),
            str(candidate.get("source") or "").strip(),
        )
        for candidate in name_candidates
    }
    for spelling in spelled_sequences:
        letters = str(spelling.get("letters") or "").strip()
        corrected_token = _title_name(letters)
        if len(letters) < 3 or not corrected_token:
            continue
        span = spelling.get("span") or [0, 0]
        try:
            spell_start = int(span[0])
            spell_end = int(span[1])
        except (TypeError, ValueError):
            continue

        best_candidate: dict[str, Any] | None = None
        best_index = -1
        best_corrected_value = ""
        best_score = 0.0
        best_distance = 10_000
        for candidate in name_candidates:
            candidate_span = candidate.get("span") or [0, 0]
            try:
                candidate_end = int(candidate_span[1])
            except (TypeError, ValueError):
                continue
            distance = spell_start - candidate_end
            if distance < 0 or distance > 120:
                continue
            tokens = str(candidate.get("value") or candidate.get("raw") or "").split()
            candidate_value = _title_name(" ".join(tokens))
            if len(tokens) >= 2 and _name_key(candidate_value) == _name_key(corrected_token):
                score = 1.0
                if score > best_score or (score == best_score and distance < best_distance):
                    best_candidate = candidate
                    best_index = -1
                    best_corrected_value = candidate_value
                    best_score = score
                    best_distance = distance
                continue
            for index, token in enumerate(tokens):
                score = _name_similarity(token, corrected_token)
                if _rough_name_key(token) == _rough_name_key(corrected_token):
                    score = max(score, 0.95)
                if score < 0.50:
                    continue
                if score > best_score or (score == best_score and distance < best_distance):
                    best_candidate = candidate
                    best_index = index
                    best_score = score
                    best_distance = distance

        if best_candidate is None:
            continue

        corrected_value = best_corrected_value
        if not corrected_value:
            current_tokens = str(best_candidate.get("value") or best_candidate.get("raw") or "").split()
            if best_index < 0 or best_index >= len(current_tokens):
                continue
            current_tokens[best_index] = corrected_token
            corrected_value = _title_name(" ".join(current_tokens))
        if not corrected_value:
            continue
        raw = str(best_candidate.get("raw") or best_candidate.get("value") or "").strip()
        key = (raw.lower(), corrected_value.lower(), "transcript_spelling_corrected")
        if key in seen:
            continue
        candidate_span = best_candidate.get("span") or [0, 0]
        try:
            evidence_start = max(0, int(candidate_span[0]))
        except (TypeError, ValueError):
            evidence_start = 0
        evidence_end = min(len(transcript), spell_end)
        corrected = dict(best_candidate)
        corrected["id"] = f"name:{len(name_candidates)}"
        corrected["raw"] = raw
        corrected["value"] = corrected_value
        corrected["source"] = "transcript_spelling_corrected"
        corrected["caller_id_used"] = ""
        corrected["evidence_text"] = transcript[evidence_start:evidence_end].strip(" ,.;:")
        corrected["confidence_hint"] = "high"
        nearby_cues = corrected.setdefault("nearby_cues", [])
        if "spelled_sequence" not in nearby_cues:
            nearby_cues.append("spelled_sequence")
        name_candidates.append(corrected)
        seen.add(key)


def _semantic_events(transcript: str, number_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not number_candidates and re.search(r"\b(?:call\s+me\s+back|call\s+back|number\s+on\s+file)\b", transcript, re.IGNORECASE):
        events.append(
            {
                "event_type": "callback_request_no_number",
                "evidence_text": transcript.strip(),
                "confidence_hint": "medium",
            }
        )

    for match in DATE_RE.finditer(transcript):
        parsed = _parse_date(match.group(0))
        if not parsed:
            continue
        cues, _phrases = _cue_window(transcript, match.start(), match.end())
        if "appointment" in cues and "dob" not in cues:
            events.append(
                {
                    "event_type": "appointment_date",
                    "raw": match.group(0),
                    "normalized": _format_date(parsed),
                    "span": [match.start(), match.end()],
                    "evidence_text": match.group(0),
                    "confidence_hint": "high",
                }
            )
    return events


def extract_candidates(
    transcript: str,
    *,
    caller_id: str = "",
    mailbox: str = "",
    transcript_id: str = "",
) -> dict[str, Any]:
    """Collect high-recall field candidates without deciding final truth."""

    text = normalize_transcript(transcript)
    number_candidates = _extract_numeric_phones(text)
    _extract_spoken_phones(text, number_candidates)
    dob_candidates = _extract_dobs(text)
    name_candidates = _extract_names(text)
    spelled_sequences = _extract_spelled_sequences(text, name_candidates)
    _add_interleaved_spelling_corrections(text, name_candidates)
    _apply_spelling_corrections(text, name_candidates, spelled_sequences)
    name_correction_candidates = _add_caller_id_name_candidates(name_candidates, caller_id or "")

    return {
        "transcript_id": transcript_id or "",
        "caller_id": caller_id or "",
        "mailbox": mailbox or "",
        "number_candidates": number_candidates,
        "dob_candidates": dob_candidates,
        "name_candidates": name_candidates,
        "name_correction_candidates": name_correction_candidates,
        "spelled_sequences": spelled_sequences,
        "semantic_events": _semantic_events(text, number_candidates),
        "possible_errors": [],
    }
