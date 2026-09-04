from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from alignment import align_word_streams
from verification import extract_numbers_from_text, format_phone_digits, normalize_phone_candidate


@dataclass
class AsrToken:
    run_id: str
    engine: str
    audio_view: str
    word: str
    start: Optional[float]
    end: Optional[float]
    confidence: Optional[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "engine": self.engine,
            "audio_view": self.audio_view,
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class DisagreementSpan:
    span_id: str
    start: Optional[float]
    end: Optional[float]
    primary_text: str
    alternatives: list[dict[str, Any]]
    reasons: list[str] = field(default_factory=list)
    contains_digits: bool = False
    field_hint: str = "general"
    token_length: int = 0
    primary_confidence_mean: Optional[float] = None
    primary_confidence_min: Optional[float] = None
    is_insertion: bool = False
    insert_after_text: str = ""
    insert_before_text: str = ""
    primary_word_start: Optional[int] = None
    primary_word_end: Optional[int] = None
    local_context: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "start": self.start,
            "end": self.end,
            "primary_text": self.primary_text,
            "alternatives": list(self.alternatives or []),
            "reasons": list(self.reasons or []),
            "contains_digits": bool(self.contains_digits),
            "field_hint": self.field_hint,
            "token_length": self.token_length,
            "primary_confidence_mean": self.primary_confidence_mean,
            "primary_confidence_min": self.primary_confidence_min,
            "is_insertion": bool(self.is_insertion),
            "insert_after_text": self.insert_after_text,
            "insert_before_text": self.insert_before_text,
            "primary_word_start": self.primary_word_start,
            "primary_word_end": self.primary_word_end,
            "local_context": self.local_context,
        }


@dataclass
class TranscriptCorrection:
    span_id: str
    old_text: str
    new_text: str
    decision_type: str
    confidence: Optional[float]
    sources: list[str]
    needs_review: bool
    reason_code: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    reason_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "decision_type": self.decision_type,
            "confidence": self.confidence,
            "sources": list(self.sources or []),
            "needs_review": bool(self.needs_review),
            "reason_code": self.reason_code,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "reason_codes": list(self.reason_codes or ([self.reason_code] if self.reason_code else [])),
        }


MEDICAL_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"call|callback|phone|number|fax|dob|date of birth|birth|birthday|"
    r"this is|my name|patient|doctor|dr|clinic|appointment|urgent|"
    r"prescription|refill|insurance"
    r")\b"
)

DOB_CONTEXT_RE = re.compile(r"(?i)\b(dob|date of birth|birth date|birthday|born)\b")
PHONE_CONTEXT_RE = re.compile(r"(?i)\b(call|callback|call back|phone|number|reach me|area code)\b")
FAX_CONTEXT_RE = re.compile(r"(?i)\b(fax|send it to)\b")
NAME_CONTEXT_RE = re.compile(r"(?i)\b(this is|my name|patient name|calling for|regarding|on behalf of)\b")
PROVIDER_CONTEXT_RE = re.compile(r"(?i)\b(dr|doctor|provider|surgeon|physician|pa|np)\b")
LOCATION_CONTEXT_RE = re.compile(r"(?i)\b(clinic|office|location|hospital|medical center|suite|building)\b")

CRITICAL_FIELD_HINTS = {
    "callback_number",
    "fax_number",
    "dob",
    "name",
    "provider",
    "location",
    "digits",
}


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _word_text(word: dict[str, Any]) -> str:
    return str(word.get("word", word.get("text", "")) or "").strip()


def _run_text(run: dict[str, Any]) -> str:
    return str(run.get("processed_text") or run.get("transcript") or run.get("text") or "")


def _words_from_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    words = run.get("words") or run.get("word_timestamps") or []
    if isinstance(words, list) and words:
        return [dict(item) for item in words if isinstance(item, dict)]
    tokens = re.findall(r"\S+", _run_text(run))
    return [{"word": token} for token in tokens]


def _run_source(run: dict[str, Any]) -> str:
    engine = str(run.get("engine") or "asr")
    role = str(run.get("role") or "peer")
    audio_view = str(run.get("audio_view") or "canonical")
    return f"{engine}:{role}:{audio_view}"


def _confidence_values(words: list[dict[str, Any]]) -> list[float]:
    values = [
        _optional_float(word.get("probability", word.get("confidence")))
        for word in words
    ]
    return [value for value in values if value is not None]


def _mean_confidence(words: list[dict[str, Any]]) -> Optional[float]:
    usable = _confidence_values(words)
    if not usable:
        return None
    return round(sum(usable) / len(usable), 6)


def _min_confidence(words: list[dict[str, Any]]) -> Optional[float]:
    usable = _confidence_values(words)
    if not usable:
        return None
    return round(min(usable), 6)


def _bounds(words: list[dict[str, Any]], start: int, end: int) -> tuple[Optional[float], Optional[float]]:
    if not words or start >= end:
        return None, None
    start_value = _optional_float(words[start].get("start")) if start < len(words) else None
    end_index = min(len(words), end) - 1
    end_value = _optional_float(words[end_index].get("end")) if end_index >= 0 else None
    return start_value, end_value


def _slice_text(words: list[dict[str, Any]], start: int, end: int) -> str:
    return " ".join(_word_text(word) for word in words[start:end]).strip()


def _context_window(words: list[dict[str, Any]], start: int, end: int, padding: int = 4) -> str:
    if not words:
        return ""
    anchor_start = max(0, min(start, len(words)))
    anchor_end = max(anchor_start, min(end, len(words)))
    left = max(0, anchor_start - padding)
    right = min(len(words), max(anchor_end, anchor_start + 1) + padding)
    return _slice_text(words, left, right)


def _token_length(value: Any) -> int:
    return len(re.findall(r"\S+", str(value or "").strip()))


def _contains_digits(value: Any) -> bool:
    return bool(re.search(r"\d", str(value or "")))


def _field_hint(primary_text: str, peer_text: str, context_text: str = "") -> str:
    combined = f"{context_text} {primary_text} {peer_text}"
    if FAX_CONTEXT_RE.search(combined):
        return "fax_number"
    if PHONE_CONTEXT_RE.search(combined) and _contains_digits(combined):
        return "callback_number"
    if DOB_CONTEXT_RE.search(combined):
        return "dob"
    if NAME_CONTEXT_RE.search(combined):
        return "name"
    if PROVIDER_CONTEXT_RE.search(combined):
        return "provider"
    if LOCATION_CONTEXT_RE.search(combined):
        return "location"
    if _contains_digits(combined):
        return "digits"
    return "general"


def _is_relevant(primary_text: str, peer_text: str, reasons: list[str], context_text: str = "") -> bool:
    combined = f"{context_text} {primary_text} {peer_text}"
    return bool(
        re.search(r"\d", combined)
        or MEDICAL_CONTEXT_RE.search(combined)
        or "low_confidence" in reasons
        or "low_whisper_confidence" in reasons
    )


def _alternative_from_run(
    run: dict[str, Any],
    words: list[dict[str, Any]],
    start: int,
    end: int,
    text: str,
    reason: str,
    is_primary: bool = False,
) -> dict[str, Any]:
    sliced = words[start:end]
    alt_start, alt_end = _bounds(words, start, end)
    engine = str(run.get("engine") or "asr")
    role = str(run.get("role") or "peer")
    audio_view = str(run.get("audio_view") or "canonical")
    return {
        "text": text,
        "source": f"{engine}:{role}:{audio_view}",
        "run_id": run.get("run_id"),
        "engine": engine,
        "role": role,
        "audio_view": audio_view,
        "start": alt_start,
        "end": alt_end,
        "confidence": _mean_confidence(sliced),
        "confidence_mean": _mean_confidence(sliced),
        "confidence_min": _min_confidence(sliced),
        "token_length": _token_length(text),
        "reason": reason,
        "is_primary": is_primary,
    }


def build_disagreement_spans(
    primary_run: dict[str, Any],
    peer_runs: list[dict[str, Any]],
    settings: Any,
) -> list[DisagreementSpan]:
    primary_words = _words_from_run(primary_run)
    spans: list[DisagreementSpan] = []
    low_word_threshold = float(getattr(settings, "router_low_word_prob_threshold", 0.65))

    for peer_index, peer_run in enumerate(peer_runs or []):
        if not isinstance(peer_run, dict):
            continue
        peer_words = _words_from_run(peer_run)
        if not primary_words or not peer_words:
            continue
        alignment = align_word_streams(primary_words, peer_words)
        for item_index, item in enumerate(alignment):
            if item.get("tag") == "equal":
                continue
            primary_start, primary_end = item.get("a", [0, 0])
            peer_start, peer_end = item.get("b", [0, 0])
            primary_text = _slice_text(primary_words, primary_start, primary_end)
            peer_text = _slice_text(peer_words, peer_start, peer_end)
            local_context = " ".join(
                item
                for item in (
                    _context_window(primary_words, primary_start, primary_end),
                    _context_window(peer_words, peer_start, peer_end),
                )
                if item
            )
            is_insertion = primary_start == primary_end and bool(peer_text)
            insert_after_text = _word_text(primary_words[primary_start - 1]) if is_insertion and primary_start > 0 else ""
            insert_before_text = (
                _word_text(primary_words[primary_start])
                if is_insertion and primary_start < len(primary_words)
                else ""
            )
            reasons = ["asr_disagreement"]
            if is_insertion:
                reasons.append("parakeet_insertion")
            if re.search(r"\d", f"{primary_text} {peer_text}"):
                reasons.append("digit_context")
            primary_slice = primary_words[primary_start:primary_end]
            primary_confidence = _mean_confidence(primary_slice)
            primary_min_confidence = _min_confidence(primary_slice)
            if primary_confidence is not None and primary_confidence < low_word_threshold:
                reasons.extend(["low_confidence", "low_whisper_confidence"])
            hint = _field_hint(primary_text, peer_text, local_context)
            if hint != "general":
                reasons.append("medical_context" if hint in CRITICAL_FIELD_HINTS else "context")
            if not _is_relevant(primary_text, peer_text, reasons, local_context):
                continue
            start, end = _bounds(primary_words, primary_start, primary_end)
            if start is None or end is None:
                start, end = _bounds(peer_words, peer_start, peer_end)
            alternatives = []
            if primary_text:
                alternatives.append(
                    _alternative_from_run(
                        primary_run,
                        primary_words,
                        primary_start,
                        primary_end,
                        primary_text,
                        "primary",
                        is_primary=True,
                    )
                )
            if peer_text:
                alternatives.append(
                    _alternative_from_run(
                        peer_run,
                        peer_words,
                        peer_start,
                        peer_end,
                        peer_text,
                        "peer_asr",
                    )
                )
            spans.append(
                DisagreementSpan(
                    span_id=f"lattice:{primary_run.get('run_id') or 'primary'}:{peer_index}:{item_index}",
                    start=start,
                    end=end,
                    primary_text=primary_text,
                    alternatives=alternatives,
                    reasons=list(dict.fromkeys(reasons)),
                    contains_digits=_contains_digits(f"{primary_text} {peer_text}"),
                    field_hint=hint,
                    token_length=max(_token_length(primary_text), _token_length(peer_text)),
                    primary_confidence_mean=primary_confidence,
                    primary_confidence_min=primary_min_confidence,
                    is_insertion=is_insertion,
                    insert_after_text=insert_after_text,
                    insert_before_text=insert_before_text,
                    primary_word_start=primary_start,
                    primary_word_end=primary_end,
                    local_context=local_context,
                )
            )

    for index, word in enumerate(primary_words):
        probability = _optional_float(word.get("probability", word.get("confidence")))
        if probability is None or probability >= low_word_threshold:
            continue
        text = _word_text(word)
        local_context = _context_window(primary_words, index, index + 1)
        if not _is_relevant(text, "", ["low_confidence"], local_context):
            continue
        start, end = _bounds(primary_words, index, index + 1)
        hint = _field_hint(text, "", local_context)
        spans.append(
            DisagreementSpan(
                span_id=f"lattice:{primary_run.get('run_id') or 'primary'}:low:{index}",
                start=start,
                end=end,
                primary_text=text,
                alternatives=[
                    _alternative_from_run(
                        primary_run,
                        primary_words,
                        index,
                        index + 1,
                        text,
                        "primary_low_confidence",
                        is_primary=True,
                    )
                ],
                reasons=list(dict.fromkeys(["low_confidence", "low_whisper_confidence"])),
                contains_digits=_contains_digits(text),
                field_hint=hint,
                token_length=_token_length(text),
                primary_confidence_mean=probability,
                primary_confidence_min=probability,
                primary_word_start=index,
                primary_word_end=index + 1,
                local_context=local_context,
            )
        )

    return spans


def _text_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _word_sequence_key(value: Any) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(value or "").casefold())
    return tuple(re.sub(r"[^a-z0-9]+", "", token) for token in tokens if re.sub(r"[^a-z0-9]+", "", token))


def _source_label(alternative: dict[str, Any]) -> str:
    return str(alternative.get("source") or alternative.get("engine") or "unknown")


def _alt_text(alternative: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(alternative.get("text") or alternative.get("value") or "").strip())


def _alt_normalized_number(alternative: dict[str, Any]) -> Optional[str]:
    normalized = normalize_phone_candidate(alternative.get("normalized_value") or alternative.get("normalized"))
    if normalized.valid and normalized.normalized:
        return normalized.normalized
    numbers = extract_numbers_from_text(_alt_text(alternative))
    if len(numbers) == 1 and numbers[0].normalized:
        return numbers[0].normalized
    return None


def _has_numeric_context(span: DisagreementSpan, alternative: dict[str, Any]) -> bool:
    text = f"{span.primary_text} {_alt_text(alternative)}"
    return bool(re.search(r"\d", text) or re.search(r"(?i)\b(call|callback|phone|fax|number)\b", text))


def _is_context_bias(alternative: dict[str, Any]) -> bool:
    return "context_bias" in _source_label(alternative) or alternative.get("source") == "bias"


def _is_grounded_choice(span: DisagreementSpan, alternative: dict[str, Any]) -> tuple[bool, str]:
    if alternative.get("is_primary"):
        return False, "primary_already_selected"
    if alternative.get("needs_review"):
        return False, "alternative_needs_review"
    if _is_context_bias(alternative) and not alternative.get("spoken_evidence"):
        return False, "context_bias_without_spoken_evidence"
    if alternative.get("llm_adjudicated"):
        return True, str(alternative.get("reason_code") or "llm_adjudicated_grounded_alternative")
    if _has_numeric_context(span, alternative):
        if alternative.get("strong") or alternative.get("verified") or alternative.get("reason_code") in {
            "verified_digit_consensus",
            "two_acoustic_views_agree",
            "clip_full_pass_agree",
        }:
            return True, str(alternative.get("reason_code") or "verified_numeric_alternative")
        return False, "numeric_alternative_not_strong"
    if alternative.get("strong") or alternative.get("verified"):
        return True, str(alternative.get("reason_code") or "verified_text_alternative")
    return False, "uncertain_alternative"


def _chosen_text(alternative: dict[str, Any]) -> str:
    normalized = _alt_normalized_number(alternative)
    if normalized:
        return alternative.get("formatted_value") or format_phone_digits(normalized) or normalized
    return _alt_text(alternative)


def _is_parakeet_alternative(alternative: dict[str, Any]) -> bool:
    source = _source_label(alternative).casefold()
    engine = str(alternative.get("engine") or "").casefold()
    return "parakeet" in source or engine == "parakeet"


def _is_critical_span(span: DisagreementSpan) -> bool:
    return bool(span.contains_digits or str(span.field_hint or "general") in CRITICAL_FIELD_HINTS)


def _is_time_aligned(span: DisagreementSpan, alternative: dict[str, Any]) -> bool:
    span_start = _optional_float(span.start)
    span_end = _optional_float(span.end)
    alt_start = _optional_float(alternative.get("start"))
    alt_end = _optional_float(alternative.get("end"))
    if None in {span_start, span_end, alt_start, alt_end}:
        return False
    if span_end <= span_start or alt_end <= alt_start:
        return False
    overlap = min(span_end, alt_end) - max(span_start, alt_start)
    if overlap > 0:
        return True
    span_mid = (span_start + span_end) / 2
    alt_mid = (alt_start + alt_end) / 2
    return abs(span_mid - alt_mid) <= max(0.35, (span_end - span_start), (alt_end - alt_start))


def _anchor_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _transcript_tokens(text: str) -> list[dict[str, Any]]:
    return [
        {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "key": _anchor_key(match.group(0)),
        }
        for match in re.finditer(r"\S+", text or "")
    ]


def _split_boundary_punctuation(token: str) -> tuple[str, str, str]:
    match = re.match(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9]*)$", str(token or ""))
    if not match:
        return "", str(token or ""), ""
    leading, core, trailing = match.groups()
    if not core:
        return "", str(token or ""), ""
    return leading, core, trailing


def _apply_case_style(primary_core: str, candidate_core: str) -> str:
    if not re.search(r"[A-Za-z]", candidate_core or ""):
        return candidate_core
    if primary_core.isupper():
        return candidate_core.upper()
    if primary_core[:1].isupper() and candidate_core.islower():
        return candidate_core[:1].upper() + candidate_core[1:]
    return candidate_core


def _merge_token_with_primary_punctuation(primary_token: str, candidate_token: str) -> str:
    primary_leading, primary_core, primary_trailing = _split_boundary_punctuation(primary_token)
    _candidate_leading, candidate_core, _candidate_trailing = _split_boundary_punctuation(candidate_token)
    if not candidate_core:
        return candidate_token
    return primary_leading + _apply_case_style(primary_core, candidate_core) + primary_trailing


def _preserve_boundary_punctuation(primary_text: str, candidate_text: str) -> str:
    primary_tokens = re.findall(r"\S+", str(primary_text or ""))
    candidate_tokens = re.findall(r"\S+", str(candidate_text or ""))
    if not primary_tokens or not candidate_tokens:
        return re.sub(r"\s+", " ", str(candidate_text or "").strip())

    primary_leading, _primary_first_core, _primary_first_trailing = _split_boundary_punctuation(primary_tokens[0])
    _primary_last_leading, _primary_last_core, primary_trailing = _split_boundary_punctuation(primary_tokens[-1])

    first_leading, first_core, first_trailing = _split_boundary_punctuation(candidate_tokens[0])
    if primary_leading and first_core:
        candidate_tokens[0] = primary_leading + first_core + first_trailing
    elif primary_leading and not first_leading:
        candidate_tokens[0] = primary_leading + candidate_tokens[0]

    last_leading, last_core, _last_trailing = _split_boundary_punctuation(candidate_tokens[-1])
    if primary_trailing and last_core:
        candidate_tokens[-1] = last_leading + last_core + primary_trailing
    elif primary_trailing:
        candidate_tokens[-1] = candidate_tokens[-1] + primary_trailing
    return " ".join(candidate_tokens)


def _candidate_text_with_primary_punctuation(primary_text: str, candidate_text: str) -> tuple[str, Optional[str]]:
    primary = re.sub(r"\s+", " ", str(primary_text or "").strip())
    candidate = re.sub(r"\s+", " ", str(candidate_text or "").strip())
    if not primary or not candidate:
        return candidate, None
    if _word_sequence_key(primary) == _word_sequence_key(candidate):
        return primary, "punctuation_only_candidate_rejected"

    primary_tokens = re.findall(r"\S+", primary)
    candidate_tokens = re.findall(r"\S+", candidate)
    if primary_tokens and len(primary_tokens) == len(candidate_tokens):
        preserved = " ".join(
            _merge_token_with_primary_punctuation(primary_token, candidate_token)
            for primary_token, candidate_token in zip(primary_tokens, candidate_tokens)
        )
        if preserved != candidate:
            return preserved, "primary_punctuation_preserved"
        return candidate, None

    preserved = _preserve_boundary_punctuation(primary, candidate)
    if preserved != candidate:
        return preserved, "primary_punctuation_preserved"
    return candidate, None


def _apply_local_insertion(transcript: str, span: DisagreementSpan, new_text: str) -> tuple[str, Optional[str]]:
    insert = re.sub(r"\s+", " ", str(new_text or "").strip())
    if not insert:
        return transcript, "empty_insert_text"
    after_key = _anchor_key(span.insert_after_text)
    before_key = _anchor_key(span.insert_before_text)
    if not after_key and not before_key:
        return transcript, "insertion_anchor_missing"

    tokens = _transcript_tokens(transcript)
    matches: list[int] = []
    if after_key and before_key:
        for index in range(len(tokens) - 1):
            if tokens[index]["key"] == after_key and tokens[index + 1]["key"] == before_key:
                matches.append(index)
        if len(matches) != 1:
            return transcript, "insertion_anchor_not_unique" if matches else "insertion_anchor_not_found"
        position = tokens[matches[0]]["end"]
        return f"{transcript[:position]} {insert}{transcript[position:]}", None

    if before_key:
        matches = [index for index, token in enumerate(tokens) if token["key"] == before_key]
        if len(matches) != 1:
            return transcript, "insertion_anchor_not_unique" if matches else "insertion_anchor_not_found"
        position = tokens[matches[0]]["start"]
        return f"{transcript[:position]}{insert} {transcript[position:]}", None

    matches = [index for index, token in enumerate(tokens) if token["key"] == after_key]
    if len(matches) != 1:
        return transcript, "insertion_anchor_not_unique" if matches else "insertion_anchor_not_found"
    position = tokens[matches[0]]["end"]
    return f"{transcript[:position]} {insert}{transcript[position:]}", None


def _has_parakeet_confirmation(alternative: dict[str, Any]) -> bool:
    reason = str(alternative.get("reason_code") or alternative.get("reason") or "").casefold()
    return bool(
        alternative.get("strong")
        or alternative.get("verified")
        or reason in {
            "verified_digit_consensus",
            "two_acoustic_views_agree",
            "clip_full_pass_agree",
            "parakeet_clip_agreement",
        }
        or "digit_consensus" in reason
        or "clip_full_pass" in reason
    )


def _base_decision(
    span: DisagreementSpan,
    decision_type: str,
    old_text: str,
    new_text: str,
    score: float,
    sources: list[str],
    reason_codes: list[str],
    needs_review: bool,
) -> TranscriptCorrection:
    deduped = list(dict.fromkeys(reason_codes or ["no_reason"]))
    return TranscriptCorrection(
        span_id=span.span_id,
        old_text=old_text,
        new_text=new_text,
        decision_type=decision_type,
        confidence=round(score, 6),
        sources=sources,
        needs_review=needs_review,
        reason_code=deduped[0],
        start=span.start,
        end=span.end,
        score=round(score, 6),
        reason_codes=deduped,
    )


def score_parakeet_replacement(span: DisagreementSpan, settings: Any) -> TranscriptCorrection:
    """Score one guarded replacement candidate.

    Whisper is the default winner. Parakeet can win when local evidence is
    short, grounded, and strong enough; a constrained LLM-selected alternative
    can also win, but only after the same local scoring gate accepts it.
    """

    old_text = re.sub(r"\s+", " ", str(span.primary_text or "").strip())
    is_insertion = bool(span.is_insertion and not old_text)
    if not old_text and not is_insertion:
        return _base_decision(span, "keep_primary", "", "", 0.0, [], ["empty_primary_span"], False)

    min_score = float(getattr(settings, "ensemble_min_replace_score", 0.85) or 0.85)
    max_tokens = int(getattr(settings, "ensemble_max_replace_tokens", 6) or 6)
    allow_general = bool(getattr(settings, "ensemble_allow_general_text", False))
    critical_only = bool(getattr(settings, "ensemble_critical_fields_only", True))
    low_threshold = float(getattr(settings, "router_low_word_prob_threshold", 0.65) or 0.65)

    guarded_alternatives = [
        alternative
        for alternative in span.alternatives or []
        if isinstance(alternative, dict)
        and not alternative.get("is_primary")
        and (_is_parakeet_alternative(alternative) or alternative.get("llm_adjudicated"))
    ]
    if not guarded_alternatives:
        return _base_decision(
            span,
            "keep_primary",
            old_text,
            old_text,
            0.0,
            [],
            ["no_parakeet_alternative"],
            False,
        )

    best: Optional[TranscriptCorrection] = None
    for alternative in guarded_alternatives:
        new_text = _chosen_text(alternative)
        source = _source_label(alternative)
        llm_adjudicated = bool(alternative.get("llm_adjudicated"))
        if not new_text or (old_text and _text_key(new_text) == _text_key(old_text)):
            candidate = _base_decision(
                span,
                "keep_primary",
                old_text,
                old_text,
                0.0,
                [source],
                ["alternative_matches_primary"],
                False,
            )
        else:
            old_tokens = _token_length(old_text)
            new_tokens = _token_length(new_text)
            critical = _is_critical_span(span)
            time_aligned = _is_time_aligned(span, alternative)
            score = 0.2
            reasons = ["asr_disagreement"]
            hard_review = False
            hard_keep = False
            if is_insertion:
                reasons.append("parakeet_insertion")
                if not span.insert_after_text and not span.insert_before_text:
                    hard_review = True
                    reasons.append("insertion_anchor_missing")

            if _is_context_bias(alternative) and not alternative.get("spoken_evidence"):
                hard_review = True
                reasons.append("context_bias_without_spoken_evidence")
            if critical_only and not critical and not llm_adjudicated:
                hard_keep = True
                reasons.append("general_text_not_allowed")
            elif not allow_general and str(span.field_hint or "general") == "general" and not llm_adjudicated:
                hard_keep = True
                reasons.append("general_text_not_allowed")
            elif llm_adjudicated:
                reasons.append(str(alternative.get("reason_code") or "llm_adjudicated_grounded_alternative"))

            if new_tokens <= max_tokens:
                score += 0.15
                reasons.append("short_local_insertion" if is_insertion else "short_local_replacement")
            else:
                hard_review = True
                score -= 0.2
                reasons.append("replacement_too_long")

            if not is_insertion and new_tokens > max(old_tokens + 2, old_tokens * 2, 2):
                hard_review = True
                score -= 0.25
                reasons.append("unsupported_extra_content")

            if time_aligned:
                score += 0.15
                reasons.append("time_aligned")
            else:
                reasons.append("timing_missing")
                if not critical:
                    hard_review = True
                else:
                    score -= 0.1

            low_confidence = False
            for value in (span.primary_confidence_mean, span.primary_confidence_min):
                if value is not None and value < low_threshold:
                    low_confidence = True
                    break
            if "low_whisper_confidence" in span.reasons or "low_confidence" in span.reasons:
                low_confidence = True
            if low_confidence:
                score += 0.2
                reasons.append("low_whisper_confidence")

            hint = str(span.field_hint or "general")
            if hint in {"callback_number", "fax_number", "dob"}:
                score += 0.25
                reasons.append(f"{hint}_context")
            elif hint in {"name", "provider", "location"}:
                score += 0.18
                reasons.append(f"{hint}_context")
            elif span.contains_digits:
                score += 0.1
                reasons.append("digit_context")

            if _has_parakeet_confirmation(alternative):
                score += 0.25
                reasons.append(str(alternative.get("reason_code") or "parakeet_clip_agreement"))
            if llm_adjudicated:
                score += 0.25

            score = max(0.0, min(1.0, score))
            if hard_keep:
                decision_type = "keep_primary"
                needs_review = False
                new_value = old_text
            elif hard_review:
                decision_type = "needs_review"
                needs_review = True
                new_value = new_text
            elif score >= min_score:
                decision_type = "replace_with_parakeet" if _is_parakeet_alternative(alternative) else "replace_with_alternative"
                needs_review = False
                new_value = new_text
            elif critical and score >= 0.55:
                decision_type = "needs_review"
                needs_review = True
                new_value = new_text
                reasons.append("score_below_replace_threshold")
            else:
                decision_type = "keep_primary"
                needs_review = False
                new_value = old_text
                reasons.append("score_below_replace_threshold")

            candidate = _base_decision(
                span,
                decision_type,
                old_text,
                new_value,
                score,
                [source],
                reasons,
                needs_review,
            )

        if best is None or (candidate.score or 0.0) > (best.score or 0.0):
            best = candidate

    return best or _base_decision(span, "keep_primary", old_text, old_text, 0.0, [], ["no_usable_parakeet_alternative"], False)


def apply_ensemble_corrections(
    primary_transcript: str,
    disagreement_spans: list[DisagreementSpan],
    settings: Any,
    adjudicator: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
) -> tuple[str, list[TranscriptCorrection]]:
    corrected = str(primary_transcript or "")
    corrections: list[TranscriptCorrection] = []
    apply_enabled = bool(getattr(settings, "ensemble_correction_apply_enabled", False))

    for span in disagreement_spans or []:
        if not isinstance(span, DisagreementSpan):
            continue
        scoring_span = span
        adjudicated = _adjudicated_alternative(span, adjudicator, settings)
        if adjudicated is not None:
            scoring_span = replace(span, alternatives=[adjudicated, *(span.alternatives or [])])
        decision = score_parakeet_replacement(scoring_span, settings)
        if decision.decision_type in {"replace_with_parakeet", "replace_with_alternative"} and apply_enabled:
            old_text = decision.old_text
            new_text = decision.new_text
            if not old_text and span.is_insertion:
                updated, error = _apply_local_insertion(corrected, span, new_text)
                if error:
                    reasons = list(dict.fromkeys([*decision.reason_codes, error]))
                    decision = TranscriptCorrection(
                        span_id=decision.span_id,
                        old_text=old_text,
                        new_text=new_text,
                        decision_type="needs_review",
                        confidence=decision.confidence,
                        sources=decision.sources,
                        needs_review=True,
                        reason_code=error,
                        start=decision.start,
                        end=decision.end,
                        score=decision.score,
                        reason_codes=reasons,
                    )
                else:
                    corrected = updated
            elif not old_text or corrected.count(old_text) != 1:
                reason = "source_span_not_found" if old_text not in corrected else "source_span_not_unique"
                reasons = list(dict.fromkeys([*decision.reason_codes, reason]))
                decision = TranscriptCorrection(
                    span_id=decision.span_id,
                    old_text=old_text,
                    new_text=new_text,
                    decision_type="needs_review",
                    confidence=decision.confidence,
                    sources=decision.sources,
                    needs_review=True,
                    reason_code=reason,
                    start=decision.start,
                    end=decision.end,
                    score=decision.score,
                    reason_codes=reasons,
                )
            else:
                adjusted_text, punctuation_reason = _candidate_text_with_primary_punctuation(old_text, new_text)
                if punctuation_reason == "punctuation_only_candidate_rejected":
                    reasons = list(dict.fromkeys([*decision.reason_codes, punctuation_reason]))
                    decision = TranscriptCorrection(
                        span_id=decision.span_id,
                        old_text=old_text,
                        new_text=old_text,
                        decision_type="keep_primary",
                        confidence=decision.confidence,
                        sources=decision.sources,
                        needs_review=False,
                        reason_code=punctuation_reason,
                        start=decision.start,
                        end=decision.end,
                        score=decision.score,
                        reason_codes=reasons,
                    )
                else:
                    if punctuation_reason:
                        new_text = adjusted_text
                        reasons = list(dict.fromkeys([*decision.reason_codes, punctuation_reason]))
                        decision = TranscriptCorrection(
                            span_id=decision.span_id,
                            old_text=old_text,
                            new_text=new_text,
                            decision_type=decision.decision_type,
                            confidence=decision.confidence,
                            sources=decision.sources,
                            needs_review=decision.needs_review,
                            reason_code=decision.reason_code,
                            start=decision.start,
                            end=decision.end,
                            score=decision.score,
                            reason_codes=reasons,
                        )
                    corrected = corrected.replace(old_text, new_text, 1)
        corrections.append(decision)

    if not apply_enabled:
        return str(primary_transcript or ""), corrections
    if corrected != str(primary_transcript or "") and _word_sequence_key(corrected) == _word_sequence_key(primary_transcript):
        corrections.append(
            TranscriptCorrection(
                span_id="transcript:punctuation_guard",
                old_text=str(primary_transcript or ""),
                new_text=corrected,
                decision_type="not_applied",
                confidence=None,
                sources=[],
                needs_review=False,
                reason_code="punctuation_only_candidate_rejected",
                reason_codes=["punctuation_only_candidate_rejected"],
            )
        )
        return str(primary_transcript or ""), corrections
    return corrected, corrections


def enrich_spans_with_clip_confirmations(
    disagreement_spans: list[DisagreementSpan],
    audit_rows: list[dict[str, Any]],
) -> list[DisagreementSpan]:
    confirmed_numbers: set[str] = set()
    for row in audit_rows or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        normalized = normalize_phone_candidate(row.get("normalized_value"))
        if normalized.valid and normalized.normalized and status in {
            "verified_digit_consensus",
            "verified",
            "parakeet_override",
        }:
            confirmed_numbers.add(normalized.normalized)
        parakeet_items = row.get("parakeet_json") or []
        if not isinstance(parakeet_items, list):
            continue
        for item in parakeet_items:
            if not isinstance(item, dict):
                continue
            for value in item.get("normalized_numbers") or []:
                number = normalize_phone_candidate(value)
                if number.valid and number.normalized:
                    confirmed_numbers.add(number.normalized)
            raw_output = item.get("raw_output") if isinstance(item.get("raw_output"), dict) else item
            consensus = raw_output.get("digit_consensus") if isinstance(raw_output, dict) else None
            if isinstance(consensus, dict):
                summary = consensus.get("summary") if isinstance(consensus.get("summary"), dict) else consensus
                number = normalize_phone_candidate(summary.get("consensus") if isinstance(summary, dict) else None)
                if number.valid and number.normalized and summary.get("strong"):
                    confirmed_numbers.add(number.normalized)

    if not confirmed_numbers:
        return disagreement_spans

    for span in disagreement_spans or []:
        if not isinstance(span, DisagreementSpan):
            continue
        for alternative in span.alternatives or []:
            if not isinstance(alternative, dict) or not _is_parakeet_alternative(alternative):
                continue
            number = _alt_normalized_number(alternative)
            if number and number in confirmed_numbers:
                alternative["verified"] = True
                alternative["strong"] = True
                alternative["reason_code"] = "verified_digit_consensus"
                alternative["spoken_evidence"] = True
    return disagreement_spans


def _adjudicated_alternative(
    span: DisagreementSpan,
    adjudicator: Optional[Callable[[dict[str, Any]], dict[str, Any]]],
    settings: Any,
) -> Optional[dict[str, Any]]:
    if not adjudicator or not getattr(settings, "transcript_lattice_llm_adjudication_enabled", False):
        return None
    decision = adjudicator(span.as_dict())
    if not isinstance(decision, dict):
        return None
    if decision.get("decision_type") in {"keep_primary", "uncertain"}:
        return None
    if "chosen_alternative_index" in decision:
        try:
            index = int(decision.get("chosen_alternative_index"))
        except (TypeError, ValueError):
            index = -1
        if 0 <= index < len(span.alternatives):
            alternative = span.alternatives[index]
            selected = dict(alternative)
            selected["llm_adjudicated"] = True
            selected["confidence"] = decision.get("confidence", selected.get("confidence"))
            selected["reason_code"] = str(decision.get("reason_code") or "llm_adjudicated_grounded_alternative")
            return selected
    selected_text = _text_key(decision.get("text") or decision.get("new_text"))
    selected_source = str(decision.get("source") or "")
    for alternative in span.alternatives:
        if selected_text and _text_key(_alt_text(alternative)) == selected_text:
            if not selected_source or selected_source == _source_label(alternative):
                selected = dict(alternative)
                selected["llm_adjudicated"] = True
                selected["confidence"] = decision.get("confidence", selected.get("confidence"))
                selected["reason_code"] = str(decision.get("reason_code") or "llm_adjudicated_grounded_alternative")
                return selected
    return None


def correct_transcript_constrained(
    primary_transcript: str,
    disagreement_spans: list[DisagreementSpan],
    settings: Any,
    adjudicator: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
) -> tuple[str, list[TranscriptCorrection]]:
    corrected = str(primary_transcript or "")
    corrections: list[TranscriptCorrection] = []

    for span in disagreement_spans or []:
        if not isinstance(span, DisagreementSpan):
            continue
        old_text = re.sub(r"\s+", " ", str(span.primary_text or "").strip())
        if not old_text:
            continue

        selected = _adjudicated_alternative(span, adjudicator, settings)
        selected_reason = ""
        if selected is not None:
            grounded, selected_reason = _is_grounded_choice(span, selected)
            if not grounded:
                selected = None

        if selected is None:
            for alternative in span.alternatives:
                grounded, reason = _is_grounded_choice(span, alternative)
                if grounded:
                    selected = alternative
                    selected_reason = reason
                    break

        if selected is None:
            review_reason = "uncertain_alternative"
            for alternative in span.alternatives:
                if not alternative.get("is_primary"):
                    _, review_reason = _is_grounded_choice(span, alternative)
                    break
            corrections.append(
                TranscriptCorrection(
                    span_id=span.span_id,
                    old_text=old_text,
                    new_text=old_text,
                    decision_type="needs_review",
                    confidence=None,
                    sources=[],
                    needs_review=True,
                    reason_code=review_reason,
                    start=span.start,
                    end=span.end,
                )
            )
            continue

        raw_new_text = _chosen_text(selected)
        if not raw_new_text:
            continue
        new_text, punctuation_reason = _candidate_text_with_primary_punctuation(old_text, raw_new_text)
        if punctuation_reason == "punctuation_only_candidate_rejected":
            corrections.append(
                TranscriptCorrection(
                    span_id=span.span_id,
                    old_text=old_text,
                    new_text=old_text,
                    decision_type="keep_primary",
                    confidence=_optional_float(selected.get("confidence")),
                    sources=[_source_label(selected)],
                    needs_review=False,
                    reason_code=punctuation_reason,
                    start=span.start,
                    end=span.end,
                    reason_codes=list(dict.fromkeys([selected_reason or "grounded_alternative", punctuation_reason])),
                )
            )
            continue
        if _text_key(new_text) == _text_key(old_text):
            continue
        if old_text not in corrected:
            corrections.append(
                TranscriptCorrection(
                    span_id=span.span_id,
                    old_text=old_text,
                    new_text=new_text,
                    decision_type="not_applied",
                    confidence=_optional_float(selected.get("confidence")),
                    sources=[_source_label(selected)],
                    needs_review=True,
                    reason_code="source_span_not_found",
                    start=span.start,
                    end=span.end,
                )
            )
            continue

        corrected = corrected.replace(old_text, new_text, 1)
        reason_codes = list(dict.fromkeys([selected_reason or "grounded_alternative"]))
        if punctuation_reason:
            reason_codes.append(punctuation_reason)
        corrections.append(
            TranscriptCorrection(
                span_id=span.span_id,
                old_text=old_text,
                new_text=new_text,
                decision_type="grounded_replacement",
                confidence=_optional_float(selected.get("confidence")),
                sources=[_source_label(selected)],
                needs_review=False,
                reason_code=reason_codes[0],
                start=span.start,
                end=span.end,
                reason_codes=reason_codes,
            )
        )

    if corrected != str(primary_transcript or "") and _word_sequence_key(corrected) == _word_sequence_key(primary_transcript):
        corrections.append(
            TranscriptCorrection(
                span_id="transcript:punctuation_guard",
                old_text=str(primary_transcript or ""),
                new_text=corrected,
                decision_type="not_applied",
                confidence=None,
                sources=[],
                needs_review=False,
                reason_code="punctuation_only_candidate_rejected",
                reason_codes=["punctuation_only_candidate_rejected"],
            )
        )
        return str(primary_transcript or ""), corrections
    return corrected, corrections
