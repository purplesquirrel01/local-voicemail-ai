from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from agent_constraints import build_agent_constraint_schema
from candidate_extractor import extract_candidates
from final_resolver import (
    FINAL_SCHEMA_KEYS,
    empty_final_json,
    format_dob,
    merge_agent_outputs,
    merge_split_agent_outputs,
    normalize_phone_candidate,
    parse_dob,
    validate_final_json,
)
from gemma_agents import CachedAgent


logger = logging.getLogger("extraction_orchestrator")

AgentTraceSink = Callable[[dict[str, Any]], None]


def _emit_agent_trace(
    trace_sink: AgentTraceSink | None,
    agent: str,
    output: dict[str, Any],
    duration_ms: int,
    constraint_mode: str,
    *,
    status: str = "completed",
    error: str = "",
) -> None:
    if trace_sink is None:
        return
    try:
        trace_sink(
            {
                "agent": str(agent),
                "output": dict(output),
                "duration_ms": max(0, int(duration_ms)),
                "constraint_mode": str(constraint_mode or "disabled"),
                "status": str(status),
                "error": str(error),
            }
        )
    except Exception:
        # Optional diagnostic tracing is observational and must never alter extraction.
        return

INPUT_JSON_DELIMITER = "\n\nInput JSON:\n"
VALID_MODES = {"legacy", "candidate_agents", "shadow_candidate_agents"}
VALID_OUTPUT_SCHEMAS = {"compact", "verbose"}
VALID_TOPOLOGIES = {
    "identity",
    "numbers_only",
    "custom",
    "scout_subject_general_fallback",
    "split_identity",
    "split_identity_correction",
    "split_identity_dual_correction",
    "split_identity_subject_fallback_dual_correction",
}
VALID_EXECUTION_MODES = {"sequential_conversation", "parallel_http"}
SPLIT_TOPOLOGIES = {
    "scout_subject_general_fallback",
    "split_identity",
    "split_identity_correction",
    "split_identity_dual_correction",
    "split_identity_subject_fallback_dual_correction",
}
CORRECTION_TOPOLOGIES = {
    "split_identity_correction",
    "split_identity_dual_correction",
    "split_identity_subject_fallback_dual_correction",
}
SUBJECT_FALLBACK_TOPOLOGIES = {"split_identity_subject_fallback_dual_correction"}
SCOUT_SUBJECT_GENERAL_FALLBACK_TOPOLOGIES = {"scout_subject_general_fallback"}
DUAL_CORRECTION_TOPOLOGIES = {
    "split_identity_dual_correction",
    "split_identity_subject_fallback_dual_correction",
}
CUSTOM_AGENT_ORDER = (
    "numbers",
    "name",
    "dob",
    "subject_fallback",
    "spelling_correction",
    "caller_id_correction",
)

COMPACT_NUMBER_FIELDS = ["callback_ids", "fax_ids", "uncertain_ids", "errors"]
VERBOSE_NUMBER_FIELDS = ["callback_numbers", "fax_numbers", "uncertain_numbers", "possible_errors"]
COMPACT_IDENTITY_FIELDS = ["name_ids", "name_correction_ids", "dob_ids", "errors"]
VERBOSE_IDENTITY_FIELDS = ["patient_names", "name_correction_candidates", "dob_candidates", "possible_errors"]
COMPACT_NAME_FIELDS = ["name_ids", "name_correction_ids", "errors"]
VERBOSE_NAME_FIELDS = ["patient_names", "name_correction_candidates", "possible_errors"]
COMPACT_NAME_CORRECTION_FIELDS = ["name_ids", "name_correction_ids", "errors"]
VERBOSE_NAME_CORRECTION_FIELDS = ["patient_names", "name_correction_candidates", "possible_errors"]
COMPACT_DOB_FIELDS = ["dob_ids", "errors"]
VERBOSE_DOB_FIELDS = ["dob_candidates", "possible_errors"]
COMPACT_SCOUT_FIELDS = ["name_candidates", "errors"]
SPELLING_CORRECTED_SOURCE = "transcript_spelling_corrected"
CALLER_ID_CORRECTED_SOURCE = "caller_id_corrected"
BROAD_NAME_RECALL_SOURCE = "broad_name_recall"
CORRECTED_NAME_SOURCES = {SPELLING_CORRECTED_SOURCE, CALLER_ID_CORRECTED_SOURCE}
SUBJECT_NAME_SOURCES = {"explicit_patient", "relationship_subject"}
SPLIT_NAME_SHARED_SOURCES = {BROAD_NAME_RECALL_SOURCE}
SCOUT_NAME_SOURCES = {
    "explicit_patient",
    "relationship_subject",
    "self_identification",
    BROAD_NAME_RECALL_SOURCE,
    SPELLING_CORRECTED_SOURCE,
    CALLER_ID_CORRECTED_SOURCE,
}
SCOUT_ORG_TERMS = {
    "aetna",
    "clinic",
    "company",
    "department",
    "disability",
    "health",
    "hospital",
    "insurance",
    "medical",
    "nurse",
    "office",
    "pulmonary",
    "rehab",
    "team",
}
SCOUT_RELATIONSHIP_RE = re.compile(r"\b(?:father|mother|dad|mom|husband|wife|son|daughter|parent|child|spouse)\b", re.I)

candidate_agent_failures = 0
legacy_fallbacks = 0
last_shadow_comparison: dict[str, Any] = {}
last_agent_timings_ms: dict[str, int] = {}
last_agent_skipped: list[str] = []
last_agent_auto_accepted: list[str] = []
last_agent_constraint_modes: dict[str, str] = {}
last_parallel_total_ms = 0
last_e2b_scout_timing_ms = 0
last_e2b_scout_added_counts: dict[str, int] = {}
last_e2b_scout_rejected_counts: dict[str, int] = {}
last_e2b_scout_errors: list[str] = []
last_name_candidate_total = 0
last_name_candidate_counts_by_source: dict[str, int] = {}


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def candidate_agent_mode() -> str:
    mode = os.environ.get("GEMMA_EXTRACT_MODE", "legacy").strip().lower() or "legacy"
    return mode if mode in VALID_MODES else "legacy"


def candidate_extractor_enabled() -> bool:
    return env_bool("CANDIDATE_EXTRACTOR_ENABLED", True)


def candidate_agents_enabled() -> bool:
    return env_bool("CANDIDATE_AGENTS_ENABLED", True)


def fallback_to_legacy_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_FALLBACK_TO_LEGACY", True)


def include_full_transcript_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_INCLUDE_FULL_TRANSCRIPT", False)


def verifier_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_RUN_VERIFIER", False)


def auto_accept_deterministic_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_AUTO_ACCEPT_DETERMINISTIC", False)


def auto_accept_dob_enabled() -> bool:
    return auto_accept_deterministic_enabled() and env_bool("CANDIDATE_AGENT_AUTO_ACCEPT_DOB", True)


def auto_accept_transcript_spelling_enabled() -> bool:
    return auto_accept_deterministic_enabled() and env_bool(
        "CANDIDATE_AGENT_AUTO_ACCEPT_TRANSCRIPT_SPELLING",
        True,
    )


def auto_accept_self_identification_enabled() -> bool:
    return auto_accept_deterministic_enabled() and env_bool(
        "CANDIDATE_AGENT_AUTO_ACCEPT_SELF_IDENTIFICATION",
        False,
    )


def candidate_agent_output_schema() -> str:
    schema = os.environ.get("CANDIDATE_AGENT_OUTPUT_SCHEMA", "compact").strip().lower() or "compact"
    return schema if schema in VALID_OUTPUT_SCHEMAS else "compact"


def candidate_agent_constrained_decoding_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_CONSTRAINED_DECODING", False) and candidate_agent_output_schema() == "compact"


def candidate_agent_e2b_scout_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_E2B_SCOUT_ENABLED", False)


def candidate_agent_broad_name_recall_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_BROAD_NAME_RECALL", True)


def spelling_correction_agent_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_SPELLING_CORRECTION_ENABLED", True)


def caller_id_correction_agent_enabled() -> bool:
    return env_bool("CANDIDATE_AGENT_CALLER_ID_CORRECTION_ENABLED", True)


def candidate_agent_topology() -> str:
    topology = os.environ.get("CANDIDATE_AGENT_TOPOLOGY", "identity").strip().lower() or "identity"
    return topology if topology in VALID_TOPOLOGIES else "identity"


def candidate_agent_selection() -> tuple[str, ...]:
    raw = os.environ.get("CANDIDATE_AGENT_SELECTION", "").strip().lower()
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    unsupported = sorted(selected - set(CUSTOM_AGENT_ORDER))
    if unsupported:
        raise RuntimeError("unsupported candidate agent selection: " + ", ".join(unsupported))
    ordered = tuple(item for item in CUSTOM_AGENT_ORDER if item in selected)
    if candidate_agent_topology() == "custom" and not ordered:
        raise RuntimeError("custom candidate agent selection is empty")
    return ordered


def candidate_agent_execution_mode() -> str:
    mode = os.environ.get("CANDIDATE_AGENT_EXECUTION", "sequential_conversation").strip().lower()
    return mode if mode in VALID_EXECUTION_MODES else "sequential_conversation"


def parse_extraction_request_from_prompt(message: str) -> dict[str, Any] | None:
    before, delimiter, after = str(message or "").partition(INPUT_JSON_DELIMITER)
    del before
    if not delimiter:
        return None
    try:
        payload = json.loads(after)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "transcript" not in payload:
        return None
    return payload


def request_payload(request: dict[str, Any]) -> dict[str, Any]:
    if "transcript" in request:
        return dict(request)
    message = request.get("message") or request.get("prompt")
    if isinstance(message, str):
        parsed = parse_extraction_request_from_prompt(message)
        if parsed is not None:
            return parsed
    return dict(request)


def _short_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _dedupe(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _short_text(value, limit=80)
        if text and text.lower() not in seen:
            result.append(text)
            seen.add(text.lower())
    return result


def _compact_cues(candidate: dict[str, Any]) -> list[str]:
    nearby = candidate.get("nearby_cues") if isinstance(candidate.get("nearby_cues"), list) else []
    phrases = candidate.get("cue_phrases") if isinstance(candidate.get("cue_phrases"), list) else []
    return _dedupe([*nearby, *phrases])


def _compact_context(candidate: dict[str, Any]) -> str:
    return _short_text(
        candidate.get("window")
        or f"{candidate.get('context_before', '')}{candidate.get('raw', '')}{candidate.get('context_after', '')}",
        limit=220,
    )


def _compact_number_candidate(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    return {
        "id": str(candidate.get("id") or ""),
        "raw": _short_text(candidate.get("raw")),
        "normalized": str(candidate.get("normalized") or ""),
        "evidence_text": _short_text(candidate.get("evidence_text")),
        "cues": _compact_cues(candidate),
        "context": _compact_context(candidate),
    }


def _compact_identity_candidate(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    result = {
        "id": str(candidate.get("id") or ""),
        "raw": _short_text(candidate.get("raw")),
        "evidence_text": _short_text(candidate.get("evidence_text")),
        "source": str(candidate.get("source") or ""),
        "cues": _compact_cues(candidate),
        "context": _compact_context(candidate),
    }
    sentence_context = _short_text(candidate.get("sentence_context"), limit=260)
    if sentence_context:
        result["sentence_context"] = sentence_context
    for key in ("value", "normalized", "letters", "caller_id_used", "suggested_value", "reason"):
        value = _short_text(candidate.get(key), limit=120)
        if value:
            result[key] = value
    return result


def _is_corrected_name_candidate(candidate: Any) -> bool:
    return isinstance(candidate, dict) and str(candidate.get("source") or "") in CORRECTED_NAME_SOURCES


def _ordinary_name_candidates(candidates: dict[str, Any]) -> list[Any]:
    return [
        candidate
        for candidate in _candidate_items(candidates, "name_candidates")
        if not _is_corrected_name_candidate(candidate)
    ]


def _corrected_name_candidates(candidates: dict[str, Any]) -> list[Any]:
    return [
        candidate
        for candidate in _candidate_items(candidates, "name_candidates")
        if _is_corrected_name_candidate(candidate)
    ]


def _spelling_corrected_name_candidates(candidates: dict[str, Any]) -> list[Any]:
    return [
        candidate
        for candidate in _candidate_items(candidates, "name_candidates")
        if isinstance(candidate, dict) and str(candidate.get("source") or "") == SPELLING_CORRECTED_SOURCE
    ]


def _caller_id_corrected_name_candidates(candidates: dict[str, Any]) -> list[Any]:
    return [
        candidate
        for candidate in _candidate_items(candidates, "name_candidates")
        if isinstance(candidate, dict) and str(candidate.get("source") or "") == CALLER_ID_CORRECTED_SOURCE
    ]


def _subject_name_candidates(candidates: dict[str, Any]) -> list[Any]:
    return [
        candidate
        for candidate in _ordinary_name_candidates(candidates)
        if isinstance(candidate, dict)
        and str(candidate.get("source") or "") in SUBJECT_NAME_SOURCES | SPLIT_NAME_SHARED_SOURCES
    ]


def _fallback_name_candidates(candidates: dict[str, Any]) -> list[Any]:
    return [
        candidate
        for candidate in _ordinary_name_candidates(candidates)
        if isinstance(candidate, dict)
        and (
            str(candidate.get("source") or "") not in SUBJECT_NAME_SOURCES
            or str(candidate.get("source") or "") in SPLIT_NAME_SHARED_SOURCES
        )
    ]


def _compact_semantic_event(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("event_type", "raw", "normalized", "evidence_text", "confidence_hint"):
        value = _short_text(event.get(key), limit=160)
        if value:
            result[key] = value
    return result


def _build_candidate_scout_payload(candidates: dict[str, Any], transcript: str) -> dict[str, Any]:
    return {
        "transcript_id": candidates.get("transcript_id", ""),
        "caller_id": candidates.get("caller_id", ""),
        "mailbox": candidates.get("mailbox", ""),
        "transcript": transcript,
        "existing_candidates": {
            "name_candidates": [
                item
                for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("name_candidates", []))
                if item.get("id")
            ],
            "spelled_sequences": [
                item
                for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("spelled_sequences", []))
                if item.get("id")
            ],
            "semantic_events": [
                item
                for item in (_compact_semantic_event(event) for event in candidates.get("semantic_events", []))
                if item
            ],
        },
    }


def _bump_count(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key) or 0) + 1


def _record_name_candidate_counts(candidates: dict[str, Any]) -> None:
    global last_name_candidate_total, last_name_candidate_counts_by_source
    counts: dict[str, int] = {}
    total = 0
    for candidate in _candidate_items(candidates, "name_candidates"):
        if not isinstance(candidate, dict):
            continue
        total += 1
        source = str(candidate.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    last_name_candidate_total = total
    last_name_candidate_counts_by_source = counts


def _find_case_insensitive_span(text: str, needle: str) -> tuple[int, int] | None:
    if not needle:
        return None
    index = text.find(needle)
    if index < 0:
        index = text.lower().find(needle.lower())
    if index < 0:
        return None
    return index, index + len(needle)


def _scout_context_fields(transcript: str, span: tuple[int, int]) -> dict[str, Any]:
    start, end = span
    before_start = max(0, start - 64)
    after_end = min(len(transcript), end + 64)
    window = transcript[before_start:after_end]
    return {
        "span": [start, end],
        "context_before": transcript[before_start:start],
        "context_after": transcript[end:after_end],
        "window": window,
        "sentence_context": window.strip(),
    }


def _scout_evidence_span(proposal: Any, transcript: str, rejected: dict[str, int]) -> tuple[str, tuple[int, int]] | None:
    if not isinstance(proposal, dict):
        _bump_count(rejected, "invalid_item")
        return None
    evidence = _short_text(proposal.get("evidence_text"), limit=400)
    span = _find_case_insensitive_span(transcript, evidence)
    if not evidence or span is None:
        _bump_count(rejected, "missing_evidence_text")
        return None
    return evidence, span


def _scout_name_key(value: Any) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").lower())


def _scout_name_tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'.-]*", value)


def _clean_scout_name(value: Any) -> str:
    tokens = _scout_name_tokens(str(value or ""))
    while tokens and tokens[0].lower().strip(".") in {"a", "client", "miss", "mr", "mrs", "ms", "patient", "the"}:
        tokens.pop(0)
    return " ".join(tokens).strip()


def _scout_name_source(proposal: dict[str, Any]) -> str:
    source = str(proposal.get("source") or "").strip()
    if source in SCOUT_NAME_SOURCES:
        return source
    role = str(proposal.get("role") or proposal.get("type") or "").strip().lower()
    if role in {"patient", "subject", "client"}:
        return "explicit_patient"
    if role in {"relationship", "relationship_subject"}:
        return "relationship_subject"
    if role in {"caller", "speaker", "self"}:
        return "self_identification"
    return "explicit_patient"


def _name_looks_like_org(value: str) -> bool:
    tokens = {token.lower().strip(".") for token in _scout_name_tokens(value)}
    return bool(tokens & SCOUT_ORG_TERMS)


def _merge_scout_name_candidate(
    candidates: dict[str, Any],
    proposal: Any,
    transcript: str,
    added: dict[str, int],
    rejected: dict[str, int],
) -> None:
    evidence = _scout_evidence_span(proposal, transcript, rejected)
    if evidence is None:
        return
    evidence_text, evidence_span = evidence
    assert isinstance(proposal, dict)
    source = _scout_name_source(proposal)
    raw = _clean_scout_name(proposal.get("raw") or proposal.get("value") or "")
    value = _clean_scout_name(proposal.get("value") or raw)
    if not raw or not value:
        _bump_count(rejected, "invalid_name")
        return
    if _name_looks_like_org(value):
        _bump_count(rejected, "organization_name")
        return
    token_count = len(_scout_name_tokens(value))
    if token_count < 2 and not (
        source == "self_identification"
        or (source == "relationship_subject" and SCOUT_RELATIONSHIP_RE.search(evidence_text))
    ):
        _bump_count(rejected, "first_name_only")
        return
    if source == CALLER_ID_CORRECTED_SOURCE and not str(proposal.get("caller_id_used") or "").strip():
        _bump_count(rejected, "caller_id_missing")
        return
    name_span = _find_case_insensitive_span(evidence_text, raw)
    if name_span is None:
        name_span = _find_case_insensitive_span(evidence_text, value)
    if name_span is None:
        span = evidence_span
    else:
        span = (evidence_span[0] + name_span[0], evidence_span[0] + name_span[1])
    existing = {
        (_scout_name_key(item.get("value") or item.get("raw")), str(item.get("source") or ""))
        for item in candidates.get("name_candidates", [])
        if isinstance(item, dict)
    }
    key = (_scout_name_key(value), source)
    if key in existing:
        return
    item = {
        "id": f"name:{len(candidates.get('name_candidates', []))}",
        "raw": raw,
        "value": value,
        "source": source,
        "confidence_hint": "medium",
        "evidence_text": evidence_text,
        "nearby_cues": ["e2b_scout"],
        "cue_phrases": [],
        **_scout_context_fields(transcript, span),
    }
    if source == CALLER_ID_CORRECTED_SOURCE:
        item["caller_id_used"] = str(proposal.get("caller_id_used") or candidates.get("caller_id") or "")
    candidates.setdefault("name_candidates", []).append(item)
    _bump_count(added, "name_candidates")


def _merge_scout_dob_candidate(
    candidates: dict[str, Any],
    proposal: Any,
    transcript: str,
    added: dict[str, int],
    rejected: dict[str, int],
) -> None:
    evidence = _scout_evidence_span(proposal, transcript, rejected)
    if evidence is None:
        return
    evidence_text, evidence_span = evidence
    assert isinstance(proposal, dict)
    raw = _short_text(proposal.get("raw") or proposal.get("normalized") or evidence_text, limit=80)
    normalized_input = _short_text(proposal.get("normalized") or raw, limit=80).replace("-", "/")
    parsed = parse_dob(normalized_input)
    if parsed is None:
        parsed = parse_dob(raw.replace("-", "/"))
    if parsed is None:
        _bump_count(rejected, "invalid_dob")
        return
    normalized = format_dob(parsed)
    existing = {
        str(item.get("normalized") or "")
        for item in candidates.get("dob_candidates", [])
        if isinstance(item, dict)
    }
    if normalized in existing:
        return
    item = {
        "id": f"dob:{len(candidates.get('dob_candidates', []))}",
        "raw": raw,
        "normalized": normalized,
        "source": "e2b_scout",
        "confidence_hint": "medium",
        "evidence_text": evidence_text,
        "nearby_cues": ["e2b_scout", "dob"],
        "cue_phrases": [],
        **_scout_context_fields(transcript, evidence_span),
    }
    candidates.setdefault("dob_candidates", []).append(item)
    _bump_count(added, "dob_candidates")


def _merge_scout_number_candidate(
    candidates: dict[str, Any],
    proposal: Any,
    transcript: str,
    added: dict[str, int],
    rejected: dict[str, int],
) -> None:
    evidence = _scout_evidence_span(proposal, transcript, rejected)
    if evidence is None:
        return
    evidence_text, evidence_span = evidence
    assert isinstance(proposal, dict)
    raw = _short_text(proposal.get("raw") or proposal.get("normalized") or evidence_text, limit=120)
    normalized_source = proposal.get("normalized") or raw
    phone = normalize_phone_candidate(normalized_source)
    if not phone.valid:
        phone = normalize_phone_candidate(raw)
    if not phone.valid or not phone.normalized:
        _bump_count(rejected, "invalid_phone")
        return
    existing = {
        str(item.get("normalized") or "")
        for item in candidates.get("number_candidates", [])
        if isinstance(item, dict)
    }
    if phone.normalized in existing:
        return
    cue = str(proposal.get("label_cue") or proposal.get("cue") or "").strip().lower()
    cues = ["e2b_scout"]
    if cue in {"callback", "fax"}:
        cues.append(cue)
    if re.search(r"\bfax", evidence_text, re.I) and "fax" not in cues:
        cues.append("fax")
    if re.search(r"\b(?:call|phone|number|reach)", evidence_text, re.I) and "callback" not in cues:
        cues.append("callback")
    item = {
        "id": f"number:{len(candidates.get('number_candidates', []))}",
        "raw": raw,
        "normalized": phone.normalized,
        "formatted": phone.formatted,
        "source": "e2b_scout",
        "confidence_hint": "medium",
        "evidence_text": evidence_text,
        "nearby_cues": cues,
        "cue_phrases": [],
        **_scout_context_fields(transcript, evidence_span),
    }
    candidates.setdefault("number_candidates", []).append(item)
    _bump_count(added, "number_candidates")


def _merge_scout_spelled_sequence(
    candidates: dict[str, Any],
    proposal: Any,
    transcript: str,
    added: dict[str, int],
    rejected: dict[str, int],
) -> None:
    evidence = _scout_evidence_span(proposal, transcript, rejected)
    if evidence is None:
        return
    evidence_text, evidence_span = evidence
    assert isinstance(proposal, dict)
    raw = _short_text(proposal.get("raw") or evidence_text, limit=120)
    letters = re.sub(r"[^A-Za-z]", "", str(proposal.get("letters") or raw)).upper()
    if len(letters) < 3:
        _bump_count(rejected, "invalid_spelling")
        return
    existing = {
        str(item.get("letters") or "")
        for item in candidates.get("spelled_sequences", [])
        if isinstance(item, dict)
    }
    if letters in existing:
        return
    item = {
        "id": f"spelling:{len(candidates.get('spelled_sequences', []))}",
        "raw": raw,
        "letters": letters,
        "source": "spelled_letters",
        "confidence_hint": "high",
        "evidence_text": evidence_text,
        "nearby_cues": ["e2b_scout", "spelled_sequence"],
        "cue_phrases": [],
        **_scout_context_fields(transcript, evidence_span),
    }
    candidates.setdefault("spelled_sequences", []).append(item)
    _bump_count(added, "spelled_sequences")


def _merge_scout_output(candidates: dict[str, Any], scout_output: dict[str, Any], transcript: str) -> tuple[dict[str, int], dict[str, int]]:
    added: dict[str, int] = {}
    rejected: dict[str, int] = {}
    for proposal in _candidate_items(scout_output, "name_candidates"):
        _merge_scout_name_candidate(candidates, proposal, transcript, added, rejected)
    return added, rejected


def _reset_e2b_scout_health() -> None:
    global last_e2b_scout_timing_ms, last_e2b_scout_added_counts, last_e2b_scout_rejected_counts, last_e2b_scout_errors
    last_e2b_scout_timing_ms = 0
    last_e2b_scout_added_counts = {}
    last_e2b_scout_rejected_counts = {}
    last_e2b_scout_errors = []


def _run_e2b_candidate_scout(
    active_agents: dict[str, Any],
    candidates: dict[str, Any],
    transcript: str,
    trace_sink: AgentTraceSink | None = None,
) -> None:
    global last_e2b_scout_timing_ms, last_e2b_scout_added_counts, last_e2b_scout_rejected_counts, last_e2b_scout_errors
    _reset_e2b_scout_health()
    if (
        not candidate_agent_e2b_scout_enabled()
        and candidate_agent_topology() not in SCOUT_SUBJECT_GENERAL_FALLBACK_TOPOLOGIES
    ):
        return
    scout = active_agents.get("candidate_scout")
    if scout is None:
        last_e2b_scout_errors = ["candidate_scout_not_configured"]
        return
    start = time.perf_counter()
    try:
        output = scout.run(_build_candidate_scout_payload(candidates, transcript))
        last_e2b_scout_timing_ms = int((time.perf_counter() - start) * 1000)
        last_e2b_scout_added_counts, last_e2b_scout_rejected_counts = _merge_scout_output(
            candidates,
            output,
            transcript,
        )
        last_e2b_scout_errors = [str(item) for item in _candidate_items(output, "errors") if str(item).strip()]
        _emit_agent_trace(
            trace_sink,
            "candidate_scout",
            output,
            last_e2b_scout_timing_ms,
            _agent_constraint_mode(scout),
        )
    except Exception as exc:  # Scout is recall help; baseline extraction should continue if it fails.
        last_e2b_scout_timing_ms = int((time.perf_counter() - start) * 1000)
        last_e2b_scout_errors = [str(exc)]
        _emit_agent_trace(
            trace_sink,
            "candidate_scout",
            {"name_candidates": [], "errors": ["candidate_scout_failed"]},
            last_e2b_scout_timing_ms,
            _agent_constraint_mode(scout),
            status="failed",
            error="candidate_scout_failed",
        )


def _build_numbers_payload(candidates: dict[str, Any], transcript: str, include_full_transcript: bool) -> dict[str, Any]:
    payload = {
        "transcript_id": candidates.get("transcript_id", ""),
        "caller_id": candidates.get("caller_id", ""),
        "mailbox": candidates.get("mailbox", ""),
        "number_candidates": [
            item
            for item in (_compact_number_candidate(candidate) for candidate in candidates.get("number_candidates", []))
            if item.get("id")
        ],
        "semantic_events": [
            item
            for item in (_compact_semantic_event(event) for event in candidates.get("semantic_events", []))
            if item
        ],
    }
    if include_full_transcript:
        payload["transcript"] = transcript
    return payload


def _build_identity_payload(candidates: dict[str, Any], transcript: str, include_full_transcript: bool) -> dict[str, Any]:
    payload = {
        "transcript_id": candidates.get("transcript_id", ""),
        "caller_id": candidates.get("caller_id", ""),
        "mailbox": candidates.get("mailbox", ""),
        "name_candidates": [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("name_candidates", []))
            if item.get("id")
        ],
        "name_correction_candidates": [
            item
            for item in (
                _compact_identity_candidate(candidate)
                for candidate in candidates.get("name_correction_candidates", [])
            )
            if item.get("id")
        ],
        "dob_candidates": [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("dob_candidates", []))
            if item.get("id")
        ],
        "spelled_sequences": [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("spelled_sequences", []))
            if item.get("id")
        ],
        "semantic_events": [
            item
            for item in (_compact_semantic_event(event) for event in candidates.get("semantic_events", []))
            if item
        ],
    }
    if include_full_transcript:
        payload["transcript"] = transcript
    return payload


def _build_name_payload(candidates: dict[str, Any], transcript: str, include_full_transcript: bool) -> dict[str, Any]:
    caller_id = str(candidates.get("caller_id") or "")
    topology = candidate_agent_topology()
    source_candidates = (
        _ordinary_name_candidates(candidates)
        if topology in CORRECTION_TOPOLOGIES
        else candidates.get("name_candidates", [])
    )
    payload = {
        "transcript_id": candidates.get("transcript_id", ""),
        "caller_id": "" if topology in CORRECTION_TOPOLOGIES else caller_id if caller_id_name_shaped(caller_id) else "",
        "mailbox": candidates.get("mailbox", ""),
        "name_candidates": [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in source_candidates)
            if item.get("id")
        ],
        "semantic_events": [
            item
            for item in (_compact_semantic_event(event) for event in candidates.get("semantic_events", []))
            if item
        ],
    }
    if topology not in CORRECTION_TOPOLOGIES:
        name_corrections = [
            item
            for item in (
                _compact_identity_candidate(candidate)
                for candidate in candidates.get("name_correction_candidates", [])
            )
            if item.get("id")
        ]
        if name_corrections:
            payload["name_correction_candidates"] = name_corrections
        spelled_sequences = [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("spelled_sequences", []))
            if item.get("id")
        ]
        if spelled_sequences:
            payload["spelled_sequences"] = spelled_sequences
    if include_full_transcript:
        payload["transcript"] = transcript
    return payload


def _build_split_name_payload(
    candidates: dict[str, Any],
    transcript: str,
    include_full_transcript: bool,
    source_candidates: list[Any],
) -> dict[str, Any]:
    payload = {
        "transcript_id": candidates.get("transcript_id", ""),
        "caller_id": "",
        "mailbox": candidates.get("mailbox", ""),
        "name_candidates": [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in source_candidates)
            if item.get("id")
        ],
        "semantic_events": [
            item
            for item in (_compact_semantic_event(event) for event in candidates.get("semantic_events", []))
            if item
        ],
    }
    if include_full_transcript:
        payload["transcript"] = transcript
    return payload


def _build_subject_name_payload(
    candidates: dict[str, Any],
    transcript: str,
    include_full_transcript: bool,
) -> dict[str, Any]:
    return _build_split_name_payload(
        candidates,
        transcript,
        include_full_transcript,
        _subject_name_candidates(candidates),
    )


def _build_caller_name_fallback_payload(
    candidates: dict[str, Any],
    transcript: str,
    include_full_transcript: bool,
) -> dict[str, Any]:
    return _build_split_name_payload(
        candidates,
        transcript,
        include_full_transcript,
        _fallback_name_candidates(candidates),
    )


def _build_name_correction_payload(
    candidates: dict[str, Any],
    transcript: str,
    include_full_transcript: bool,
) -> dict[str, Any]:
    caller_id = str(candidates.get("caller_id") or "")
    caller_id_for_worker = caller_id_display_name(caller_id) if caller_id_name_shaped(caller_id) else ""
    payload = {
        "transcript_id": candidates.get("transcript_id", ""),
        "caller_id": caller_id_for_worker,
        "mailbox": candidates.get("mailbox", ""),
        "name_candidates": [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in _corrected_name_candidates(candidates))
            if item.get("id")
        ],
        "semantic_events": [
            item
            for item in (_compact_semantic_event(event) for event in candidates.get("semantic_events", []))
            if item
        ],
    }
    spelled_sequences = [
        item
        for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("spelled_sequences", []))
        if item.get("id")
    ]
    if spelled_sequences:
        payload["spelled_sequences"] = spelled_sequences
    if caller_id_for_worker:
        corrections = [
            item
            for item in (
                _compact_identity_candidate(candidate)
                for candidate in candidates.get("name_correction_candidates", [])
            )
            if item.get("id")
        ]
        if corrections:
            payload["name_correction_candidates"] = corrections
    if include_full_transcript:
        payload["transcript"] = transcript
    return payload


def _build_spelling_correction_payload(
    candidates: dict[str, Any],
    transcript: str,
    include_full_transcript: bool,
) -> dict[str, Any]:
    payload = {
        "transcript_id": candidates.get("transcript_id", ""),
        "caller_id": "",
        "mailbox": candidates.get("mailbox", ""),
        "name_candidates": [
            item
            for item in (
                _compact_identity_candidate(candidate)
                for candidate in _spelling_corrected_name_candidates(candidates)
            )
            if item.get("id")
        ],
        "spelled_sequences": [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("spelled_sequences", []))
            if item.get("id")
        ],
        "semantic_events": [
            item
            for item in (_compact_semantic_event(event) for event in candidates.get("semantic_events", []))
            if item
        ],
    }
    if include_full_transcript:
        payload["transcript"] = transcript
    return payload


def _build_caller_id_correction_payload(
    candidates: dict[str, Any],
    transcript: str,
    include_full_transcript: bool,
) -> dict[str, Any]:
    caller_id = str(candidates.get("caller_id") or "")
    caller_id_for_worker = caller_id_display_name(caller_id) if caller_id_name_shaped(caller_id) else ""
    payload = {
        "transcript_id": candidates.get("transcript_id", ""),
        "caller_id": caller_id_for_worker,
        "mailbox": candidates.get("mailbox", ""),
        "name_candidates": [
            item
            for item in (
                _compact_identity_candidate(candidate)
                for candidate in _caller_id_corrected_name_candidates(candidates)
            )
            if item.get("id")
        ],
        "semantic_events": [
            item
            for item in (_compact_semantic_event(event) for event in candidates.get("semantic_events", []))
            if item
        ],
    }
    if caller_id_for_worker:
        corrections = [
            item
            for item in (
                _compact_identity_candidate(candidate)
                for candidate in candidates.get("name_correction_candidates", [])
            )
            if item.get("id")
        ]
        if corrections:
            payload["name_correction_candidates"] = corrections
    if include_full_transcript:
        payload["transcript"] = transcript
    return payload


def _build_dob_payload(candidates: dict[str, Any], transcript: str, include_full_transcript: bool) -> dict[str, Any]:
    payload = {
        "transcript_id": candidates.get("transcript_id", ""),
        "caller_id": candidates.get("caller_id", ""),
        "mailbox": candidates.get("mailbox", ""),
        "dob_candidates": [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("dob_candidates", []))
            if item.get("id")
        ],
        "name_candidates": [
            item
            for item in (_compact_identity_candidate(candidate) for candidate in candidates.get("name_candidates", []))
            if item.get("id")
        ],
        "semantic_events": [
            item
            for item in (_compact_semantic_event(event) for event in candidates.get("semantic_events", []))
            if item
        ],
    }
    if include_full_transcript:
        payload["transcript"] = transcript
    return payload


def default_prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def _prompt_fragment(filename: str) -> str:
    return (default_prompt_dir() / filename).read_text(encoding="utf-8").strip()


def caller_id_name_shaped(caller_id: Any) -> bool:
    text = caller_id_display_name(caller_id)
    if not text:
        return False
    upper = text.upper()
    digits = re.findall(r"\d", upper)
    letters = re.findall(r"[A-Z]", upper)
    if not letters:
        return False
    if len(digits) >= 3 or (digits and len(digits) >= max(1, len(letters) // 2)):
        return False
    generic_values = {
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
    }
    if upper in generic_values:
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


def caller_id_display_name(caller_id: Any) -> str:
    text = " ".join(str(caller_id or "").replace("_", " ").split()).strip()
    quoted_name = re.match(r'^\s*"([^"]+)"', text)
    if quoted_name:
        text = quoted_name.group(1).strip()
    else:
        text = re.sub(r"\([^)]*\d[^)]*\)", " ", text)
    quoted = re.match(r'^"?([^"<]+?)"?\s*<[^>]+>$', text)
    if quoted:
        text = quoted.group(1).strip()
    text = re.sub(r"\b\d[\d\s().+-]{6,}\d\b", " ", text)
    return re.sub(r"\s+", " ", text).strip(" \"'<>")


def _auto_name_key(value: Any) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").lower())


def _caller_id_name_orders_for_auto_accept(caller_id: Any) -> list[str]:
    if not caller_id_name_shaped(caller_id):
        return []
    text = caller_id_display_name(caller_id).strip(" ,.;:\"'")
    if "," in text:
        last, first = [part.strip(" ,.;:\"'") for part in text.split(",", 1)]
        if first and last:
            return [f"{first} {last}"]
    tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", text)
    tokens = [token for token in tokens if token]
    if len(tokens) < 2:
        return []
    if len(tokens) == 2:
        return [f"{tokens[0]} {tokens[1]}", f"{tokens[1]} {tokens[0]}"]
    if len(tokens) == 3 and len(tokens[1].strip(".-'")) == 1:
        return [f"{tokens[0]} {tokens[2]}"]
    if len(tokens) == 3 and len(tokens[2].strip(".-'")) == 1:
        return [f"{tokens[1]} {tokens[0]}"]
    return [f"{tokens[0]} {tokens[-1]}", f"{tokens[-1]} {tokens[0]}"]


def _caller_id_confirms_name_for_auto_accept(name: Any, caller_id: Any) -> bool:
    name_key = _auto_name_key(name)
    return bool(
        name_key
        and any(_auto_name_key(candidate) == name_key for candidate in _caller_id_name_orders_for_auto_accept(caller_id))
    )


def build_name_prompt(dynamic_payload: dict[str, Any]) -> str:
    fragments = [_prompt_fragment("name_agent.md")]
    if candidate_agent_topology() in CORRECTION_TOPOLOGIES:
        return fragments[0].strip()
    if dynamic_payload.get("spelled_sequences"):
        fragments.append(_prompt_fragment("name_agent_spelling.md"))
    if caller_id_name_shaped(dynamic_payload.get("caller_id")):
        fragments.append(_prompt_fragment("name_agent_caller_id.md"))
    return "\n\n".join(fragment for fragment in fragments if fragment).strip()


def build_name_correction_prompt(dynamic_payload: dict[str, Any]) -> str:
    fragments = [_prompt_fragment("name_correction_agent.md")]
    if dynamic_payload.get("spelled_sequences"):
        fragments.append(_prompt_fragment("name_agent_spelling.md"))
    if caller_id_name_shaped(dynamic_payload.get("caller_id")):
        fragments.append(_prompt_fragment("name_agent_caller_id.md"))
    return "\n\n".join(fragment for fragment in fragments if fragment).strip()


def build_spelling_correction_prompt(dynamic_payload: dict[str, Any]) -> str:
    del dynamic_payload
    return "\n\n".join(
        fragment
        for fragment in (
            _prompt_fragment("spelling_correction_agent.md"),
            _prompt_fragment("name_agent_spelling.md"),
        )
        if fragment
    ).strip()


def build_caller_id_correction_prompt(dynamic_payload: dict[str, Any]) -> str:
    del dynamic_payload
    return "\n\n".join(
        fragment
        for fragment in (
            _prompt_fragment("caller_id_correction_agent.md"),
            _prompt_fragment("name_agent_caller_id.md"),
        )
        if fragment
    ).strip()


def _agent_expected_fields(compact_fields: list[str], verbose_fields: list[str]) -> list[str]:
    if candidate_agent_output_schema() == "verbose":
        return [*verbose_fields, *compact_fields]
    return [*compact_fields, *verbose_fields]


def _candidate_items(candidates: dict[str, Any], key: str) -> list[Any]:
    value = candidates.get(key)
    return value if isinstance(value, list) else []


def _has_number_candidates(candidates: dict[str, Any]) -> bool:
    return bool(_candidate_items(candidates, "number_candidates"))


def _has_name_candidates(candidates: dict[str, Any]) -> bool:
    if candidate_agent_topology() in CORRECTION_TOPOLOGIES:
        return bool(_ordinary_name_candidates(candidates))
    return bool(
        _candidate_items(candidates, "name_candidates")
        or _candidate_items(candidates, "spelled_sequences")
        or _candidate_items(candidates, "name_correction_candidates")
    )


def _has_subject_name_candidates(candidates: dict[str, Any]) -> bool:
    return bool(_subject_name_candidates(candidates))


def _has_fallback_name_candidates(candidates: dict[str, Any]) -> bool:
    return bool(_fallback_name_candidates(candidates))


def _has_name_correction_candidates(candidates: dict[str, Any]) -> bool:
    return bool(
        _corrected_name_candidates(candidates)
        or _candidate_items(candidates, "spelled_sequences")
        or _candidate_items(candidates, "name_correction_candidates")
    )


def _has_spelling_correction_candidates(candidates: dict[str, Any]) -> bool:
    return bool(
        _spelling_corrected_name_candidates(candidates)
        or _candidate_items(candidates, "spelled_sequences")
    )


def _has_caller_id_correction_candidates(candidates: dict[str, Any]) -> bool:
    return bool(
        caller_id_name_shaped(candidates.get("caller_id"))
        and (
            _caller_id_corrected_name_candidates(candidates)
            or _candidate_items(candidates, "name_correction_candidates")
        )
    )


def _has_dob_candidates(candidates: dict[str, Any]) -> bool:
    return bool(_candidate_items(candidates, "dob_candidates"))


def _single_auto_accepted_dob_id(candidates: dict[str, Any]) -> str:
    if not auto_accept_dob_enabled():
        return ""
    items = [item for item in _candidate_items(candidates, "dob_candidates") if isinstance(item, dict)]
    if len(items) != 1:
        return ""
    item = items[0]
    if str(item.get("source") or "") != "date_numeric":
        return ""
    if str(item.get("confidence_hint") or "") != "high":
        return ""
    evidence = str(item.get("evidence_text") or "").strip()
    if not evidence or not re.search(r"\b(?:date\s+of\s+birth|birth\s+date|dob|d\.o\.b\.|born)\b", evidence, re.I):
        return ""
    if parse_dob(item.get("normalized")) is None:
        return ""
    return str(item.get("id") or "")


def _single_auto_accepted_spelling_name_id(candidates: dict[str, Any]) -> str:
    if not auto_accept_transcript_spelling_enabled():
        return ""
    if not _candidate_items(candidates, "spelled_sequences"):
        return ""
    items = [
        item
        for item in _candidate_items(candidates, "name_candidates")
        if isinstance(item, dict) and str(item.get("source") or "") == SPELLING_CORRECTED_SOURCE
    ]
    if len(items) != 1:
        return ""
    item = items[0]
    if str(item.get("confidence_hint") or "") != "high":
        return ""
    for key in ("id", "raw", "value", "evidence_text"):
        if not str(item.get(key) or "").strip():
            return ""
    return str(item.get("id") or "")


def _single_auto_accepted_self_identification_name_id(candidates: dict[str, Any]) -> str:
    if not auto_accept_self_identification_enabled():
        return ""
    if _candidate_items(candidates, "spelled_sequences") or _candidate_items(candidates, "name_correction_candidates"):
        return ""
    if _corrected_name_candidates(candidates):
        return ""
    ordinary = _ordinary_name_candidates(candidates)
    items = [
        item
        for item in ordinary
        if isinstance(item, dict) and str(item.get("source") or "") == "self_identification"
    ]
    if len(ordinary) != 1 or len(items) != 1:
        return ""
    item = items[0]
    if str(item.get("confidence_hint") or "") not in {"medium", "high"}:
        return ""
    evidence = str(item.get("evidence_text") or "").strip()
    if not re.search(r"\b(?:this\s+is|my\s+name\s+is|i\s+am|i'm|it's|the\s+name\s+is)\b", evidence, re.I):
        return ""
    for key in ("id", "raw", "value"):
        if not str(item.get(key) or "").strip():
            return ""
    caller_id = candidates.get("caller_id")
    if caller_id_name_shaped(caller_id) and not _caller_id_confirms_name_for_auto_accept(
        item.get("value") or item.get("raw"),
        caller_id,
    ):
        return ""
    return str(item.get("id") or "")


def _empty_numbers_output() -> dict[str, list[Any]]:
    return {"callback_ids": [], "fax_ids": [], "uncertain_ids": [], "errors": []}


def _empty_name_output() -> dict[str, list[Any]]:
    return {"name_ids": [], "name_correction_ids": [], "errors": []}


def _name_output_has_selection(output: dict[str, Any]) -> bool:
    return bool(
        _candidate_items(output, "name_ids")
        or _candidate_items(output, "patient_names")
        or _candidate_items(output, "name_correction_ids")
        or _candidate_items(output, "name_correction_candidates")
    )


def _empty_dob_output() -> dict[str, list[Any]]:
    return {"dob_ids": [], "errors": []}


def build_default_agents(text_generator=None) -> dict[str, CachedAgent]:
    prompt_dir = default_prompt_dir()
    constraint_builder = build_agent_constraint_schema if candidate_agent_constrained_decoding_enabled() else None
    topology = candidate_agent_topology()
    if topology == "custom":
        selected = set(candidate_agent_selection())
        agents: dict[str, CachedAgent] = {}

        def add(name: str, filename: str, tokens: int, fields: list[str]) -> None:
            agents[name] = CachedAgent.from_prompt_file(
                name,
                prompt_dir / filename,
                tokens,
                fields,
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )

        if "numbers" in selected:
            add(
                "numbers",
                "numbers_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NUMBERS", 192),
                _agent_expected_fields(COMPACT_NUMBER_FIELDS, VERBOSE_NUMBER_FIELDS),
            )
        if "name" in selected:
            add(
                "name",
                "name_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NAME", 220),
                _agent_expected_fields(COMPACT_NAME_FIELDS, VERBOSE_NAME_FIELDS),
            )
            agents["name"].prompt_builder = build_name_prompt
        if "dob" in selected:
            add(
                "dob",
                "dob_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_DOB", 160),
                _agent_expected_fields(COMPACT_DOB_FIELDS, VERBOSE_DOB_FIELDS),
            )
        if "subject_fallback" in selected:
            for name, filename, env_name in (
                ("subject_name", "subject_name_agent.md", "SUBJECT_NAME"),
                ("caller_name_fallback", "caller_name_fallback_agent.md", "CALLER_NAME_FALLBACK"),
            ):
                add(
                    name,
                    filename,
                    env_int(f"CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_{env_name}", 160),
                    _agent_expected_fields(COMPACT_NAME_FIELDS, VERBOSE_NAME_FIELDS),
                )
        if "spelling_correction" in selected:
            add(
                "spelling_correction",
                "spelling_correction_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_SPELLING_CORRECTION", 64),
                _agent_expected_fields(COMPACT_NAME_CORRECTION_FIELDS, VERBOSE_NAME_CORRECTION_FIELDS),
            )
            agents["spelling_correction"].prompt_builder = build_spelling_correction_prompt
        if "caller_id_correction" in selected:
            add(
                "caller_id_correction",
                "caller_id_correction_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_CALLER_ID_CORRECTION", 96),
                _agent_expected_fields(COMPACT_NAME_CORRECTION_FIELDS, VERBOSE_NAME_CORRECTION_FIELDS),
            )
            agents["caller_id_correction"].prompt_builder = build_caller_id_correction_prompt
        return agents
    agents = {
        "numbers": CachedAgent.from_prompt_file(
            "numbers",
            prompt_dir / "numbers_agent.md",
            env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NUMBERS", 192),
            _agent_expected_fields(COMPACT_NUMBER_FIELDS, VERBOSE_NUMBER_FIELDS),
            text_generator=text_generator,
            constraint_builder=constraint_builder,
        ),
    }
    if candidate_agent_e2b_scout_enabled() or topology in SCOUT_SUBJECT_GENERAL_FALLBACK_TOPOLOGIES:
        agents["candidate_scout"] = CachedAgent.from_prompt_file(
            "candidate_scout",
            prompt_dir / "candidate_scout_agent.md",
            env_int("CANDIDATE_AGENT_E2B_SCOUT_MAX_OUTPUT_TOKENS", 256),
            COMPACT_SCOUT_FIELDS,
            text_generator=text_generator,
            constraint_builder=None,
        )
    if topology == "numbers_only":
        return agents
    if topology in SPLIT_TOPOLOGIES:
        if topology in SCOUT_SUBJECT_GENERAL_FALLBACK_TOPOLOGIES:
            agents["subject_name"] = CachedAgent.from_prompt_file(
                "subject_name",
                prompt_dir / "subject_name_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_SUBJECT_NAME", 160),
                _agent_expected_fields(COMPACT_NAME_FIELDS, VERBOSE_NAME_FIELDS),
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )
            agents["name"] = CachedAgent.from_prompt_file(
                "name",
                prompt_dir / "name_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NAME", 220),
                _agent_expected_fields(COMPACT_NAME_FIELDS, VERBOSE_NAME_FIELDS),
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )
            agents["name"].prompt_builder = build_name_prompt
            agents["caller_name_fallback"] = CachedAgent.from_prompt_file(
                "caller_name_fallback",
                prompt_dir / "caller_name_fallback_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_CALLER_NAME_FALLBACK", 160),
                _agent_expected_fields(COMPACT_NAME_FIELDS, VERBOSE_NAME_FIELDS),
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )
        elif topology in SUBJECT_FALLBACK_TOPOLOGIES:
            agents["subject_name"] = CachedAgent.from_prompt_file(
                "subject_name",
                prompt_dir / "subject_name_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_SUBJECT_NAME", 160),
                _agent_expected_fields(COMPACT_NAME_FIELDS, VERBOSE_NAME_FIELDS),
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )
            agents["caller_name_fallback"] = CachedAgent.from_prompt_file(
                "caller_name_fallback",
                prompt_dir / "caller_name_fallback_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_CALLER_NAME_FALLBACK", 160),
                _agent_expected_fields(COMPACT_NAME_FIELDS, VERBOSE_NAME_FIELDS),
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )
        else:
            agents["name"] = CachedAgent.from_prompt_file(
                "name",
                prompt_dir / "name_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NAME", 220),
                _agent_expected_fields(COMPACT_NAME_FIELDS, VERBOSE_NAME_FIELDS),
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )
            agents["name"].prompt_builder = build_name_prompt
        if topology == "split_identity_correction":
            agents["name_correction"] = CachedAgent.from_prompt_file(
                "name_correction",
                prompt_dir / "name_correction_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_NAME_CORRECTION", 128),
                _agent_expected_fields(COMPACT_NAME_CORRECTION_FIELDS, VERBOSE_NAME_CORRECTION_FIELDS),
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )
            agents["name_correction"].prompt_builder = build_name_correction_prompt
        if topology in DUAL_CORRECTION_TOPOLOGIES and spelling_correction_agent_enabled():
            agents["spelling_correction"] = CachedAgent.from_prompt_file(
                "spelling_correction",
                prompt_dir / "spelling_correction_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_SPELLING_CORRECTION", 64),
                _agent_expected_fields(COMPACT_NAME_CORRECTION_FIELDS, VERBOSE_NAME_CORRECTION_FIELDS),
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )
            agents["spelling_correction"].prompt_builder = build_spelling_correction_prompt
        if topology in DUAL_CORRECTION_TOPOLOGIES and caller_id_correction_agent_enabled():
            agents["caller_id_correction"] = CachedAgent.from_prompt_file(
                "caller_id_correction",
                prompt_dir / "caller_id_correction_agent.md",
                env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_CALLER_ID_CORRECTION", 96),
                _agent_expected_fields(COMPACT_NAME_CORRECTION_FIELDS, VERBOSE_NAME_CORRECTION_FIELDS),
                text_generator=text_generator,
                constraint_builder=constraint_builder,
            )
            agents["caller_id_correction"].prompt_builder = build_caller_id_correction_prompt
        agents["dob"] = CachedAgent.from_prompt_file(
            "dob",
            prompt_dir / "dob_agent.md",
            env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_DOB", 160),
            _agent_expected_fields(COMPACT_DOB_FIELDS, VERBOSE_DOB_FIELDS),
            text_generator=text_generator,
            constraint_builder=constraint_builder,
        )
    else:
        agents["identity"] = CachedAgent.from_prompt_file(
            "identity",
            prompt_dir / "identity_agent.md",
            env_int("CANDIDATE_AGENT_MAX_OUTPUT_TOKENS_IDENTITY", 220),
            _agent_expected_fields(COMPACT_IDENTITY_FIELDS, VERBOSE_IDENTITY_FIELDS),
            text_generator=text_generator,
            constraint_builder=constraint_builder,
        )
    return agents


def _run_custom_candidate_agents(
    active_agents: dict[str, Any],
    candidates: dict[str, Any],
    transcript: str,
    include_full_transcript: bool,
    trace_sink: AgentTraceSink | None = None,
) -> dict[str, Any]:
    global last_agent_timings_ms, last_agent_skipped, last_agent_auto_accepted
    global last_agent_constraint_modes, last_parallel_total_ms

    selected = set(candidate_agent_selection())
    specifications: dict[str, tuple[dict[str, Any], dict[str, Any], bool]] = {}
    if "numbers" in selected:
        specifications["numbers"] = (
            _build_numbers_payload(candidates, transcript, include_full_transcript),
            _empty_numbers_output(),
            _has_number_candidates(candidates),
        )
    if "name" in selected:
        specifications["name"] = (
            _build_name_payload(candidates, transcript, include_full_transcript),
            _empty_name_output(),
            _has_name_candidates(candidates),
        )
    if "dob" in selected:
        specifications["dob"] = (
            _build_dob_payload(candidates, transcript, include_full_transcript),
            _empty_dob_output(),
            _has_dob_candidates(candidates),
        )
    if "subject_fallback" in selected:
        specifications["subject_name"] = (
            _build_subject_name_payload(candidates, transcript, include_full_transcript),
            _empty_name_output(),
            _has_subject_name_candidates(candidates),
        )
        specifications["caller_name_fallback"] = (
            _build_caller_name_fallback_payload(candidates, transcript, include_full_transcript),
            _empty_name_output(),
            False,
        )
    if "spelling_correction" in selected:
        specifications["spelling_correction"] = (
            _build_spelling_correction_payload(candidates, transcript, include_full_transcript),
            _empty_name_output(),
            _has_spelling_correction_candidates(candidates),
        )
    if "caller_id_correction" in selected:
        specifications["caller_id_correction"] = (
            _build_caller_id_correction_payload(candidates, transcript, include_full_transcript),
            _empty_name_output(),
            _has_caller_id_correction_candidates(candidates),
        )

    outputs = {name: dict(default) for name, (_payload, default, _relevant) in specifications.items()}
    timings = {name: 0 for name in specifications}
    modes = {name: "disabled" for name in specifications}
    skipped = [name for name, (_payload, _default, relevant) in specifications.items() if not relevant]
    run_names = [
        name
        for name, (_payload, _default, relevant) in specifications.items()
        if relevant and name != "caller_name_fallback"
    ]
    total_start = time.perf_counter()

    def run(name: str) -> tuple[str, dict[str, Any], int, str]:
        if name not in active_agents:
            raise RuntimeError(f"candidate agent {name!r} is not configured")
        payload, default, _relevant = specifications[name]
        output, duration, mode = _run_specialist_safely(name, active_agents, payload, default)
        return name, output, duration, mode

    if candidate_agent_execution_mode() == "parallel_http" and run_names:
        with ThreadPoolExecutor(max_workers=len(run_names)) as executor:
            futures = {executor.submit(run, name): name for name in run_names}
            results = (future.result() for future in as_completed(futures))
            for name, output, duration, mode in results:
                outputs[name], timings[name], modes[name] = output, duration, mode
                _emit_agent_trace(trace_sink, name, output, duration, mode)
    else:
        for name in run_names:
            _, output, duration, mode = run(name)
            outputs[name], timings[name], modes[name] = output, duration, mode
            _emit_agent_trace(trace_sink, name, output, duration, mode)

    subject_selected = _name_output_has_usable_selection(
        outputs.get("subject_name", _empty_name_output()),
        specifications.get("subject_name", ({}, {}, False))[0],
    )
    general_selected = _name_output_has_usable_selection(
        outputs.get("name", _empty_name_output()),
        specifications.get("name", ({}, {}, False))[0],
    )
    if "caller_name_fallback" in specifications and not subject_selected and not general_selected:
        name, output, duration, mode = run("caller_name_fallback")
        outputs[name], timings[name], modes[name] = output, duration, mode
        _emit_agent_trace(trace_sink, name, output, duration, mode)
        if name in skipped:
            skipped.remove(name)

    name_output = (
        outputs.get("subject_name", _empty_name_output())
        if subject_selected
        else outputs.get("name", _empty_name_output())
        if general_selected
        else outputs.get("caller_name_fallback", _empty_name_output())
    )
    merged = merge_split_agent_outputs(
        outputs.get("numbers", _empty_numbers_output()),
        name_output,
        outputs.get("dob", _empty_dob_output()),
        outputs.get("spelling_correction"),
        outputs.get("caller_id_correction"),
    )
    last_agent_timings_ms = timings
    last_agent_skipped = skipped
    last_agent_auto_accepted = []
    last_agent_constraint_modes = modes
    last_parallel_total_ms = int((time.perf_counter() - total_start) * 1000)
    return merged


def _agent_constraint_mode(agent: Any) -> str:
    metrics = getattr(agent, "metrics", None)
    if isinstance(metrics, dict) and isinstance(metrics.get("last_constraint_mode"), str):
        return metrics["last_constraint_mode"]
    return "disabled"


def _run_named_agent(agent_name: str, agent: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any], int, str]:
    start = time.perf_counter()
    output = agent.run(payload)
    duration_ms = int((time.perf_counter() - start) * 1000)
    return agent_name, output, duration_ms, _agent_constraint_mode(agent)


def _run_specialist_safely(
    agent_name: str,
    active_agents: dict[str, Any],
    payload: dict[str, Any],
    default: dict[str, Any],
) -> tuple[dict[str, Any], int, str]:
    agent = active_agents.get(agent_name)
    if agent is None:
        failed = dict(default)
        failed["errors"] = [f"{agent_name}_not_configured"]
        return failed, 0, "disabled"

    start = time.perf_counter()
    try:
        _, output, duration_ms, constraint_mode = _run_named_agent(agent_name, agent, payload)
        return output, duration_ms, constraint_mode
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.warning("candidate specialist failed agent=%s error=%s", agent_name, exc)
        failed = dict(default)
        failed["errors"] = [f"{agent_name}_failed"]
        return failed, duration_ms, _agent_constraint_mode(agent)


def _selected_name_ids(output: dict[str, Any]) -> list[str]:
    selected = [str(item) for item in _candidate_items(output, "name_ids") if str(item)]
    for key in ("patient_names", "name_correction_candidates"):
        for item in _candidate_items(output, key):
            if isinstance(item, dict) and item.get("candidate_id"):
                selected.append(str(item["candidate_id"]))
    return selected


def _name_output_has_usable_selection(
    output: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    allowed = {
        str(item.get("id"))
        for item in payload.get("name_candidates", [])
        if isinstance(item, dict) and item.get("id") and item.get("evidence_text")
    }
    return any(candidate_id in allowed for candidate_id in _selected_name_ids(output))


def _run_parallel_six_agent_wave(
    active_agents: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    defaults: dict[str, dict[str, Any]],
    relevant: dict[str, bool],
    outputs: dict[str, dict[str, Any]],
    timings: dict[str, int],
    constraint_modes: dict[str, str],
    trace_sink: AgentTraceSink | None = None,
) -> None:
    agent_names = [
        agent_name
        for agent_name in ("numbers", "dob", "subject_name", "name")
        if relevant[agent_name]
    ]
    if not agent_names:
        return

    with ThreadPoolExecutor(max_workers=min(4, len(agent_names))) as executor:
        futures = {
            executor.submit(
                _run_specialist_safely,
                agent_name,
                active_agents,
                payloads[agent_name],
                defaults[agent_name],
            ): agent_name
            for agent_name in agent_names
        }
        for future in as_completed(futures):
            agent_name = futures[future]
            (
                outputs[agent_name],
                timings[agent_name],
                constraint_modes[agent_name],
            ) = future.result()
            _emit_agent_trace(
                trace_sink,
                agent_name,
                outputs[agent_name],
                timings[agent_name],
                constraint_modes[agent_name],
            )


def _run_six_agent_topology(
    active_agents: dict[str, Any],
    candidates: dict[str, Any],
    transcript: str,
    include_full_transcript: bool,
    trace_sink: AgentTraceSink | None = None,
) -> dict[str, Any]:
    global last_agent_timings_ms, last_agent_skipped, last_agent_auto_accepted
    global last_agent_constraint_modes, last_parallel_total_ms

    payloads = {
        "numbers": _build_numbers_payload(candidates, transcript, include_full_transcript),
        "dob": _build_dob_payload(candidates, transcript, include_full_transcript),
        "subject_name": _build_subject_name_payload(candidates, transcript, include_full_transcript),
        "name": _build_name_payload(candidates, transcript, include_full_transcript),
        "caller_name_fallback": _build_caller_name_fallback_payload(
            candidates,
            transcript,
            include_full_transcript,
        ),
    }
    relevant = {
        "numbers": _has_number_candidates(candidates),
        "dob": _has_dob_candidates(candidates),
        "subject_name": _has_subject_name_candidates(candidates),
        "name": _has_name_candidates(candidates),
        "caller_name_fallback": _has_fallback_name_candidates(candidates),
    }
    defaults = {
        "numbers": _empty_numbers_output(),
        "dob": _empty_dob_output(),
        "subject_name": _empty_name_output(),
        "name": _empty_name_output(),
        "caller_name_fallback": _empty_name_output(),
    }
    outputs = {key: dict(value) for key, value in defaults.items()}
    timings = {key: 0 for key in defaults}
    constraint_modes = {key: "disabled" for key in defaults}
    skipped = [key for key in defaults if not relevant[key]]
    total_start = time.perf_counter()

    if candidate_agent_execution_mode() == "parallel_http":
        _run_parallel_six_agent_wave(
            active_agents,
            payloads,
            defaults,
            relevant,
            outputs,
            timings,
            constraint_modes,
            trace_sink,
        )
    else:
        for agent_name in ("numbers", "dob", "subject_name", "name"):
            if not relevant[agent_name]:
                continue
            (
                outputs[agent_name],
                timings[agent_name],
                constraint_modes[agent_name],
            ) = _run_specialist_safely(
                agent_name,
                active_agents,
                payloads[agent_name],
                defaults[agent_name],
            )
            _emit_agent_trace(
                trace_sink,
                agent_name,
                outputs[agent_name],
                timings[agent_name],
                constraint_modes[agent_name],
            )

    subject_selected = _name_output_has_usable_selection(
        outputs["subject_name"],
        payloads["subject_name"],
    )
    general_selected = _name_output_has_usable_selection(
        outputs["name"],
        payloads["name"],
    )
    fallback_required = not subject_selected and not general_selected
    if fallback_required and relevant["caller_name_fallback"]:
        (
            outputs["caller_name_fallback"],
            timings["caller_name_fallback"],
            constraint_modes["caller_name_fallback"],
        ) = _run_specialist_safely(
            "caller_name_fallback",
            active_agents,
            payloads["caller_name_fallback"],
            defaults["caller_name_fallback"],
        )
        _emit_agent_trace(
            trace_sink,
            "caller_name_fallback",
            outputs["caller_name_fallback"],
            timings["caller_name_fallback"],
            constraint_modes["caller_name_fallback"],
        )
    elif "caller_name_fallback" not in skipped:
        skipped.append("caller_name_fallback")

    winning_name = (
        outputs["subject_name"]
        if subject_selected
        else outputs["name"]
        if general_selected
        else outputs["caller_name_fallback"]
    )

    last_agent_timings_ms = timings
    last_agent_skipped = skipped
    last_agent_auto_accepted = []
    last_agent_constraint_modes = constraint_modes
    last_parallel_total_ms = int((time.perf_counter() - total_start) * 1000)
    logger.info("candidate fallback required=%s", fallback_required)

    merged = merge_split_agent_outputs(outputs["numbers"], winning_name, outputs["dob"])
    specialist_errors = [
        str(error)
        for output in outputs.values()
        for error in _candidate_items(output, "errors") + _candidate_items(output, "possible_errors")
        if str(error).strip()
    ]
    merged["possible_errors"] = list(
        dict.fromkeys([*merged.get("possible_errors", []), *specialist_errors])
    )
    return merged


def _run_split_candidate_agents(
    active_agents: dict[str, Any],
    candidates: dict[str, Any],
    transcript: str,
    include_full_transcript: bool,
    trace_sink: AgentTraceSink | None = None,
) -> dict[str, Any]:
    global last_agent_timings_ms, last_agent_skipped, last_agent_auto_accepted, last_agent_constraint_modes, last_parallel_total_ms

    topology = candidate_agent_topology()
    if topology in SCOUT_SUBJECT_GENERAL_FALLBACK_TOPOLOGIES:
        return _run_six_agent_topology(
            active_agents,
            candidates,
            transcript,
            include_full_transcript,
            trace_sink,
        )

    auto_accepted: list[str] = []
    auto_dob_id = _single_auto_accepted_dob_id(candidates)
    auto_spelling_name_id = _single_auto_accepted_spelling_name_id(candidates)
    auto_self_name_id = _single_auto_accepted_self_identification_name_id(candidates)
    payloads = {
        "numbers": _build_numbers_payload(candidates, transcript, include_full_transcript),
        "dob": _build_dob_payload(candidates, transcript, include_full_transcript),
    }
    defaults = {
        "numbers": _empty_numbers_output(),
        "dob": _empty_dob_output(),
    }
    relevant = {
        "numbers": _has_number_candidates(candidates),
        "dob": _has_dob_candidates(candidates),
    }
    if topology in SUBJECT_FALLBACK_TOPOLOGIES:
        payloads["subject_name"] = _build_subject_name_payload(candidates, transcript, include_full_transcript)
        payloads["caller_name_fallback"] = _build_caller_name_fallback_payload(
            candidates,
            transcript,
            include_full_transcript,
        )
        defaults["subject_name"] = _empty_name_output()
        defaults["caller_name_fallback"] = _empty_name_output()
        relevant["subject_name"] = _has_subject_name_candidates(candidates)
        relevant["caller_name_fallback"] = _has_fallback_name_candidates(candidates)
    else:
        payloads["name"] = _build_name_payload(candidates, transcript, include_full_transcript)
        defaults["name"] = _empty_name_output()
        relevant["name"] = _has_name_candidates(candidates)
    if auto_dob_id:
        defaults["dob"] = {"dob_ids": [auto_dob_id], "errors": []}
        relevant["dob"] = False
        auto_accepted.append("dob")
    if auto_self_name_id:
        if topology in SUBJECT_FALLBACK_TOPOLOGIES:
            defaults["caller_name_fallback"] = {
                "name_ids": [auto_self_name_id],
                "name_correction_ids": [],
                "errors": [],
            }
            relevant["caller_name_fallback"] = False
            auto_accepted.append("caller_name_fallback")
        else:
            defaults["name"] = {"name_ids": [auto_self_name_id], "name_correction_ids": [], "errors": []}
            relevant["name"] = False
            auto_accepted.append("name")
    if topology == "split_identity_correction":
        payloads["name_correction"] = _build_name_correction_payload(
            candidates,
            transcript,
            include_full_transcript,
        )
        defaults["name_correction"] = _empty_name_output()
        relevant["name_correction"] = _has_name_correction_candidates(candidates)
        if auto_spelling_name_id and not _has_caller_id_correction_candidates(candidates):
            defaults["name_correction"] = {"name_ids": [auto_spelling_name_id], "name_correction_ids": [], "errors": []}
            relevant["name_correction"] = False
            auto_accepted.append("name_correction")
    if topology in DUAL_CORRECTION_TOPOLOGIES:
        defaults["spelling_correction"] = _empty_name_output()
        defaults["caller_id_correction"] = _empty_name_output()
        if spelling_correction_agent_enabled():
            payloads["spelling_correction"] = _build_spelling_correction_payload(
                candidates,
                transcript,
                include_full_transcript,
            )
            relevant["spelling_correction"] = _has_spelling_correction_candidates(candidates)
        else:
            payloads["spelling_correction"] = {}
            relevant["spelling_correction"] = False
        if caller_id_correction_agent_enabled():
            payloads["caller_id_correction"] = _build_caller_id_correction_payload(
                candidates,
                transcript,
                include_full_transcript,
            )
            relevant["caller_id_correction"] = _has_caller_id_correction_candidates(candidates)
        else:
            payloads["caller_id_correction"] = {}
            relevant["caller_id_correction"] = False
        if auto_spelling_name_id and spelling_correction_agent_enabled():
            defaults["spelling_correction"] = {"name_ids": [auto_spelling_name_id], "name_correction_ids": [], "errors": []}
            relevant["spelling_correction"] = False
            auto_accepted.append("spelling_correction")
    run_payloads = {name: payload for name, payload in payloads.items() if relevant[name]}
    skipped = [name for name in payloads if not relevant[name]]

    for name in run_payloads:
        if name not in active_agents:
            raise RuntimeError(f"candidate agent {name!r} is not configured")

    outputs: dict[str, dict[str, Any]] = {name: dict(defaults[name]) for name in skipped}
    timings: dict[str, int] = {name: 0 for name in skipped}
    constraint_modes: dict[str, str] = {name: "disabled" for name in skipped}
    total_start = time.perf_counter()
    if candidate_agent_execution_mode() == "parallel_http" and run_payloads:
        with ThreadPoolExecutor(max_workers=max(1, len(run_payloads))) as executor:
            futures = {
                executor.submit(_run_named_agent, name, active_agents[name], payload): name
                for name, payload in run_payloads.items()
            }
            for future in as_completed(futures):
                name, output, duration_ms, constraint_mode = future.result()
                outputs[name] = output
                timings[name] = duration_ms
                constraint_modes[name] = constraint_mode
                _emit_agent_trace(trace_sink, name, output, duration_ms, constraint_mode)
    else:
        for name, payload in run_payloads.items():
            agent_name, output, duration_ms, constraint_mode = _run_named_agent(name, active_agents[name], payload)
            outputs[agent_name] = output
            timings[agent_name] = duration_ms
            constraint_modes[agent_name] = constraint_mode
            _emit_agent_trace(trace_sink, agent_name, output, duration_ms, constraint_mode)

    last_agent_timings_ms = timings
    last_agent_skipped = skipped
    last_agent_auto_accepted = auto_accepted
    last_agent_constraint_modes = constraint_modes
    last_parallel_total_ms = int((time.perf_counter() - total_start) * 1000)
    if topology == "split_identity_correction":
        return merge_split_agent_outputs(
            outputs["numbers"],
            outputs["name"],
            outputs["dob"],
            outputs["name_correction"],
        )
    if topology == "split_identity_dual_correction":
        return merge_split_agent_outputs(
            outputs["numbers"],
            outputs["name"],
            outputs["dob"],
            outputs["spelling_correction"],
            outputs["caller_id_correction"],
        )
    if topology in SUBJECT_FALLBACK_TOPOLOGIES:
        subject_selected = _name_output_has_selection(outputs["subject_name"])
        name_output = (
            outputs["subject_name"]
            if subject_selected
            else outputs["caller_name_fallback"]
        )
        caller_id_correction_output = (
            _empty_name_output()
            if subject_selected
            else outputs["caller_id_correction"]
        )
        return merge_split_agent_outputs(
            outputs["numbers"],
            name_output,
            outputs["dob"],
            outputs["spelling_correction"],
            caller_id_correction_output,
        )
    return merge_split_agent_outputs(outputs["numbers"], outputs["name"], outputs["dob"])


def _run_candidate_agents(
    request: dict[str, Any],
    *,
    agents: dict[str, Any] | None,
    include_full_transcript: bool,
    trace_sink: AgentTraceSink | None = None,
) -> dict[str, Any]:
    if not candidate_extractor_enabled():
        raise RuntimeError("candidate extractor is disabled")
    if not candidate_agents_enabled():
        raise RuntimeError("candidate agents are disabled")

    payload = request_payload(request)
    transcript = str(payload.get("transcript") or "").strip()
    if not transcript:
        raise RuntimeError("candidate extraction skipped: transcript is empty")
    candidates = extract_candidates(
        transcript,
        caller_id=str(payload.get("caller_id") or ""),
        mailbox=str(payload.get("mailbox") or ""),
        transcript_id=str(payload.get("transcript_id") or payload.get("request_id") or ""),
    )

    global last_agent_timings_ms, last_agent_skipped, last_agent_auto_accepted, last_agent_constraint_modes, last_parallel_total_ms

    active_agents = agents if agents is not None else build_default_agents()
    _run_e2b_candidate_scout(active_agents, candidates, transcript, trace_sink)
    _record_name_candidate_counts(candidates)
    topology = candidate_agent_topology()
    if topology == "custom":
        merged = _run_custom_candidate_agents(
            active_agents,
            candidates,
            transcript,
            include_full_transcript,
            trace_sink,
        )
    elif topology == "numbers_only":
        outputs: dict[str, dict[str, Any]] = {}
        timings: dict[str, int] = {}
        constraint_modes: dict[str, str] = {}
        skipped: list[str] = []
        total_start = time.perf_counter()
        if _has_number_candidates(candidates):
            if "numbers" not in active_agents:
                raise RuntimeError("candidate agent 'numbers' is not configured")
            _, outputs["numbers"], timings["numbers"], constraint_modes["numbers"] = _run_named_agent(
                "numbers",
                active_agents["numbers"],
                _build_numbers_payload(candidates, transcript, include_full_transcript),
            )
            _emit_agent_trace(
                trace_sink,
                "numbers",
                outputs["numbers"],
                timings["numbers"],
                constraint_modes["numbers"],
            )
        else:
            outputs["numbers"] = _empty_numbers_output()
            timings["numbers"] = 0
            constraint_modes["numbers"] = "disabled"
            skipped.append("numbers")
        last_agent_timings_ms = timings
        last_agent_skipped = skipped
        last_agent_auto_accepted = []
        last_agent_constraint_modes = constraint_modes
        last_parallel_total_ms = int((time.perf_counter() - total_start) * 1000)
        merged = merge_agent_outputs(outputs["numbers"], {})
    elif topology in SPLIT_TOPOLOGIES:
        merged = _run_split_candidate_agents(
            active_agents,
            candidates,
            transcript,
            include_full_transcript,
            trace_sink,
        )
    else:
        identity_agent = active_agents["identity"]
        outputs: dict[str, dict[str, Any]] = {}
        timings: dict[str, int] = {}
        constraint_modes: dict[str, str] = {}
        skipped: list[str] = []
        total_start = time.perf_counter()
        if _has_number_candidates(candidates):
            if "numbers" not in active_agents:
                raise RuntimeError("candidate agent 'numbers' is not configured")
            _, outputs["numbers"], timings["numbers"], constraint_modes["numbers"] = _run_named_agent(
                "numbers",
                active_agents["numbers"],
                _build_numbers_payload(candidates, transcript, include_full_transcript),
            )
            _emit_agent_trace(
                trace_sink,
                "numbers",
                outputs["numbers"],
                timings["numbers"],
                constraint_modes["numbers"],
            )
        else:
            outputs["numbers"] = _empty_numbers_output()
            timings["numbers"] = 0
            constraint_modes["numbers"] = "disabled"
            skipped.append("numbers")
        _, outputs["identity"], timings["identity"], constraint_modes["identity"] = _run_named_agent(
            "identity",
            identity_agent,
            _build_identity_payload(candidates, transcript, include_full_transcript),
        )
        _emit_agent_trace(
            trace_sink,
            "identity",
            outputs["identity"],
            timings["identity"],
            constraint_modes["identity"],
        )
        numbers_output = outputs["numbers"]
        identity_output = outputs["identity"]
        last_agent_timings_ms = timings
        last_agent_skipped = skipped
        last_agent_auto_accepted = []
        last_agent_constraint_modes = constraint_modes
        last_parallel_total_ms = int((time.perf_counter() - total_start) * 1000)
        merged = merge_agent_outputs(numbers_output, identity_output)
    return validate_final_json(merged, candidates, transcript)


def _call_legacy(legacy_extractor, request: dict[str, Any]) -> dict[str, Any]:
    if legacy_extractor is None:
        return empty_final_json()
    result = legacy_extractor(request)
    if isinstance(result, dict):
        return result
    return empty_final_json()


def log_shadow_comparison(legacy: dict[str, Any], candidate: dict[str, Any]) -> None:
    global last_shadow_comparison
    last_shadow_comparison = {
        "legacy_counts": {key: len(legacy.get(key, [])) for key in FINAL_SCHEMA_KEYS},
        "candidate_counts": {key: len(candidate.get(key, [])) for key in FINAL_SCHEMA_KEYS},
    }
    logger.info("candidate-agent shadow comparison counts=%s", last_shadow_comparison)


def extract_with_candidate_agents(
    request: dict[str, Any],
    legacy_extractor=None,
    *,
    agents: dict[str, Any] | None = None,
    mode: str | None = None,
    fallback_to_legacy: bool | None = None,
    include_full_transcript: bool | None = None,
    trace_sink: AgentTraceSink | None = None,
) -> dict[str, Any]:
    global candidate_agent_failures, legacy_fallbacks

    active_mode = (mode or candidate_agent_mode()).strip().lower()
    if active_mode not in VALID_MODES:
        active_mode = "legacy"
    fallback = fallback_to_legacy_enabled() if fallback_to_legacy is None else fallback_to_legacy
    include_transcript = include_full_transcript_enabled() if include_full_transcript is None else include_full_transcript

    if active_mode == "legacy":
        return _call_legacy(legacy_extractor, request)

    if active_mode == "shadow_candidate_agents":
        legacy = _call_legacy(legacy_extractor, request)
        try:
            candidate = _run_candidate_agents(
                request,
                agents=agents,
                include_full_transcript=include_transcript,
                trace_sink=trace_sink,
            )
            log_shadow_comparison(legacy, candidate)
        except Exception as exc:
            candidate_agent_failures += 1
            logger.warning("candidate-agent shadow run failed error=%s", exc)
        return legacy

    try:
        return _run_candidate_agents(
            request,
            agents=agents,
            include_full_transcript=include_transcript,
            trace_sink=trace_sink,
        )
    except Exception:
        candidate_agent_failures += 1
        if fallback:
            legacy_fallbacks += 1
            return _call_legacy(legacy_extractor, request)
        raise


def candidate_agent_health(agents_loaded: list[str] | None = None, execution_mode: str | None = None) -> dict[str, Any]:
    return {
        "candidate_extractor_enabled": candidate_extractor_enabled(),
        "candidate_agent_mode": candidate_agent_mode(),
        "agents_loaded": agents_loaded or [],
        "agent_execution_mode": execution_mode or candidate_agent_execution_mode(),
        "candidate_agent_topology": candidate_agent_topology(),
        "candidate_agent_failures": candidate_agent_failures,
        "legacy_fallbacks": legacy_fallbacks,
        "last_agent_timings_ms": dict(last_agent_timings_ms),
        "last_agent_skipped": list(last_agent_skipped),
        "last_agent_auto_accepted": list(last_agent_auto_accepted),
        "last_agent_constraint_modes": dict(last_agent_constraint_modes),
        "last_parallel_total_ms": last_parallel_total_ms,
        "candidate_agents_enabled": candidate_agents_enabled(),
        "candidate_agent_include_full_transcript": include_full_transcript_enabled(),
        "candidate_agent_run_verifier": verifier_enabled(),
        "candidate_agent_output_schema": candidate_agent_output_schema(),
        "candidate_agent_constrained_decoding": candidate_agent_constrained_decoding_enabled(),
        "candidate_agent_e2b_scout_enabled": candidate_agent_e2b_scout_enabled(),
        "candidate_agent_broad_name_recall": candidate_agent_broad_name_recall_enabled(),
        "last_e2b_scout_timing_ms": last_e2b_scout_timing_ms,
        "last_e2b_scout_added_counts": dict(last_e2b_scout_added_counts),
        "last_e2b_scout_rejected_counts": dict(last_e2b_scout_rejected_counts),
        "last_e2b_scout_errors": list(last_e2b_scout_errors),
        "last_name_candidate_total": last_name_candidate_total,
        "last_name_candidate_counts_by_source": dict(last_name_candidate_counts_by_source),
        "candidate_agent_auto_accept_deterministic": auto_accept_deterministic_enabled(),
        "candidate_agent_auto_accept_dob": auto_accept_dob_enabled(),
        "candidate_agent_auto_accept_transcript_spelling": auto_accept_transcript_spelling_enabled(),
        "candidate_agent_auto_accept_self_identification": auto_accept_self_identification_enabled(),
    }


def record_legacy_fallback() -> None:
    global legacy_fallbacks
    legacy_fallbacks += 1
