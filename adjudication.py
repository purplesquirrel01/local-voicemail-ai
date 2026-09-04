from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional


logger = logging.getLogger("voicemail_adjudication")


def _candidate_values(candidate: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("normalized", "normalized_value", "value", "final_value", "raw"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            values.add(str(value).strip())
    return values


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").lower())


def _name_tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z]+", str(value or "").lower()) if len(token) > 1}


def _names_plausibly_match(left: Any, right: Any) -> bool:
    left_key = _name_key(left)
    right_key = _name_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key or left_key in right_key or right_key in left_key:
        return True
    return bool(_name_tokens(left) & _name_tokens(right))


def _candidate_has_spoken_name_evidence(candidate: dict[str, Any]) -> bool:
    source = str(candidate.get("source") or candidate.get("candidate_source") or "").lower()
    evidence = str(candidate.get("evidence_text") or candidate.get("raw") or candidate.get("value") or "").strip()
    return bool(evidence) and source not in {"caller_id", "metadata_prior"}


def _bias_name_values(request: dict[str, Any], chosen_candidate: dict[str, Any]) -> set[str]:
    context = request.get("context") if isinstance(request.get("context"), dict) else {}
    if not _candidate_has_spoken_name_evidence(chosen_candidate):
        return set()
    spoken_values = _candidate_values(chosen_candidate)
    allowed: set[str] = set()
    for item in context.get("bias_candidates") or []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or "").strip()
        kind = str(item.get("type") or "").lower()
        if not value or kind not in {"doctor", "name", "patient", "provider", "common_name", "person", "staff"}:
            continue
        if any(_names_plausibly_match(spoken, value) for spoken in spoken_values):
            allowed.add(value)
    return allowed


def validate_adjudication_decision(request: dict[str, Any], decision: Any, require_source_match: bool = True) -> Optional[dict[str, Any]]:
    if not isinstance(decision, dict):
        return None
    field = str(request.get("field") or "")
    if decision.get("field") != field:
        return None
    decision_type = decision.get("decision_type")
    if decision_type not in {"choose_candidate", "not_included", "uncertain"}:
        return None
    candidates = [item for item in request.get("candidates") or [] if isinstance(item, dict)]
    candidates_by_id = {str(item.get("candidate_id")): item for item in candidates}
    if decision_type == "choose_candidate":
        chosen = str(decision.get("chosen_candidate_id") or "")
        final_value = str(decision.get("final_value") or "").strip()
        _ = require_source_match
        if chosen not in candidates_by_id:
            return None
        chosen_candidate = candidates_by_id[chosen]
        allowed_values = _candidate_values(chosen_candidate)
        if field == "name":
            allowed_values.update(_bias_name_values(request, chosen_candidate))
        if final_value not in allowed_values:
            return None
    return {
        "field": field,
        "decision_type": decision_type,
        "chosen_candidate_id": decision.get("chosen_candidate_id"),
        "final_value": decision.get("final_value"),
        "needs_review": bool(decision.get("needs_review", decision_type != "choose_candidate")),
        "reason_code": str(decision.get("reason_code") or decision_type),
        "confidence": decision.get("confidence"),
    }


def compact_adjudication_prompt(payload: dict[str, Any]) -> str:
    safe_payload = {
        "field": payload.get("field"),
        "candidates": payload.get("candidates") or [],
        "context": payload.get("context") or {},
        "allowed_decisions": ["choose_candidate", "not_included", "uncertain"],
    }
    return (
        "Return JSON only. Choose only one provided candidate id/value; do not invent values.\n"
        "For callback_number, fax_number, and dob, final_value must come from the chosen candidate, not context bias.\n"
        "For name, context bias can only correct spelling when the chosen candidate has spoken evidence.\n"
        "Schema: {\"field\":\"\",\"decision_type\":\"choose_candidate|not_included|uncertain\","
        "\"chosen_candidate_id\":null,\"final_value\":null,\"needs_review\":true,"
        "\"reason_code\":\"\",\"confidence\":0.0}\n"
        f"Input JSON:\n{json.dumps(safe_payload, ensure_ascii=True)}"
    )


def _compact_span_alternatives(span: dict[str, Any]) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    for index, item in enumerate(span.get("alternatives") or []):
        if not isinstance(item, dict):
            continue
        alternatives.append(
            {
                "index": index,
                "text": re.sub(r"\s+", " ", str(item.get("text") or item.get("value") or "").strip()),
                "source": str(item.get("source") or item.get("engine") or "unknown"),
                "confidence": item.get("confidence"),
                "is_primary": bool(item.get("is_primary")),
                "strong": bool(item.get("strong") or item.get("verified")),
                "reason": item.get("reason"),
                "reason_code": item.get("reason_code"),
            }
        )
    return alternatives


def compact_transcript_adjudication_prompt(span: dict[str, Any]) -> str:
    safe_payload = {
        "span_id": span.get("span_id"),
        "start": span.get("start"),
        "end": span.get("end"),
        "primary_text": re.sub(r"\s+", " ", str(span.get("primary_text") or "").strip()),
        "reasons": list(span.get("reasons") or []),
        "alternatives": _compact_span_alternatives(span),
        "allowed_decisions": ["choose_alternative", "keep_primary", "uncertain"],
    }
    return (
        "Return JSON only. You are adjudicating one transcript disagreement span.\n"
        "Choose only one provided non-primary alternative when it is clearly more accurate than primary.\n"
        "Do not invent text, add words, remove words, normalize names, or combine alternatives.\n"
        "Do not choose an alternative only because it changes punctuation or casing; primary punctuation is preferred.\n"
        "If unsure, choose keep_primary or uncertain.\n"
        "For choose_alternative, final_text must exactly equal the chosen alternative text.\n"
        "Schema: {\"decision_type\":\"choose_alternative|keep_primary|uncertain\","
        "\"chosen_alternative_index\":null,\"final_text\":null,\"source\":null,"
        "\"reason_code\":\"\",\"confidence\":0.0}\n"
        f"Input JSON:\n{json.dumps(safe_payload, ensure_ascii=True)}"
    )


def validate_transcript_adjudication_decision(
    span: dict[str, Any],
    decision: Any,
    require_source_match: bool = True,
    min_confidence: float = 0.0,
) -> Optional[dict[str, Any]]:
    if not isinstance(decision, dict):
        return None
    decision_type = str(decision.get("decision_type") or "")
    if decision_type not in {"choose_alternative", "keep_primary", "uncertain"}:
        return None
    if decision_type != "choose_alternative":
        return {
            "decision_type": decision_type,
            "needs_review": decision_type == "uncertain",
            "reason_code": str(decision.get("reason_code") or decision_type),
            "confidence": decision.get("confidence"),
        }

    try:
        chosen_index = int(decision.get("chosen_alternative_index"))
    except (TypeError, ValueError):
        return None
    alternatives = [item for item in span.get("alternatives") or [] if isinstance(item, dict)]
    if chosen_index < 0 or chosen_index >= len(alternatives):
        return None
    chosen = alternatives[chosen_index]
    if chosen.get("is_primary"):
        return None

    alternative_text = re.sub(r"\s+", " ", str(chosen.get("text") or chosen.get("value") or "").strip())
    final_text = re.sub(r"\s+", " ", str(decision.get("final_text") or "").strip())
    if not alternative_text or final_text != alternative_text:
        return None

    decision_source = str(decision.get("source") or "").strip()
    chosen_source = str(chosen.get("source") or chosen.get("engine") or "unknown")
    if require_source_match and decision_source and decision_source != chosen_source:
        return None

    try:
        confidence = float(decision.get("confidence"))
    except (TypeError, ValueError):
        return None
    if confidence < float(min_confidence or 0.0):
        return None

    return {
        "decision_type": "choose_alternative",
        "chosen_alternative_index": chosen_index,
        "text": alternative_text,
        "new_text": alternative_text,
        "source": chosen_source,
        "confidence": confidence,
        "reason_code": str(decision.get("reason_code") or "llm_adjudicated_grounded_alternative"),
        "needs_review": False,
    }


def adjudicate_with_litert(
    request: dict[str, Any],
    settings: Any,
    requests_module: Any,
    base_url: str,
) -> Optional[dict[str, Any]]:
    prompt = compact_adjudication_prompt(request)
    try:
        response = requests_module.post(
            base_url.rstrip("/") + "/api/chat",
            json={"message": prompt, "history": [], "show_thinking": False},
            timeout=(5, float(getattr(settings, "llm_adjudication_timeout_seconds", 30) or 30)),
        )
        if response.status_code >= 400:
            return None
        payload = response.json()
    except Exception:
        return None
    text = payload.get("response") or payload.get("reply") or payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        return None
    try:
        decision = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return validate_adjudication_decision(
        request,
        decision,
        require_source_match=bool(getattr(settings, "llm_adjudication_require_source_match", True)),
    )
