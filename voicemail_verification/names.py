"""Name verification compatibility surface."""

from __future__ import annotations

from verification import (
    caller_id_can_correct_spoken_name,
    clean_fallback_person_name,
    clean_name,
    corrected_name_from_spelling,
    evidence_is_addressee_only,
    extract_relationship_name_candidates,
    extract_self_identification_name_candidates,
    extract_spelled_name_candidates,
    extract_subject_reference_name_candidates,
    name_candidate_score,
    name_key,
    names_plausibly_match,
    person_like_name,
    resolve_name_field,
    subject_reference_looks_like_topic,
)

__all__ = [
    "caller_id_can_correct_spoken_name",
    "clean_fallback_person_name",
    "clean_name",
    "corrected_name_from_spelling",
    "evidence_is_addressee_only",
    "extract_relationship_name_candidates",
    "extract_self_identification_name_candidates",
    "extract_spelled_name_candidates",
    "extract_subject_reference_name_candidates",
    "name_candidate_score",
    "name_key",
    "names_plausibly_match",
    "person_like_name",
    "resolve_name_field",
    "subject_reference_looks_like_topic",
]
