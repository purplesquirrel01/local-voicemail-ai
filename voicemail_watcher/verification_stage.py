"""Field verification compatibility surface."""

from __future__ import annotations

from watcher import (
    VerificationRunResult,
    add_compact_dob_fallback_candidates,
    add_relationship_name_fallback_candidates,
    add_self_identification_name_fallback_candidates,
    add_spelled_name_fallback_candidates,
    add_subject_reference_name_fallback_candidates,
    apply_resolutions_to_entities,
    build_candidate_records,
    build_gemma_input_payload,
    call_gemma_field_extraction,
    gemma_candidates_for_field,
    patient_name_values_for_compact_dob,
    run_parakeet_for_record,
    safe_verify_voicemail_fields,
    select_entities_for_output,
    timeout_audit_rows,
    unavailable_audit_rows,
    verification_apply_gate_satisfied,
    verify_voicemail_fields,
)

__all__ = [
    "VerificationRunResult",
    "add_compact_dob_fallback_candidates",
    "add_relationship_name_fallback_candidates",
    "add_self_identification_name_fallback_candidates",
    "add_spelled_name_fallback_candidates",
    "add_subject_reference_name_fallback_candidates",
    "apply_resolutions_to_entities",
    "build_candidate_records",
    "build_gemma_input_payload",
    "call_gemma_field_extraction",
    "gemma_candidates_for_field",
    "patient_name_values_for_compact_dob",
    "run_parakeet_for_record",
    "safe_verify_voicemail_fields",
    "select_entities_for_output",
    "timeout_audit_rows",
    "unavailable_audit_rows",
    "verification_apply_gate_satisfied",
    "verify_voicemail_fields",
]
