"""Gemma field-extraction compatibility surface."""

from __future__ import annotations

from watcher import (
    build_gemma_http_request,
    build_gemma_input_payload,
    call_gemma_field_extraction,
    extract_gemma_response_text,
    gemma_api_mode,
    gemma_payload_log_text,
    load_gemma_prompt,
    normalize_gemma_http_payload,
    response_error_excerpt,
)

__all__ = [
    "build_gemma_http_request",
    "build_gemma_input_payload",
    "call_gemma_field_extraction",
    "extract_gemma_response_text",
    "gemma_api_mode",
    "gemma_payload_log_text",
    "load_gemma_prompt",
    "normalize_gemma_http_payload",
    "response_error_excerpt",
]
