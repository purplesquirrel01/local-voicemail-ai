"""Verification attribution compatibility surface."""

from __future__ import annotations

from verification import (
    create_verification_clip,
    map_evidence_to_timestamps,
    normalize_segment_item,
    normalize_word_item,
    safe_clip_filename,
    segment_bounds_for_span,
    set_padded_clip_bounds,
)

__all__ = [
    "create_verification_clip",
    "map_evidence_to_timestamps",
    "normalize_segment_item",
    "normalize_word_item",
    "safe_clip_filename",
    "segment_bounds_for_span",
    "set_padded_clip_bounds",
]
