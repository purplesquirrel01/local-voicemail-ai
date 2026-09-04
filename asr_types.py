from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SpanCandidate:
    span_id: str
    file_key: str
    field_type: str
    source: str
    source_run_id: str = ""
    start: Optional[float] = None
    end: Optional[float] = None
    word_start: Optional[int] = None
    word_end: Optional[int] = None
    text: str = ""
    normalized_value: Optional[str] = None
    confidence: Optional[float] = None
    reasons: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "file_key": self.file_key,
            "field_type": self.field_type,
            "source": self.source,
            "source_run_id": self.source_run_id,
            "start": self.start,
            "end": self.end,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "text": self.text,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "reason_json": list(self.reasons or []),
            "reasons": list(self.reasons or []),
        }
