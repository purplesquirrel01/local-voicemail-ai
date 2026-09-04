#!/usr/bin/env python3
"""
Durable voicemail watcher for adapter-supported Asterisk systems.

The watcher observes Asterisk voicemail metadata files, waits until the matching
audio and metadata are complete, transcribes the audio through the local Whisper
API, and emails the transcript plus WAV attachment to the mailbox owner.

Production guarantees provided here:
- durable SQLite status table instead of an append-only text file
- startup reconciliation for missed filesystem events
- retry/dead-letter state instead of silent one-shot failures
- no transcript/PHI logging
- environment-driven configuration
- event handlers that enqueue work instead of blocking on transcription/email
"""

from __future__ import annotations

import glob
import json
import logging
import os
import queue
import re
import smtplib
import sqlite3
import ssl
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from adjudication import compact_transcript_adjudication_prompt, validate_transcript_adjudication_decision
from asr_lattice import build_disagreement_spans, correct_transcript_constrained
from verification import (
    CandidateRecord,
    FieldResolution,
    GemmaSchemaError,
    VerificationBudgetExceeded,
    call_parakeet_cli,
    call_parakeet_http,
    check_budget,
    constrain_phone_clip_bounds_for_neighbors,
    create_verification_clip,
    extract_compact_dob_candidates,
    extract_explicit_patient_name_candidates,
    extract_numbers_from_text,
    extract_relationship_name_candidates,
    extract_self_identification_name_candidates,
    extract_spelled_name_candidates,
    extract_subject_reference_name_candidates,
    format_dob,
    format_phone_digits,
    map_evidence_to_timestamps,
    normalize_phone_candidate,
    parse_dob,
    parse_gemma_response,
    remaining_budget,
    resolve_dob_field,
    resolve_legacy_field,
    resolve_name_field,
    resolve_phone_field,
)
from voicemail_common import env as common_env
from voicemail_common import formatting as common_formatting
from voicemail_common import keys as common_keys
from voicemail_common import spool as common_spool
from voicemail_common import time as common_time
from voicemail_watcher import transcript_corrections as watcher_transcript_corrections
from voicemail_watcher.mailbox_spelling_rules import apply_mailbox_spelling_rules, load_mailbox_spelling_rules

try:
    import requests
except ImportError:  # pragma: no cover - production dependency
    requests = None

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # pragma: no cover - production dependency
    FileSystemEventHandler = object
    Observer = None


logger = logging.getLogger("voicemail_watcher")


STATUS_DISCOVERED = "discovered"
STATUS_PROCESSING = "processing"
STATUS_RETRY = "retry"
STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"
STATUS_DEAD = "dead"

TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_SKIPPED, STATUS_DEAD}

EMAIL_RE = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$")
INBOX_EXT_RE = re.compile(r"/(?P<extension>\d{3,6})/INBOX/")
CALLERID_RE = re.compile(r'^\s*"?(?P<name>[^"<]*)"?\s*(?:<(?P<number>[^>]+)>)?\s*$')
VERIFICATION_FIELD_NAMES = ("name", "dob", "callback_number", "fax_number")


class RetryableProcessingError(Exception):
    """The voicemail should be retried later."""


class SkippedProcessing(Exception):
    """The voicemail should be skipped without retrying."""


class PermanentProcessingError(Exception):
    """The voicemail should not be retried."""


def utc_now_iso() -> str:
    return common_time.utc_now_iso()


def env_bool(name: str, default: bool) -> bool:
    return common_env.env_bool(name, default)


def env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    return common_env.env_int(name, default, minimum)


def env_float(name: str, default: float, minimum: Optional[float] = None) -> float:
    return common_env.env_float(name, default, minimum)


def parse_retry_delays(raw: str) -> tuple[float, ...]:
    delays: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        delay = float(item)
        if delay < 0:
            raise ValueError("Retry delays must be non-negative")
        delays.append(delay)
    return tuple(delays or [2, 5, 15, 60, 300])


def parse_verification_fields(raw: str | None) -> tuple[str, ...]:
    value = str(raw or "").strip()
    if not value:
        return VERIFICATION_FIELD_NAMES

    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not requested:
        return VERIFICATION_FIELD_NAMES

    invalid = sorted(requested.difference(VERIFICATION_FIELD_NAMES))
    if invalid:
        supported = ",".join(VERIFICATION_FIELD_NAMES)
        raise ValueError(
            "Unsupported VOICEMAIL_VERIFICATION_FIELDS value(s): "
            f"{','.join(invalid)}; supported values: {supported}"
        )

    return tuple(field for field in VERIFICATION_FIELD_NAMES if field in requested)


def parse_service_urls(primary_url: str, raw_urls: str = "", label: str = "service") -> tuple[str, ...]:
    source = raw_urls.strip() or primary_url.strip()
    urls: list[str] = []
    seen: set[str] = set()

    for item in re.split(r"[\s,;]+", source):
        url = item.strip().rstrip("/")
        if not url or url in seen:
            continue
        urls.append(url)
        seen.add(url)

    if not urls:
        raise ValueError(f"At least one {label} URL must be configured")

    return tuple(urls)


def parse_whisper_urls(primary_url: str, raw_urls: str = "") -> tuple[str, ...]:
    return parse_service_urls(primary_url, raw_urls, "Whisper")


def whisper_ready_url(transcribe_url: str) -> str:
    parsed = urlsplit(transcribe_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Whisper URL must be absolute, got {transcribe_url!r}")
    return urlunsplit((parsed.scheme, parsed.netloc, "/ready", "", ""))


def service_ready_url(service_url: str) -> str:
    parsed = urlsplit(service_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Service URL must be absolute, got {service_url!r}")
    return urlunsplit((parsed.scheme, parsed.netloc, "/ready", "", ""))


def _api_key_headers(api_key: str) -> dict[str, str]:
    key = str(api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}", "X-API-Key": key}


def gemma_headers(settings: "Settings") -> dict[str, str]:
    return _api_key_headers(getattr(settings, "gemma_api_key", "") or os.environ.get("GEMMA_API_KEY", ""))


def parakeet_headers(settings: "Settings") -> dict[str, str]:
    return _api_key_headers(getattr(settings, "parakeet_api_key", "") or os.environ.get("PARAKEET_API_KEY", ""))


class HostLoadTracker:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._in_flight: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def host_key(url: str) -> str:
        parsed = urlsplit(str(url or ""))
        host = parsed.hostname or parsed.netloc or str(url or "")
        return host.strip().lower()

    def mark_started(self, url: str, kind: str) -> None:
        if not self.enabled:
            return
        host = self.host_key(url)
        if not host:
            return
        key = str(kind or "").strip().lower()
        if not key:
            return
        with self._lock:
            by_kind = self._in_flight.setdefault(host, {})
            by_kind[key] = by_kind.get(key, 0) + 1

    def mark_finished(self, url: str, kind: str) -> None:
        if not self.enabled:
            return
        host = self.host_key(url)
        key = str(kind or "").strip().lower()
        if not host or not key:
            return
        with self._lock:
            by_kind = self._in_flight.get(host)
            if not by_kind:
                return
            by_kind[key] = max(0, by_kind.get(key, 0) - 1)
            if by_kind[key] == 0:
                by_kind.pop(key, None)
            if not by_kind:
                self._in_flight.pop(host, None)

    def in_flight(self, url: str, kind: str | None = None) -> int:
        if not self.enabled:
            return 0
        host = self.host_key(url)
        if not host:
            return 0
        with self._lock:
            by_kind = self._in_flight.get(host, {})
            if kind is None:
                return sum(by_kind.values())
            return by_kind.get(str(kind or "").strip().lower(), 0)


@dataclass(frozen=True)
class Settings:
    watch_dir: str
    voicemail_config: str
    state_db: str
    whisper_url: str
    whisper_urls: tuple[str, ...]
    whisper_api_key: str
    whisper_timeout_seconds: int
    whisper_ready_timeout_seconds: float
    accuracy_profile: str
    smtp_host: str
    smtp_port: int
    smtp_timeout_seconds: int
    smtp_starttls: bool
    email_enabled: bool
    from_address: str
    from_name: str
    fallback_recipient: str
    local_timezone: str
    date_timezone_label: str
    min_duration_seconds: int
    max_attempts: int
    retry_delays: tuple[float, ...]
    readiness_timeout_seconds: float
    stable_check_interval_seconds: float
    workers: int
    startup_scan: bool
    process_after_origtime: int
    log_level: str
    host_aware_routing: bool
    mailbox_spelling_rules_enabled: bool
    mailbox_spelling_rules_path: str
    gemma_field_extraction_enabled: bool
    gemma_base_url: str
    gemma_base_urls: tuple[str, ...]
    gemma_api_key: str
    gemma_model: str
    gemma_api_mode: str
    gemma_timeout_seconds: int
    gemma_ready_timeout_seconds: float
    gemma_prompt_path: str
    gemma_max_retries: int
    gemma_fail_open: bool
    gemma_log_raw_response: bool
    gemma_raw_log_max_chars: int
    parakeet_verification_enabled: bool
    parakeet_verification_mode: str
    parakeet_verification_url: str
    parakeet_verification_urls: tuple[str, ...]
    parakeet_api_key: str
    parakeet_verification_cmd: str
    parakeet_verification_timeout_seconds: int
    parakeet_ready_timeout_seconds: float
    parakeet_verification_max_retries: int
    parakeet_expected_sample_rate: int
    parakeet_fail_open: bool
    verification_clip_dir: str
    verification_clip_retention_days: int
    verification_total_timeout_seconds: int
    verification_apply_resolved_values: bool
    verification_fields: tuple[str, ...]
    verification_require_audit_for_apply: bool
    verification_ffmpeg_bin: str
    asr_runs_enabled: bool
    parakeet_full_pass_enabled: bool
    parakeet_full_pass_url: str
    parakeet_full_pass_urls: tuple[str, ...]
    parakeet_full_pass_timeout_seconds: int
    transcript_lattice_enabled: bool
    transcript_lattice_apply_enabled: bool
    llm_adjudication_enabled: bool
    transcript_lattice_llm_adjudication_enabled: bool
    transcript_lattice_llm_min_confidence: float
    transcript_lattice_llm_max_spans: int
    llm_adjudication_timeout_seconds: int
    llm_adjudication_require_source_match: bool
    router_low_word_prob_threshold: float

    @classmethod
    def from_env(cls) -> "Settings":
        primary_whisper_url = os.environ.get(
            "WHISPER_URL",
            "http://127.0.0.1:8765/transcribe/voicemail",
        ).strip()
        whisper_urls = parse_whisper_urls(
            primary_whisper_url,
            os.environ.get("WHISPER_URLS", ""),
        )
        primary_gemma_base_url = os.environ.get("GEMMA_BASE_URL", "http://127.0.0.1:11434").strip()
        gemma_base_urls = parse_service_urls(
            primary_gemma_base_url,
            os.environ.get("GEMMA_BASE_URLS", ""),
            "Gemma",
        )
        primary_parakeet_url = os.environ.get(
            "PARAKEET_VERIFICATION_URL",
            "http://127.0.0.1:8766/transcribe",
        ).strip()
        parakeet_verification_urls = parse_service_urls(
            primary_parakeet_url,
            os.environ.get("PARAKEET_VERIFICATION_URLS", ""),
            "Parakeet",
        )
        primary_parakeet_full_url = os.environ.get(
            "PARAKEET_FULL_PASS_URL",
            primary_parakeet_url,
        ).strip() or primary_parakeet_url
        parakeet_full_pass_urls = parse_service_urls(
            primary_parakeet_full_url,
            os.environ.get("PARAKEET_FULL_PASS_URLS", ""),
            "Parakeet full pass",
        )
        accuracy_profile = os.environ.get("VOICEMAIL_ACCURACY_PROFILE", "default").strip().lower() or "default"
        if accuracy_profile in {"v1.5_max", "v15_max", "hybrid_v1_5", "hybrid-v1.5"}:
            accuracy_profile = "v1_5_max"
        v1_5_max = accuracy_profile == "v1_5_max"

        def profile_bool(name: str, default: bool, profile_default: bool) -> bool:
            return env_bool(name, profile_default if v1_5_max else default)

        return cls(
            watch_dir=os.environ.get("VOICEMAIL_WATCH_DIR", "/var/spool/asterisk/voicemail"),
            voicemail_config=os.environ.get(
                "VOICEMAIL_CONFIG",
                "/etc/asterisk/vitalpbx/voicemail__50-1-main.conf",
            ),
            state_db=os.environ.get(
                "VOICEMAIL_STATE_DB",
                "/var/lib/local-voicemail-transcription/pbx/state.sqlite3",
            ),
            whisper_url=whisper_urls[0],
            whisper_urls=whisper_urls,
            whisper_api_key=os.environ.get("WHISPER_API_KEY", "").strip(),
            whisper_timeout_seconds=env_int("WHISPER_REQUEST_TIMEOUT", 120, minimum=1),
            whisper_ready_timeout_seconds=env_float("WHISPER_READY_TIMEOUT", 2.0, minimum=0.1),
            accuracy_profile=accuracy_profile,
            smtp_host=os.environ.get("SMTP_HOST", ""),
            smtp_port=env_int("SMTP_PORT", 25, minimum=1),
            smtp_timeout_seconds=env_int("SMTP_TIMEOUT", 30, minimum=1),
            smtp_starttls=env_bool("SMTP_STARTTLS", True),
            email_enabled=env_bool("VOICEMAIL_EMAIL_ENABLED", False),
            from_address=os.environ.get("VOICEMAIL_FROM_ADDRESS", ""),
            from_name=os.environ.get("VOICEMAIL_FROM_NAME", "Local Voicemail Transcription"),
            fallback_recipient=os.environ.get("VOICEMAIL_FALLBACK_RECIPIENT", "").strip(),
            local_timezone=os.environ.get("VOICEMAIL_TIMEZONE", "America/Chicago"),
            date_timezone_label=os.environ.get("VOICEMAIL_TIMEZONE_LABEL", "").strip(),
            min_duration_seconds=env_int("VOICEMAIL_MIN_DURATION_SECONDS", 3, minimum=0),
            max_attempts=env_int("VOICEMAIL_MAX_ATTEMPTS", 5, minimum=1),
            retry_delays=parse_retry_delays(os.environ.get("VOICEMAIL_RETRY_DELAYS", "2,5,15,60,300")),
            readiness_timeout_seconds=env_float("VOICEMAIL_READY_TIMEOUT", 30, minimum=1),
            stable_check_interval_seconds=env_float("VOICEMAIL_STABLE_INTERVAL", 0.5, minimum=0.1),
            workers=env_int("VOICEMAIL_WATCHER_WORKERS", 1, minimum=1),
            startup_scan=env_bool("VOICEMAIL_STARTUP_SCAN", True),
            process_after_origtime=env_int("VOICEMAIL_PROCESS_AFTER_ORIGTIME", 0, minimum=0),
            log_level=os.environ.get("VOICEMAIL_LOG_LEVEL", "INFO").upper(),
            host_aware_routing=env_bool("WATCHER_HOST_AWARE_ROUTING", True),
            mailbox_spelling_rules_enabled=env_bool("VOICEMAIL_MAILBOX_SPELLING_RULES_ENABLED", True),
            mailbox_spelling_rules_path=os.environ.get(
                "VOICEMAIL_MAILBOX_SPELLING_RULES_PATH",
                "/etc/local-voicemail-transcription/mailbox-spelling-rules.json",
            ).strip(),
            gemma_field_extraction_enabled=env_bool("GEMMA_FIELD_EXTRACTION_ENABLED", False),
            gemma_base_url=gemma_base_urls[0],
            gemma_base_urls=gemma_base_urls,
            gemma_api_key=os.environ.get("GEMMA_API_KEY", "").strip(),
            gemma_model=os.environ.get("GEMMA_MODEL", "gemma4:e4b").strip(),
            gemma_api_mode=os.environ.get("GEMMA_API_MODE", "ollama").strip().lower(),
            gemma_timeout_seconds=env_int("GEMMA_TIMEOUT_SECONDS", 180, minimum=1),
            gemma_ready_timeout_seconds=env_float("GEMMA_READY_TIMEOUT", 2.0, minimum=0.1),
            gemma_prompt_path=os.environ.get("GEMMA_PROMPT_PATH", "").strip(),
            gemma_max_retries=env_int("GEMMA_MAX_RETRIES", 1, minimum=0),
            gemma_fail_open=env_bool("GEMMA_FAIL_OPEN", True),
            gemma_log_raw_response=env_bool("GEMMA_LOG_RAW_RESPONSE", False),
            gemma_raw_log_max_chars=env_int("GEMMA_RAW_LOG_MAX_CHARS", 2000, minimum=1),
            parakeet_verification_enabled=env_bool("PARAKEET_VERIFICATION_ENABLED", False),
            parakeet_verification_mode=os.environ.get("PARAKEET_VERIFICATION_MODE", "http").strip().lower(),
            parakeet_verification_url=parakeet_verification_urls[0],
            parakeet_verification_urls=parakeet_verification_urls,
            parakeet_api_key=os.environ.get("PARAKEET_API_KEY", "").strip(),
            parakeet_verification_cmd=os.environ.get("PARAKEET_VERIFICATION_CMD", "").strip(),
            parakeet_verification_timeout_seconds=env_int("PARAKEET_VERIFICATION_TIMEOUT_SECONDS", 30, minimum=1),
            parakeet_ready_timeout_seconds=env_float("PARAKEET_READY_TIMEOUT", 2.0, minimum=0.1),
            parakeet_verification_max_retries=env_int("PARAKEET_VERIFICATION_MAX_RETRIES", 1, minimum=0),
            parakeet_expected_sample_rate=env_int("PARAKEET_EXPECTED_SAMPLE_RATE", 16000, minimum=8000),
            parakeet_fail_open=env_bool("PARAKEET_FAIL_OPEN", True),
            verification_clip_dir=os.environ.get(
                "VERIFICATION_CLIP_DIR",
                "/var/lib/voicemail-verification/clips",
            ),
            verification_clip_retention_days=env_int("VERIFICATION_CLIP_RETENTION_DAYS", 30, minimum=0),
            verification_total_timeout_seconds=env_int("VERIFICATION_TOTAL_TIMEOUT_SECONDS", 240, minimum=1),
            verification_apply_resolved_values=env_bool("VERIFICATION_APPLY_RESOLVED_VALUES", False),
            verification_fields=parse_verification_fields(
                os.environ.get("VOICEMAIL_VERIFICATION_FIELDS")
            ),
            verification_require_audit_for_apply=env_bool("VERIFICATION_REQUIRE_AUDIT_FOR_APPLY", True),
            verification_ffmpeg_bin=os.environ.get("VERIFICATION_FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg",
            asr_runs_enabled=profile_bool("ASR_RUNS_ENABLED", False, True),
            parakeet_full_pass_enabled=profile_bool("PARAKEET_FULL_PASS_ENABLED", False, True),
            parakeet_full_pass_url=parakeet_full_pass_urls[0],
            parakeet_full_pass_urls=parakeet_full_pass_urls,
            parakeet_full_pass_timeout_seconds=env_int("PARAKEET_FULL_PASS_TIMEOUT_SECONDS", 60, minimum=1),
            transcript_lattice_enabled=profile_bool("TRANSCRIPT_LATTICE_ENABLED", False, True),
            transcript_lattice_apply_enabled=env_bool("TRANSCRIPT_LATTICE_APPLY_ENABLED", False),
            llm_adjudication_enabled=profile_bool("LLM_ADJUDICATION_ENABLED", False, True),
            transcript_lattice_llm_adjudication_enabled=profile_bool(
                "TRANSCRIPT_LATTICE_LLM_ADJUDICATION_ENABLED",
                False,
                True,
            ),
            transcript_lattice_llm_min_confidence=env_float(
                "TRANSCRIPT_LATTICE_LLM_MIN_CONFIDENCE",
                0.8,
                minimum=0.0,
            ),
            transcript_lattice_llm_max_spans=env_int("TRANSCRIPT_LATTICE_LLM_MAX_SPANS", 8, minimum=0),
            llm_adjudication_timeout_seconds=env_int("LLM_ADJUDICATION_TIMEOUT_SECONDS", 30, minimum=1),
            llm_adjudication_require_source_match=env_bool("LLM_ADJUDICATION_REQUIRE_SOURCE_MATCH", True),
            router_low_word_prob_threshold=env_float("ROUTER_LOW_WORD_PROB_THRESHOLD", 0.65, minimum=0.0),
        )


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    entities: dict[str, Any]
    raw_text: str = ""
    processed_text: str = ""
    segments: list[dict[str, Any]] = None  # type: ignore[assignment]
    words: list[dict[str, Any]] = None  # type: ignore[assignment]
    confidence: dict[str, Any] = None  # type: ignore[assignment]
    asr_run_id: str = ""
    endpoint_url: str = ""
    alternative_runs: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.segments is None:
            object.__setattr__(self, "segments", [])
        if self.words is None:
            words = self.entities.get("_word_timestamps") if isinstance(self.entities, dict) else []
            object.__setattr__(self, "words", list(words) if isinstance(words, list) else [])
        if self.confidence is None:
            object.__setattr__(self, "confidence", {})
        if self.alternative_runs is None:
            object.__setattr__(self, "alternative_runs", [])


def optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_word_timestamp(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    raw_word = item.get("word", item.get("text", item.get("token", "")))
    word = str(raw_word or "").strip()
    if not word:
        return None

    start = optional_float(item.get("start", item.get("start_time", item.get("timestamp_start"))))
    end = optional_float(item.get("end", item.get("end_time", item.get("timestamp_end"))))
    timestamp = item.get("timestamp")
    if (start is None or end is None) and isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        start = optional_float(timestamp[0])
        end = optional_float(timestamp[1])

    if start is None or end is None or start < 0 or end < start:
        return None

    return {"word": word, "start": round(start, 3), "end": round(end, 3)}


def extract_word_timestamps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one normalized word-timestamp stream without duplicating segment words.

    New Whisper responses may include both top-level ``words`` and
    ``segments[].words``.  Prefer the top-level list when present, and only fall
    back to segment words for older response shapes that do not expose a usable
    top-level stream.
    """

    for key in ("words", "word_timestamps", "word_segments"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        normalized = [
            word
            for word in (normalize_word_timestamp(item) for item in value)
            if word is not None
        ]
        if normalized:
            return normalized

    candidates: list[Any] = []
    segments = payload.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            words = segment.get("words")
            if isinstance(words, list):
                candidates.extend(words)

    return [
        word
        for word in (normalize_word_timestamp(item) for item in candidates)
        if word is not None
    ]


def normalize_segment(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    start = optional_float(item.get("start"))
    end = optional_float(item.get("end"))
    if start is None or end is None or start < 0 or end < start:
        return None
    text = str(item.get("text", "") or "")
    words = []
    raw_words = item.get("words")
    if isinstance(raw_words, list):
        for word in raw_words:
            normalized = normalize_word_timestamp(word)
            if normalized is not None:
                words.append(normalized)
    return {"start": round(start, 3), "end": round(end, 3), "text": text, "words": words}


def extract_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in raw_segments:
        segment = normalize_segment(item)
        if segment is not None:
            segments.append(segment)
    return segments


class WhisperEndpointPool:
    def __init__(self, urls: tuple[str, ...], host_load_tracker: HostLoadTracker | None = None):
        if not urls:
            raise ValueError("At least one Whisper endpoint is required")
        self._urls = urls
        self._url_priority = {url: index for index, url in enumerate(urls)}
        self._in_flight = {url: 0 for url in urls}
        self._host_load_tracker = host_load_tracker
        self._lock = threading.Lock()

    def mark_started(self, url: str) -> None:
        with self._lock:
            self._in_flight[url] = self._in_flight.get(url, 0) + 1
        if self._host_load_tracker is not None:
            self._host_load_tracker.mark_started(url, "whisper")

    def mark_finished(self, url: str) -> None:
        with self._lock:
            self._in_flight[url] = max(0, self._in_flight.get(url, 0) - 1)
        if self._host_load_tracker is not None:
            self._host_load_tracker.mark_finished(url, "whisper")

    def _local_in_flight(self, url: str) -> int:
        with self._lock:
            return self._in_flight.get(url, 0)

    def _cross_host_load(self, url: str) -> int:
        if self._host_load_tracker is None:
            return 0
        return self._host_load_tracker.in_flight(url, "gemma")

    def _voicemail_queue_depth(self, url: str, settings: Settings, request_id: str) -> Optional[int]:
        if requests is None:
            return None

        ready_url = whisper_ready_url(url)
        headers = {"X-Request-ID": request_id}
        if settings.whisper_api_key:
            headers["Authorization"] = f"Bearer {settings.whisper_api_key}"

        try:
            response = requests.get(
                ready_url,
                headers=headers,
                timeout=(2, settings.whisper_ready_timeout_seconds),
            )
            if response.status_code >= 400:
                logger.warning(
                    "Whisper ready check returned status url=%s request_id=%s status=%s",
                    ready_url,
                    request_id,
                    response.status_code,
                )
                return None

            payload = response.json()
            queue_depths = payload.get("queue_depths", {})
            return max(0, int(queue_depths.get("voicemail", 0)))
        except Exception as exc:
            logger.warning(
                "Whisper ready check failed url=%s request_id=%s error=%s",
                ready_url,
                request_id,
                exc,
            )
            return None

    def least_busy_order(self, settings: Settings, request_id: str) -> tuple[str, ...]:
        scored: list[tuple[int, int, int, int, str]] = []

        for url in self._urls:
            remote_depth = self._voicemail_queue_depth(url, settings, request_id)
            local_in_flight = self._local_in_flight(url)
            cross_host_load = self._cross_host_load(url)
            priority = self._url_priority[url]
            if remote_depth is None:
                scored.append((1, local_in_flight + cross_host_load, local_in_flight, priority, url))
                continue

            scored.append((0, remote_depth + local_in_flight + cross_host_load, local_in_flight, priority, url))

        ordered = tuple(item[-1] for item in sorted(scored))
        if len(ordered) > 1:
            logger.info(
                "Selected Whisper endpoint order request_id=%s endpoints=%s",
                request_id,
                ", ".join(ordered),
            )
        return ordered


class ServiceEndpointPool:
    def __init__(
        self,
        name: str,
        urls: tuple[str, ...],
        ready_timeout_attr: str,
        host_load_tracker: HostLoadTracker | None = None,
        kind: str = "",
        cross_kinds: tuple[str, ...] = (),
    ):
        if not urls:
            raise ValueError(f"At least one {name} endpoint is required")
        self._name = name
        self._urls = urls
        self._ready_timeout_attr = ready_timeout_attr
        self._url_priority = {url: index for index, url in enumerate(urls)}
        self._in_flight = {url: 0 for url in urls}
        self._host_load_tracker = host_load_tracker
        self._kind = str(kind or "").strip().lower()
        self._cross_kinds = tuple(str(item or "").strip().lower() for item in cross_kinds if str(item or "").strip())
        self._lock = threading.Lock()

    def mark_started(self, url: str) -> None:
        with self._lock:
            self._in_flight[url] = self._in_flight.get(url, 0) + 1
        if self._host_load_tracker is not None and self._kind:
            self._host_load_tracker.mark_started(url, self._kind)

    def mark_finished(self, url: str) -> None:
        with self._lock:
            self._in_flight[url] = max(0, self._in_flight.get(url, 0) - 1)
        if self._host_load_tracker is not None and self._kind:
            self._host_load_tracker.mark_finished(url, self._kind)

    def _local_in_flight(self, url: str) -> int:
        with self._lock:
            return self._in_flight.get(url, 0)

    def _cross_host_load(self, url: str) -> int:
        if self._host_load_tracker is None:
            return 0
        return sum(self._host_load_tracker.in_flight(url, kind) for kind in self._cross_kinds)

    @staticmethod
    def _remote_load_from_payload(payload: Any) -> int:
        if not isinstance(payload, dict):
            return 0
        load = payload.get("load")
        if isinstance(load, dict):
            for key in ("in_flight", "active_requests", "queue_depth", "queued"):
                try:
                    return max(0, int(load.get(key, 0)))
                except (TypeError, ValueError):
                    continue
        for key in ("in_flight", "active_requests", "queue_depth", "queued"):
            try:
                return max(0, int(payload.get(key, 0)))
            except (TypeError, ValueError):
                continue
        return 0

    def _remote_load(self, url: str, settings: Settings, request_id: str) -> Optional[int]:
        if requests is None:
            return None

        ready_url = service_ready_url(url)
        timeout = float(getattr(settings, self._ready_timeout_attr, 2.0) or 2.0)
        try:
            response = requests.get(
                ready_url,
                headers={"X-Request-ID": request_id},
                timeout=(2, timeout),
            )
            if response.status_code >= 400:
                logger.warning(
                    "%s ready check returned status url=%s request_id=%s status=%s",
                    self._name,
                    ready_url,
                    request_id,
                    response.status_code,
                )
                return None
            return self._remote_load_from_payload(response.json())
        except Exception as exc:
            logger.warning(
                "%s ready check failed url=%s request_id=%s error=%s",
                self._name,
                ready_url,
                request_id,
                exc,
            )
            return None

    def least_busy_order(self, settings: Settings, request_id: str) -> tuple[str, ...]:
        scored: list[tuple[int, int, int, int, str]] = []
        for url in self._urls:
            remote_load = self._remote_load(url, settings, request_id)
            local_in_flight = self._local_in_flight(url)
            cross_host_load = self._cross_host_load(url)
            priority = self._url_priority[url]
            if remote_load is None:
                scored.append((1, local_in_flight + cross_host_load, local_in_flight, priority, url))
                continue
            scored.append((0, remote_load + local_in_flight + cross_host_load, local_in_flight, priority, url))

        ordered = tuple(item[-1] for item in sorted(scored))
        if len(ordered) > 1:
            logger.info(
                "Selected %s endpoint order request_id=%s endpoints=%s",
                self._name,
                request_id,
                ", ".join(ordered),
            )
        return ordered


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class VoicemailStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_parent_directory()
        self._init_schema()

    def _ensure_parent_directory(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _transaction(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voicemails (
                    file_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    txt_path TEXT NOT NULL,
                    wav_path TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    first_seen_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    last_error TEXT,
                    emailed_utc TEXT,
                    transcript_chars INTEGER
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_voicemails_status ON voicemails(status)"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_voicemails_extension_txt_path
                ON voicemails(extension, txt_path)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voicemail_transcripts (
                    file_key TEXT PRIMARY KEY,
                    extension TEXT NOT NULL,
                    mailbox TEXT,
                    folder TEXT NOT NULL DEFAULT 'INBOX',
                    msg_name TEXT NOT NULL,
                    txt_path TEXT NOT NULL,
                    wav_path TEXT NOT NULL,
                    callerid TEXT,
                    origtime INTEGER,
                    origdate TEXT,
                    duration INTEGER,
                    transcript TEXT,
                    entities_json TEXT NOT NULL DEFAULT '{}',
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    deleted_utc TEXT,
                    deleted_by TEXT,
                    deleted_comment TEXT
                )
                """
            )
            self._ensure_column(conn, "voicemail_transcripts", "deleted_by", "TEXT")
            self._ensure_column(conn, "voicemail_transcripts", "deleted_comment", "TEXT")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_voicemail_transcripts_extension
                ON voicemail_transcripts(extension, deleted_utc, origtime DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voicemail_field_verification (
                    file_key TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    final_value TEXT,
                    normalized_value TEXT,
                    status TEXT NOT NULL,
                    needs_review INTEGER NOT NULL DEFAULT 0,
                    review_reasons_json TEXT NOT NULL DEFAULT '[]',
                    attribution_json TEXT NOT NULL DEFAULT '[]',
                    whisper_json TEXT NOT NULL DEFAULT '{}',
                    gemma_json TEXT NOT NULL DEFAULT '[]',
                    parakeet_json TEXT NOT NULL DEFAULT '[]',
                    clip_json TEXT NOT NULL DEFAULT '[]',
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    PRIMARY KEY (file_key, field_name)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asr_runs (
                    run_id TEXT PRIMARY KEY,
                    file_key TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    role TEXT NOT NULL,
                    audio_view TEXT NOT NULL DEFAULT 'canonical',
                    endpoint_url TEXT,
                    model TEXT,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    transcript TEXT,
                    raw_text TEXT,
                    processed_text TEXT,
                    segments_json TEXT NOT NULL DEFAULT '[]',
                    words_json TEXT NOT NULL DEFAULT '[]',
                    confidence_json TEXT NOT NULL DEFAULT '{}',
                    numbers_json TEXT NOT NULL DEFAULT '[]',
                    audio_quality_json TEXT NOT NULL DEFAULT '{}',
                    duration_seconds REAL,
                    error TEXT,
                    created_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_asr_runs_file_key
                ON asr_runs(file_key, engine, role, created_utc DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asr_span_candidates (
                    span_id TEXT PRIMARY KEY,
                    file_key TEXT NOT NULL,
                    field_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_run_id TEXT,
                    start REAL,
                    end REAL,
                    word_start INTEGER,
                    word_end INTEGER,
                    text TEXT,
                    normalized_value TEXT,
                    confidence REAL,
                    reason_json TEXT NOT NULL DEFAULT '[]',
                    routed_to_json TEXT NOT NULL DEFAULT '[]',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_asr_span_candidates_file_key
                ON asr_span_candidates(file_key, field_type, status)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcript_corrections (
                    correction_id TEXT PRIMARY KEY,
                    file_key TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    start REAL,
                    end REAL,
                    old_text TEXT,
                    new_text TEXT,
                    decision_type TEXT NOT NULL,
                    confidence REAL,
                    score REAL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    needs_review INTEGER NOT NULL DEFAULT 0,
                    reason_code TEXT,
                    reason_codes_json TEXT NOT NULL DEFAULT '[]',
                    created_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_transcript_corrections_file_key
                ON transcript_corrections(file_key, created_utc DESC)
                """
            )

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        column_type: str,
    ) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def _table_has_file_key(self, conn: sqlite3.Connection, table: str) -> bool:
        try:
            return "file_key" in {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        except sqlite3.OperationalError:
            return False

    def _delete_auxiliary_rows(self, conn: sqlite3.Connection, file_key: str) -> None:
        for table in (
            "voicemail_field_verification",
            "asr_runs",
            "asr_span_candidates",
            "transcript_corrections",
        ):
            if self._table_has_file_key(conn, table):
                conn.execute(f"DELETE FROM {table} WHERE file_key = ?", (file_key,))

    def _rekey_auxiliary_rows(self, conn: sqlite3.Connection, source_key: str, target_key: str) -> None:
        for table in (
            "voicemail_field_verification",
            "asr_runs",
            "asr_span_candidates",
            "transcript_corrections",
        ):
            if not self._table_has_file_key(conn, table):
                continue
            try:
                conn.execute(f"UPDATE {table} SET file_key = ? WHERE file_key = ?", (target_key, source_key))
            except sqlite3.IntegrityError:
                logger.warning(
                    "Could not re-key auxiliary watcher rows table=%s source_key=%s target_key=%s",
                    table,
                    source_key,
                    target_key,
                )

    def reset_interrupted_jobs(self) -> int:
        now = utc_now_iso()
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """
                UPDATE voicemails
                SET status = ?, updated_utc = ?, last_error = ?
                WHERE status = ?
                """,
                (
                    STATUS_RETRY,
                    now,
                    "Service stopped while processing; retrying after restart",
                    STATUS_PROCESSING,
                ),
            )
            return cur.rowcount

    def _retire_stale_path_duplicates(
        self,
        conn: sqlite3.Connection,
        extension: str,
        txt_path: str,
        file_key: str,
        now: str,
    ) -> int:
        rows = conn.execute(
            """
            SELECT file_key
            FROM voicemails
            WHERE extension = ?
              AND txt_path = ?
              AND file_key <> ?
            """,
            (extension, txt_path, file_key),
        ).fetchall()
        stale_keys = [str(row["file_key"]) for row in rows]
        if not stale_keys:
            return 0

        reason = "Retired stale duplicate for same current INBOX txt_path"
        for stale_key in stale_keys:
            self._delete_auxiliary_rows(conn, stale_key)
            conn.execute(
                """
                UPDATE voicemail_transcripts
                SET deleted_utc = COALESCE(deleted_utc, ?),
                    deleted_by = COALESCE(deleted_by, 'deduped_by_current_inbox_path'),
                    updated_utc = ?
                WHERE file_key = ?
                """,
                (now, now, stale_key),
            )
            conn.execute(
                """
                UPDATE voicemails
                SET status = ?,
                    attempts = CASE WHEN attempts < 1 THEN 1 ELSE attempts END,
                    updated_utc = ?,
                    last_error = ?
                WHERE file_key = ?
                """,
                (STATUS_DEAD, now, reason, stale_key),
            )
        return len(stale_keys)

    def discover(self, file_key: str, extension: str, txt_path: str, wav_path: str) -> str:
        txt_path = os.path.abspath(txt_path)
        wav_path = os.path.abspath(wav_path)
        now = utc_now_iso()
        with self._lock, self._transaction() as conn:
            stale_count = self._retire_stale_path_duplicates(conn, extension, txt_path, file_key, now)
            if stale_count:
                logger.warning(
                    "Retired stale duplicate voicemail row(s) extension=%s txt_path=%s count=%s",
                    extension,
                    txt_path,
                    stale_count,
                )
            conn.execute(
                """
                INSERT INTO voicemails (
                    file_key, status, extension, txt_path, wav_path,
                    attempts, first_seen_utc, updated_utc
                )
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(file_key) DO UPDATE SET
                    extension = excluded.extension,
                    txt_path = excluded.txt_path,
                    wav_path = excluded.wav_path,
                    updated_utc = CASE
                        WHEN voicemails.status IN ('completed', 'skipped', 'dead')
                        THEN voicemails.updated_utc
                        ELSE excluded.updated_utc
                    END
                """,
                (
                    file_key,
                    STATUS_DISCOVERED,
                    extension,
                    txt_path,
                    wav_path,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT status FROM voicemails WHERE file_key = ?",
                (file_key,),
            ).fetchone()
            return str(row["status"])

    def mark_dead_by_path(self, txt_path: str, reason: str) -> int:
        now = utc_now_iso()
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """
                UPDATE voicemails
                SET status = ?, attempts = CASE WHEN attempts < 1 THEN 1 ELSE attempts END,
                    updated_utc = ?, last_error = ?
                WHERE txt_path = ? AND status NOT IN (?, ?, ?)
                """,
                (
                    STATUS_DEAD,
                    now,
                    reason[:1000],
                    os.path.abspath(txt_path),
                    STATUS_COMPLETED,
                    STATUS_SKIPPED,
                    STATUS_DEAD,
                ),
            )
            return cur.rowcount

    def claim(self, file_key: str) -> bool:
        now = utc_now_iso()
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                """
                UPDATE voicemails
                SET status = ?, attempts = attempts + 1, updated_utc = ?, last_error = NULL
                WHERE file_key = ? AND status IN (?, ?)
                """,
                (STATUS_PROCESSING, now, file_key, STATUS_DISCOVERED, STATUS_RETRY),
            )
            return cur.rowcount == 1

    def mark_completed(self, file_key: str, transcript_chars: int) -> None:
        now = utc_now_iso()
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE voicemails
                SET status = ?, updated_utc = ?, emailed_utc = COALESCE(emailed_utc, ?), transcript_chars = ?,
                    last_error = NULL
                WHERE file_key = ?
                """,
                (STATUS_COMPLETED, now, now, transcript_chars, file_key),
            )

    def mark_emailed(self, file_key: str) -> None:
        now = utc_now_iso()
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE voicemails
                SET emailed_utc = COALESCE(emailed_utc, ?), updated_utc = ?
                WHERE file_key = ?
                """,
                (now, now, file_key),
            )

    def already_emailed(self, file_key: str) -> bool:
        with self._lock, self._transaction() as conn:
            row = conn.execute(
                "SELECT emailed_utc FROM voicemails WHERE file_key = ?",
                (file_key,),
            ).fetchone()
        return bool(row and row["emailed_utc"])

    def mark_skipped(self, file_key: str, reason: str) -> None:
        now = utc_now_iso()
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE voicemails
                SET status = ?, updated_utc = ?, last_error = ?
                WHERE file_key = ?
                """,
                (STATUS_SKIPPED, now, reason[:1000], file_key),
            )

    def mark_skipped_by_path(self, txt_path: str, reason: str) -> None:
        now = utc_now_iso()
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE voicemails
                SET status = ?, updated_utc = ?, last_error = ?
                WHERE txt_path = ? AND status NOT IN (?, ?, ?)
                """,
                (
                    STATUS_SKIPPED,
                    now,
                    reason[:1000],
                    os.path.abspath(txt_path),
                    STATUS_COMPLETED,
                    STATUS_SKIPPED,
                    STATUS_DEAD,
                ),
            )

    def migrate_legacy_key(self, legacy_key: str, file_key: str) -> None:
        if legacy_key == file_key:
            return

        with self._lock, self._transaction() as conn:
            legacy_row = conn.execute(
                "SELECT file_key FROM voicemails WHERE file_key = ?",
                (legacy_key,),
            ).fetchone()
            if not legacy_row:
                return

            existing_row = conn.execute(
                "SELECT file_key FROM voicemails WHERE file_key = ?",
                (file_key,),
            ).fetchone()
            if existing_row:
                return

            conn.execute(
                "UPDATE voicemails SET file_key = ? WHERE file_key = ?",
                (file_key, legacy_key),
            )

            legacy_transcript = conn.execute(
                "SELECT file_key FROM voicemail_transcripts WHERE file_key = ?",
                (legacy_key,),
            ).fetchone()
            existing_transcript = conn.execute(
                "SELECT file_key FROM voicemail_transcripts WHERE file_key = ?",
                (file_key,),
            ).fetchone()
            if legacy_transcript and not existing_transcript:
                conn.execute(
                    "UPDATE voicemail_transcripts SET file_key = ? WHERE file_key = ?",
                    (file_key, legacy_key),
                )
            self._rekey_auxiliary_rows(conn, legacy_key, file_key)

        logger.info("Migrated legacy voicemail key legacy_key=%s file_key=%s", legacy_key, file_key)

    def mark_retry_or_dead(self, file_key: str, error: str, max_attempts: int) -> str:
        now = utc_now_iso()
        with self._lock, self._transaction() as conn:
            row = conn.execute(
                "SELECT attempts FROM voicemails WHERE file_key = ?",
                (file_key,),
            ).fetchone()
            attempts = int(row["attempts"]) if row else max_attempts
            status = STATUS_DEAD if attempts >= max_attempts else STATUS_RETRY
            conn.execute(
                """
                UPDATE voicemails
                SET status = ?, updated_utc = ?, last_error = ?
                WHERE file_key = ?
                """,
                (status, now, error[:1000], file_key),
            )
            return status

    def upsert_transcript(
        self,
        file_key: str,
        extension: str,
        txt_path: str,
        wav_path: str,
        info: dict[str, str],
        transcript: str,
        entities: dict[str, Any],
    ) -> None:
        now = utc_now_iso()
        mailbox = info.get("origmailbox", extension).strip() or extension
        msg_name = os.path.splitext(os.path.basename(txt_path))[0]

        def optional_int(value: Any) -> Optional[int]:
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                return None

        entities_json = json.dumps(entities or {}, ensure_ascii=True, sort_keys=True)

        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO voicemail_transcripts (
                    file_key, extension, mailbox, folder, msg_name, txt_path, wav_path,
                    callerid, origtime, origdate, duration, transcript, entities_json,
                    created_utc, updated_utc, deleted_utc, deleted_comment
                )
                VALUES (?, ?, ?, 'INBOX', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                ON CONFLICT(file_key) DO UPDATE SET
                    extension = excluded.extension,
                    mailbox = excluded.mailbox,
                    folder = excluded.folder,
                    msg_name = excluded.msg_name,
                    txt_path = excluded.txt_path,
                    wav_path = excluded.wav_path,
                    callerid = excluded.callerid,
                    origtime = excluded.origtime,
                    origdate = excluded.origdate,
                    duration = excluded.duration,
                    transcript = excluded.transcript,
                    entities_json = excluded.entities_json,
                    updated_utc = excluded.updated_utc,
                    deleted_utc = NULL,
                    deleted_by = NULL
                """,
                (
                    file_key,
                    extension,
                    mailbox,
                    msg_name,
                    txt_path,
                    wav_path,
                    info.get("callerid", "").strip(),
                    optional_int(info.get("origtime")),
                    info.get("origdate", "").strip(),
                    optional_int(info.get("duration")),
                    transcript,
                    entities_json,
                    now,
                    now,
                ),
            )

    def upsert_field_verifications(self, file_key: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        now = utc_now_iso()

        def dump(value: Any) -> str:
            return json.dumps(value, ensure_ascii=True, sort_keys=True)

        with self._lock, self._transaction() as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO voicemail_field_verification (
                        file_key, field_name, final_value, normalized_value, status,
                        needs_review, review_reasons_json, attribution_json,
                        whisper_json, gemma_json, parakeet_json, clip_json,
                        warnings_json, created_utc, updated_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_key, field_name) DO UPDATE SET
                        final_value = excluded.final_value,
                        normalized_value = excluded.normalized_value,
                        status = excluded.status,
                        needs_review = excluded.needs_review,
                        review_reasons_json = excluded.review_reasons_json,
                        attribution_json = excluded.attribution_json,
                        whisper_json = excluded.whisper_json,
                        gemma_json = excluded.gemma_json,
                        parakeet_json = excluded.parakeet_json,
                        clip_json = excluded.clip_json,
                        warnings_json = excluded.warnings_json,
                        updated_utc = excluded.updated_utc
                    """,
                    (
                        file_key,
                        row.get("field_name"),
                        row.get("final_value"),
                        row.get("normalized_value"),
                        row.get("status") or "not_included",
                        1 if row.get("needs_review") else 0,
                        dump(row.get("review_reasons") or []),
                        dump(row.get("attribution_json") or []),
                        dump(row.get("whisper_json") or {}),
                        dump(row.get("gemma_json") or []),
                        dump(row.get("parakeet_json") or []),
                        dump(row.get("clip_json") or []),
                        dump(row.get("warnings_json") or []),
                        now,
                        now,
                    ),
                )

    def insert_asr_run(self, file_key: str, run: dict[str, Any]) -> None:
        now = utc_now_iso()

        def dump(value: Any, default: Any) -> str:
            if value is None:
                value = default
            return json.dumps(value, ensure_ascii=True, sort_keys=True)

        run_id = str(run.get("run_id") or f"asr_{uuid.uuid4().hex}")
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO asr_runs (
                    run_id, file_key, engine, role, audio_view, endpoint_url, model,
                    params_json, transcript, raw_text, processed_text, segments_json,
                    words_json, confidence_json, numbers_json, audio_quality_json,
                    duration_seconds, error, created_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    transcript = excluded.transcript,
                    raw_text = excluded.raw_text,
                    processed_text = excluded.processed_text,
                    segments_json = excluded.segments_json,
                    words_json = excluded.words_json,
                    confidence_json = excluded.confidence_json,
                    numbers_json = excluded.numbers_json,
                    audio_quality_json = excluded.audio_quality_json,
                    duration_seconds = excluded.duration_seconds,
                    error = excluded.error
                """,
                (
                    run_id,
                    file_key,
                    str(run.get("engine") or "unknown"),
                    str(run.get("role") or "primary"),
                    str(run.get("audio_view") or "canonical"),
                    run.get("endpoint_url"),
                    run.get("model"),
                    dump(run.get("params"), {}),
                    run.get("transcript") or run.get("text"),
                    run.get("raw_text"),
                    run.get("processed_text"),
                    dump(run.get("segments"), []),
                    dump(run.get("words"), []),
                    dump(run.get("confidence"), {}),
                    dump(run.get("numbers"), []),
                    dump(run.get("audio_quality"), {}),
                    run.get("duration_seconds"),
                    run.get("error"),
                    str(run.get("created_utc") or now),
                ),
            )

    def insert_lattice_spans(self, file_key: str, spans: list[Any]) -> None:
        if not spans:
            return

        now = utc_now_iso()

        def dump(value: Any, default: Any) -> str:
            if value is None:
                value = default
            return json.dumps(value, ensure_ascii=True, sort_keys=True)

        with self._lock, self._transaction() as conn:
            for span in spans:
                span_dict = span.as_dict() if hasattr(span, "as_dict") else span
                if not isinstance(span_dict, dict):
                    continue
                alternatives = [item for item in span_dict.get("alternatives") or [] if isinstance(item, dict)]
                for index, alternative in enumerate(alternatives):
                    span_id = f"{span_dict.get('span_id')}:alt:{index}"
                    conn.execute(
                        """
                        INSERT INTO asr_span_candidates (
                            span_id, file_key, field_type, source, source_run_id, start, end,
                            word_start, word_end, text, normalized_value, confidence, reason_json,
                            routed_to_json, result_json, status, created_utc, updated_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(span_id) DO UPDATE SET
                            text = excluded.text,
                            confidence = excluded.confidence,
                            reason_json = excluded.reason_json,
                            result_json = excluded.result_json,
                            status = excluded.status,
                            updated_utc = excluded.updated_utc
                        """,
                        (
                            span_id,
                            file_key,
                            str(span_dict.get("field_hint") or "general"),
                            str(alternative.get("source") or alternative.get("engine") or "unknown"),
                            alternative.get("run_id"),
                            alternative.get("start", span_dict.get("start")),
                            alternative.get("end", span_dict.get("end")),
                            span_dict.get("primary_word_start"),
                            span_dict.get("primary_word_end"),
                            alternative.get("text"),
                            alternative.get("normalized_value"),
                            alternative.get("confidence"),
                            dump(span_dict.get("reasons"), []),
                            dump([], []),
                            dump({"span": span_dict.get("span_id"), "alternative_index": index}, {}),
                            "audited",
                            now,
                            now,
                        ),
                    )

    def insert_transcript_corrections(self, file_key: str, corrections: list[dict[str, Any]]) -> None:
        if not corrections:
            return

        now = utc_now_iso()

        def dump(value: Any) -> str:
            return json.dumps(value or [], ensure_ascii=True, sort_keys=True)

        with self._lock, self._transaction() as conn:
            for row in corrections:
                if not isinstance(row, dict):
                    continue
                correction_id = str(row.get("correction_id") or f"corr_{uuid.uuid4().hex}")
                reason_codes = row.get("reason_codes") or ([row.get("reason_code")] if row.get("reason_code") else [])
                conn.execute(
                    """
                    INSERT INTO transcript_corrections (
                        correction_id, file_key, span_id, start, end, old_text, new_text,
                        decision_type, confidence, score, sources_json, needs_review,
                        reason_code, reason_codes_json, created_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(correction_id) DO UPDATE SET
                        old_text = excluded.old_text,
                        new_text = excluded.new_text,
                        decision_type = excluded.decision_type,
                        confidence = excluded.confidence,
                        score = excluded.score,
                        sources_json = excluded.sources_json,
                        needs_review = excluded.needs_review,
                        reason_code = excluded.reason_code,
                        reason_codes_json = excluded.reason_codes_json
                    """,
                    (
                        correction_id,
                        file_key,
                        str(row.get("span_id") or ""),
                        row.get("start"),
                        row.get("end"),
                        row.get("old_text"),
                        row.get("new_text"),
                        str(row.get("decision_type") or "unknown"),
                        row.get("confidence"),
                        row.get("score"),
                        dump(row.get("sources")),
                        1 if row.get("needs_review") else 0,
                        row.get("reason_code"),
                        dump(reason_codes),
                        now,
                    ),
                )


def normalize_path(path: str) -> str:
    return common_spool.normalize_path(path)


def extract_extension(path: str) -> Optional[str]:
    return common_spool.extract_extension(path)


def is_voicemail_txt(path: str) -> bool:
    return common_spool.is_voicemail_txt(path)


def moved_out_of_inbox_paths(txt_path: str) -> list[str]:
    inbox_dir = os.path.dirname(os.path.abspath(txt_path))
    if os.path.basename(inbox_dir).upper() != "INBOX":
        return []

    extension_dir = os.path.dirname(inbox_dir)
    if not os.path.isdir(extension_dir):
        return []

    msg_stem = os.path.splitext(os.path.basename(txt_path))[0]
    matches: list[str] = []
    for candidate in glob.glob(os.path.join(extension_dir, "*", f"{msg_stem}.*")):
        candidate_dir = os.path.dirname(os.path.abspath(candidate))
        if normalize_path(candidate_dir) == normalize_path(inbox_dir):
            continue
        if os.path.isfile(candidate):
            matches.append(os.path.abspath(candidate))

    return sorted(matches)


def moved_out_of_inbox_reason(txt_path: str) -> Optional[str]:
    moved_paths = moved_out_of_inbox_paths(txt_path)
    if not moved_paths:
        return None

    moved_dirs = sorted({os.path.dirname(path) for path in moved_paths})
    preview = ", ".join(moved_dirs[:3])
    if len(moved_dirs) > 3:
        preview = f"{preview}, ..."
    return f"INBOX voicemail files were moved before processing; found matching files in: {preview}"


def parse_txt(txt_path: str) -> dict[str, str]:
    """Parse an Asterisk voicemail metadata file into a dict."""
    return common_spool.parse_txt(txt_path)


def format_duration(seconds: str) -> str:
    return common_formatting.format_duration(seconds, empty_on_invalid=False)


def central_fallback_timezone(dt_utc: datetime) -> timezone:
    def nth_sunday(year: int, month: int, n: int) -> datetime:
        first = datetime(year, month, 1, tzinfo=timezone.utc)
        days_until_sunday = (6 - first.weekday()) % 7
        return first + timedelta(days=days_until_sunday + (n - 1) * 7)

    year = dt_utc.year
    dst_start_utc = nth_sunday(year, 3, 2).replace(hour=8)
    dst_end_utc = nth_sunday(year, 11, 1).replace(hour=7)

    if dst_start_utc <= dt_utc < dst_end_utc:
        return timezone(timedelta(hours=-5), "CDT")
    return timezone(timedelta(hours=-6), "CST")


def get_local_timezone(timezone_name: str, dt_utc: datetime) -> timezone:
    try:
        return ZoneInfo(timezone_name)
    except Exception:
        if timezone_name == "America/Chicago":
            return central_fallback_timezone(dt_utc)
        raise


def format_date(origdate: str, timezone_name: str, timezone_label: str = "") -> str:
    if not origdate:
        return "Unknown"

    try:
        dt = datetime.strptime(origdate, "%a %b %d %I:%M:%S %p UTC %Y")
        dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(get_local_timezone(timezone_name, dt))
        hour = local.strftime("%I").lstrip("0") or "0"
        label = timezone_label or local.strftime("%Z")
        return f"{local:%A, %B %d %Y}, {hour}:{local:%M} {local:%p} {label}".rstrip()
    except Exception as exc:
        logger.warning("Could not convert voicemail date %r: %s", origdate, exc)
        return origdate


def format_phone_number(value: str) -> Optional[str]:
    return common_formatting.format_phone_number(value)


def parse_callerid(callerid: str) -> tuple[str, str]:
    match = CALLERID_RE.match(callerid or "")
    if not match:
        return (callerid or "Unknown").strip() or "Unknown", "Not Included"

    raw_name = (match.group("name") or "").strip().strip('"')
    raw_number = (match.group("number") or "").strip()
    name = raw_name or "Unknown"
    number = format_phone_number(raw_number) or "Not Included"
    return name, number


def included_or_default(value: Any) -> str:
    if value is None:
        return "Not Included"
    value = str(value).strip()
    return value if value else "Not Included"


def callback_match_status(caller_number: str, callback_number: Any) -> str:
    caller = format_phone_number(caller_number)
    callback = format_phone_number(str(callback_number or ""))

    if not callback:
        return "No Callback Number Found"
    if not caller:
        return "Caller Number Not Included"
    if caller == callback:
        return "Yes"
    return "No - Review"


def format_caller_id(name: str) -> str:
    if not name or name == "Unknown":
        return "Unknown"
    return f'"{name}"'


def sanitize_header(value: str) -> str:
    return re.sub(r"[\r\n]+", " ", value).strip()


def validate_recipients(recipients: list[str], fallback: str) -> list[str]:
    valid: list[str] = []
    seen: set[str] = set()
    for addr in recipients:
        normalized = addr.strip()
        key = normalized.lower()
        if not normalized or key in seen or not EMAIL_RE.match(normalized):
            continue
        seen.add(key)
        valid.append(normalized)

    if valid:
        return valid
    if fallback and EMAIL_RE.match(fallback):
        logger.warning("No valid mailbox recipients found; using fallback recipient")
        return [fallback]
    raise PermanentProcessingError("No valid email recipient and fallback is invalid")


def get_email(extension: str, settings: Settings) -> list[str]:
    recipients: list[str] = []

    try:
        with open(settings.voicemail_config, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = re.match(rf"^\s*{re.escape(extension)}\s*=>\s*(?P<body>.*)$", line)
                if not match:
                    continue

                parts = [part.strip() for part in match.group("body").split(",")]
                if len(parts) >= 3:
                    candidate_fields = [parts[2]]
                    candidate_fields.extend(
                        part for part in parts[3:] if "@" in part and "=" not in part
                    )
                    raw = ";".join(candidate_fields)
                    recipients = [
                        addr.strip()
                        for addr in re.split(r"[|;,]", raw)
                        if addr.strip()
                    ]
                break
    except FileNotFoundError:
        logger.error("Voicemail config not found: %s", settings.voicemail_config)
    except OSError as exc:
        logger.error("Could not read voicemail config %s: %s", settings.voicemail_config, exc)

    return validate_recipients(recipients, settings.fallback_recipient)


def metadata_file_hash(txt_path: str) -> str:
    return common_keys.metadata_file_hash(txt_path)


def build_legacy_file_key(extension: str, info: dict[str, str], txt_path: str) -> Optional[str]:
    return common_keys.build_legacy_file_key(extension, info, txt_path)


def build_file_key(extension: str, info: dict[str, str], txt_path: str) -> Optional[str]:
    return common_keys.build_file_key(extension, info, txt_path)


def is_before_process_cutoff(info: dict[str, str], settings: Settings) -> bool:
    if settings.process_after_origtime <= 0:
        return False

    try:
        origtime = int(info.get("origtime", "").strip())
    except ValueError:
        return False

    return origtime < settings.process_after_origtime


def matching_wav_path(txt_path: str) -> str:
    return common_spool.matching_wav_path(txt_path)


def is_file_stable(path: str, interval_seconds: float) -> bool:
    if not os.path.exists(path):
        return False
    size_one = os.path.getsize(path)
    if size_one <= 0:
        return False
    time.sleep(interval_seconds)
    if not os.path.exists(path):
        return False
    size_two = os.path.getsize(path)
    return size_one == size_two and size_two > 0


def wait_for_ready_files(txt_path: str, settings: Settings) -> tuple[dict[str, str], str]:
    wav_path = matching_wav_path(txt_path)
    deadline = time.monotonic() + settings.readiness_timeout_seconds
    last_error = "Voicemail files were not ready before timeout"

    while time.monotonic() < deadline:
        try:
            txt_exists = os.path.exists(txt_path)
            wav_exists = os.path.exists(wav_path)
            if not txt_exists or not wav_exists:
                moved_reason = moved_out_of_inbox_reason(txt_path)
                if moved_reason:
                    raise SkippedProcessing(moved_reason)

            if not txt_exists:
                last_error = "TXT metadata file is missing"
            elif not wav_exists:
                last_error = "WAV audio file is missing"
            else:
                info = parse_txt(txt_path)
                if not info.get("origtime"):
                    last_error = "TXT metadata does not contain origtime yet"
                elif not info.get("duration"):
                    last_error = "TXT metadata does not contain duration yet"
                elif not is_file_stable(wav_path, settings.stable_check_interval_seconds):
                    last_error = "WAV audio file is still changing"
                else:
                    return info, wav_path
        except OSError as exc:
            last_error = f"Could not read voicemail files yet: {exc}"

        time.sleep(min(1.0, settings.stable_check_interval_seconds))

    raise RetryableProcessingError(last_error)


def transcribe(
    path: str,
    request_id: str,
    settings: Settings,
    endpoint_urls: Optional[tuple[str, ...]] = None,
    endpoint_pool: Optional[WhisperEndpointPool] = None,
) -> TranscriptionResult:
    if requests is None:
        raise RuntimeError("The requests package is required to call the Whisper API")

    headers = {"X-Request-ID": request_id}
    if settings.whisper_api_key:
        headers["Authorization"] = f"Bearer {settings.whisper_api_key}"

    urls = endpoint_urls or settings.whisper_urls or (settings.whisper_url,)
    retryable_errors: list[str] = []

    for url in urls:
        try:
            if endpoint_pool is not None:
                endpoint_pool.mark_started(url)
            try:
                with open(path, "rb") as f:
                    response = requests.post(
                        url,
                        files={"file": (os.path.basename(path), f, "audio/wav")},
                        headers=headers,
                        timeout=(10, settings.whisper_timeout_seconds),
                    )
            finally:
                if endpoint_pool is not None:
                    endpoint_pool.mark_finished(url)
        except requests.RequestException as exc:
            retryable_errors.append(f"{url}: request failed: {exc}")
            logger.warning("Whisper endpoint request failed url=%s request_id=%s error=%s", url, request_id, exc)
            continue

        if 500 <= response.status_code < 600 or response.status_code in {408, 429}:
            retryable_errors.append(f"{url}: retryable status {response.status_code}")
            logger.warning(
                "Whisper endpoint returned retryable status url=%s request_id=%s status=%s",
                url,
                request_id,
                response.status_code,
            )
            continue

        if response.status_code >= 400:
            raise PermanentProcessingError(
                f"Whisper API rejected request with status {response.status_code} from {url}"
            )

        try:
            payload = response.json()
        except ValueError:
            retryable_errors.append(f"{url}: invalid JSON")
            logger.warning("Whisper endpoint returned invalid JSON url=%s request_id=%s", url, request_id)
            continue

        transcript = str(payload.get("text", "")).strip()
        if not transcript:
            retryable_errors.append(f"{url}: empty transcript")
            logger.warning("Whisper endpoint returned empty transcript url=%s request_id=%s", url, request_id)
            continue

        entities = payload.get("entities")
        if not isinstance(entities, dict):
            entities = {}
        word_timestamps = extract_word_timestamps(payload)
        segments = extract_segments(payload)
        if word_timestamps:
            entities = dict(entities)
            entities["_word_timestamps"] = word_timestamps
        raw_text = str(payload.get("raw_text") or transcript).strip()
        processed_text = str(payload.get("processed_text") or transcript).strip()
        confidence = payload.get("confidence") if isinstance(payload.get("confidence"), dict) else {}
        raw_alternative_runs = payload.get("alternative_runs")
        alternative_runs = (
            [dict(item) for item in raw_alternative_runs if isinstance(item, dict)]
            if isinstance(raw_alternative_runs, list)
            else []
        )
        for run in alternative_runs:
            run.setdefault("run_id", f"asr_whisper_alt_{uuid.uuid4().hex}")
            run.setdefault("engine", "whisper")
            run.setdefault("role", "secondary_full_pass")
            run.setdefault("audio_view", "canonical")
            run.setdefault("endpoint_url", url)
        return TranscriptionResult(
            text=transcript,
            entities=entities,
            raw_text=raw_text,
            processed_text=processed_text,
            segments=segments,
            words=word_timestamps,
            confidence=confidence,
            asr_run_id=f"asr_whisper_{uuid.uuid4().hex}",
            endpoint_url=url,
            alternative_runs=alternative_runs,
        )

    detail = "; ".join(retryable_errors) if retryable_errors else "no Whisper endpoints configured"
    raise RetryableProcessingError(f"All Whisper endpoints failed: {detail}")


def whisper_run_from_transcription(transcription: TranscriptionResult) -> dict[str, Any]:
    numbers: list[str] = []
    entities = transcription.entities if isinstance(transcription.entities, dict) else {}
    for key in ("callback_number", "fax_number"):
        value = entities.get(key)
        if value:
            numbers.append(str(value))
    raw_verifications = entities.get("phone_number_verifications")
    if isinstance(raw_verifications, list):
        for item in raw_verifications:
            if isinstance(item, dict) and item.get("number"):
                numbers.append(str(item.get("number")))

    return {
        "run_id": transcription.asr_run_id or f"asr_whisper_{uuid.uuid4().hex}",
        "engine": "whisper",
        "role": "primary",
        "audio_view": "canonical",
        "endpoint_url": transcription.endpoint_url,
        "transcript": transcription.text,
        "raw_text": transcription.raw_text,
        "processed_text": transcription.processed_text,
        "segments": transcription.segments,
        "words": transcription.words,
        "confidence": transcription.confidence,
        "numbers": list(dict.fromkeys(numbers)),
        "created_utc": utc_now_iso(),
    }


def call_parakeet_full_transcription(
    wav_path: str,
    file_key: str,
    settings: Settings,
    endpoint_pool: Optional[ServiceEndpointPool],
) -> dict[str, Any]:
    run_id = f"asr_parakeet_full_{uuid.uuid4().hex}"
    urls = tuple(url for url in (settings.parakeet_full_pass_urls or settings.parakeet_verification_urls) if url)
    if requests is None:
        return {
            "run_id": run_id,
            "engine": "parakeet",
            "role": "full_pass",
            "audio_view": "canonical",
            "error": "requests_unavailable",
            "created_utc": utc_now_iso(),
        }
    if not urls:
        return {
            "run_id": run_id,
            "engine": "parakeet",
            "role": "full_pass",
            "audio_view": "canonical",
            "error": "no_endpoint",
            "created_utc": utc_now_iso(),
        }

    endpoint_order = urls
    pool_urls = tuple(getattr(endpoint_pool, "_urls", ()) or ()) if endpoint_pool is not None else ()
    if endpoint_pool is not None and set(pool_urls) == set(urls):
        endpoint_order = endpoint_pool.least_busy_order(settings, file_key)

    last_error = ""
    for url in endpoint_order:
        timeout = settings.parakeet_full_pass_timeout_seconds
        try:
            if endpoint_pool is not None and url in pool_urls:
                endpoint_pool.mark_started(url)
            try:
                with open(wav_path, "rb") as audio:
                    response = requests.post(
                        url,
                        files={"file": (os.path.basename(wav_path), audio, "audio/wav")},
                        headers=parakeet_headers(settings) or None,
                        timeout=(min(5, timeout), timeout),
                    )
            finally:
                if endpoint_pool is not None and url in pool_urls:
                    endpoint_pool.mark_finished(url)
            if response.status_code >= 400:
                last_error = f"http_{response.status_code}"
                continue
            try:
                payload = response.json()
            except ValueError:
                payload = {"text": response.text}
            if not isinstance(payload, dict):
                payload = {"raw": payload, "error": "invalid_payload"}
            text = str(payload.get("text") or payload.get("transcript") or "").strip()
            return {
                "run_id": run_id,
                "engine": "parakeet",
                "role": "full_pass",
                "audio_view": "canonical",
                "endpoint_url": url,
                "model": payload.get("model"),
                "transcript": text,
                "raw_text": text,
                "processed_text": text,
                "segments": payload.get("segments") if isinstance(payload.get("segments"), list) else [],
                "words": (
                    payload.get("words")
                    if isinstance(payload.get("words"), list)
                    else payload.get("timestamps")
                    if isinstance(payload.get("timestamps"), list)
                    else []
                ),
                "confidence": payload.get("confidence") if isinstance(payload.get("confidence"), dict) else {},
                "numbers": (
                    payload.get("normalized_numbers")
                    if isinstance(payload.get("normalized_numbers"), list)
                    else []
                ),
                "duration_seconds": payload.get("duration_seconds"),
                "created_utc": utc_now_iso(),
                "raw_output": payload,
            }
        except Exception as exc:
            last_error = str(exc)[:200]
            logger.warning("Parakeet full-pass failed key=%s endpoint=%s error=%s", file_key, url, last_error)

    return {
        "run_id": run_id,
        "engine": "parakeet",
        "role": "full_pass",
        "audio_view": "canonical",
        "error": last_error or "unavailable",
        "created_utc": utc_now_iso(),
    }


def call_gemma_transcript_lattice_adjudication(
    span_payload: dict[str, Any],
    settings: Settings,
    endpoint_pool: Optional[ServiceEndpointPool] = None,
    request_id: str = "",
) -> Optional[dict[str, Any]]:
    if requests is None:
        logger.warning("Transcript lattice Gemma adjudication skipped: requests package is unavailable")
        return None

    prompt_text = compact_transcript_adjudication_prompt(span_payload)
    endpoint_order = (
        endpoint_pool.least_busy_order(settings, request_id or str(span_payload.get("span_id") or "lattice"))
        if endpoint_pool is not None
        else (settings.gemma_base_urls or (settings.gemma_base_url,))
    )
    min_confidence = settings.transcript_lattice_llm_min_confidence
    timeout = settings.llm_adjudication_timeout_seconds

    for base_url in endpoint_order:
        try:
            url, request_payload = build_gemma_http_request(settings, prompt_text, base_url)
            if endpoint_pool is not None:
                endpoint_pool.mark_started(base_url)
            try:
                response = requests.post(
                    url,
                    json=request_payload,
                    headers=gemma_headers(settings) or None,
                    timeout=(min(5, timeout), timeout),
                )
            finally:
                if endpoint_pool is not None:
                    endpoint_pool.mark_finished(base_url)
            if response.status_code >= 400:
                excerpt = response_error_excerpt(response)
                suffix = f": {excerpt}" if excerpt else ""
                raise RuntimeError(f"Gemma returned HTTP {response.status_code}{suffix}")
            payload = normalize_gemma_http_payload(response.json(), settings)
            text = extract_gemma_response_text(payload)
            if isinstance(text, str):
                decision = json.loads(text.strip())
            else:
                decision = payload
            validated = validate_transcript_adjudication_decision(
                span_payload,
                decision,
                require_source_match=settings.llm_adjudication_require_source_match,
                min_confidence=min_confidence,
            )
            if validated is None:
                logger.warning(
                    "Transcript lattice Gemma adjudication returned invalid decision span_id=%s endpoint=%s",
                    span_payload.get("span_id"),
                    base_url,
                )
                return None
            logger.info(
                "Transcript lattice Gemma adjudication decision span_id=%s decision=%s confidence=%s",
                span_payload.get("span_id"),
                validated.get("decision_type"),
                validated.get("confidence"),
            )
            return validated
        except Exception as exc:
            logger.warning(
                "Transcript lattice Gemma adjudication failed span_id=%s endpoint=%s error=%s",
                span_payload.get("span_id"),
                base_url,
                exc,
            )
    return None


def transcript_correction_dicts(corrections: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for correction in corrections or []:
        if hasattr(correction, "as_dict"):
            row = correction.as_dict()
        elif isinstance(correction, dict):
            row = dict(correction)
        else:
            continue
        row.setdefault("correction_id", f"corr_{uuid.uuid4().hex}")
        rows.append(row)
    return rows


def run_transcript_lattice_audit(
    file_key: str,
    wav_path: str,
    transcription: TranscriptionResult,
    transcript_for_output: str,
    settings: Settings,
    store: VoicemailStore,
    gemma_endpoint_pool: ServiceEndpointPool,
    parakeet_endpoint_pool: ServiceEndpointPool,
) -> tuple[str, list[dict[str, Any]]]:
    if not (settings.asr_runs_enabled or settings.parakeet_full_pass_enabled or settings.transcript_lattice_enabled):
        return transcript_for_output, []

    primary_run = whisper_run_from_transcription(transcription)
    if settings.asr_runs_enabled:
        store.insert_asr_run(file_key, primary_run)
        for run in transcription.alternative_runs or []:
            store.insert_asr_run(file_key, run)

    peer_runs = [dict(run) for run in (transcription.alternative_runs or []) if isinstance(run, dict)]
    if settings.parakeet_full_pass_enabled:
        parakeet_run = call_parakeet_full_transcription(wav_path, file_key, settings, parakeet_endpoint_pool)
        if settings.asr_runs_enabled:
            store.insert_asr_run(file_key, parakeet_run)
        if not parakeet_run.get("error") and str(parakeet_run.get("transcript") or "").strip():
            peer_runs.append(parakeet_run)

    if not (settings.transcript_lattice_enabled and peer_runs):
        return transcript_for_output, []

    lattice_spans = build_disagreement_spans(primary_run, peer_runs, settings)
    store.insert_lattice_spans(file_key, lattice_spans)

    adjudication_count = 0

    def adjudicator(span_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        nonlocal adjudication_count
        if not (
            settings.llm_adjudication_enabled
            and settings.transcript_lattice_llm_adjudication_enabled
            and adjudication_count < settings.transcript_lattice_llm_max_spans
        ):
            return None
        adjudication_count += 1
        return call_gemma_transcript_lattice_adjudication(
            span_payload,
            settings,
            endpoint_pool=gemma_endpoint_pool,
            request_id=file_key,
        )

    corrected_transcript, corrections = correct_transcript_constrained(
        transcript_for_output,
        lattice_spans,
        settings,
        adjudicator=adjudicator,
    )
    correction_rows = transcript_correction_dicts(corrections)
    store.insert_transcript_corrections(file_key, correction_rows)
    return corrected_transcript, correction_rows


def build_email_body(
    extension: str,
    info: dict[str, str],
    transcript: str,
    entities: dict[str, Any],
    settings: Settings,
) -> str:
    caller_name, caller_number = parse_callerid(info.get("callerid", "Unknown"))
    duration = format_duration(info.get("duration", "0"))
    date_str = format_date(
        info.get("origdate", ""),
        settings.local_timezone,
        settings.date_timezone_label,
    )
    mailbox = info.get("origmailbox", extension)
    separator = "─" * 40
    callback_number = entities.get("callback_number")

    return (
        "Voicemail Transcript\n"
        f"{separator}\n"
        f"Extension: {mailbox}\n"
        f"Caller ID: {format_caller_id(caller_name)}\n"
        f"Caller Number: {caller_number}\n"
        f"Duration: {duration}\n"
        f"Date: {date_str}\n\n"
        f"Name: {included_or_default(entities.get('name'))}\n"
        f"DOB: {included_or_default(entities.get('dob'))}\n"
        f"Callback Number: {included_or_default(callback_number)}\n"
        f"Callback Matches Caller ID: {callback_match_status(caller_number, callback_number)}\n"
        f"Fax Number: {included_or_default(entities.get('fax_number'))}\n\n"
        f"{transcript}\n\n"
        f"{separator}\n"
        "This transcript was generated by AI.\n"
        "Accuracy may vary based on audio quality.\n"
        "Please verify all clinical details.\n"
    )


def apply_final_mailbox_spelling_rules(
    mailbox: str,
    transcript: str,
    entities: dict[str, Any],
    settings: Settings,
    *,
    file_key: str = "",
    stage: str = "final",
) -> tuple[str, dict[str, Any]]:
    rules = load_mailbox_spelling_rules(settings.mailbox_spelling_rules_path)
    corrected_transcript, corrected_entities, count = apply_mailbox_spelling_rules(
        mailbox,
        transcript,
        entities,
        rules,
        enabled=settings.mailbox_spelling_rules_enabled,
    )
    if count:
        logger.info(
            "Mailbox spelling rules applied key=%s extension=%s stage=%s count=%s",
            file_key,
            mailbox,
            stage,
            count,
        )
    return corrected_transcript, corrected_entities


def send_email(
    recipients: list[str],
    file_key: str,
    extension: str,
    wav_path: str,
    info: dict[str, str],
    transcript: str,
    entities: dict[str, Any],
    settings: Settings,
) -> None:
    callerid = sanitize_header(info.get("callerid", "Unknown"))
    msg = EmailMessage()
    msg["Subject"] = f"New Voicemail Message from {callerid}"
    msg["From"] = f"{sanitize_header(settings.from_name)} <{settings.from_address}>"
    msg["To"] = ", ".join(recipients)
    msg["Message-ID"] = f"<voicemail-{file_key}@local.invalid>"
    msg.set_content(build_email_body(extension, info, transcript, entities, settings))

    with open(wav_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="audio",
            subtype="wav",
            filename=os.path.basename(wav_path),
        )

    with smtplib.SMTP(
        settings.smtp_host,
        settings.smtp_port,
        timeout=settings.smtp_timeout_seconds,
    ) as smtp:
        smtp.ehlo()
        if settings.smtp_starttls:
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
        smtp.send_message(msg, from_addr=settings.from_address, to_addrs=recipients)

    logger.info("Email sent key=%s extension=%s recipient_count=%s", file_key, extension, len(recipients))


DEFAULT_GEMMA_FIELD_PROMPT = """
You are a deterministic medical voicemail entity extractor.

Return ONLY valid JSON. Do not include explanations, markdown, comments, or extra text.

Required JSON schema:
{
  "patient_names": [
    {
      "raw": null,
      "value": null,
      "evidence_text": null,
      "source": null,
      "caller_id_used": null,
      "confidence": "candidate_only"
    }
  ],
  "name_correction_candidates": [
    {
      "raw": null,
      "suggested_value": null,
      "evidence_text": null,
      "caller_id_used": null,
      "reason": null
    }
  ],
  "dob_candidates": [
    {
      "raw": null,
      "normalized": null,
      "evidence_text": null,
      "confidence": "candidate_only"
    }
  ],
  "callback_numbers": [
    {
      "raw": null,
      "normalized": null,
      "formatted": null,
      "label_cue": null,
      "evidence_text": null,
      "confidence": "candidate_only"
    }
  ],
  "fax_numbers": [
    {
      "raw": null,
      "normalized": null,
      "formatted": null,
      "label_cue": null,
      "evidence_text": null,
      "confidence": "candidate_only"
    }
  ],
  "uncertain_numbers": [],
  "possible_errors": []
}

Rules:
- Gemma identifies candidate fields and exact evidence phrases only.
- Do not invent missing values.
- Do not return a patient_names candidate unless evidence_text contains the raw name or a close ASR form of the raw name.
- Do not use generic phrases as evidence_text for a name.
- Patient/caller names must identify the voicemail speaker or the patient/subject of the call.
- Name priority: first find the patient/subject of the voicemail. Patient/subject cues include "patient's name is", "calling on a patient", "calling for", "on behalf of", "regarding", "about", and relationship phrases such as "my husband", "my wife", "my mother", "my father", "my son", or "my daughter".
- If no patient/subject name is clearly present, use the caller/speaker name from self-identification phrases such as "this is", "my name is", or "the name is".
- Do not treat greeted/addressee names, provider/doctor names, staff names, clinic/company names, or incidental story names as the patient/subject.
- Never extract a name that is only being greeted or addressed. Names after greetings like "Hey Casey", "Hi Casey", "Hello Casey", or "Dear Casey" are addressees, not patient names.
- Never extract an uncertain recipient/addressee phrase as a patient name. Examples: "Casey, I think that's your name", "if I heard that correctly", or "is that your name".
- Use self-identification phrases for caller names only when no patient/subject name is clearly present: "this is", "it's", "my name is", "I am", or "the name is".
- If the transcript says "Hey, Casey. It's Mark Exampel again" and caller_id is "MARK EXAMPLE", exclude Casey and return one patient_names item with raw "Mark Exampel", value "Mark Exampel", evidence_text "It's Mark Exampel again", source "transcript", caller_id_used "".
- If there are repeated self-identifications with noisy variants, prefer the clearest complete person name from a self-identification phrase, especially a later correction such as "Again, my name is Taylor Example".
- Do not infer or repair uncertain phone/fax digits.
- Callback and fax numbers must be present in the transcript evidence.
- Caller ID phone numbers are metadata only and must not become callback/fax candidates.
- Do not use Caller ID to correct name spelling. Caller ID names are often truncated or incomplete.
- Caller ID may not create a name by itself.
- If Caller ID may match the transcript name but is not strong enough to overwrite value, keep patient_names transcript-supported and add name_correction_candidates as an audit/review hint.
- name_correction_candidates must not replace patient_names.
- Allowed name_correction_candidates reasons are "phonetic_last_first_match", "last_name_phonetic_match", and "weak_phonetic_match".
- Explicit spelling immediately after a spoken name may correct the name. If first and last name are spelled separately, return the full name, not only the last spelled word. Synthetic example: "my name is Bailey, B-A-I-L-E-Y, Example, E-X-A-M-P-L-E" means raw "Bailey Example", value "Bailey Example", source "transcript_spelling_corrected".
- If a name is corrected by spelling, use the spelled letters even when ASR heard a similar name. Synthetic example: "calling for Avery Exampel, S-A-M-P-L-E" means raw "Avery Exampel", value "Avery Sample", source "transcript_spelling_corrected".
- When no patient/subject name is present, prefer the clearest self-identification over later incidental names or phrases. "my name is Bailey Example" beats greeting "Hey Avery"; "this is Caller Example" beats later "the staff member was Example".
- Do not extract ordinary phrases as names, such as "probably safest", "having problems", "home right now", "scheduled as soon as", "that said", "is due for the rooster", or "three weeks ago".
- Do not extract organization/company names as patient names unless explicitly the patient/client name.
- Explicit patient phrases beat later filler: "calling on a patient, Patient Example" means name "Patient Example", not "that said".
- DOB must be clearly identified as a date of birth, birth date, or DOB.
- Normalize DOB to MM/DD/YYYY only when complete and plausible.
- Normalize callback and fax numbers as digits only and format 10-digit US phone/fax numbers as "(###) ###-####".
- Put ambiguous or incomplete values in possible_errors or uncertain_numbers, not final candidate arrays.
""".strip()


@dataclass(frozen=True)
class VerificationRunResult:
    proposed_entities: dict[str, Any]
    audit_rows: list[dict[str, Any]]
    should_apply: bool
    timed_out: bool = False
    complete: bool = True


def load_gemma_prompt(settings: Settings) -> str:
    if settings.gemma_prompt_path:
        try:
            with open(settings.gemma_prompt_path, "r", encoding="utf-8") as f:
                value = f.read().strip()
            if value:
                return value
        except OSError as exc:
            logger.warning("Could not read GEMMA_PROMPT_PATH path=%s error=%s", settings.gemma_prompt_path, exc)
    return DEFAULT_GEMMA_FIELD_PROMPT


ADAPTIVE_GEMMA_PROMPT_MARKER = "ADAPTIVE_GEMMA_FIELD_PROMPT_V1"
ADAPTIVE_GEMMA_PROMPT_CAPSULE_ORDER = (
    "spelled_name",
    "relationship_subject",
    "compact_dob",
    "addressee_name_exclusion",
    "organization_exclusion",
    "callback_fax",
)
ADAPTIVE_GEMMA_PROMPT_CAPSULES = {
    "spelled_name": (
        'Synthetic spelling example: "my name is Bailey, B-A-I-L-E-Y, Example, E-X-A-M-P-L-E" -> '
        '{"n":[["Bailey Example","Bailey Example","my name is Bailey, B-A-I-L-E-Y, Example, E-X-A-M-P-L-E",'
        '"transcript_spelling_corrected",""]],"r":[],"d":[],"c":[],"f":[],"u":[],"e":[]}.'
    ),
    "relationship_subject": (
        'Relationship example: "calling for my husband Rick Sample" -> extract Rick Sample as the subject, '
        'not the caller: {"n":[["Rick Sample","Rick Sample","my husband Rick Sample","relationship_subject",""]],'
        '"r":[],"d":[],"c":[],"f":[],"u":[],"e":[]}.'
    ),
    "compact_dob": (
        'Compact DOB example: "patient Jane Example DOB 62554" -> '
        '{"n":[["Jane Example","Jane Example","patient Jane Example","transcript",""]],'
        '"r":[],"d":[["62554","06/25/1954","DOB 62554"]],"c":[],"f":[],"u":[],"e":[]}.'
    ),
    "addressee_name_exclusion": (
        'Addressee example: "Hi Casey, this is Taylor Sample" -> Casey is only greeted; extract Taylor Sample from '
        'self-identification evidence.'
    ),
    "organization_exclusion": (
        'Organization example: "calling from Example Clinic about a client" -> Example Clinic is not a '
        'patient name; return no name unless a clear person subject appears.'
    ),
    "callback_fax": (
        'Phone/fax example: "call back at 217-555-0100, fax 217-555-0199" -> '
        '{"n":[],"r":[],"d":[],"c":[["217-555-0100","2175550100","(217) 555-0100","call back",'
        '"call back at 217-555-0100"]],"f":[["217-555-0199","2175550199","(217) 555-0199","fax",'
        '"fax 217-555-0199"]],"u":[],"e":[]}.'
    ),
}


def select_adaptive_gemma_prompt_capsules(transcript_text: str) -> list[str]:
    text = str(transcript_text or "")
    selected: set[str] = set()

    if re.search(r"\b(?:spelled|spelling)\b", text, re.IGNORECASE) or re.search(
        r"\b[A-Z](?:[-\s]+[A-Z]){2,}\b",
        text,
    ):
        selected.add("spelled_name")

    if re.search(
        r"\b(?:husband|wife|mother|father|son|daughter|spouse|mom|dad|brother|sister|parent|child)\b",
        text,
        re.IGNORECASE,
    ):
        selected.add("relationship_subject")

    if re.search(r"\b(?:dob|d\.?\s*o\.?\s*b\.?|date of birth|birth ?date|born)\b", text, re.IGNORECASE) or re.search(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}[,\s]+(?:\d{4,6}|\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?)\b",
        text,
    ):
        selected.add("compact_dob")

    if re.search(
        r"^\s*(?:hi|hello|hey|dear|good morning|good afternoon|good evening)[,\s]+"
        r"(?!(?:this|it'?s|my|i)\b)[A-Za-z][A-Za-z'-]+(?:\b|,)",
        text,
        re.IGNORECASE,
    ):
        selected.add("addressee_name_exclusion")

    if re.search(
        r"\b(?:clinic|hospital|insurance|healthcare|health care|medical|center|centre|company|department|office|"
        r"benefits|billing|pharmacy|provider|doctor|dr\.|agency)\b",
        text,
        re.IGNORECASE,
    ):
        selected.add("organization_exclusion")

    if re.search(
        r"\b(?:call back|callback|call me|phone|telephone|cell|mobile|contact|reach me|fax|number|extension)\b",
        text,
        re.IGNORECASE,
    ) or re.search(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", text):
        selected.add("callback_fax")

    return [capsule for capsule in ADAPTIVE_GEMMA_PROMPT_CAPSULE_ORDER if capsule in selected]


def build_adaptive_gemma_prompt(prompt: str, transcript_text: str) -> str:
    if ADAPTIVE_GEMMA_PROMPT_MARKER not in str(prompt or ""):
        return prompt

    selected_capsules = select_adaptive_gemma_prompt_capsules(transcript_text)
    if not selected_capsules:
        return prompt.rstrip()

    capsule_text = "\n".join(ADAPTIVE_GEMMA_PROMPT_CAPSULES[name] for name in selected_capsules)
    return f"{prompt.rstrip()}\n\nAdaptive examples:\n{capsule_text}"


def gemma_api_mode(settings: Any) -> str:
    return str(getattr(settings, "gemma_api_mode", "ollama") or "ollama").strip().lower()


def build_gemma_http_request(
    settings: Any,
    prompt_text: str,
    base_url: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    mode = gemma_api_mode(settings)
    service_base_url = (base_url or settings.gemma_base_url).rstrip("/")
    if mode in {"litert", "litert_chat", "chat"}:
        request_payload = {
            "message": prompt_text,
            "history": [],
            "show_thinking": False,
        }
        return (
            f"{service_base_url}/api/chat",
            request_payload,
        )

    return (
        f"{service_base_url}/api/generate",
        {
            "model": settings.gemma_model,
            "prompt": prompt_text,
            "stream": False,
            "format": "json",
        },
    )


def extract_gemma_response_text(payload: Any) -> Optional[str]:
    if isinstance(payload, str):
        return payload

    if not isinstance(payload, dict):
        return None

    for key in ("response", "text", "reply", "content", "output", "generated_text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value

    message = payload.get("message")
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                return text
            choice_message = first.get("message")
            if isinstance(choice_message, dict):
                content = choice_message.get("content")
                if isinstance(content, str):
                    return content

    return None


def normalize_gemma_http_payload(payload: Any, settings: Any) -> Any:
    if gemma_api_mode(settings) not in {"litert", "litert_chat", "chat"}:
        return payload

    text = extract_gemma_response_text(payload)
    if text is not None:
        return {"response": text}
    return payload


def gemma_payload_log_text(payload: Any, settings: Any) -> str:
    text = extract_gemma_response_text(payload)
    if text is not None:
        return text
    try:
        return json.dumps(payload, ensure_ascii=True)
    except (TypeError, ValueError):
        return str(payload)


def response_error_excerpt(response: Any, max_chars: int = 500) -> str:
    try:
        text = str(response.text or "")
    except Exception:
        return ""
    text = text.strip()
    if not text:
        return ""

    try:
        payload = json.loads(text)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        for key in ("detail", "error", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break

    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def build_gemma_input_payload(
    transcription: TranscriptionResult,
    info: dict[str, str],
) -> dict[str, Any]:
    """
    Keep Gemma input small and transcript-centered.

    Word timestamps and segments are intentionally not sent to Gemma. The model
    only needs the transcript plus metadata to choose candidate values and exact
    evidence phrases; the watcher maps evidence back to local timing data later.

    Use Whisper's final processed transcript as the only transcript Gemma sees.
    That text already includes Whisper's targeted phone pass corrections. Sending
    raw_text or duplicate transcript fields lets Gemma reintroduce digits that
    Whisper already corrected.
    """
    transcript_text = str(
        transcription.processed_text
        or transcription.text
        or transcription.raw_text
        or ""
    ).strip()

    payload: dict[str, Any] = {
        "transcript": transcript_text,
        "caller_id": info.get("callerid", ""),
        "mailbox": info.get("origmailbox", ""),
    }
    return payload


def call_gemma_field_extraction(
    transcription: TranscriptionResult,
    info: dict[str, str],
    settings: Settings,
    deadline: Optional[float],
    endpoint_pool: Optional[ServiceEndpointPool] = None,
    request_id: str = "",
    event_observer: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests package is unavailable")

    transcript_text = str(
        transcription.processed_text
        or transcription.text
        or transcription.raw_text
        or ""
    ).strip()
    if not transcript_text:
        raise RuntimeError("Gemma field extraction skipped: transcript is empty")

    prompt = build_adaptive_gemma_prompt(load_gemma_prompt(settings), transcript_text)
    input_payload = build_gemma_input_payload(transcription, info)
    prompt_text = f"{prompt}\n\nInput JSON:\n{json.dumps(input_payload, ensure_ascii=True)}"
    last_error: Optional[Exception] = None
    schema_error: Optional[GemmaSchemaError] = None
    non_schema_error_seen = False
    words = transcription.entities.get("_word_timestamps", [])
    logger.info(
        "Gemma field extraction request processed_transcript_chars=%s word_count=%s input_payload_chars=%s prompt_chars=%s",
        len(str(input_payload["transcript"] or "")),
        len(words) if isinstance(words, list) else 0,
        len(json.dumps(input_payload, ensure_ascii=True)),
        len(prompt_text),
    )

    for _attempt in range(settings.gemma_max_retries + 1):
        check_budget(deadline)
        endpoint_order = (
            endpoint_pool.least_busy_order(settings, request_id or uuid.uuid4().hex[:8])
            if endpoint_pool is not None
            else (getattr(settings, "gemma_base_urls", None) or (settings.gemma_base_url,))
        )
        for base_url in endpoint_order:
            try:
                timeout = remaining_budget(deadline, settings.gemma_timeout_seconds)
                url, request_payload = build_gemma_http_request(settings, prompt_text, base_url)
                if endpoint_pool is not None:
                    endpoint_pool.mark_started(base_url)
                try:
                    response = requests.post(
                        url,
                        json=request_payload,
                        headers=gemma_headers(settings) or None,
                        timeout=(min(5, timeout), timeout),
                    )
                finally:
                    if endpoint_pool is not None:
                        endpoint_pool.mark_finished(base_url)

                if response.status_code >= 400:
                    excerpt = response_error_excerpt(response)
                    suffix = f": {excerpt}" if excerpt else ""
                    raise RuntimeError(f"Gemma returned HTTP {response.status_code}{suffix}")
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise GemmaSchemaError("Gemma returned non-JSON HTTP response") from exc
                normalized_payload = normalize_gemma_http_payload(payload, settings)
                if getattr(settings, "gemma_log_raw_response", False):
                    raw_text = gemma_payload_log_text(normalized_payload, settings)
                    max_chars = int(getattr(settings, "gemma_raw_log_max_chars", 2000) or 2000)
                    excerpt = raw_text[:max_chars]
                    if len(raw_text) > max_chars:
                        excerpt += "...[truncated]"
                    logger.warning(
                        "Gemma raw response chars=%s excerpt=%s",
                        len(raw_text),
                        json.dumps(excerpt, ensure_ascii=True),
                    )
                parsed_payload = parse_gemma_response(normalized_payload)
                return parsed_payload
            except GemmaSchemaError as exc:
                # Schema-invalid output is semantically different from an unavailable
                # Gemma server. Preserve the exception type so the resolver records
                # gemma_invalid_json instead of a generic unavailable fallback when
                # every configured endpoint fails the same way.
                schema_error = exc
                last_error = exc
                logger.warning(
                    "Gemma field extraction returned invalid schema endpoint=%s error=%s",
                    base_url,
                    exc,
                )
            except Exception as exc:
                non_schema_error_seen = True
                last_error = exc
                logger.warning(
                    "Gemma field extraction attempt failed endpoint=%s error=%s",
                    base_url,
                    exc,
                )

    if schema_error is not None and not non_schema_error_seen:
        raise schema_error
    raise RuntimeError(f"Gemma field extraction failed: {last_error}")


def gemma_candidates_for_field(gemma_payload: dict[str, Any], field_name: str) -> list[dict[str, Any]]:
    key_by_field = {
        "name": "patient_names",
        "dob": "dob_candidates",
        "callback_number": "callback_numbers",
        "fax_number": "fax_numbers",
    }
    return [dict(item) for item in gemma_payload.get(key_by_field[field_name], []) if isinstance(item, dict)]


def empty_gemma_field_payload() -> dict[str, list[Any]]:
    return {
        "patient_names": [],
        "name_correction_candidates": [],
        "dob_candidates": [],
        "callback_numbers": [],
        "fax_numbers": [],
        "uncertain_numbers": [],
        "possible_errors": [],
    }


def patient_name_values_for_compact_dob(gemma_payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for candidate in gemma_payload.get("patient_names", []):
        if not isinstance(candidate, dict):
            continue
        source = str(candidate.get("source") or "").lower()
        evidence_text = str(candidate.get("evidence_text") or "").strip()
        if source == "caller_id" and not evidence_text:
            continue
        for key in ("raw", "value"):
            value = re.sub(r"\s+", " ", str(candidate.get(key) or "").strip(" ,.;:"))
            if value and value not in names:
                names.append(value)
    return names


def add_compact_dob_fallback_candidates(
    gemma_payload: dict[str, Any],
    transcript_text: str,
) -> int:
    candidates = gemma_payload.setdefault("dob_candidates", [])
    if not isinstance(candidates, list):
        return 0

    existing: set[tuple[str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        existing.add(
            (
                str(candidate.get("normalized") or "").strip(),
                str(candidate.get("evidence_text") or "").strip().lower(),
            )
        )

    added = 0
    patient_names = patient_name_values_for_compact_dob(gemma_payload)
    for candidate in extract_compact_dob_candidates(transcript_text, patient_names):
        key = (
            str(candidate.get("normalized") or "").strip(),
            str(candidate.get("evidence_text") or "").strip().lower(),
        )
        if key in existing:
            continue
        candidates.append(candidate)
        existing.add(key)
        added += 1
    return added


NEAR_PHONE_FRAGMENT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*[A-Za-z0-9]{1,8})"
    r"(?![A-Za-z0-9])"
)
NEAR_PHONE_CALLBACK_CUE_RE = re.compile(
    r"\b(?:callback(?:\s+number)?|call\s+back|call\s+me\s+back|"
    r"return\s+call|phone\s+number|contact\s+number|reach\s+me|call\s+me)\b",
    re.IGNORECASE,
)
NEAR_PHONE_FAX_CUE_RE = re.compile(
    r"\b(?:fax(?:\s+number)?|faxed|faxing|fax\s+this\s+to|fax\s+to)\b",
    re.IGNORECASE,
)
NEAR_PHONE_EXCLUSION_RE = re.compile(
    r"\b(?:date\s+of\s+birth|birth\s+date|dob|d\.o\.b\.|born|"
    r"extension|ext\.?|x)\b",
    re.IGNORECASE,
)


def near_phone_fragment_field(
    transcript_text: str,
    fragment_start: int,
) -> tuple[Optional[str], Optional[int], str]:
    left_start = max(0, fragment_start - 90)
    left_context = transcript_text[left_start:fragment_start]
    if NEAR_PHONE_EXCLUSION_RE.search(left_context[-45:]):
        return None, None, ""

    matches: list[tuple[int, str, str]] = []
    for match in NEAR_PHONE_CALLBACK_CUE_RE.finditer(left_context):
        matches.append((left_start + match.start(), "callback_number", match.group(0)))
    for match in NEAR_PHONE_FAX_CUE_RE.finditer(left_context):
        matches.append((left_start + match.start(), "fax_number", match.group(0)))
    if not matches:
        return None, None, ""
    cue_start, field_name, cue_text = max(matches, key=lambda item: item[0])
    return field_name, cue_start, cue_text


def near_phone_fragment_digit_count(raw: str) -> int:
    return len(re.sub(r"\D", "", raw))


def add_near_phone_fallback_candidates(
    gemma_payload: dict[str, Any],
    transcript_text: str,
) -> int:
    added = 0
    existing: set[tuple[str, str]] = set()
    for field_key in ("callback_numbers", "fax_numbers", "uncertain_numbers"):
        for candidate in gemma_payload.get(field_key, []) or []:
            if not isinstance(candidate, dict):
                continue
            existing.add(
                (
                    field_key,
                    re.sub(r"\s+", " ", str(candidate.get("evidence_text") or "").strip()).lower(),
                )
            )

    for match in NEAR_PHONE_FRAGMENT_RE.finditer(str(transcript_text or "")):
        raw = match.group(0).strip()
        if "/" in raw or normalize_phone_candidate(raw).valid:
            continue
        digit_count = near_phone_fragment_digit_count(raw)
        if digit_count < 7 or digit_count == 10 or digit_count > 14:
            continue

        field_name, cue_start, cue_text = near_phone_fragment_field(transcript_text, match.start())
        if field_name is None or cue_start is None:
            continue

        field_key = "callback_numbers" if field_name == "callback_number" else "fax_numbers"
        evidence_text = transcript_text[cue_start : match.end()].strip(" \t\r\n,.;:")
        evidence_key = re.sub(r"\s+", " ", evidence_text).lower()
        if not evidence_text or (field_key, evidence_key) in existing:
            continue

        gemma_payload.setdefault(field_key, []).append(
            {
                "raw": raw,
                "normalized": "",
                "formatted": "",
                "label_cue": cue_text,
                "evidence_text": evidence_text,
                "source": "near_phone_fragment",
            }
        )
        existing.add((field_key, evidence_key))
        added += 1
    return added


def add_spelled_name_fallback_candidates(
    gemma_payload: dict[str, Any],
    transcript_text: str,
) -> int:
    candidates = gemma_payload.setdefault("patient_names", [])
    if not isinstance(candidates, list):
        return 0

    existing: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        existing.add(
            (
                str(candidate.get("raw") or "").strip().lower(),
                str(candidate.get("value") or "").strip().lower(),
                str(candidate.get("evidence_text") or "").strip().lower(),
            )
        )

    added = 0
    patient_names = patient_name_values_for_compact_dob(gemma_payload)
    for candidate in extract_spelled_name_candidates(transcript_text, patient_names):
        key = (
            str(candidate.get("raw") or "").strip().lower(),
            str(candidate.get("value") or "").strip().lower(),
            str(candidate.get("evidence_text") or "").strip().lower(),
        )
        if key in existing:
            continue
        candidates.append(candidate)
        existing.add(key)
        added += 1
    return added


def add_explicit_patient_name_fallback_candidates(
    gemma_payload: dict[str, Any],
    transcript_text: str,
) -> int:
    candidates = gemma_payload.setdefault("patient_names", [])
    if not isinstance(candidates, list):
        return 0

    existing: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        existing.add(
            (
                str(candidate.get("raw") or "").strip().lower(),
                str(candidate.get("value") or "").strip().lower(),
                str(candidate.get("evidence_text") or "").strip().lower(),
            )
        )

    added = 0
    for candidate in extract_explicit_patient_name_candidates(transcript_text):
        key = (
            str(candidate.get("raw") or "").strip().lower(),
            str(candidate.get("value") or "").strip().lower(),
            str(candidate.get("evidence_text") or "").strip().lower(),
        )
        if key in existing:
            continue
        candidates.append(candidate)
        existing.add(key)
        added += 1
    return added


def add_self_identification_name_fallback_candidates(
    gemma_payload: dict[str, Any],
    transcript_text: str,
) -> int:
    candidates = gemma_payload.setdefault("patient_names", [])
    if not isinstance(candidates, list):
        return 0

    existing: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        existing.add(
            (
                str(candidate.get("raw") or "").strip().lower(),
                str(candidate.get("value") or "").strip().lower(),
                str(candidate.get("evidence_text") or "").strip().lower(),
            )
        )

    added = 0
    for candidate in extract_self_identification_name_candidates(transcript_text):
        key = (
            str(candidate.get("raw") or "").strip().lower(),
            str(candidate.get("value") or "").strip().lower(),
            str(candidate.get("evidence_text") or "").strip().lower(),
        )
        if key in existing:
            continue
        candidates.append(candidate)
        existing.add(key)
        added += 1
    return added


def add_relationship_name_fallback_candidates(
    gemma_payload: dict[str, Any],
    transcript_text: str,
) -> int:
    candidates = gemma_payload.setdefault("patient_names", [])
    if not isinstance(candidates, list):
        return 0

    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = (
            str(candidate.get("raw") or "").strip().lower(),
            str(candidate.get("value") or "").strip().lower(),
            str(candidate.get("evidence_text") or "").strip().lower(),
        )
        existing[key] = candidate

    added = 0
    for candidate in extract_relationship_name_candidates(transcript_text):
        key = (
            str(candidate.get("raw") or "").strip().lower(),
            str(candidate.get("value") or "").strip().lower(),
            str(candidate.get("evidence_text") or "").strip().lower(),
        )
        if key in existing:
            existing_candidate = existing[key]
            if str(existing_candidate.get("source") or "").lower() != "relationship_subject":
                existing_candidate.update(candidate)
                added += 1
            continue
        candidates.append(candidate)
        existing[key] = candidate
        added += 1
    return added


def add_subject_reference_name_fallback_candidates(
    gemma_payload: dict[str, Any],
    transcript_text: str,
) -> int:
    candidates = gemma_payload.setdefault("patient_names", [])
    if not isinstance(candidates, list):
        return 0

    existing: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        existing.add(
            (
                str(candidate.get("raw") or "").strip().lower(),
                str(candidate.get("value") or "").strip().lower(),
                str(candidate.get("evidence_text") or "").strip().lower(),
            )
        )

    added = 0
    for candidate in extract_subject_reference_name_candidates(transcript_text):
        key = (
            str(candidate.get("raw") or "").strip().lower(),
            str(candidate.get("value") or "").strip().lower(),
            str(candidate.get("evidence_text") or "").strip().lower(),
        )
        if key in existing:
            continue
        candidates.append(candidate)
        existing.add(key)
        added += 1
    return added


def build_candidate_records(
    field_name: str,
    gemma_payload: dict[str, Any],
    transcription: TranscriptionResult,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    words = transcription.entities.get("_word_timestamps", [])
    if not isinstance(words, list):
        words = []

    for index, candidate in enumerate(gemma_candidates_for_field(gemma_payload, field_name)):
        cid = "name:%s" % index if field_name == "name" else "dob:%s" % index if field_name == "dob" else f"{field_name}:{index}"
        evidence_text = str(candidate.get("evidence_text") or "")
        # Gemma is authoritative for field attribution, so attribution is
        # grounded through Gemma's field-specific evidence_text. Do not map a
        # phone/fax candidate by digits alone; that would reintroduce weak
        # regex-style attribution.
        attribution = map_evidence_to_timestamps(
            field_name=field_name,
            candidate_id_value=cid,
            evidence_text=evidence_text,
            words=words,
            segments=transcription.segments,
            candidate_digits=None,
        )
        whisper_numbers = [
            number.normalized
            for number in extract_numbers_from_text(attribution.matched_text)
            if number.normalized
        ]
        candidate["candidate_id"] = cid
        records.append(
            CandidateRecord(
                candidate_id=cid,
                field_name=field_name,
                gemma=candidate,
                attribution=attribution,
                whisper_numbers=whisper_numbers,
            )
        )
    return records


def run_parakeet_for_record(
    wav_path: str,
    record: CandidateRecord,
    settings: Settings,
    deadline: Optional[float],
    endpoint_pool: Optional[ServiceEndpointPool] = None,
) -> None:
    if not settings.parakeet_verification_enabled:
        return
    if not record.attribution.mapped:
        return

    check_budget(deadline)
    clip_timeout = int(max(1, remaining_budget(deadline, min(30, settings.parakeet_verification_timeout_seconds))))
    clip = create_verification_clip(
        wav_path,
        record.attribution,
        settings.verification_clip_dir,
        expected_sample_rate=settings.parakeet_expected_sample_rate,
        ffmpeg_bin=settings.verification_ffmpeg_bin,
        timeout_seconds=clip_timeout,
    )
    record.clip = clip
    if clip.get("error"):
        record.attribution.review_reasons.append("clip_failure")
        logger.warning(
            "Parakeet verification clip failed candidate_id=%s error=%s",
            record.candidate_id,
            clip.get("error"),
        )
        return

    clip_path = str(clip.get("path") or "")
    if not clip_path:
        record.attribution.review_reasons.append("clip_failure")
        logger.warning("Parakeet verification clip missing path candidate_id=%s", record.candidate_id)
        return

    last_result = None
    for _attempt in range(settings.parakeet_verification_max_retries + 1):
        check_budget(deadline)
        timeout = remaining_budget(deadline, settings.parakeet_verification_timeout_seconds)
        if settings.parakeet_verification_mode == "cli":
            last_result = call_parakeet_cli(
                record.candidate_id,
                settings.parakeet_verification_cmd,
                clip_path,
                timeout,
            )
        else:
            endpoint_order = (
                endpoint_pool.least_busy_order(settings, record.candidate_id)
                if endpoint_pool is not None
                else (
                    getattr(settings, "parakeet_verification_urls", None)
                    or (settings.parakeet_verification_url,)
                )
            )
            for url in endpoint_order:
                if endpoint_pool is not None:
                    endpoint_pool.mark_started(url)
                try:
                    last_result = call_parakeet_http(
                        record.candidate_id,
                        url,
                        clip_path,
                        timeout,
                        requests,
                        parakeet_headers(settings),
                    )
                finally:
                    if endpoint_pool is not None:
                        endpoint_pool.mark_finished(url)
                if last_result.error is None:
                    break
        if last_result is not None and last_result.error is None:
            break
    record.parakeet = last_result
    if last_result is None:
        logger.info("Parakeet verification produced no result candidate_id=%s", record.candidate_id)
    elif last_result.error:
        logger.warning(
            "Parakeet verification failed candidate_id=%s error=%s",
            record.candidate_id,
            last_result.error,
        )
    else:
        logger.info(
            "Parakeet verification completed candidate_id=%s normalized_number_count=%s",
            record.candidate_id,
            len(last_result.normalized_numbers),
        )


def timeout_audit_rows(original_entities: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field_name, legacy_key in (
        ("name", "name"),
        ("dob", "dob"),
        ("callback_number", "callback_number"),
        ("fax_number", "fax_number"),
    ):
        resolution = resolve_legacy_field(
            field_name,
            original_entities.get(legacy_key),
            True,
            "timeout",
        )
        resolution.needs_review = True
        resolution.review_reasons = list(dict.fromkeys([*resolution.review_reasons, "timeout"]))
        rows.append(resolution.as_audit_row())
    return rows


def unavailable_audit_rows(
    original_entities: dict[str, Any],
    reason: str,
    fail_open: bool,
) -> list[dict[str, Any]]:
    return [
        resolve_legacy_field("name", original_entities.get("name"), fail_open, reason).as_audit_row(),
        resolve_legacy_field("dob", original_entities.get("dob"), fail_open, reason).as_audit_row(),
        resolve_legacy_field("callback_number", original_entities.get("callback_number"), fail_open, reason).as_audit_row(),
        resolve_legacy_field("fax_number", original_entities.get("fax_number"), fail_open, reason).as_audit_row(),
    ]


PHONE_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:"
    r"(?:\+?1[\s().-]*)?(?:\d[\s().-]*){9,11}\d"
    r"|"
    r"(?:\+?1[\s.-]*)?"
    r"\(?[A-Za-z0-9]{3}\)?[\s.-]*"
    r"[A-Za-z0-9]{3}[\s.-]*"
    r"[A-Za-z0-9]{4}"
    r")"
    r"(?![A-Za-z0-9])"
)

PHONE_TRANSCRIPT_CORRECTION_STATUSES = {
    "verified",
    "parakeet_override",
    "whisper_caller_id_verified",
    "whisper_span_fallback",
}

DOB_TEXT_RE = re.compile(
    r"(?<!\d)(?:\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{2,4}|\d{6,8})(?!\d)"
)


def phone_alnum(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()
    if len(text) == 11 and text.startswith("1"):
        text = text[1:]
    return text


def loose_phone_text_matches(candidate: Any, normalized_digits: Any) -> bool:
    normalized = re.sub(r"\D", "", str(normalized_digits or ""))
    if len(normalized) == 11 and normalized.startswith("1"):
        normalized = normalized[1:]
    if len(normalized) != 10:
        return False

    candidate_text = str(candidate or "")
    candidate_digits = re.sub(r"\D", "", candidate_text)
    if len(candidate_digits) == 11 and candidate_digits.startswith("1"):
        candidate_digits = candidate_digits[1:]
    if candidate_digits == normalized:
        return True
    if candidate_digits == normalized[:3] + normalized:
        return True
    if len(candidate_digits) > len(normalized) and len(candidate_digits) <= len(normalized) + 2:
        extra = candidate_digits[len(normalized) :]
        if candidate_digits.startswith(normalized) and extra and set(extra) == {normalized[-1]}:
            return True
    if len(candidate_digits) >= 7 and (normalized.startswith(candidate_digits) or candidate_digits in normalized):
        return True

    candidate_key = phone_alnum(candidate_text)
    if len(candidate_key) != 10:
        return False

    positional_matches = sum(1 for left, right in zip(candidate_key, normalized) if left == right)
    digit_matches = sum(
        1
        for left, right in zip(candidate_key, normalized)
        if left.isdigit() and left == right
    )
    return positional_matches >= 8 or digit_matches >= 8


def repeated_area_code_prefix_matches(
    words: list[dict[str, Any]],
    prefix_index: int,
    span_end: int,
    normalized_digits: str,
) -> bool:
    normalized = re.sub(r"\D", "", str(normalized_digits or ""))
    if len(normalized) == 11 and normalized.startswith("1"):
        normalized = normalized[1:]
    if len(normalized) != 10 or prefix_index < 0:
        return False

    prefix_digits = re.sub(r"\D", "", str(words[prefix_index].get("word", "")))
    if prefix_digits != normalized[:3]:
        return False

    candidate = " ".join(str(words[i].get("word", "")) for i in range(prefix_index, span_end + 1))
    candidate_digits = re.sub(r"\D", "", candidate)
    if len(candidate_digits) == 11 and candidate_digits.startswith("1"):
        candidate_digits = candidate_digits[1:]
    return candidate_digits in {normalized, normalized[:3] + normalized}


def replace_phone_text_in_transcript(transcript: str, normalized_digits: str, formatted_number: str) -> tuple[str, int]:
    replacements = 0

    def replace_match(match: re.Match[str]) -> str:
        nonlocal replacements
        candidate = match.group(0)
        if not loose_phone_text_matches(candidate, normalized_digits):
            return candidate
        replacements += 1
        return formatted_number

    return PHONE_TEXT_RE.sub(replace_match, transcript), replacements


def attributed_word_ranges(
    audit_row: dict[str, Any],
    field_name: Optional[str] = None,
    word_count: Optional[int] = None,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for attribution in audit_row.get("attribution_json") or []:
        if not isinstance(attribution, dict):
            continue
        attribution_field = attribution.get("field_name")
        if field_name and attribution_field and attribution_field != field_name:
            continue
        try:
            start_index = int(attribution.get("word_start"))
            end_index = int(attribution.get("word_end"))
        except (TypeError, ValueError):
            continue
        if start_index > end_index:
            continue
        if word_count is not None:
            start_index = max(0, start_index)
            end_index = min(word_count - 1, end_index)
            if start_index > end_index:
                continue
        ranges.append((start_index, end_index))
    return ranges


def corrected_phone_word_span(
    words: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    normalized_digits: str,
) -> Optional[tuple[int, int]]:
    if not words:
        return None

    start_index = max(0, start_index)
    end_index = min(len(words) - 1, end_index)
    if start_index > end_index:
        return None

    best: Optional[tuple[int, int, int]] = None
    max_width = min(5, end_index - start_index + 1)
    for width in range(1, max_width + 1):
        for index in range(start_index, end_index - width + 2):
            candidate = " ".join(str(words[i].get("word", "")) for i in range(index, index + width))
            if not loose_phone_text_matches(candidate, normalized_digits):
                continue
            score = len(re.sub(r"\D", "", candidate))
            current = (index, index + width - 1, score)
            if best is None or width < best[1] - best[0] + 1 or score > best[2]:
                best = current
        if best is not None:
            break

    if best is None:
        return None

    span_start, span_end, _score = best
    if span_start > start_index and repeated_area_code_prefix_matches(
        words,
        span_start - 1,
        span_end,
        normalized_digits,
    ):
        return span_start - 1, span_end

    return span_start, span_end


def transcript_word_char_spans(
    transcript: str,
    words: list[dict[str, Any]],
) -> list[Optional[tuple[int, int]]]:
    spans: list[Optional[tuple[int, int]]] = []
    cursor = 0
    for item in words:
        token = str(item.get("word", "") or "")
        if not token:
            spans.append(None)
            continue
        start = transcript.find(token, cursor)
        if start < 0:
            stripped = token.strip(" \t\r\n")
            start = transcript.find(stripped, cursor) if stripped else -1
            token = stripped
        if start < 0:
            spans.append(None)
            continue
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


def phone_span_replacement_text(original_text: str, formatted_number: str) -> str:
    trailing = re.search(r"([^\w\s]+)$", original_text or "")
    if trailing and not formatted_number.endswith(trailing.group(1)):
        return formatted_number + trailing.group(1)
    return formatted_number


def replace_attributed_phone_text_in_transcript(
    transcript: str,
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    normalized_digits: str,
    formatted_number: str,
) -> tuple[str, int]:
    if not transcript or not words:
        return transcript, 0

    field_name = str(audit_row.get("field_name") or "")
    ranges = attributed_word_ranges(audit_row, field_name=field_name, word_count=len(words))
    if not ranges:
        return transcript, 0

    char_spans = transcript_word_char_spans(transcript, words)
    replacements: list[tuple[int, int, str]] = []
    for start_index, end_index in ranges:
        span = corrected_phone_word_span(words, start_index, end_index, normalized_digits)
        if span is None:
            continue

        span_start, span_end = span
        start_span = char_spans[span_start]
        end_span = char_spans[span_end]
        if start_span is None or end_span is None:
            continue

        char_start = start_span[0]
        char_end = end_span[1]
        original_text = transcript[char_start:char_end]
        if not loose_phone_text_matches(original_text, normalized_digits):
            continue
        replacements.append((char_start, char_end, phone_span_replacement_text(original_text, formatted_number)))

    if not replacements:
        return transcript, 0

    replacements.sort(key=lambda item: (item[0], item[1]))
    non_overlapping: list[tuple[int, int, str]] = []
    previous_end = -1
    for char_start, char_end, replacement in replacements:
        if char_start < previous_end:
            continue
        non_overlapping.append((char_start, char_end, replacement))
        previous_end = char_end

    corrected = transcript
    for char_start, char_end, replacement in reversed(non_overlapping):
        corrected = corrected[:char_start] + replacement + corrected[char_end:]
    return corrected, len(non_overlapping)


def phone_row_sort_key(audit_row: dict[str, Any]) -> int:
    field_name = str(audit_row.get("field_name") or "")
    ranges = attributed_word_ranges(audit_row, field_name=field_name)
    if not ranges:
        return -1
    return min(start_index for start_index, _end_index in ranges)


def merge_corrected_phone_words(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    normalized_digits: str,
    formatted_number: str,
) -> tuple[list[dict[str, Any]], bool]:
    if not words:
        return words, False

    field_name = str(audit_row.get("field_name") or "")
    ranges = attributed_word_ranges(audit_row, field_name=field_name, word_count=len(words))
    if not ranges:
        return words, False

    for start_index, end_index in ranges:
        span = corrected_phone_word_span(words, start_index, end_index, normalized_digits)
        if span is None:
            continue

        span_start, span_end = span
        first = words[span_start]
        last = words[span_end]
        merged = dict(first)
        merged["word"] = formatted_number
        if last.get("end") is not None:
            merged["end"] = last.get("end")
        return words[:span_start] + [merged] + words[span_end + 1 :], True

    return words, False


def dob_values_from_audit_row(audit_row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in audit_row.get("gemma_json") or []:
        if not isinstance(item, dict):
            continue
        for key in ("normalized", "raw", "value", "formatted"):
            parsed = parse_dob(item.get(key))
            if parsed:
                values.add(format_dob(parsed))
    return values


def dob_text_matches(candidate: Any, accepted_dobs: set[str]) -> bool:
    parsed = parse_dob(candidate)
    return bool(parsed and format_dob(parsed) in accepted_dobs)


def dob_span_replacement_text(original_text: str, final_dob: str) -> str:
    trailing = re.search(r"([^\w\s]+)$", original_text or "")
    if trailing and not final_dob.endswith(trailing.group(1)):
        return final_dob + trailing.group(1)
    return final_dob


def replace_attributed_dob_text_in_transcript(
    transcript: str,
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    final_dob: str,
) -> tuple[str, int]:
    if not transcript or not words or not final_dob:
        return transcript, 0

    accepted_dobs = dob_values_from_audit_row(audit_row)
    if not accepted_dobs:
        return transcript, 0

    ranges = attributed_word_ranges(audit_row, field_name="dob", word_count=len(words))
    if not ranges:
        return transcript, 0

    char_spans = transcript_word_char_spans(transcript, words)
    replacements: list[tuple[int, int, str]] = []
    for start_index, end_index in ranges:
        start_span = char_spans[start_index]
        end_span = char_spans[end_index]
        if start_span is None or end_span is None:
            continue

        char_start = start_span[0]
        char_end = end_span[1]
        span_text = transcript[char_start:char_end]
        matches = [
            match
            for match in DOB_TEXT_RE.finditer(span_text)
            if dob_text_matches(match.group(0), accepted_dobs)
        ]
        if len(matches) != 1:
            continue
        match = matches[0]
        replacements.append(
            (
                char_start + match.start(),
                char_start + match.end(),
                dob_span_replacement_text(match.group(0), final_dob),
            )
        )

    if not replacements:
        return transcript, 0

    replacements.sort(key=lambda item: (item[0], item[1]))
    non_overlapping: list[tuple[int, int, str]] = []
    previous_end = -1
    for char_start, char_end, replacement in replacements:
        if char_start < previous_end:
            continue
        non_overlapping.append((char_start, char_end, replacement))
        previous_end = char_end

    corrected = transcript
    for char_start, char_end, replacement in reversed(non_overlapping):
        corrected = corrected[:char_start] + replacement + corrected[char_end:]
    return corrected, len(non_overlapping)


def corrected_dob_word_span(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    accepted_dobs: set[str],
) -> Optional[tuple[int, int]]:
    if not words or not accepted_dobs:
        return None

    ranges = attributed_word_ranges(audit_row, field_name="dob", word_count=len(words))
    for start_index, end_index in ranges:
        max_width = min(3, end_index - start_index + 1)
        for width in range(1, max_width + 1):
            for index in range(start_index, end_index - width + 2):
                candidate = " ".join(str(words[i].get("word", "")) for i in range(index, index + width))
                if dob_text_matches(candidate, accepted_dobs):
                    return index, index + width - 1
    return None


def merge_corrected_dob_words(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    final_dob: str,
) -> tuple[list[dict[str, Any]], bool]:
    accepted_dobs = dob_values_from_audit_row(audit_row)
    span = corrected_dob_word_span(words, audit_row, accepted_dobs)
    if span is None:
        return words, False

    span_start, span_end = span
    original_text = " ".join(str(item.get("word", "")) for item in words[span_start : span_end + 1])
    first = words[span_start]
    last = words[span_end]
    merged = dict(first)
    merged["word"] = dob_span_replacement_text(original_text, final_dob)
    if last.get("end") is not None:
        merged["end"] = last.get("end")
    return words[:span_start] + [merged] + words[span_end + 1 :], True


def replacement_name_values(audit_row: dict[str, Any]) -> list[tuple[str, str]]:
    if audit_row.get("field_name") != "name":
        return []
    status = str(audit_row.get("status") or "")
    if status not in {"caller_id_spelling_corrected", "transcript_spelling_corrected"}:
        return []

    final_value = re.sub(r"\s+", " ", str(audit_row.get("final_value") or "")).strip()
    if not final_value:
        return []

    gemma_items = [item for item in audit_row.get("gemma_json") or [] if isinstance(item, dict)]
    replacements: list[tuple[str, str]] = []
    for item in gemma_items:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip().lower()
        raw = re.sub(r"\s+", " ", str(item.get("raw") or "")).strip()
        value = re.sub(r"\s+", " ", str(item.get("value") or "")).strip()
        caller_id_used = re.sub(r"\s+", " ", str(item.get("caller_id_used") or "")).strip()
        if status == "caller_id_spelling_corrected":
            if source:
                if source != "caller_id_corrected":
                    continue
                corrected_value = value
            elif len(gemma_items) != 1:
                continue
            else:
                corrected_value = caller_id_used or value
            if corrected_value != final_value:
                continue
        elif status == "transcript_spelling_corrected":
            if source:
                if source != "transcript_spelling_corrected":
                    continue
            elif len(gemma_items) != 1:
                continue
            if value != final_value:
                continue
        if raw and raw != final_value:
            replacements.append((raw, final_value))
        elif value and value != final_value:
            replacements.append((value, final_value))

    return list(dict.fromkeys(replacements))


def replace_name_text_in_transcript(transcript: str, raw_name: str, final_name: str) -> tuple[str, int]:
    raw_name = re.sub(r"\s+", " ", raw_name or "").strip()
    final_name = re.sub(r"\s+", " ", final_name or "").strip()
    if not raw_name or not final_name or raw_name == final_name:
        return transcript, 0

    pattern = re.compile(r"(?<![A-Za-z])" + re.escape(raw_name) + r"(?![A-Za-z])")
    matches = list(pattern.finditer(transcript))
    if len(matches) != 1:
        return transcript, 0
    match = matches[0]
    return transcript[: match.start()] + final_name + transcript[match.end() :], 1


def name_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"\s+", str(value or "").strip()) if token]


def normalized_name_token(value: Any) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").lower())


def word_span_matches_name(words: list[dict[str, Any]], start_index: int, tokens: list[str]) -> bool:
    if not tokens or start_index + len(tokens) > len(words):
        return False
    for offset, token in enumerate(tokens):
        if normalized_name_token(words[start_index + offset].get("word")) != normalized_name_token(token):
            return False
    return True


def corrected_name_word_span(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    raw_name: str,
) -> Optional[tuple[int, int]]:
    tokens = name_tokens(raw_name)
    if not words or not tokens:
        return None

    ranges: list[tuple[int, int]] = []
    for attribution in audit_row.get("attribution_json") or []:
        if not isinstance(attribution, dict):
            continue
        try:
            start_index = int(attribution.get("word_start"))
            end_index = int(attribution.get("word_end"))
        except (TypeError, ValueError):
            continue
        if start_index <= end_index:
            ranges.append((max(0, start_index), min(len(words) - 1, end_index)))

    if not ranges:
        return None

    width = len(tokens)
    for start_index, end_index in ranges:
        for index in range(start_index, end_index - width + 2):
            if word_span_matches_name(words, index, tokens):
                return index, index + width - 1

    return None


def name_word_occurrence_index(
    words: list[dict[str, Any]],
    span_start: int,
    raw_name: str,
) -> Optional[int]:
    tokens = name_tokens(raw_name)
    if not words or not tokens:
        return None

    occurrence_index = 0
    for index in range(0, max(0, span_start) + 1):
        if word_span_matches_name(words, index, tokens):
            if index == span_start:
                return occurrence_index
            occurrence_index += 1
    return None


def replace_attributed_name_text_in_transcript(
    transcript: str,
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    raw_name: str,
    final_name: str,
) -> tuple[str, int]:
    raw_name = re.sub(r"\s+", " ", raw_name or "").strip()
    final_name = re.sub(r"\s+", " ", final_name or "").strip()
    if not raw_name or not final_name or raw_name == final_name:
        return transcript, 0

    span = corrected_name_word_span(words, audit_row, raw_name) if words else None
    if span is None:
        return replace_name_text_in_transcript(transcript, raw_name, final_name)

    pattern = re.compile(r"(?<![A-Za-z])" + re.escape(raw_name) + r"(?![A-Za-z])")
    matches = list(pattern.finditer(transcript))
    occurrence_index = name_word_occurrence_index(words, span[0], raw_name)
    if occurrence_index is None or occurrence_index >= len(matches):
        return transcript, 0

    match = matches[occurrence_index]
    return transcript[: match.start()] + final_name + transcript[match.end() :], 1


def corrected_name_word_items(
    original_words: list[dict[str, Any]],
    final_name: str,
) -> list[dict[str, Any]]:
    tokens = name_tokens(final_name)
    if not original_words or not tokens:
        return original_words

    trailing = re.search(r"([^\w\s]+)$", str(original_words[-1].get("word", "")))
    if trailing and tokens and not str(tokens[-1]).endswith(trailing.group(1)):
        tokens[-1] = tokens[-1] + trailing.group(1)

    if len(tokens) == len(original_words):
        return [
            {
                **dict(original_words[index]),
                "word": token,
            }
            for index, token in enumerate(tokens)
        ]

    first = original_words[0]
    last = original_words[-1]
    start = optional_float(first.get("start"))
    end = optional_float(last.get("end"))
    if len(tokens) == 1:
        merged = dict(first)
        merged["word"] = tokens[0]
        if last.get("end") is not None:
            merged["end"] = last.get("end")
        return [merged]

    duration = (end - start) if start is not None and end is not None and end >= start else None
    replacement: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        item = dict(first)
        item["word"] = token
        if duration is not None and start is not None:
            item["start"] = start + duration * index / len(tokens)
            item["end"] = start + duration * (index + 1) / len(tokens)
        replacement.append(item)
    return replacement


def merge_corrected_name_words(
    words: list[dict[str, Any]],
    audit_row: dict[str, Any],
    raw_name: str,
    final_name: str,
) -> tuple[list[dict[str, Any]], bool]:
    span = corrected_name_word_span(words, audit_row, raw_name)
    if span is None:
        return words, False
    span_start, span_end = span
    replacement = corrected_name_word_items(words[span_start : span_end + 1], final_name)
    return words[:span_start] + replacement + words[span_end + 1 :], True


def apply_verified_phone_corrections_to_transcript(
    transcript: str,
    entities: dict[str, Any],
    audit_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    corrected_transcript = transcript
    corrected_entities = dict(entities or {})
    words = corrected_entities.get("_word_timestamps")
    corrected_words = [dict(item) for item in words] if isinstance(words, list) else []
    corrections: list[dict[str, Any]] = []
    phone_rows: list[dict[str, Any]] = []

    for row in audit_rows:
        if not isinstance(row, dict):
            continue
        if row.get("field_name") == "name":
            for raw_name, final_name in replacement_name_values(row):
                corrected_transcript, replacements = replace_attributed_name_text_in_transcript(
                    corrected_transcript,
                    corrected_words,
                    row,
                    raw_name,
                    final_name,
                )
                words_changed = False
                if corrected_words:
                    corrected_words, words_changed = merge_corrected_name_words(
                        corrected_words,
                        row,
                        raw_name,
                        final_name,
                    )
                if replacements or words_changed:
                    corrections.append(
                        {
                            "field_name": "name",
                            "raw": raw_name,
                            "value": final_name,
                            "status": row.get("status"),
                            "transcript_replacements": replacements,
                            "word_timestamps_updated": words_changed,
                        }
                    )
            continue
        if row.get("field_name") == "dob":
            continue
        if row.get("field_name") not in {"callback_number", "fax_number"}:
            continue
        if row.get("status") not in PHONE_TRANSCRIPT_CORRECTION_STATUSES:
            continue
        phone_rows.append(row)

    for row in sorted(phone_rows, key=phone_row_sort_key, reverse=True):
        normalized = str(row.get("normalized_value") or "")
        formatted = str(row.get("final_value") or format_phone_digits(normalized) or "")
        if not normalized or not formatted:
            continue

        corrected_transcript, replacements = replace_attributed_phone_text_in_transcript(
            corrected_transcript,
            corrected_words,
            row,
            normalized,
            formatted,
        )
        words_changed = False
        if corrected_words:
            corrected_words, words_changed = merge_corrected_phone_words(
                corrected_words,
                row,
                normalized,
                formatted,
            )

        if replacements or words_changed:
            corrections.append(
                {
                    "field_name": row.get("field_name"),
                    "number": formatted,
                    "normalized": normalized,
                    "status": row.get("status"),
                    "transcript_replacements": replacements,
                    "word_timestamps_updated": words_changed,
                }
            )

    if corrected_words and corrections:
        corrected_entities["_word_timestamps"] = corrected_words
    if corrections:
        corrected_entities["transcript_corrections"] = corrections

    return corrected_transcript, corrected_entities


apply_verified_phone_corrections_to_transcript = watcher_transcript_corrections.apply_verified_phone_corrections_to_transcript


def parakeet_verification_fields(fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(field for field in fields if field in {"dob", "callback_number", "fax_number"})


def build_scoped_verification_records(
    fields: tuple[str, ...],
    gemma_payload: dict[str, Any],
    transcription: TranscriptionResult,
) -> dict[str, list[CandidateRecord]]:
    return {
        field: build_candidate_records(field, gemma_payload, transcription)
        for field in fields
    }


def scoped_audit_rows(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    allowed = set(fields)
    return [row for row in rows if row.get("field_name") in allowed]


def apply_resolutions_to_entities(
    original_entities: dict[str, Any],
    resolutions: list[FieldResolution],
) -> dict[str, Any]:
    updated = dict(original_entities)
    for resolution in resolutions:
        if resolution.field_name == "name":
            updated["name"] = resolution.final_value
        elif resolution.field_name == "dob":
            updated["dob"] = resolution.final_value
        elif resolution.field_name == "callback_number":
            updated["callback_number"] = resolution.final_value
        elif resolution.field_name == "fax_number":
            updated["fax_number"] = resolution.final_value
    return updated


def verify_voicemail_fields(
    wav_path: str,
    transcription: TranscriptionResult,
    info: dict[str, str],
    settings: Settings,
    gemma_endpoint_pool: Optional[ServiceEndpointPool] = None,
    parakeet_endpoint_pool: Optional[ServiceEndpointPool] = None,
    request_id: str = "",
    event_observer: Optional[Callable[..., Any]] = None,
) -> VerificationRunResult:
    original_entities = dict(transcription.entities or {})
    verification_fields = tuple(
        getattr(settings, "verification_fields", VERIFICATION_FIELD_NAMES)
    )

    if not settings.gemma_field_extraction_enabled:
        logger.info("Field verification disabled")
        return VerificationRunResult(original_entities, [], should_apply=False, complete=False)

    logger.info("Field verification scope fields=%s", ",".join(verification_fields))

    deadline = time.monotonic() + settings.verification_total_timeout_seconds
    gemma_error_reason = ""
    gemma_status = "completed"

    notify_event_observer(event_observer, "gemma.started", metrics={"service": "Gemma"})
    try:
        gemma_payload = call_gemma_field_extraction(
            transcription,
            info,
            settings,
            deadline,
            endpoint_pool=gemma_endpoint_pool,
            request_id=request_id,
            event_observer=event_observer,
        )
    except VerificationBudgetExceeded:
        notify_event_observer(event_observer, "gemma.completed", status="failed")
        logger.warning("Field verification timed out before Gemma completed")
        return VerificationRunResult(
            original_entities,
            scoped_audit_rows(timeout_audit_rows(original_entities), verification_fields),
            should_apply=False,
            timed_out=True,
            complete=False,
        )
    except GemmaSchemaError as exc:
        logger.warning("Gemma field extraction returned invalid schema error=%s", exc)
        gemma_payload = empty_gemma_field_payload()
        gemma_error_reason = "gemma_invalid_json"
        gemma_status = "degraded"
    except Exception as exc:
        notify_event_observer(event_observer, "gemma.completed", status="failed")
        logger.warning("Gemma field extraction unavailable error=%s", exc)
        rows = scoped_audit_rows(
            unavailable_audit_rows(
                original_entities,
                "gemma_unavailable",
                settings.gemma_fail_open,
            ),
            verification_fields,
        )
        return VerificationRunResult(original_entities, rows, should_apply=False)

    compact_dob_count = (
        add_compact_dob_fallback_candidates(gemma_payload, transcription.text)
        if "dob" in verification_fields
        else 0
    )
    if compact_dob_count:
        logger.info("Compact DOB fallback added candidate_count=%s", compact_dob_count)
    near_phone_count = (
        add_near_phone_fallback_candidates(gemma_payload, transcription.text)
        if {"callback_number", "fax_number"}.intersection(verification_fields)
        else 0
    )
    if near_phone_count:
        logger.info("Near-phone fallback added candidate_count=%s", near_phone_count)
    logger.info(
        "Gemma field extraction completed candidate_counts=%s",
        {
            "patient_names": len(gemma_payload.get("patient_names", [])),
            "dob_candidates": len(gemma_payload.get("dob_candidates", [])),
            "callback_numbers": len(gemma_payload.get("callback_numbers", [])),
            "fax_numbers": len(gemma_payload.get("fax_numbers", [])),
            "uncertain_numbers": len(gemma_payload.get("uncertain_numbers", [])),
        },
    )
    candidate_count = sum(
        len(gemma_payload.get(field, []))
        for field in ("patient_names", "dob_candidates", "callback_numbers", "fax_numbers")
    )
    notify_event_observer(
        event_observer,
        "gemma.completed",
        status=gemma_status,
        metrics={"service": "Gemma", "candidate_count": candidate_count},
    )

    parakeet_started = False
    try:
        check_budget(deadline)
        records_by_field = build_scoped_verification_records(
            verification_fields,
            gemma_payload,
            transcription,
        )
        phone_records = [
            record
            for field_name in ("callback_number", "fax_number")
            for record in records_by_field.get(field_name, [])
        ]
        constrain_phone_clip_bounds_for_neighbors(phone_records)

        eligible_parakeet_records = [
            record
            for field_name in parakeet_verification_fields(verification_fields)
            for record in records_by_field.get(field_name, [])
            if record.attribution.mapped
        ]
        parakeet_event_enabled = bool(
            getattr(settings, "parakeet_verification_enabled", False)
            and eligible_parakeet_records
        )
        if parakeet_event_enabled:
            notify_event_observer(event_observer, "parakeet.started", metrics={"service": "Parakeet"})
            parakeet_started = True
        for field_name in parakeet_verification_fields(verification_fields):
            for record in records_by_field.get(field_name, []):
                check_budget(deadline)
                run_parakeet_for_record(
                    wav_path,
                    record,
                    settings,
                    deadline,
                    endpoint_pool=parakeet_endpoint_pool,
                )
        if parakeet_event_enabled:
            verified_count = sum(
                1
                for record in eligible_parakeet_records
                if record.parakeet is not None and not record.parakeet.error
            )
            parakeet_status = (
                "completed"
                if verified_count == len(eligible_parakeet_records)
                else "degraded" if verified_count else "failed"
            )
            notify_event_observer(
                event_observer,
                "parakeet.completed",
                status=parakeet_status,
                metrics={
                    "service": "Parakeet",
                    "candidate_count": len(eligible_parakeet_records),
                    "verified_count": verified_count,
                },
            )

        caller_name, caller_number = parse_callerid(info.get("callerid", ""))
        resolvers = {
            "name": lambda: resolve_name_field(records_by_field["name"], caller_name),
            "dob": lambda: resolve_dob_field(records_by_field["dob"]),
            "callback_number": lambda: resolve_phone_field(
                "callback_number",
                records_by_field["callback_number"],
                original_entities.get("callback_number"),
                caller_id_value=caller_number,
                gemma_unavailable=False,
                fail_open=settings.gemma_fail_open,
            ),
            "fax_number": lambda: resolve_phone_field(
                "fax_number",
                records_by_field["fax_number"],
                original_entities.get("fax_number"),
                gemma_unavailable=False,
                fail_open=settings.gemma_fail_open,
            ),
        }
        resolutions = [resolvers[field_name]() for field_name in verification_fields]
        if gemma_error_reason:
            adjusted_resolutions: list[FieldResolution] = []
            for resolution in resolutions:
                if not resolution.final_value and settings.gemma_fail_open:
                    resolution = resolve_legacy_field(
                        resolution.field_name,
                        original_entities.get(resolution.field_name),
                        True,
                        gemma_error_reason,
                    )
                resolution.needs_review = True
                resolution.review_reasons = list(
                    dict.fromkeys([*resolution.review_reasons, gemma_error_reason])
                )
                adjusted_resolutions.append(resolution)
            resolutions = adjusted_resolutions
    except VerificationBudgetExceeded:
        if parakeet_started:
            notify_event_observer(event_observer, "parakeet.completed", status="failed")
        logger.warning("Field verification timed out after partial work; preserving original entities")
        return VerificationRunResult(
            original_entities,
            scoped_audit_rows(timeout_audit_rows(original_entities), verification_fields),
            should_apply=False,
            timed_out=True,
            complete=False,
        )

    proposed_entities = apply_resolutions_to_entities(original_entities, resolutions)
    audit_rows = [resolution.as_audit_row() for resolution in resolutions]
    return VerificationRunResult(
        proposed_entities=proposed_entities,
        audit_rows=audit_rows,
        should_apply=settings.verification_apply_resolved_values,
        complete=True,
    )


def safe_verify_voicemail_fields(
    file_key: str,
    wav_path: str,
    transcription: TranscriptionResult,
    info: dict[str, str],
    settings: Settings,
    gemma_endpoint_pool: Optional[ServiceEndpointPool] = None,
    parakeet_endpoint_pool: Optional[ServiceEndpointPool] = None,
    event_observer: Optional[Callable[..., Any]] = None,
) -> VerificationRunResult:
    """Run verification without allowing verifier failures to block delivery."""

    try:
        return verify_voicemail_fields(
            wav_path,
            transcription,
            info,
            settings,
            gemma_endpoint_pool=gemma_endpoint_pool,
            parakeet_endpoint_pool=parakeet_endpoint_pool,
            request_id=file_key,
            event_observer=event_observer,
        )
    except Exception as exc:
        logger.warning(
            "Field verification failed; preserving original entities key=%s error=%s",
            file_key,
            exc,
        )
        original_entities = dict(transcription.entities or {})
        verification_fields = tuple(
            getattr(settings, "verification_fields", VERIFICATION_FIELD_NAMES)
        )
        return VerificationRunResult(
            proposed_entities=original_entities,
            audit_rows=scoped_audit_rows(
                unavailable_audit_rows(
                    original_entities,
                    "gemma_unavailable",
                    settings.gemma_fail_open,
                ),
                verification_fields,
            ),
            should_apply=False,
            complete=False,
        )


def notify_event_observer(
    observer: Optional[Callable[..., Any]],
    phase: str,
    **fields: Any,
) -> bool:
    """Best-effort observer hook that can never affect voicemail delivery."""
    if not callable(observer):
        return False
    try:
        observer(phase, **fields)
        return True
    except Exception:
        return False


def select_entities_for_output(
    original_entities: dict[str, Any],
    verification_result: VerificationRunResult,
    audit_written: bool,
    require_audit_for_apply: bool,
) -> dict[str, Any]:
    """Apply resolved entities only when the end-only apply gate is satisfied."""

    if not verification_apply_gate_satisfied(verification_result, audit_written, require_audit_for_apply):
        return dict(original_entities)
    return dict(verification_result.proposed_entities)


def verification_apply_gate_satisfied(
    verification_result: VerificationRunResult,
    audit_written: bool,
    require_audit_for_apply: bool,
) -> bool:
    if not verification_result.should_apply:
        return False
    if verification_result.timed_out or not verification_result.complete:
        return False
    if require_audit_for_apply and not audit_written:
        return False
    return True


class VoicemailProcessor:
    def __init__(
        self,
        settings: Settings,
        store: VoicemailStore,
    ):
        self.settings = settings
        self.store = store
        self.host_load_tracker = HostLoadTracker(settings.host_aware_routing)
        self.whisper_endpoints = WhisperEndpointPool(
            settings.whisper_urls,
            host_load_tracker=self.host_load_tracker,
        )
        self.gemma_endpoints = ServiceEndpointPool(
            "Gemma",
            settings.gemma_base_urls,
            "gemma_ready_timeout_seconds",
            host_load_tracker=self.host_load_tracker,
            kind="gemma",
            cross_kinds=("whisper",),
        )
        self.parakeet_endpoints = ServiceEndpointPool(
            "Parakeet",
            settings.parakeet_verification_urls,
            "parakeet_ready_timeout_seconds",
        )
        self.work_queue: queue.Queue[str] = queue.Queue()
        self.pending_paths: set[str] = set()
        self.pending_lock = threading.Lock()
        self.stop_event = threading.Event()

    def enqueue(self, txt_path: str, reason: str = "event") -> None:
        if not is_voicemail_txt(txt_path):
            return

        absolute_path = os.path.abspath(txt_path)
        with self.pending_lock:
            if absolute_path in self.pending_paths:
                return
            self.pending_paths.add(absolute_path)

        self.work_queue.put(absolute_path)
        logger.info("Queued voicemail metadata path reason=%s path=%s", reason, absolute_path)

    def scan_existing_inbox(self) -> int:
        count = 0
        for root, _, files in os.walk(self.settings.watch_dir):
            if "/INBOX" not in normalize_path(root):
                continue
            for name in files:
                if name.endswith(".txt") and "msg" in name:
                    self.enqueue(os.path.join(root, name), reason="startup_scan")
                    count += 1
        logger.info("Startup scan queued %s existing INBOX voicemail metadata file(s)", count)
        return count

    def start_workers(self) -> list[threading.Thread]:
        threads: list[threading.Thread] = []
        for worker_id in range(self.settings.workers):
            thread = threading.Thread(
                target=self.worker_loop,
                args=(worker_id,),
                name=f"voicemail-worker-{worker_id}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        return threads

    def worker_loop(self, worker_id: int) -> None:
        logger.info("Voicemail worker %s started", worker_id)
        while not self.stop_event.is_set():
            try:
                txt_path = self.work_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                self.process_with_retries(txt_path)
            finally:
                with self.pending_lock:
                    self.pending_paths.discard(txt_path)
                self.work_queue.task_done()

    def process_with_retries(self, txt_path: str) -> None:
        for attempt in range(1, self.settings.max_attempts + 1):
            try:
                self.process_once(txt_path, attempt=attempt)
                return
            except SkippedProcessing as exc:
                logger.info("Skipped voicemail path=%s reason=%s", txt_path, exc)
                return
            except PermanentProcessingError as exc:
                logger.error("Permanent voicemail processing failure path=%s error=%s", txt_path, exc)
                return
            except RetryableProcessingError as exc:
                if attempt >= self.settings.max_attempts:
                    logger.error(
                        "Voicemail processing exhausted local retries path=%s attempts=%s error=%s",
                        txt_path,
                        attempt,
                        exc,
                    )
                    return

                delay = self.settings.retry_delays[min(attempt - 1, len(self.settings.retry_delays) - 1)]
                logger.warning(
                    "Retryable voicemail processing failure path=%s attempt=%s delay=%ss error=%s",
                    txt_path,
                    attempt,
                    delay,
                    exc,
                )
                time.sleep(delay)
            except Exception as exc:
                logger.exception("Unexpected voicemail processing error path=%s error=%s", txt_path, exc)
                return

    def process_once(self, txt_path: str, attempt: int = 1) -> None:
        extension = extract_extension(txt_path)
        if not extension:
            return

        try:
            info, wav_path = wait_for_ready_files(txt_path, self.settings)
        except SkippedProcessing as exc:
            self.store.mark_skipped_by_path(txt_path, str(exc))
            raise
        except RetryableProcessingError as exc:
            if not os.path.exists(txt_path):
                reason = f"Source voicemail TXT missing after readiness wait: {exc}"
                marked = self.store.mark_dead_by_path(txt_path, reason)
                logger.warning(
                    "Marked missing voicemail metadata path dead path=%s rows=%s error=%s",
                    txt_path,
                    marked,
                    exc,
                )
                raise PermanentProcessingError(reason) from exc
            raise

        file_key = build_file_key(extension, info, txt_path)
        if not file_key:
            raise RetryableProcessingError("Voicemail metadata does not contain a stable file key yet")
        legacy_key = build_legacy_file_key(extension, info, txt_path)
        if legacy_key:
            self.store.migrate_legacy_key(legacy_key, file_key)

        status = self.store.discover(file_key, extension, txt_path, wav_path)
        if status in TERMINAL_STATUSES:
            logger.info("Voicemail already terminal status=%s key=%s", status, file_key)
            return

        if is_before_process_cutoff(info, self.settings):
            reason = (
                "Voicemail origtime "
                f"{info.get('origtime', '').strip()} is before configured cutoff "
                f"{self.settings.process_after_origtime}"
            )
            self.store.mark_skipped(file_key, reason)
            logger.info("Skipped old voicemail key=%s reason=%s", file_key, reason)
            return

        if not self.store.claim(file_key):
            logger.info("Voicemail is already being handled by another worker key=%s", file_key)
            return

        try:
            duration = int(info.get("duration", "0"))
            if duration < self.settings.min_duration_seconds:
                reason = f"Voicemail duration {duration}s below minimum {self.settings.min_duration_seconds}s"
                self.store.mark_skipped(file_key, reason)
                logger.info("Skipped short voicemail key=%s duration=%ss", file_key, duration)
                return

            recipients = get_email(extension, self.settings) if self.settings.email_enabled else []
            logger.info(
                "Processing voicemail key=%s extension=%s recipient_count=%s duration=%ss",
                file_key,
                extension,
                len(recipients),
                duration,
            )
            transcription = transcribe(
                wav_path,
                file_key,
                self.settings,
                self.whisper_endpoints.least_busy_order(self.settings, file_key),
                self.whisper_endpoints,
            )
            original_entities = dict(transcription.entities or {})
            # Persist the Whisper transcript before slower verification work. This
            # keeps the portal/diagnostics useful if Gemma or Parakeet is slow or
            # the watcher is stopped mid-verification. Spelling corrections are
            # applied only to copies so downstream verification still receives the
            # untouched Whisper result.
            provisional_transcript, provisional_entities = apply_final_mailbox_spelling_rules(
                info.get("origmailbox", extension) or extension,
                transcription.text,
                original_entities,
                self.settings,
                file_key=file_key,
                stage="provisional",
            )
            self.store.upsert_transcript(
                file_key,
                extension,
                txt_path,
                wav_path,
                info,
                provisional_transcript,
                provisional_entities,
            )
            verification_result = safe_verify_voicemail_fields(
                file_key,
                wav_path,
                transcription,
                info,
                self.settings,
                gemma_endpoint_pool=self.gemma_endpoints,
                parakeet_endpoint_pool=self.parakeet_endpoints,
            )
            audit_written = False
            if verification_result.audit_rows:
                try:
                    self.store.upsert_field_verifications(file_key, verification_result.audit_rows)
                    audit_written = True
                except Exception as exc:
                    logger.warning("Could not write field verification audit key=%s error=%s", file_key, exc)

            apply_gate_ok = verification_apply_gate_satisfied(
                verification_result,
                audit_written,
                self.settings.verification_require_audit_for_apply,
            )
            entities_for_output = select_entities_for_output(
                original_entities,
                verification_result,
                audit_written,
                self.settings.verification_require_audit_for_apply,
            )
            if verification_result.should_apply and entities_for_output == original_entities:
                logger.warning(
                    "Preserving original entities because verification apply gate was not satisfied key=%s",
                    file_key,
                )
            transcript_for_output = transcription.text
            if apply_gate_ok:
                transcript_for_output, entities_for_output = apply_verified_phone_corrections_to_transcript(
                    transcript_for_output,
                    entities_for_output,
                    verification_result.audit_rows,
                )
            corrected_transcript, transcript_corrections = run_transcript_lattice_audit(
                file_key,
                wav_path,
                transcription,
                transcript_for_output,
                self.settings,
                self.store,
                self.gemma_endpoints,
                self.parakeet_endpoints,
            )
            if transcript_corrections:
                entities_for_output = dict(entities_for_output)
                entities_for_output["_transcript_corrections"] = transcript_corrections
                entities_for_output["_corrected_transcript"] = corrected_transcript
                if self.settings.transcript_lattice_apply_enabled:
                    transcript_for_output = corrected_transcript

            transcript_for_output, entities_for_output = apply_final_mailbox_spelling_rules(
                info.get("origmailbox", extension) or extension,
                transcript_for_output,
                entities_for_output,
                self.settings,
                file_key=file_key,
                stage="final",
            )

            self.store.upsert_transcript(
                file_key,
                extension,
                txt_path,
                wav_path,
                info,
                transcript_for_output,
                entities_for_output,
            )
            if not self.settings.email_enabled:
                logger.info("Email delivery disabled by VOICEMAIL_EMAIL_ENABLED=false key=%s", file_key)
            elif self.store.already_emailed(file_key):
                logger.warning("Skipping duplicate email for already emailed voicemail key=%s", file_key)
            else:
                send_email(
                    recipients,
                    file_key,
                    extension,
                    wav_path,
                    info,
                    transcript_for_output,
                    entities_for_output,
                    self.settings,
                )
                self.store.mark_emailed(file_key)
            self.store.mark_completed(file_key, len(transcript_for_output))
            logger.info(
                "Completed voicemail key=%s transcript_chars=%s",
                file_key,
                len(transcript_for_output),
            )

        except PermanentProcessingError as exc:
            self.store.mark_retry_or_dead(file_key, str(exc), 1)
            raise
        except RetryableProcessingError as exc:
            status = self.store.mark_retry_or_dead(file_key, str(exc), self.settings.max_attempts)
            if status == STATUS_DEAD:
                raise PermanentProcessingError(f"Voicemail moved to dead-letter: {exc}") from exc
            raise
        except Exception as exc:
            status = self.store.mark_retry_or_dead(file_key, str(exc), self.settings.max_attempts)
            if status == STATUS_DEAD:
                raise PermanentProcessingError(f"Voicemail moved to dead-letter: {exc}") from exc
            raise RetryableProcessingError(str(exc)) from exc


class VoicemailHandler(FileSystemEventHandler):
    def __init__(self, processor: VoicemailProcessor):
        self.processor = processor

    def _handle_path(self, path: str, reason: str) -> None:
        try:
            self.processor.enqueue(path, reason=reason)
        except Exception as exc:
            logger.exception("Unhandled watcher event error path=%s reason=%s error=%s", path, reason, exc)

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._handle_path(event.src_path, "created")

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._handle_path(event.src_path, "modified")

    def on_moved(self, event) -> None:
        if not event.is_directory:
            self._handle_path(event.dest_path, "moved")


def validate_startup_dependencies(settings: Settings) -> None:
    if requests is None:
        raise RuntimeError("Missing required dependency: requests")
    if Observer is None:
        raise RuntimeError("Missing required dependency: watchdog")
    if not os.path.isdir(settings.watch_dir):
        raise RuntimeError(f"Voicemail watch directory does not exist: {settings.watch_dir}")

    # Validate the configured timezone during startup, not during the first email.
    get_local_timezone(settings.local_timezone, datetime.now(timezone.utc))

    if settings.email_enabled and not settings.fallback_recipient:
        raise RuntimeError(
            "VOICEMAIL_FALLBACK_RECIPIENT must be explicitly set when VOICEMAIL_EMAIL_ENABLED=true"
        )

    if not settings.whisper_api_key:
        logger.warning("WHISPER_API_KEY is not set; Whisper requests will be unauthenticated")


def cleanup_verification_clips(settings: Settings) -> int:
    if settings.verification_clip_retention_days <= 0:
        return 0
    clip_dir = settings.verification_clip_dir
    if not clip_dir or not os.path.isdir(clip_dir):
        return 0
    cutoff = time.time() - settings.verification_clip_retention_days * 24 * 3600
    removed = 0
    for path in Path(clip_dir).glob("*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as exc:
            logger.warning("Could not remove old verification clip path=%s error=%s", path, exc)
    return removed


def main() -> int:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    from voicemail_watcher.providers import build_provider_registry

    providers = build_provider_registry(os.environ)
    if providers.external_phi_warning:
        logger.warning(providers.external_phi_warning)
    validate_startup_dependencies(settings)
    removed_clips = cleanup_verification_clips(settings)
    if removed_clips:
        logger.info("Removed %s old verification clip(s)", removed_clips)

    logger.info(
        "Starting voicemail watcher watch_dir=%s state_db=%s whisper_endpoints=%s gemma_endpoints=%s parakeet_endpoints=%s providers=%s/%s/%s",
        settings.watch_dir,
        settings.state_db,
        len(settings.whisper_urls),
        len(settings.gemma_base_urls),
        len(settings.parakeet_verification_urls),
        providers.transcription.name,
        providers.extraction.name,
        providers.verification.name,
    )
    store = VoicemailStore(settings.state_db)
    reset_count = store.reset_interrupted_jobs()
    if reset_count:
        logger.warning("Reset %s interrupted voicemail job(s) to retry", reset_count)

    processor = VoicemailProcessor(settings, store)
    workers = processor.start_workers()

    if settings.startup_scan:
        processor.scan_existing_inbox()

    observer = Observer()
    observer.schedule(VoicemailHandler(processor), settings.watch_dir, recursive=True)
    observer.start()
    logger.info("Watching %s with %s worker(s)", settings.watch_dir, len(workers))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        processor.stop_event.set()
        observer.stop()
        observer.join(timeout=15)
        logger.info("Voicemail watcher stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
