#!/usr/bin/env python3
"""
Local Whisper API Server for VitalPBX.

The service accepts voicemail/call-recording audio over HTTP, transcribes it
with faster-whisper, and prioritizes voicemail jobs over call recordings.

Production hardening included here:
- environment-validated runtime settings
- optional bearer/API-key authentication
- bounded queues with explicit 429 rejection
- request handlers that do not block the FastAPI event loop
- worker-owned temp-file cleanup, including client-timeout cases
- readiness/health endpoints with operational state
- PHI-safe logging: no transcript text is logged
- safer upload streaming, size limits, and basic container signature checks
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import queue
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

try:
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse
except ImportError:  # pragma: no cover - production dependency
    FastAPI = None
    File = Form = Header = None
    UploadFile = Any
    JSONResponse = PlainTextResponse = None

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

try:
    import uvicorn
except ImportError:  # pragma: no cover - production dependency
    uvicorn = None


APP_VERSION = "3.0.0"
TEMP_PREFIX = "whisper-api-"


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def normalize_extensions(raw: str) -> set[str]:
    extensions: set[str] = set()
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        extensions.add(item)
    if not extensions:
        raise ValueError("WHISPER_ALLOWED_EXTENSIONS must contain at least one extension")
    return extensions


@dataclass(frozen=True)
class Settings:
    model_size: str
    model_device: str
    model_compute_type: str
    model_local_files_only: bool
    model_cpu_threads: int
    model_num_workers: int
    total_workers: int
    voicemail_workers: int
    recording_workers: int
    host: str
    port: int
    environment: str
    expected_ip: str
    api_key: str
    require_auth: bool
    voicemail_timeout_seconds: int
    recording_timeout_seconds: int
    max_upload_mb: int
    max_upload_bytes: int
    upload_chunk_size: int
    max_queue_depth: int
    min_tmp_free_bytes: int
    stale_temp_max_age_seconds: int
    allowed_extensions: set[str]
    default_language: Optional[str]
    initial_prompt: str
    initial_prompt_file: str
    model_lock_enabled: bool
    phone_second_pass_enabled: bool
    phone_second_pass_max_windows: int
    phone_second_pass_padding_ms: int
    word_timestamps_enabled: bool
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        total_workers = env_int("WHISPER_WORKERS", 1, minimum=1)
        voicemail_workers = env_int("VOICEMAIL_WORKERS", 1, minimum=1)
        if voicemail_workers > total_workers:
            voicemail_workers = total_workers
        recording_workers = max(0, total_workers - voicemail_workers)

        max_upload_mb = env_int("WHISPER_MAX_UPLOAD_MB", 50, minimum=1)
        api_key = os.environ.get("WHISPER_API_KEY", "").strip()

        default_prompt = (
            "Common voicemail words: appointment, referral, prescription, "
            "therapy, callback, date of birth, and fax."
        )

        return cls(
            model_size=os.environ.get("WHISPER_MODEL", "base"),
            model_device=os.environ.get("WHISPER_DEVICE", "cpu"),
            model_compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
            model_local_files_only=env_bool("WHISPER_LOCAL_FILES_ONLY", False),
            model_cpu_threads=env_int("WHISPER_CPU_THREADS", 16, minimum=1),
            model_num_workers=env_int("WHISPER_MODEL_NUM_WORKERS", 1, minimum=1),
            total_workers=total_workers,
            voicemail_workers=voicemail_workers,
            recording_workers=recording_workers,
            host=os.environ.get("WHISPER_HOST", "127.0.0.1"),
            port=env_int("WHISPER_PORT", 8765, minimum=1),
            environment=os.environ.get("WHISPER_ENV", "unspecified"),
            expected_ip=os.environ.get("WHISPER_EXPECTED_IP", ""),
            api_key=api_key,
            require_auth=env_bool("WHISPER_REQUIRE_AUTH", bool(api_key)),
            voicemail_timeout_seconds=env_int("WHISPER_VOICEMAIL_TIMEOUT", 120, minimum=1),
            recording_timeout_seconds=env_int("WHISPER_RECORDING_TIMEOUT", 300, minimum=1),
            max_upload_mb=max_upload_mb,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            upload_chunk_size=env_int("WHISPER_UPLOAD_CHUNK_SIZE", 1024 * 1024, minimum=1024),
            max_queue_depth=env_int("WHISPER_MAX_QUEUE_DEPTH", 25, minimum=1),
            min_tmp_free_bytes=env_int("WHISPER_MIN_TMP_FREE_MB", 1024, minimum=1) * 1024 * 1024,
            stale_temp_max_age_seconds=env_int("WHISPER_STALE_TEMP_MAX_AGE_SECONDS", 24 * 3600, minimum=60),
            allowed_extensions=normalize_extensions(
                os.environ.get(
                    "WHISPER_ALLOWED_EXTENSIONS",
                    ".wav,.mp3,.m4a,.flac,.ogg,.oga,.webm",
                )
            ),
            default_language=os.environ.get("WHISPER_DEFAULT_LANGUAGE", "en").strip() or None,
            initial_prompt=os.environ.get("WHISPER_INITIAL_PROMPT", default_prompt),
            initial_prompt_file=os.environ.get("WHISPER_INITIAL_PROMPT_FILE", "").strip(),
            model_lock_enabled=env_bool("WHISPER_MODEL_LOCK", total_workers > 1),
            phone_second_pass_enabled=env_bool("WHISPER_PHONE_SECOND_PASS", True),
            phone_second_pass_max_windows=env_int("WHISPER_PHONE_SECOND_PASS_MAX_WINDOWS", 4, minimum=0),
            phone_second_pass_padding_ms=env_int("WHISPER_PHONE_SECOND_PASS_PADDING_MS", 1800, minimum=0),
            word_timestamps_enabled=env_bool("WHISPER_WORD_TIMESTAMPS", True),
            log_level=os.environ.get("WHISPER_LOG_LEVEL", "INFO").upper(),
        )


SETTINGS = Settings.from_env()

logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("whisper_api")


PRIORITY_VOICEMAIL = 1
PRIORITY_RECORDING = 10

SERVICE_STARTED_MONOTONIC = time.monotonic()
SERVICE_STARTED_UTC = datetime.now(timezone.utc).isoformat()

stats_lock = threading.Lock()
stats = {
    "submitted": 0,
    "completed": 0,
    "failed": 0,
    "timed_out": 0,
    "rejected_queue_full": 0,
    "rejected_upload": 0,
    "bytes_received": 0,
    "last_success_utc": None,
    "last_error_utc": None,
    "last_error": None,
    "last_job_seconds": None,
    "last_language": None,
    "last_language_probability": None,
}

job_sequence = itertools.count()


@dataclass(order=True)
class TranscriptionJob:
    priority: int
    sequence: int

    job_id: str = field(compare=False)
    audio_path: str = field(compare=False)
    language: Optional[str] = field(compare=False)
    result_event: threading.Event = field(compare=False)

    kind: str = field(compare=False, default="voicemail")
    upload_bytes: int = field(compare=False, default=0)
    created_monotonic: float = field(compare=False, default_factory=time.monotonic)

    result: Optional[str] = field(compare=False, default=None)
    raw_text: Optional[str] = field(compare=False, default=None)
    segments: list[dict[str, Any]] = field(compare=False, default_factory=list)
    entities: dict[str, Any] = field(compare=False, default_factory=dict)
    word_timestamps: list[dict[str, Any]] = field(compare=False, default_factory=list)
    error: Optional[str] = field(compare=False, default=None)


@dataclass
class PhoneVerificationCandidate:
    start: float
    end: float
    kind: str
    text: str


voicemail_queue: queue.PriorityQueue[TranscriptionJob] = queue.PriorityQueue(
    maxsize=SETTINGS.max_queue_depth
)
recording_queue: queue.PriorityQueue[TranscriptionJob] = queue.PriorityQueue(
    maxsize=SETTINGS.max_queue_depth
)

whisper_model: Any = None
transcribe_lock = threading.Lock()


class QueueFullError(Exception):
    pass


class TranscriptionFailedError(Exception):
    pass


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

PHONE_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"call me at|call us at|"
    r"call me back at|call us back at|"
    r"return my call to|return our call to|"
    r"give me a call at|give us a call at|"
    r"reach me at|you can reach me at|"
    r"my number is|number is|"
    r"my phone number is|phone number is|"
    r"my callback number is|callback number is|call back number is|"
    r"cell number is|mobile number is|"
    r"area code"
    r")\b(?P<tail>[^.\n]{0,100})"
)

PHONE_SECOND_PASS_PROMPT = (
    "Transcribe only phone numbers and fax numbers from this audio. "
    "Use digits only. If the speaker says oh, O, or zero, write 0. "
    "Keep every spoken digit in order. Do not write words."
)

PHONE_VERIFICATION_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"fax|faxed|faxing|send it to|sent over|send over|"
    r"phone|number|callback|call back|call me|call us|"
    r"give me a call|give us a call|reach me|reach us|"
    r"return my call|return our call|area code"
    r")\b"
)

PHONE_VERIFICATION_FAX_RE = re.compile(
    r"(?i)\b(fax|faxed|faxing|send it to|sent over|send over)\b"
)

PHONE_VERIFICATION_CALLBACK_RE = re.compile(
    r"(?i)\b("
    r"phone|callback|call back|call me|call us|give me a call|give us a call|"
    r"reach me|reach us|return my call|return our call|questions|number"
    r")\b"
)

NUMBER_WORD_RE = re.compile(
    r"(?i)\b(zero|oh|o|one|two|three|four|five|six|seven|eight|nine)\b"
)

TRANSCRIPT_NUMBER_WORD_RE = re.compile(
    r"(?i)\b(zero|one|two|three|four|five|six|seven|eight|nine)\b"
)

PHONE_NUMBER_JOINED_WORD_RE = re.compile(
    r"(?P<number>(?:\(\d{3}\)\s*|\b\d{3}[-.\s])\d{3}[-.\s]\d{4})(?P<word>[A-Za-z])"
)

PHONE_CANDIDATE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s().-]*)?(?:\(?\d{3}\)?[\s.-]*)\d{3}[\s.-]*\d{4}(?!\d)"
)

DOB_VALUE_RE = re.compile(
    r"(?i)\b(?:date of birth|birth date|birthdate|birthday|dob)\b[^\n\d]{0,40}"
    r"(?P<dob>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
)

IMPLICIT_DOB_AFTER_NAME_RE = re.compile(
    r"(?i)\b(?:my name is|this is)\b[^.\n]{0,80}?"
    r"(?P<dob>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
)

IMPLICIT_COMPACT_DOB_AFTER_NAME_RE = re.compile(
    r"(?i)\b(?:my name is|this is)\b[^.\n]{0,80}?"
    r"(?P<dob>\d{5,8})(?!\d)"
)

BARE_OLD_YEAR_DATE_RE = re.compile(
    r"(?P<dob>\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b)"
    r"(?=\s*[,.;!?]|\s*$)"
)

NAME_CONTEXT_RE = re.compile(
    r"(?i)\b(?:hi[, ]+)?(?:this is|my name is|name is|patient(?:'s)? name is)\s+"
    r"(?P<name>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4})"
)

PATIENT_NAME_PATTERNS = [
    re.compile(
        r"(?i)\bon\s+behalf\s+of\s+(?:my|his|her|their|our)\s+"
        r"(?:father|mother|dad|mom|parent|wife|husband|spouse|son|daughter|brother|sister)"
        r"\s*,\s*"
        r"(?P<name>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4})"
    ),
    re.compile(
        r"(?i)\bon\s+behalf\s+of\s+(?!my\b|his\b|her\b|their\b|our\b)"
        r"(?P<name>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4})"
    ),
    re.compile(
        r"\bregarding\s+"
        r"(?P<name>[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){1,3})"
        r"(?=\s*[,.;]|\s+(?:and|just|who|whose|with|for|about|because|since)\b)"
    ),
    re.compile(
        r"\brequest\s+for\s+"
        r"(?:[A-Za-z0-9][A-Za-z0-9'-]*\s+){0,8}?"
        r"for\s+"
        r"(?P<name>[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){1,3})"
        r"(?=\s*[,.;]|\s+(?:and|who|whose|with|about|because|since)\b)"
    ),
    re.compile(
        r"(?i:\b(?:received|receive|got|have|had|sent|send)?\s*"
        r"(?:an?\s+)?(?:order|referral|request|note|records?)\s+for\s+(?:a\s+|the\s+)?)"
        r"(?P<name>[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){1,3})"
        r"(?=\s*[,.;]|\s+(?:and|who|whose|with|about|because|since)\b)"
    ),
    re.compile(
        r"(?i)\b(?:mutual\s+)?patient(?:\s+whose\s+name\s+is|\s+name\s+is|,)?\s+"
        r"(?P<name>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4})"
    ),
    re.compile(
        r"(?i)\b(?:for|regarding|about)\s+(?:a\s+|the\s+)?(?:MECO\s+)?patient,?\s+"
        r"(?P<name>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4})"
    ),
    re.compile(
        r"(?i)\b(?:whose\s+name\s+is|patient(?:'s)?\s+name\s+is)\s+"
        r"(?P<name>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4})"
    ),
    re.compile(
        r"(?i)\b(?:order|referral|request|note|records?)\s+"
        r"(?:that\s+we\s+received\s+)?on\s+(?:a\s+|the\s+)?"
        r"(?P<name>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4})"
    ),
    re.compile(
        r"\b(?:post[- ]?op\s+)?report\s+on\s+"
        r"(?P<name>[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){1,3})"
        r"(?=\s*[,.;]|\s+(?:and|who|whose|with|for|about|because|since)\b)"
    ),
    re.compile(
        r"\b(?:calling\s+)?in\s+regards?\s+to\s+"
        r"(?P<name>[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){1,3})"
    ),
]

TRAILING_NAME_WORDS = {
    "again",
    "calling",
    "callin",
    "from",
    "with",
    "about",
    "regarding",
    "i",
    "im",
    "i'm",
    "need",
    "have",
    "had",
    "was",
    "male",
    "female",
    "date",
}

INVALID_NAME_STARTS = {
    "a",
    "an",
    "the",
    "calling",
    "regarding",
    "about",
    "for",
    "from",
    "of",
    "to",
    "doctor",
    "dr",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def words_to_digits(text: str) -> str:
    return NUMBER_WORD_RE.sub(lambda m: DIGIT_WORDS[m.group(1).lower()], text)


def transcript_words_to_digits(text: str) -> str:
    return TRANSCRIPT_NUMBER_WORD_RE.sub(lambda m: DIGIT_WORDS[m.group(1).lower()], text)


def optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_timestamp_word(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def timestamp_alignment_key(value: Any) -> str:
    text = clean_timestamp_word(value).lower()
    text = words_to_digits(text)
    return re.sub(r"[^a-z0-9]+", "", text)


def collect_word_timestamps(segments: list[Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []

    for segment in segments:
        segment_words = getattr(segment, "words", None)
        if not segment_words:
            continue

        for item in segment_words:
            word = clean_timestamp_word(getattr(item, "word", ""))
            start = optional_float(getattr(item, "start", None))
            end = optional_float(getattr(item, "end", None))
            if not word or start is None or end is None or start < 0 or end < start:
                continue

            words.append(
                {
                    "word": word,
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )

    return words


def collect_segments(segments: list[Any]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    for segment in segments:
        start = optional_float(getattr(segment, "start", None))
        end = optional_float(getattr(segment, "end", None))
        text = clean_timestamp_word(getattr(segment, "text", ""))
        if start is None or end is None or start < 0 or end < start:
            continue

        words: list[dict[str, Any]] = []
        segment_words = getattr(segment, "words", None)
        if segment_words:
            for item in segment_words:
                word = clean_timestamp_word(getattr(item, "word", ""))
                word_start = optional_float(getattr(item, "start", None))
                word_end = optional_float(getattr(item, "end", None))
                if not word or word_start is None or word_end is None or word_start < 0 or word_end < word_start:
                    continue
                words.append(
                    {
                        "word": word,
                        "start": round(word_start, 3),
                        "end": round(word_end, 3),
                    }
                )

        collected.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "words": words,
            }
        )

    return collected


def align_word_timestamps_to_text(
    text: str,
    raw_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not text or not raw_words:
        return []

    transcript_tokens = re.findall(r"\S+", text)
    aligned: list[dict[str, Any]] = []
    raw_index = 0
    max_exact_lookahead = 8
    max_group_lookahead = 18

    for token in transcript_tokens:
        token_key = timestamp_alignment_key(token)
        if not token_key:
            continue

        matched_start: Optional[float] = None
        matched_end: Optional[float] = None
        matched_next_index: Optional[int] = None

        for index in range(raw_index, min(len(raw_words), raw_index + max_exact_lookahead)):
            raw_key = timestamp_alignment_key(raw_words[index].get("word"))
            if raw_key and raw_key == token_key:
                matched_start = optional_float(raw_words[index].get("start"))
                matched_end = optional_float(raw_words[index].get("end"))
                matched_next_index = index + 1
                break

        if matched_start is None:
            combined = ""
            group_start: Optional[float] = None
            group_end: Optional[float] = None
            for index in range(raw_index, min(len(raw_words), raw_index + max_group_lookahead)):
                raw_key = timestamp_alignment_key(raw_words[index].get("word"))
                if not raw_key:
                    continue
                if group_start is None:
                    group_start = optional_float(raw_words[index].get("start"))
                group_end = optional_float(raw_words[index].get("end"))
                combined += raw_key

                if combined == token_key:
                    matched_start = group_start
                    matched_end = group_end
                    matched_next_index = index + 1
                    break
                if not token_key.startswith(combined):
                    break

        if matched_start is None and raw_index < len(raw_words):
            matched_start = optional_float(raw_words[raw_index].get("start"))
            matched_end = optional_float(raw_words[raw_index].get("end"))
            matched_next_index = raw_index + 1

        if matched_start is None or matched_end is None:
            continue

        raw_index = matched_next_index if matched_next_index is not None else raw_index
        aligned.append(
            {
                "word": token,
                "start": round(matched_start, 3),
                "end": round(matched_end, 3),
            }
        )

    return aligned


def format_us_phone(digits: str) -> str:
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]

    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

    if len(digits) == 7:
        return f"{digits[:3]}-{digits[3:]}"

    return digits


def normalize_callback_numbers(text: str) -> str:
    def repl(match: re.Match) -> str:
        prefix = match.group(1)
        tail = words_to_digits(match.group("tail"))

        candidates = re.findall(r"(?:\d[\s().-]*){7,11}", tail)
        if not candidates:
            return match.group(0)

        raw = candidates[0]
        trailing_space = re.search(r"\s+$", raw)
        raw_number = raw.rstrip()
        digits = re.sub(r"\D", "", raw_number)
        formatted = format_us_phone(digits)

        if formatted == digits:
            return match.group(0)

        replacement = formatted + (trailing_space.group(0) if trailing_space else "")
        new_tail = tail.replace(raw, replacement, 1)
        return f"{prefix}{new_tail}"

    return PHONE_CONTEXT_RE.sub(repl, text)


def normalize_phone_number_spacing(text: str) -> str:
    return PHONE_NUMBER_JOINED_WORD_RE.sub(
        lambda match: f"{match.group('number')} {match.group('word')}",
        text,
    )


def format_phone_digits(digits: str) -> Optional[str]:
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]

    if len(digits) != 10:
        return None

    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def extract_phone_numbers(text: str) -> list[dict[str, Any]]:
    numbers: list[dict[str, Any]] = []
    seen: set[str] = set()

    for match in PHONE_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        formatted = format_phone_digits(digits)
        if not formatted or formatted in seen:
            continue

        prior_boundaries = [
            text.rfind(boundary, 0, match.start())
            for boundary in (".", "!", "?", "\n")
        ]
        next_boundaries = [
            text.find(boundary, match.end())
            for boundary in (".", "!", "?", "\n")
        ]
        context_start = max(prior_boundaries) + 1
        future_boundaries = [pos for pos in next_boundaries if pos != -1]
        context_end = min(future_boundaries) if future_boundaries else len(text)
        context = text[context_start:context_end].lower()

        numbers.append(
            {
                "number": formatted,
                "is_fax": bool(re.search(r"\b(fax|sent over|send over|order sent|order be sent)\b", context)),
                "is_callback": bool(
                    re.search(
                        r"\b(call|callback|call back|reach|phone|cell|mobile|number|questions)\b",
                        context,
                    )
                ),
            }
        )
        seen.add(formatted)

    return numbers


def extract_phone_numbers_from_verification_text(text: str) -> list[str]:
    normalized = words_to_digits(text)
    numbers: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(r"(?:\d[\s().-]*){7,11}", normalized):
        digits = re.sub(r"\D", "", match.group(0))
        formatted = format_phone_digits(digits)
        if formatted and formatted not in seen:
            numbers.append(formatted)
            seen.add(formatted)

    return numbers


def phone_digit_distance(left: Optional[str], right: Optional[str]) -> Optional[int]:
    left_digits = re.sub(r"\D", "", left or "")
    right_digits = re.sub(r"\D", "", right or "")

    if len(left_digits) == 11 and left_digits.startswith("1"):
        left_digits = left_digits[1:]
    if len(right_digits) == 11 and right_digits.startswith("1"):
        right_digits = right_digits[1:]

    if len(left_digits) != 10 or len(right_digits) != 10:
        return None

    return sum(1 for left_digit, right_digit in zip(left_digits, right_digits) if left_digit != right_digit)


def replace_close_or_add_phone(numbers: list[str], verified_number: str) -> None:
    for index, existing in enumerate(numbers):
        distance = phone_digit_distance(existing, verified_number)
        if distance is not None and distance <= 2:
            numbers[index] = verified_number
            return

    if verified_number not in numbers:
        numbers.append(verified_number)


def is_close_phone_match(left: Optional[str], right: Optional[str]) -> bool:
    distance = phone_digit_distance(left, right)
    return distance is not None and distance <= 2


def phone_number_vote_winner(
    first_number: Optional[str],
    second_number: Optional[str],
    third_number: Optional[str],
) -> Optional[str]:
    votes = [
        number
        for number in (first_number, second_number, third_number)
        if number
    ]

    for number in votes:
        if votes.count(number) >= 2:
            return number

    return None


def select_majority_phone_numbers(
    first_numbers: list[str],
    second_numbers: list[str],
    third_numbers: Optional[list[str]] = None,
) -> list[str]:
    if third_numbers is None:
        return second_numbers

    selected: list[str] = []
    max_len = max(len(first_numbers), len(second_numbers), len(third_numbers))

    for index in range(max_len):
        first = first_numbers[index] if index < len(first_numbers) else None
        second = second_numbers[index] if index < len(second_numbers) else None
        third = third_numbers[index] if index < len(third_numbers) else None
        winner = phone_number_vote_winner(first, second, third)

        if winner and winner not in selected:
            selected.append(winner)

    return selected


def select_majority_phone_verifications(
    first_numbers: list[str],
    second_numbers: list[str],
    third_numbers: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    if third_numbers is None:
        return [{"number": number} for number in second_numbers]

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    max_len = max(len(first_numbers), len(second_numbers), len(third_numbers))

    for index in range(max_len):
        first = first_numbers[index] if index < len(first_numbers) else None
        second = second_numbers[index] if index < len(second_numbers) else None
        third = third_numbers[index] if index < len(third_numbers) else None
        winner = phone_number_vote_winner(first, second, third)

        if not winner or winner in seen:
            continue

        verification = {"number": winner}
        if first and first != winner:
            verification["original_number"] = first
        selected.append(verification)
        seen.add(winner)

    return selected


def needs_phone_tiebreaker(first_numbers: list[str], second_numbers: list[str]) -> bool:
    for index, second in enumerate(second_numbers):
        first = first_numbers[index] if index < len(first_numbers) else None
        if first and first != second:
            return True

    return False


def apply_phone_verifications(
    entities: dict[str, Any],
    verifications: list[dict[str, str]],
) -> dict[str, Any]:
    if not verifications:
        return entities

    updated = dict(entities)
    phone_numbers = list(updated.get("phone_numbers") or [])
    callback_number = updated.get("callback_number")
    fax_number = updated.get("fax_number")
    applied: list[dict[str, str]] = []

    for verification in verifications:
        number = verification.get("number", "")
        kind = verification.get("kind", "unknown")
        if not format_phone_digits(re.sub(r"\D", "", number)):
            continue

        replace_close_or_add_phone(phone_numbers, number)

        if kind == "fax":
            if fax_number:
                if not is_close_phone_match(fax_number, number):
                    continue
            elif callback_number == number:
                continue
            fax_number = number
        elif kind == "callback":
            if callback_number:
                if not is_close_phone_match(callback_number, number):
                    continue
            elif fax_number == number:
                continue
            callback_number = number
        else:
            accepted = False
            if is_close_phone_match(callback_number, number):
                callback_number = number
                accepted = True
            if is_close_phone_match(fax_number, number):
                fax_number = number
                accepted = True
            if not accepted:
                continue

        applied.append({"kind": kind, "number": number})

    updated["callback_number"] = callback_number
    updated["fax_number"] = fax_number
    updated["phone_numbers"] = phone_numbers
    updated["phone_number_verifications"] = applied
    return updated


def apply_phone_verification_transcript_corrections(
    text: str,
    verifications: list[dict[str, str]],
) -> str:
    corrections: dict[str, str] = {}

    for verification in verifications:
        original = verification.get("original_number")
        verified = verification.get("number")
        if not original or not verified or original == verified:
            continue

        original_formatted = format_phone_digits(re.sub(r"\D", "", original))
        verified_formatted = format_phone_digits(re.sub(r"\D", "", verified))
        if original_formatted and verified_formatted:
            corrections[original_formatted] = verified_formatted

    if not corrections:
        return text

    def repl(match: re.Match) -> str:
        candidate = format_phone_digits(re.sub(r"\D", "", match.group(0)))
        if candidate in corrections:
            return corrections[candidate]
        return match.group(0)

    return PHONE_CANDIDATE_RE.sub(repl, text)


def clean_extracted_name(raw_name: str) -> Optional[str]:
    name = re.sub(r"\s+", " ", raw_name).strip(" ,.;:-")
    words = name.split()

    while words and words[-1].lower().strip(".,") in TRAILING_NAME_WORDS:
        words.pop()

    if not words:
        return None

    if words[0].lower().strip(".,") in INVALID_NAME_STARTS:
        return None

    cleaned = " ".join(words).strip(" ,.;:-")
    if not cleaned or any(char.isdigit() for char in cleaned):
        return None

    return cleaned


def extract_name(text: str) -> Optional[str]:
    for pattern in PATIENT_NAME_PATTERNS:
        for match in pattern.finditer(text):
            name = clean_extracted_name(match.group("name"))
            if name:
                return name

    for match in NAME_CONTEXT_RE.finditer(text):
        name = clean_extracted_name(match.group("name"))
        if name:
            return name
    return None


def normalize_date_token(value: str) -> str:
    return value.replace("-", "/")


def parse_compact_date_token(value: str) -> Optional[tuple[int, int, int]]:
    digits = re.sub(r"\D", "", value)

    if len(digits) == 8:
        return int(digits[:2]), int(digits[2:4]), int(digits[4:])
    if len(digits) == 6:
        return int(digits[:2]), int(digits[2:4]), int(digits[4:])
    if len(digits) == 5:
        return int(digits[0]), int(digits[1:3]), int(digits[3:])

    return None


def parse_separated_date_token(value: str) -> Optional[tuple[int, int, int]]:
    parts = re.split(r"[/-]", value)
    if len(parts) != 3:
        return None

    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def is_plausible_standalone_dob(month: int, day: int, year: int) -> bool:
    normalized_year = normalize_two_digit_year(year)
    current_year = datetime.now().year
    return (
        1900 <= normalized_year <= current_year - 10
        and is_valid_calendar_date(month, day, normalized_year)
    )


def extract_dob(text: str) -> Optional[str]:
    explicit = DOB_VALUE_RE.search(text)
    if explicit:
        return normalize_date_token(explicit.group("dob"))

    implicit = IMPLICIT_DOB_AFTER_NAME_RE.search(text)
    if implicit:
        candidate = normalize_date_token(implicit.group("dob"))
        try:
            month, day, year = [int(part) for part in candidate.split("/")]
        except ValueError:
            return None

        if is_valid_calendar_date(month, day, year):
            return candidate

    compact = IMPLICIT_COMPACT_DOB_AFTER_NAME_RE.search(text)
    if compact:
        parts = parse_compact_date_token(compact.group("dob"))
        if parts:
            month, day, year = parts
            if is_valid_calendar_date(month, day, year):
                return format_mmddyyyy(month, day, year)

    for match in BARE_OLD_YEAR_DATE_RE.finditer(text):
        parts = parse_separated_date_token(match.group("dob"))
        if not parts:
            continue

        month, day, year = parts
        if is_plausible_standalone_dob(month, day, year):
            return format_mmddyyyy(month, day, year)

    return None


def extract_transcript_entities(text: str) -> dict[str, Any]:
    phones = extract_phone_numbers(text)
    fax_number = next((item["number"] for item in phones if item["is_fax"]), None)
    callback_number = next(
        (item["number"] for item in phones if item["is_callback"] and not item["is_fax"]),
        None,
    )
    if not callback_number:
        callback_number = next((item["number"] for item in phones if not item["is_fax"]), None)

    return {
        "name": extract_name(text),
        "dob": extract_dob(text),
        "callback_number": callback_number,
        "fax_number": fax_number,
        "phone_numbers": [item["number"] for item in phones],
    }


MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

DATE_CONTEXT_WORDS = (
    r"date of birth|birth date|birthdate|birthday|dob|"
    r"encounter date|visit date|office date|appointment date|surgery date"
)
DATE_PREFIX = rf"(?P<prefix>\b(?:{DATE_CONTEXT_WORDS})\b[^\n\d]{{0,40}}?)"

COLLAPSED_DOB_CONTEXT_RE = re.compile(
    rf"(?i){DATE_PREFIX}"
    r"(?P<month>\d{1,2})\s*[/-]\s*(?P<day>\d)(?P<year>\d{2})(?!\d)"
)

MONTH_NAME_CONTEXT_DATE_RE = re.compile(
    rf"(?i){DATE_PREFIX}"
    r"\b(?P<month>january|jan|february|feb|march|mar|april|apr|may|june|jun|"
    r"july|jul|august|aug|september|sept|sep|october|oct|"
    r"november|nov|december|dec)"
    r"\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{2,4})\b"
)

SEPARATED_CONTEXT_DATE_RE = re.compile(
    rf"(?i){DATE_PREFIX}"
    r"(?P<month>\d{1,2})\s*[/-]\s*(?P<day>\d{1,2})"
    r"(?:\s*[/-]\s*(?P<year>\d{2}|\d{4})(?:\s*[/-]\s*(?P<split_year>\d{2}))?)"
    r"(?!\d)"
)

COMPACT_WITH_YEAR_CONTEXT_DATE_RE = re.compile(
    rf"(?i){DATE_PREFIX}"
    r"(?P<first>\d{3,4})\s*[/-]\s*(?P<year>\d{2}|\d{4})(?!\d)"
)

COMPACT_CONTEXT_DATE_RE = re.compile(
    rf"(?i){DATE_PREFIX}"
    r"(?P<digits>\d{5}|\d{6}|\d{8})(?!\d)"
)


def normalize_two_digit_year(year: int) -> int:
    if year >= 100:
        return year

    current_two_digit_year = datetime.now().year % 100
    if year <= current_two_digit_year:
        return 2000 + year
    return 1900 + year


def is_valid_calendar_date(month: int, day: int, year: int) -> bool:
    try:
        date(normalize_two_digit_year(year), month, day)
        return True
    except ValueError:
        return False


def format_mmddyyyy(month: int, day: int, year: int) -> str:
    year = normalize_two_digit_year(year)
    return f"{month:02d}/{day:02d}/{year:04d}"


def _normalized_or_original(prefix: str, original_date: str, month: int, day: int, year: int) -> str:
    if not is_valid_calendar_date(month, day, year):
        return prefix + original_date
    return prefix + format_mmddyyyy(month, day, year)


def normalize_month_name_context_date(match: re.Match) -> str:
    try:
        prefix = match.group("prefix")
        original_date = match.group(0)[len(prefix):]
        month = MONTHS[match.group("month").lower()]
        day = int(match.group("day"))
        year = int(match.group("year"))
        return _normalized_or_original(prefix, original_date, month, day, year)
    except Exception:
        return match.group(0)


def normalize_separated_context_date(match: re.Match) -> str:
    try:
        prefix = match.group("prefix")
        original_date = match.group(0)[len(prefix):]
        month = int(match.group("month"))
        day = int(match.group("day"))
        year_part = match.group("year")
        split_year_part = match.group("split_year")

        if split_year_part and year_part in {"19", "20"}:
            year = int(year_part + split_year_part)
        else:
            year = int(year_part)

        return _normalized_or_original(prefix, original_date, month, day, year)
    except Exception:
        return match.group(0)


def normalize_compact_with_year_context_date(match: re.Match) -> str:
    try:
        prefix = match.group("prefix")
        original_date = match.group(0)[len(prefix):]
        first = match.group("first")
        year = int(match.group("year"))

        if len(first) == 3:
            month = int(first[0])
            day = int(first[1:])
        elif len(first) == 4:
            month = int(first[:2])
            day = int(first[2:])
        else:
            return match.group(0)

        return _normalized_or_original(prefix, original_date, month, day, year)
    except Exception:
        return match.group(0)


def normalize_compact_context_date(match: re.Match) -> str:
    try:
        prefix = match.group("prefix")
        original_date = match.group(0)[len(prefix):]
        digits = match.group("digits")

        if len(digits) == 8:
            month = int(digits[:2])
            day = int(digits[2:4])
            year = int(digits[4:])
        elif len(digits) == 6:
            month = int(digits[:2])
            day = int(digits[2:4])
            year = int(digits[4:])
        elif len(digits) == 5:
            month = int(digits[0])
            day = int(digits[1:3])
            year = int(digits[3:])
        else:
            return match.group(0)

        return _normalized_or_original(prefix, original_date, month, day, year)
    except Exception:
        return match.group(0)


def normalize_medical_dates(text: str) -> str:
    text = COLLAPSED_DOB_CONTEXT_RE.sub(
        lambda match: _normalized_or_original(
            match.group("prefix"),
            match.group(0)[len(match.group("prefix")):],
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("year")),
        ),
        text,
    )
    text = MONTH_NAME_CONTEXT_DATE_RE.sub(normalize_month_name_context_date, text)
    text = SEPARATED_CONTEXT_DATE_RE.sub(normalize_separated_context_date, text)
    text = COMPACT_WITH_YEAR_CONTEXT_DATE_RE.sub(normalize_compact_with_year_context_date, text)
    text = COMPACT_CONTEXT_DATE_RE.sub(normalize_compact_context_date, text)
    return text


def get_initial_prompt() -> str:
    if SETTINGS.initial_prompt_file:
        try:
            with open(SETTINGS.initial_prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
                if prompt:
                    return prompt
        except Exception as exc:
            logger.warning("Could not read WHISPER_INITIAL_PROMPT_FILE: %s", exc)

    return SETTINGS.initial_prompt


def validate_runtime_settings() -> None:
    if SETTINGS.require_auth and not SETTINGS.api_key:
        raise RuntimeError("WHISPER_REQUIRE_AUTH is true but WHISPER_API_KEY is not set")
    if SETTINGS.host in {"0.0.0.0", "::"} and not SETTINGS.require_auth:
        logger.warning("Server is binding publicly without authentication")
    if FastAPI is None:
        raise RuntimeError("Missing required dependency: fastapi")


def cleanup_audio_path(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception as exc:
        logger.warning("Could not remove temp file %s: %s", path, exc)


def cleanup_stale_temp_files() -> int:
    temp_dir = tempfile.gettempdir()
    cutoff = time.time() - SETTINGS.stale_temp_max_age_seconds
    removed = 0

    try:
        with os.scandir(temp_dir) as entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.startswith(TEMP_PREFIX):
                    continue
                try:
                    if entry.stat().st_mtime < cutoff:
                        os.unlink(entry.path)
                        removed += 1
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    logger.warning("Could not remove stale temp file %s: %s", entry.path, exc)
    except Exception as exc:
        logger.warning("Could not scan temp directory for stale files: %s", exc)

    return removed


def phone_signal_digit_count(text: str) -> int:
    return len(re.findall(r"\d", words_to_digits(text)))


def classify_phone_verification_context(text: str) -> str:
    if PHONE_VERIFICATION_FAX_RE.search(text):
        return "fax"
    if PHONE_VERIFICATION_CALLBACK_RE.search(text):
        return "callback"
    return "unknown"


def combine_phone_verification_kind(left: str, right: str) -> str:
    if left == right:
        return left
    if left == "unknown":
        return right
    if right == "unknown":
        return left
    return left


def collect_phone_verification_candidates(
    segments: list[Any],
    padding_seconds: float,
    max_windows: int,
) -> list[PhoneVerificationCandidate]:
    if max_windows <= 0:
        return []

    candidates: list[PhoneVerificationCandidate] = []
    segment_count = len(segments)

    for index, segment in enumerate(segments):
        text = str(getattr(segment, "text", "") or "")
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        if start is None or end is None:
            continue

        surrounding_text = " ".join(
            str(getattr(segments[i], "text", "") or "")
            for i in range(max(0, index - 1), min(segment_count, index + 2))
        )
        has_context = PHONE_VERIFICATION_CONTEXT_RE.search(surrounding_text) is not None
        has_number = PHONE_CANDIDATE_RE.search(words_to_digits(text)) is not None
        has_spoken_digits = phone_signal_digit_count(surrounding_text) >= 7

        if not has_number and not (has_context and has_spoken_digits):
            continue

        kind = classify_phone_verification_context(surrounding_text)
        window = PhoneVerificationCandidate(
            start=max(0.0, float(start) - padding_seconds),
            end=max(float(end) + padding_seconds, float(start) + 0.5),
            kind=kind,
            text=surrounding_text,
        )

        if candidates and window.start <= candidates[-1].end + 0.25:
            candidates[-1].end = max(candidates[-1].end, window.end)
            candidates[-1].kind = combine_phone_verification_kind(candidates[-1].kind, window.kind)
            candidates[-1].text = f"{candidates[-1].text} {window.text}"
        else:
            candidates.append(window)

        if len(candidates) >= max_windows:
            break

    return candidates


def create_audio_clip(audio_path: str, start: float, end: float) -> Optional[str]:
    duration = max(0.5, end - start)
    fd, clip_path = tempfile.mkstemp(prefix=f"{TEMP_PREFIX}phone-", suffix=".wav")
    os.close(fd)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        audio_path,
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        clip_path,
    ]

    try:
        subprocess.run(command, check=True, timeout=30)
        return clip_path
    except FileNotFoundError:
        logger.warning("ffmpeg is not installed; skipping phone-number second pass")
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out while preparing phone-number second pass clip")
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg failed while preparing phone-number second pass clip: %s", exc)

    cleanup_audio_path(clip_path)
    return None


def transcribe_phone_clip(clip_path: str, language: Optional[str]) -> str:
    if whisper_model is None:
        raise RuntimeError("Whisper model is not loaded")

    kwargs = {
        "beam_size": 10,
        "temperature": 0.0,
        "language": language or SETTINGS.default_language or "en",
        "condition_on_previous_text": False,
        "vad_filter": False,
        "initial_prompt": PHONE_SECOND_PASS_PROMPT,
    }

    lock_context = transcribe_lock if SETTINGS.model_lock_enabled else contextlib.nullcontext()
    with lock_context:
        segments, _info = whisper_model.transcribe(clip_path, **kwargs)
        return " ".join(segment.text.strip() for segment in segments)


def run_phone_second_pass(
    audio_path: str,
    segments: list[Any],
    language: Optional[str],
) -> list[dict[str, str]]:
    if not SETTINGS.phone_second_pass_enabled:
        return []

    padding_seconds = SETTINGS.phone_second_pass_padding_ms / 1000
    candidates = collect_phone_verification_candidates(
        segments,
        padding_seconds,
        SETTINGS.phone_second_pass_max_windows,
    )
    if not candidates:
        return []

    verifications: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for candidate in candidates:
        clip_path = create_audio_clip(audio_path, candidate.start, candidate.end)
        if not clip_path:
            continue

        try:
            verification_text = transcribe_phone_clip(clip_path, language)
            first_numbers = extract_phone_numbers_from_verification_text(candidate.text)
            second_numbers = extract_phone_numbers_from_verification_text(verification_text)
            third_numbers = None

            if needs_phone_tiebreaker(first_numbers, second_numbers):
                tiebreaker_text = transcribe_phone_clip(clip_path, language)
                third_numbers = extract_phone_numbers_from_verification_text(tiebreaker_text)

            for selected in select_majority_phone_verifications(first_numbers, second_numbers, third_numbers):
                number = selected["number"]
                key = (candidate.kind, number)
                if key in seen:
                    continue
                verification = {"kind": candidate.kind, "number": number}
                if selected.get("original_number"):
                    verification["original_number"] = selected["original_number"]
                verifications.append(verification)
                seen.add(key)
        except Exception as exc:
            logger.warning("Phone-number second pass failed: %s", exc)
        finally:
            cleanup_audio_path(clip_path)

    if verifications:
        logger.info(
            "Phone-number second pass found %s verified number candidate(s)",
            len(verifications),
        )

    return verifications


def load_model() -> None:
    global whisper_model

    from faster_whisper import WhisperModel

    logger.info(
        "Loading Whisper model=%r device=%s compute_type=%s cpu_threads=%s num_workers=%s",
        SETTINGS.model_size,
        SETTINGS.model_device,
        SETTINGS.model_compute_type,
        SETTINGS.model_cpu_threads,
        SETTINGS.model_num_workers,
    )

    whisper_model = WhisperModel(
        SETTINGS.model_size,
        device=SETTINGS.model_device,
        compute_type=SETTINGS.model_compute_type,
        cpu_threads=SETTINGS.model_cpu_threads,
        num_workers=SETTINGS.model_num_workers,
        local_files_only=SETTINGS.model_local_files_only,
    )

    logger.info("Model ready.")


def run_transcription(job: TranscriptionJob) -> None:
    started = time.monotonic()
    waited = started - job.created_monotonic

    try:
        if whisper_model is None:
            raise RuntimeError("Whisper model is not loaded")

        kwargs = {
            "beam_size": 8,
            "temperature": 0.0,
            "language": "en",
            "condition_on_previous_text": False,
            "hallucination_silence_threshold": 2.0,
            "no_speech_threshold": 0.7,
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.25,
                "min_silence_duration_ms": 600,
                "speech_pad_ms": 550,
                "min_speech_duration_ms": 100,
            },
            "initial_prompt": get_initial_prompt(),
        }
        if SETTINGS.word_timestamps_enabled:
            kwargs["word_timestamps"] = True

        if job.language:
            kwargs["language"] = job.language
        elif SETTINGS.default_language:
            kwargs["language"] = SETTINGS.default_language

        logger.info(
            "[%s] starting kind=%s queued_wait=%.2fs upload_bytes=%s",
            job.job_id,
            job.kind,
            waited,
            job.upload_bytes,
        )

        lock_context = transcribe_lock if SETTINGS.model_lock_enabled else contextlib.nullcontext()
        with lock_context:
            segments, info = whisper_model.transcribe(job.audio_path, **kwargs)
            segment_list = list(segments)
            raw_text = " ".join(segment.text.strip() for segment in segment_list)
            raw_word_timestamps = collect_word_timestamps(segment_list)
            raw_segments = collect_segments(segment_list)

        job.raw_text = raw_text
        job.segments = raw_segments
        text = transcript_words_to_digits(raw_text)
        text = normalize_callback_numbers(text)
        text = normalize_phone_number_spacing(text)
        job.result = normalize_medical_dates(text)

        phone_verifications = run_phone_second_pass(
            job.audio_path,
            segment_list,
            kwargs.get("language"),
        )
        job.result = apply_phone_verification_transcript_corrections(
            job.result,
            phone_verifications,
        )
        job.entities = extract_transcript_entities(job.result)
        job.entities = apply_phone_verifications(job.entities, phone_verifications)
        job.word_timestamps = align_word_timestamps_to_text(job.result, raw_word_timestamps)

        elapsed = time.monotonic() - started
        language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)

        with stats_lock:
            stats["completed"] += 1
            stats["last_success_utc"] = utc_now_iso()
            stats["last_job_seconds"] = round(elapsed, 3)
            stats["last_language"] = language
            stats["last_language_probability"] = (
                round(language_probability, 4)
                if isinstance(language_probability, float)
                else None
            )

        logger.info(
            "[%s] done kind=%s language=%s probability=%s chars=%s seconds=%.2f",
            job.job_id,
            job.kind,
            language,
            f"{language_probability:.0%}" if isinstance(language_probability, float) else "unknown",
            len(job.result or ""),
            elapsed,
        )

    except Exception as exc:
        job.error = "transcription_failed"
        with stats_lock:
            stats["failed"] += 1
            stats["last_error_utc"] = utc_now_iso()
            stats["last_error"] = str(exc)[:500]

        logger.error("[%s] transcription error kind=%s error=%s", job.job_id, job.kind, exc)

    finally:
        cleanup_audio_path(job.audio_path)
        job.result_event.set()


def voicemail_worker(worker_id: int) -> None:
    logger.info("Voicemail worker %s started (voicemail-only)", worker_id)

    while True:
        try:
            job = voicemail_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            logger.info("[VM-worker-%s] picked up job %s", worker_id, job.job_id)
            run_transcription(job)
        finally:
            voicemail_queue.task_done()


def recording_worker(worker_id: int) -> None:
    logger.info("Recording worker %s started", worker_id)

    while True:
        try:
            job = voicemail_queue.get_nowait()
        except queue.Empty:
            job = None

        if job is not None:
            try:
                logger.info("[REC-worker-%s] assisting voicemail job %s", worker_id, job.job_id)
                run_transcription(job)
            finally:
                voicemail_queue.task_done()
            continue

        try:
            job = recording_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            logger.info("[REC-worker-%s] picked up recording job %s", worker_id, job.job_id)
            run_transcription(job)
        finally:
            recording_queue.task_done()


def startup() -> None:
    validate_runtime_settings()
    removed = cleanup_stale_temp_files()
    if removed:
        logger.info("Removed %s stale temp upload file(s)", removed)

    load_model()

    for i in range(SETTINGS.voicemail_workers):
        threading.Thread(target=voicemail_worker, args=(i,), daemon=True).start()

    for i in range(SETTINGS.recording_workers):
        threading.Thread(target=recording_worker, args=(i,), daemon=True).start()

    logger.info(
        "Workers: %s voicemail-reserved + %s recording = %s total",
        SETTINGS.voicemail_workers,
        SETTINGS.recording_workers,
        SETTINGS.total_workers,
    )


def normalize_priority(priority: str) -> tuple[int, str]:
    priority_normalized = (priority or "voicemail").strip().lower()

    if priority_normalized == "voicemail":
        return PRIORITY_VOICEMAIL, "voicemail"

    if priority_normalized == "recording":
        return PRIORITY_RECORDING, "recording"

    return PRIORITY_RECORDING, "recording"


def get_queue_for_priority(priority: int) -> queue.PriorityQueue[TranscriptionJob]:
    return voicemail_queue if priority == PRIORITY_VOICEMAIL else recording_queue


def queue_is_full(priority: int) -> bool:
    return get_queue_for_priority(priority).full()


def get_safe_suffix(filename: Optional[str]) -> str:
    suffix = os.path.splitext(filename or "")[-1].lower()
    if not suffix:
        suffix = ".wav"

    if suffix not in SETTINGS.allowed_extensions:
        allowed = ", ".join(sorted(SETTINGS.allowed_extensions))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio extension {suffix!r}. Allowed: {allowed}",
        )

    return suffix


def validate_audio_signature(path: str, suffix: str) -> None:
    with open(path, "rb") as f:
        header = f.read(64)

    if not header:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    valid = True
    if suffix == ".wav":
        valid = len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    elif suffix == ".flac":
        valid = header.startswith(b"fLaC")
    elif suffix in {".ogg", ".oga"}:
        valid = header.startswith(b"OggS")
    elif suffix == ".webm":
        valid = header.startswith(b"\x1a\x45\xdf\xa3")
    elif suffix == ".mp3":
        valid = header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
        )
    elif suffix == ".m4a":
        valid = len(header) >= 12 and header[4:8] == b"ftyp"

    if not valid:
        raise HTTPException(
            status_code=415,
            detail=f"Uploaded file content does not look like a valid {suffix} audio file",
        )


async def save_upload_to_temp(file: UploadFile) -> tuple[str, int, str]:
    suffix = get_safe_suffix(getattr(file, "filename", None))
    tmp_path = None
    total = 0

    try:
        with tempfile.NamedTemporaryFile(prefix=TEMP_PREFIX, suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

            while True:
                chunk = await file.read(SETTINGS.upload_chunk_size)
                if not chunk:
                    break

                total += len(chunk)
                if total > SETTINGS.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio upload exceeds {SETTINGS.max_upload_mb} MB limit",
                    )

                tmp.write(chunk)

        if total == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        validate_audio_signature(tmp_path, suffix)
        return tmp_path, total, suffix

    except Exception:
        if tmp_path:
            cleanup_audio_path(tmp_path)
        with stats_lock:
            stats["rejected_upload"] += 1
            stats["last_error_utc"] = utc_now_iso()
            stats["last_error"] = "upload_rejected"
        raise

    finally:
        try:
            await file.close()
        except Exception:
            pass


def submit_job(
    audio_path: str,
    priority: int,
    kind: str,
    language: Optional[str] = None,
    timeout: int = 120,
    upload_bytes: int = 0,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], str, list[dict[str, Any]]]:
    job = TranscriptionJob(
        priority=priority,
        sequence=next(job_sequence),
        job_id=str(uuid.uuid4())[:8],
        audio_path=audio_path,
        language=(language or "").strip() or None,
        result_event=threading.Event(),
        kind=kind,
        upload_bytes=upload_bytes,
    )

    q = get_queue_for_priority(priority)
    try:
        q.put_nowait(job)
    except queue.Full as exc:
        cleanup_audio_path(audio_path)
        with stats_lock:
            stats["rejected_queue_full"] += 1
            stats["last_error_utc"] = utc_now_iso()
            stats["last_error"] = f"{kind} queue is full"
        raise QueueFullError(f"{kind} transcription queue is full") from exc

    with stats_lock:
        stats["submitted"] += 1
        stats["bytes_received"] += upload_bytes

    logger.info(
        "[%s] queued kind=%s priority=%s upload_bytes=%s",
        job.job_id,
        kind,
        priority,
        upload_bytes,
    )

    if not job.result_event.wait(timeout=timeout):
        with stats_lock:
            stats["timed_out"] += 1
            stats["last_error_utc"] = utc_now_iso()
            stats["last_error"] = f"Job {job.job_id} timed out after {timeout}s"

        logger.error("[%s] timed out kind=%s timeout=%ss", job.job_id, kind, timeout)
        raise TimeoutError(f"Transcription timed out after {timeout}s")

    if job.error:
        raise TranscriptionFailedError(job.error)

    return job.result or "", job.entities or {}, job.word_timestamps or [], job.raw_text or "", job.segments or []


def get_disk_free_tmp() -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(tempfile.gettempdir())
        return {
            "path": tempfile.gettempdir(),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "used_bytes": usage.used,
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_health_status() -> tuple[str, list[str]]:
    reasons: list[str] = []
    disk = get_disk_free_tmp()

    if whisper_model is None:
        reasons.append("model_not_loaded")
    if isinstance(disk.get("free_bytes"), int) and disk["free_bytes"] < SETTINGS.min_tmp_free_bytes:
        reasons.append("low_tmp_disk")
    if voicemail_queue.full():
        reasons.append("voicemail_queue_full")
    if recording_queue.full():
        reasons.append("recording_queue_full")
    if SETTINGS.require_auth and not SETTINGS.api_key:
        reasons.append("auth_misconfigured")

    return ("degraded" if reasons else "ok"), reasons


def health() -> dict[str, Any]:
    with stats_lock:
        stats_snapshot = dict(stats)

    status, degraded_reasons = get_health_status()

    return {
        "status": status,
        "degraded_reasons": degraded_reasons,
        "environment": SETTINGS.environment,
        "expected_ip": SETTINGS.expected_ip or None,
        "app_version": APP_VERSION,
        "service_started_utc": SERVICE_STARTED_UTC,
        "uptime_seconds": round(time.monotonic() - SERVICE_STARTED_MONOTONIC, 3),
        "security": {
            "auth_required": SETTINGS.require_auth,
            "public_bind": SETTINGS.host in {"0.0.0.0", "::"},
        },
        "model": {
            "name_or_path": SETTINGS.model_size,
            "loaded": whisper_model is not None,
            "device": SETTINGS.model_device,
            "compute_type": SETTINGS.model_compute_type,
            "cpu_threads": SETTINGS.model_cpu_threads,
            "model_num_workers": SETTINGS.model_num_workers,
            "default_language": SETTINGS.default_language,
            "model_lock_enabled": SETTINGS.model_lock_enabled,
            "word_timestamps_enabled": SETTINGS.word_timestamps_enabled,
        },
        "workers": {
            "voicemail_reserved": SETTINGS.voicemail_workers,
            "recording": SETTINGS.recording_workers,
            "total": SETTINGS.total_workers,
        },
        "queue_depths": {
            "voicemail": voicemail_queue.qsize(),
            "recording": recording_queue.qsize(),
        },
        "queue_limits": {
            "max_depth_per_queue": SETTINGS.max_queue_depth,
        },
        "upload_limits": {
            "max_upload_mb": SETTINGS.max_upload_mb,
            "allowed_extensions": sorted(SETTINGS.allowed_extensions),
        },
        "stats": stats_snapshot,
        "disk": {
            "tmp": get_disk_free_tmp(),
            "min_tmp_free_bytes": SETTINGS.min_tmp_free_bytes,
        },
    }


HEADER_EMPTY = Header(default="") if Header is not None else ""


def ready(
    authorization: str = HEADER_EMPTY,
    x_api_key: str = HEADER_EMPTY,
) -> dict[str, Any]:
    authorize_request(authorization, x_api_key)
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model is not loaded")

    disk = get_disk_free_tmp()
    if isinstance(disk.get("free_bytes"), int) and disk["free_bytes"] < SETTINGS.min_tmp_free_bytes:
        raise HTTPException(status_code=503, detail="Temporary disk space is below configured minimum")

    return {
        "status": "ready",
        "model": SETTINGS.model_size,
        "queue_depths": {
            "voicemail": voicemail_queue.qsize(),
            "recording": recording_queue.qsize(),
        },
    }


def authorize_request(authorization: str = "", x_api_key: str = "") -> None:
    if not SETTINGS.require_auth:
        return

    supplied = ""
    if authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    elif x_api_key:
        supplied = x_api_key.strip()

    if not supplied or not secrets.compare_digest(supplied, SETTINGS.api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")


async def handle_transcription(
    file: UploadFile,
    language: Optional[str],
    response_format: str,
    priority: str,
) -> Any:
    requested_format = (response_format or "json").strip().lower()
    if requested_format not in {"json", "text"}:
        raise HTTPException(status_code=400, detail="response_format must be 'json' or 'text'")

    p, kind = normalize_priority(priority)
    timeout = (
        SETTINGS.voicemail_timeout_seconds
        if p == PRIORITY_VOICEMAIL
        else SETTINGS.recording_timeout_seconds
    )

    if queue_is_full(p):
        with stats_lock:
            stats["rejected_queue_full"] += 1
            stats["last_error_utc"] = utc_now_iso()
            stats["last_error"] = f"{kind} queue is full"
        raise HTTPException(status_code=429, detail=f"{kind} transcription queue is full")

    tmp_path, upload_bytes, _suffix = await save_upload_to_temp(file)

    try:
        text, entities, word_timestamps, raw_text, segments = await asyncio.to_thread(
            submit_job,
            tmp_path,
            p,
            kind,
            language,
            timeout,
            upload_bytes,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except TranscriptionFailedError as exc:
        logger.error("Request failed kind=%s error=%s", kind, exc)
        raise HTTPException(status_code=500, detail="Transcription failed") from exc
    except Exception as exc:
        logger.error("Request failed kind=%s error=%s", kind, exc)
        raise HTTPException(status_code=500, detail="Transcription request failed") from exc

    if requested_format == "text":
        return PlainTextResponse(text) if PlainTextResponse is not None else text

    payload = {
        "text": text,
        "raw_text": raw_text or text,
        "processed_text": text,
        "segments": segments,
        "words": word_timestamps,
        "entities": entities,
        "meta": {
            "environment": SETTINGS.environment,
            "kind": kind,
            "bytes": upload_bytes,
            "word_timestamps": bool(word_timestamps),
        },
    }
    return JSONResponse(payload) if JSONResponse is not None else payload


FILE_REQUIRED = File(...) if File is not None else None
FORM_MODEL = Form(default="whisper-1") if Form is not None else "whisper-1"
FORM_LANGUAGE = Form(default=None) if Form is not None else None
FORM_JSON = Form(default="json") if Form is not None else "json"
FORM_VOICEMAIL = Form(default="voicemail") if Form is not None else "voicemail"
async def transcribe_openai_compat(
    file: UploadFile = FILE_REQUIRED,
    model: str = FORM_MODEL,
    language: Optional[str] = FORM_LANGUAGE,
    response_format: str = FORM_JSON,
    priority: str = FORM_VOICEMAIL,
    authorization: str = HEADER_EMPTY,
    x_api_key: str = HEADER_EMPTY,
) -> Any:
    _ = model
    authorize_request(authorization, x_api_key)
    return await handle_transcription(file, language, response_format, priority)


async def transcribe_voicemail(
    file: UploadFile = FILE_REQUIRED,
    language: Optional[str] = FORM_LANGUAGE,
    authorization: str = HEADER_EMPTY,
    x_api_key: str = HEADER_EMPTY,
) -> Any:
    authorize_request(authorization, x_api_key)
    return await handle_transcription(file, language, "json", "voicemail")


async def transcribe_recording(
    file: UploadFile = FILE_REQUIRED,
    language: Optional[str] = FORM_LANGUAGE,
    authorization: str = HEADER_EMPTY,
    x_api_key: str = HEADER_EMPTY,
) -> Any:
    authorize_request(authorization, x_api_key)
    return await handle_transcription(file, language, "json", "recording")


app = FastAPI(title="Local Whisper API", version=APP_VERSION) if FastAPI is not None else None

if app is not None:
    app.on_event("startup")(startup)
    app.get("/health")(health)
    app.get("/ready")(ready)
    app.post("/v1/audio/transcriptions")(transcribe_openai_compat)
    app.post("/transcribe/voicemail")(transcribe_voicemail)
    app.post("/transcribe/recording")(transcribe_recording)


if __name__ == "__main__":
    if uvicorn is None:
        raise RuntimeError("Missing required dependency: uvicorn")
    if app is None:
        raise RuntimeError("Missing required dependency: fastapi")
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level=SETTINGS.log_level.lower())
