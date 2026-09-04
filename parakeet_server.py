#!/usr/bin/env python3
"""
Small HTTP wrapper for NVIDIA Parakeet TDT.

The voicemail watcher posts Gemma-attributed WAV clips here as multipart
``file`` uploads. This server loads the Parakeet model once at startup and
returns compact JSON that the watcher can audit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    import uvicorn
    from fastapi import FastAPI, File, Header, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - startup dependency
    uvicorn = None

    class FastAPI:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_event(self, *_args: Any, **_kwargs: Any) -> Any:
            return lambda func: func

        def get(self, *_args: Any, **_kwargs: Any) -> Any:
            return lambda func: func

        def post(self, *_args: Any, **_kwargs: Any) -> Any:
            return lambda func: func

    def File(*_args: Any, **_kwargs: Any) -> None:  # type: ignore[no-redef]
        return None

    def Header(*_args: Any, **_kwargs: Any) -> None:  # type: ignore[no-redef]
        return None

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    UploadFile = Any  # type: ignore[assignment]

    class JSONResponse(dict):  # type: ignore[no-redef]
        pass


APP_VERSION = "0.1.0"
SERVICE_STARTED = time.time()
MODEL: Any = None
MODEL_LOCK = threading.Lock()
REQUEST_LOCK = threading.Lock()
ACTIVE_REQUESTS = 0
COMPLETED_REQUESTS = 0
PARAKEET_API_KEY = os.environ.get("PARAKEET_API_KEY", "").strip()
PARAKEET_REQUIRE_AUTH = os.environ.get(
    "PARAKEET_REQUIRE_AUTH",
    "true" if PARAKEET_API_KEY else "false",
).strip().lower() in {"1", "true", "yes", "y", "on"}

logger = logging.getLogger("parakeet_api")


def request_stats() -> dict[str, int]:
    with REQUEST_LOCK:
        return {
            "in_flight": ACTIVE_REQUESTS,
            "active_requests": ACTIVE_REQUESTS,
            "completed_requests": COMPLETED_REQUESTS,
        }


def authorize_request(authorization: Optional[str] = None, x_api_key: Optional[str] = None) -> None:
    if not PARAKEET_REQUIRE_AUTH:
        return
    if not PARAKEET_API_KEY:
        raise HTTPException(status_code=500, detail="Parakeet API auth is required but PARAKEET_API_KEY is not set")
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    supplied = (x_api_key or bearer or "").strip()
    if not supplied or not secrets.compare_digest(supplied, PARAKEET_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def mark_request_started() -> None:
    global ACTIVE_REQUESTS
    with REQUEST_LOCK:
        ACTIVE_REQUESTS += 1


def mark_request_finished() -> None:
    global ACTIVE_REQUESTS, COMPLETED_REQUESTS
    with REQUEST_LOCK:
        ACTIVE_REQUESTS = max(0, ACTIVE_REQUESTS - 1)
        COMPLETED_REQUESTS += 1


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


PARAKEET_MODEL_NAME = os.environ.get("PARAKEET_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2").strip()
PARAKEET_HOST = os.environ.get("PARAKEET_HOST", "0.0.0.0").strip()
PARAKEET_PORT = env_int("PARAKEET_PORT", 8766, minimum=1)
PARAKEET_TMP_DIR = os.environ.get("PARAKEET_TMP_DIR", tempfile.gettempdir()).strip()
PARAKEET_MAX_UPLOAD_MB = env_int("PARAKEET_MAX_UPLOAD_MB", 25, minimum=1)
PARAKEET_LOG_LEVEL = os.environ.get("PARAKEET_LOG_LEVEL", "INFO").upper()
PARAKEET_RETURN_TIMESTAMPS = env_bool("PARAKEET_RETURN_TIMESTAMPS", False)


DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "won": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
}

DIGIT_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(DIGIT_WORDS, key=len, reverse=True)) + r")\b",
    re.I,
)
PHONE_CHUNK_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-()]*)?(?:\d[\s.\-()]*){9}\d(?![\s.\-()]*\d)")
EXTENSION_DIGITS_RE = re.compile(
    r"(?:\bextension\b|\bext\b\.?|\bext\.?(?=\d)|\bx\b|\bx(?=\d))[\s:;,#.\-]*(?:\d[\s.\-]*){1,6}(?=\D|$)",
    re.I,
)


def words_to_digits(text: str) -> str:
    return DIGIT_WORD_RE.sub(lambda match: DIGIT_WORDS[match.group(1).lower()], text)


def strip_extension_digits(text: str) -> str:
    return EXTENSION_DIGITS_RE.sub(" ", text)


def normalize_phone_candidate(value: str) -> Optional[str]:
    digits = re.sub(r"\D+", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return digits
    return None


def formatted_phone(digits: str) -> str:
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def extract_numbers_from_text(text: str) -> tuple[list[str], list[str]]:
    candidates: list[str] = []
    converted = strip_extension_digits(words_to_digits(text))
    for match in PHONE_CHUNK_RE.finditer(converted):
        normalized = normalize_phone_candidate(match.group(0))
        if normalized:
            candidates.append(normalized)

    if not candidates:
        compact_digits = re.sub(r"\D+", "", converted)
        for index in range(0, max(0, len(compact_digits) - 9)):
            normalized = normalize_phone_candidate(compact_digits[index : index + 10])
            if normalized:
                candidates.append(normalized)

    unique = list(dict.fromkeys(candidates))
    return unique, [formatted_phone(item) for item in unique]


def load_model() -> None:
    global MODEL
    if MODEL is not None:
        return
    logger.info("Loading Parakeet model=%s", PARAKEET_MODEL_NAME)
    from nemo.collections.asr.models import ASRModel

    model_path = Path(PARAKEET_MODEL_NAME)
    if model_path.is_file():
        MODEL = ASRModel.restore_from(restore_path=str(model_path), map_location="cpu")
    else:
        MODEL = ASRModel.from_pretrained(PARAKEET_MODEL_NAME)
    logger.info("Parakeet model ready")


def output_to_text(output: Any) -> str:
    item = output[0] if isinstance(output, list) and output else output
    if isinstance(item, str):
        return item.strip()
    for attr in ("text", "pred_text", "transcript"):
        value = getattr(item, attr, None)
        if isinstance(value, str):
            return value.strip()
    if isinstance(item, dict):
        for key in ("text", "pred_text", "transcript"):
            value = item.get(key)
            if isinstance(value, str):
                return value.strip()
    return str(item or "").strip()


def output_to_timestamps(output: Any) -> Any:
    if not PARAKEET_RETURN_TIMESTAMPS:
        return None
    item = output[0] if isinstance(output, list) and output else output
    for attr in ("timestamp", "timestamps"):
        value = getattr(item, attr, None)
        if value:
            return value
    if isinstance(item, dict):
        return item.get("timestamp") or item.get("timestamps")
    return None


def transcribe_path(path: str) -> dict[str, Any]:
    if MODEL is None:
        raise RuntimeError("Parakeet model is not loaded")
    started = time.monotonic()
    output = MODEL.transcribe([path], timestamps=True)
    text = output_to_text(output)
    normalized_numbers, formatted_numbers = extract_numbers_from_text(text)
    payload: dict[str, Any] = {
        "text": text,
        "normalized_numbers": normalized_numbers,
        "formatted_numbers": formatted_numbers,
        "duration_seconds": round(time.monotonic() - started, 3),
        "model": PARAKEET_MODEL_NAME,
    }
    timestamps = output_to_timestamps(output)
    if timestamps is not None:
        payload["timestamps"] = timestamps
    return payload


def transcribe_path_locked(path: str) -> dict[str, Any]:
    with MODEL_LOCK:
        return transcribe_path(path)


app = FastAPI(title="Local Parakeet API", version=APP_VERSION)


@app.on_event("startup")
def startup() -> None:
    logging.basicConfig(
        level=getattr(logging, PARAKEET_LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    Path(PARAKEET_TMP_DIR).mkdir(parents=True, exist_ok=True)
    load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if MODEL is not None else "starting",
        "app_version": APP_VERSION,
        "model": PARAKEET_MODEL_NAME,
        "auth_required": PARAKEET_REQUIRE_AUTH,
        "uptime_seconds": round(time.time() - SERVICE_STARTED, 3),
        "load": request_stats(),
    }


@app.get("/ready")
def ready(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    authorize_request(authorization, x_api_key)
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Parakeet model is not loaded")
    return {"status": "ready", "model": PARAKEET_MODEL_NAME, "load": request_stats()}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> JSONResponse:
    authorize_request(authorization, x_api_key)
    mark_request_started()
    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    safe_name = f"parakeet-{uuid.uuid4().hex}{suffix}"
    tmp_path = Path(PARAKEET_TMP_DIR) / safe_name
    max_bytes = PARAKEET_MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="Upload too large")
                out.write(chunk)
        result = await asyncio.to_thread(transcribe_path_locked, str(tmp_path))
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Parakeet transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)[:200]) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            await file.close()
        except Exception:
            pass
        mark_request_finished()


if __name__ == "__main__":
    if uvicorn is None:
        raise RuntimeError(
            "Missing FastAPI runtime. Install with: pip install fastapi uvicorn[standard] python-multipart"
        )
    uvicorn.run(app, host=PARAKEET_HOST, port=PARAKEET_PORT)
