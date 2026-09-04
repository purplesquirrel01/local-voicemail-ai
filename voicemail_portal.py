#!/usr/bin/env python3
"""
Authenticated voicemail transcript portal for adapter-supported Asterisk systems.

This companion service reads the voicemail_transcripts table maintained by
watcher.py, reconciles current INBOX files from the Asterisk voicemail spool,
streams audio to the browser, and safely deletes messages by moving all matching
message files into a quarantine directory.
"""

from __future__ import annotations

import base64
import glob
import getpass
import hashlib
import hmac
import html
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import socket
import smtplib
import sqlite3
import ssl
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

from voicemail_common import env as common_env
from voicemail_common import formatting as common_formatting
from voicemail_common import keys as common_keys
from voicemail_common import spool as common_spool
from voicemail_common import time as common_time
APP_VERSION = "1.1.0"
SESSION_COOKIE = "vm_portal_session"
CSRF_HEADER = "x-csrf-token"
PASSWORD_HASH_PREFIX = "pbkdf2_sha256"
INBOX_EXT_RE = re.compile(r"/(?P<extension>\d{3,6})/INBOX/")
EMAIL_SAFE_FILE_KEY_RE = re.compile(r"^[a-f0-9]{32}$")
CALLERID_RE = re.compile(r'^\s*"?(?P<name>[^"<]*)"?\s*(?:<(?P<number>[^>]+)>)?\s*$')
MSG_STEM_RE = re.compile(r"^msg(?P<number>\d{4})$")
EMAIL_RE = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$")
DELETE_COMMENT_MAX_CHARS = 1000
EXTERNAL_RETENTION_DIR_NAME = "_external_retention"

logger = logging.getLogger("voicemail_portal")
LOGIN_FAILURES: dict[str, list[float]] = {}
LOGIN_FAILURES_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return common_time.utc_now_iso()


def env_bool(name: str, default: bool) -> bool:
    return common_env.env_bool(name, default)


def env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    return common_env.env_int(name, default, minimum)


def env_float(name: str, default: float, minimum: Optional[float] = None) -> float:
    return common_env.env_float(name, default, minimum)


def env_csv_values(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    values: list[str] = []
    for item in raw.replace("\n", ",").split(","):
        value = item.strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def normalize_path(path: str) -> str:
    return common_spool.normalize_path(path)


def extract_extension(path: str) -> Optional[str]:
    return common_spool.extract_extension(path)


def is_voicemail_txt(path: str) -> bool:
    return common_spool.is_voicemail_txt(path)


def parse_txt(txt_path: str) -> dict[str, str]:
    return common_spool.parse_txt(txt_path)


def matching_wav_path(txt_path: str) -> str:
    return common_spool.matching_wav_path(txt_path)


def metadata_file_hash(txt_path: str) -> str:
    return common_keys.metadata_file_hash(txt_path)


def build_legacy_file_key(extension: str, info: dict[str, str], txt_path: str) -> Optional[str]:
    return common_keys.build_legacy_file_key(extension, info, txt_path)


def build_file_key(extension: str, info: dict[str, str], txt_path: str) -> Optional[str]:
    return common_keys.build_file_key(extension, info, txt_path)


def optional_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_duration(seconds: Any) -> str:
    return common_formatting.format_duration(seconds)


def format_phone_number(value: Any) -> Optional[str]:
    return common_formatting.format_phone_number(value)


def caller_number_from_callerid(callerid: str) -> Optional[str]:
    match = CALLERID_RE.match(callerid or "")
    if not match:
        return format_phone_number(callerid)
    return format_phone_number(match.group("number") or callerid)


def caller_number_digits_from_callerid(callerid: str) -> str:
    formatted = caller_number_from_callerid(callerid)
    if not formatted:
        return ""
    digits = re.sub(r"\D", "", formatted)
    return digits if len(digits) == 10 else ""


def callback_matches_caller_id(callerid: str, callback_number: Any) -> str:
    caller = caller_number_from_callerid(callerid)
    callback = format_phone_number(callback_number)
    if not callback:
        return "Callback not included"
    if not caller:
        return "Caller ID not included"
    return "Yes" if caller == callback else "No - Needs Review"


def format_callerid_display(callerid: str) -> str:
    def repl(match: re.Match) -> str:
        raw = match.group("number")
        digits = re.sub(r"\D", "", raw)
        formatted = format_phone_number(digits)
        if not formatted:
            return match.group(0)
        area, prefix, line = formatted.split("-")
        return f"({area})-{prefix}-{line}"

    return re.sub(r"<(?P<number>[^>]+)>", repl, callerid or "Unknown")


def parse_callerid_for_email(callerid: str) -> tuple[str, str]:
    match = CALLERID_RE.match(callerid or "")
    if not match:
        return (callerid or "Unknown").strip() or "Unknown", "Not Included"

    raw_name = (match.group("name") or "").strip().strip('"')
    raw_number = (match.group("number") or "").strip()
    name = raw_name or "Unknown"
    number = format_phone_number(raw_number) or "Not Included"
    return name, number


def format_caller_id_for_email(name: str) -> str:
    if not name or name == "Unknown":
        return "Unknown"
    return f'"{name}"'


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


def email_timezone(timezone_name: str, dt_utc: datetime) -> timezone:
    try:
        return ZoneInfo(timezone_name)
    except Exception:
        if timezone_name == "America/Chicago":
            return central_fallback_timezone(dt_utc)
        raise


def format_email_date(origdate: str, timezone_name: str, timezone_label: str = "") -> str:
    if not origdate:
        return "Unknown"

    try:
        dt = datetime.strptime(origdate, "%a %b %d %I:%M:%S %p UTC %Y")
        dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(email_timezone(timezone_name, dt))
        hour = local.strftime("%I").lstrip("0") or "0"
        label = timezone_label or local.strftime("%Z")
        return f"{local:%A, %B %d %Y}, {hour}:{local:%M} {local:%p} {label}".rstrip()
    except Exception as exc:
        logger.warning("Could not convert voicemail date %r: %s", origdate, exc)
        return origdate


def callback_match_status_for_email(caller_number: str, callback_number: Any) -> str:
    caller = format_phone_number(caller_number)
    callback = format_phone_number(str(callback_number or ""))

    if not callback:
        return "No Callback Number Found"
    if not caller:
        return "Caller Number Not Included"
    if caller == callback:
        return "Yes"
    return "No - Review"


def normalize_word_timestamps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    words: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "") or "").strip()
        start = optional_float(item.get("start"))
        end = optional_float(item.get("end"))
        if not word or start is None or end is None or start < 0 or end < start:
            continue
        words.append({"word": word, "start": round(start, 3), "end": round(end, 3)})
    return words


def parse_json_array(value: Any) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def parse_json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def format_display_date(origtime: Any, origdate: str, timezone_name: str) -> str:
    timestamp = optional_int(origtime)
    if timestamp is not None:
        try:
            local = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(ZoneInfo(timezone_name))
            hour = local.strftime("%I").lstrip("0") or "0"
            return f"{local:%a, %b %d %Y}, {hour}:{local:%M} {local:%p} {local:%Z}"
        except Exception:
            pass
    return origdate or "Unknown"


def format_iso_display(value: Any, timezone_name: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    try:
        normalized = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(ZoneInfo(timezone_name))
        hour = local.strftime("%I").lstrip("0") or "0"
        return f"{local:%a, %b %d %Y}, {hour}:{local:%M} {local:%p} {local:%Z}"
    except Exception:
        return raw


def hash_password(password: str, iterations: int = 310000) -> str:
    salt = secrets.token_urlsafe(18)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"{PASSWORD_HASH_PREFIX}${iterations}${salt}${b64url_encode(digest)}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt, expected_raw = stored_hash.split("$", 3)
        if algorithm != PASSWORD_HASH_PREFIX:
            return False
        iterations = int(iterations_raw)
        expected = b64url_decode(expected_raw)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        if hmac.compare_digest(actual, expected):
            return True
        installer_salt = b64url_decode(salt)
        installer_actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            installer_salt,
            iterations,
        )
        return hmac.compare_digest(installer_actual, expected)
    except Exception:
        return False


def print_password_hash(argv: list[str]) -> int:
    if len(argv) >= 3:
        password = argv[2]
    else:
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match", file=sys.stderr)
            return 1
    print(hash_password(password))
    return 0


if len(sys.argv) >= 2 and sys.argv[1] == "hash-password":
    raise SystemExit(print_password_hash(sys.argv))


from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from voicemail_portal_app.api_v1 import (
    IntegrationAPIConfig,
    SQLiteIntegrationBackend,
    load_service_principals,
    mount_api_v1,
)
from pydantic import BaseModel

try:
    import uvicorn
except ImportError:  # pragma: no cover - production dependency
    uvicorn = None


@dataclass(frozen=True)
class Settings:
    state_db: str
    watch_dir: str
    trash_dir: str
    users_file: str
    voicemail_config: str
    voicemail_config_glob: str
    host: str
    port: int
    base_path: str
    session_secret: str
    shared_password_hash: str
    auto_users: bool
    session_ttl_seconds: int
    cookie_secure: bool
    login_rate_limit_attempts: int
    login_rate_limit_window_seconds: int
    local_timezone: str
    date_timezone_label: str
    sync_interval_seconds: int
    page_limit: int
    auto_delete_short_seconds: int
    delete_comment_user_extensions: tuple[str, ...]
    deleted_retention_days: int
    forward_enabled: bool
    forward_email_enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_timeout_seconds: int
    smtp_starttls: bool
    from_address: str
    from_name: str
    fallback_recipient: str
    mwi_refresh_enabled: bool
    ami_host: str
    ami_port: int
    ami_username: str
    ami_secret: str
    ami_timeout_seconds: float
    ami_default_context: str
    log_level: str
    logo_url: str
    logo_file: str
    light_logo_url: str
    light_logo_file: str
    brand_name: str
    brand_tagline: str

    @classmethod
    def from_env(cls) -> "Settings":
        base_path = os.environ.get("VOICEMAIL_PORTAL_BASE_PATH", "").strip()
        if base_path and not base_path.startswith("/"):
            base_path = f"/{base_path}"
        base_path = base_path.rstrip("/")
        ami_username = os.environ.get("VOICEMAIL_PORTAL_AMI_USERNAME", "").strip()
        ami_secret = os.environ.get("VOICEMAIL_PORTAL_AMI_SECRET", "").strip()

        return cls(
            state_db=os.environ.get(
                "VOICEMAIL_STATE_DB",
                "/var/lib/local-voicemail-transcription/pbx/state.sqlite3",
            ),
            watch_dir=os.environ.get("VOICEMAIL_WATCH_DIR", "/var/spool/asterisk/voicemail"),
            trash_dir=os.environ.get(
                "VOICEMAIL_PORTAL_TRASH_DIR",
                "/var/lib/local-voicemail-transcription/pbx/trash",
            ),
            users_file=os.environ.get(
                "VOICEMAIL_PORTAL_USERS_FILE",
                "/etc/local-voicemail-transcription/users.json",
            ),
            voicemail_config=os.environ.get(
                "VOICEMAIL_CONFIG",
                "/etc/asterisk/vitalpbx/voicemail__50-1-main.conf",
            ),
            voicemail_config_glob=os.environ.get(
                "VOICEMAIL_PORTAL_VOICEMAIL_CONFIG_GLOB",
                "/etc/asterisk/vitalpbx/voicemail__*.conf",
            ),
            host=os.environ.get("VOICEMAIL_PORTAL_HOST", "127.0.0.1"),
            port=env_int("VOICEMAIL_PORTAL_PORT", 8899, minimum=1),
            base_path=base_path,
            session_secret=os.environ.get("VOICEMAIL_PORTAL_SESSION_SECRET", "").strip(),
            shared_password_hash=os.environ.get("VOICEMAIL_PORTAL_SHARED_PASSWORD_HASH", "").strip(),
            auto_users=env_bool("VOICEMAIL_PORTAL_AUTO_USERS", True),
            session_ttl_seconds=env_int("VOICEMAIL_PORTAL_SESSION_TTL", 12 * 3600, minimum=300),
            cookie_secure=env_bool("VOICEMAIL_PORTAL_COOKIE_SECURE", True),
            login_rate_limit_attempts=env_int("VOICEMAIL_PORTAL_LOGIN_RATE_LIMIT_ATTEMPTS", 5, minimum=0),
            login_rate_limit_window_seconds=env_int("VOICEMAIL_PORTAL_LOGIN_RATE_LIMIT_WINDOW", 300, minimum=1),
            local_timezone=os.environ.get("VOICEMAIL_TIMEZONE", "America/Chicago"),
            date_timezone_label=os.environ.get("VOICEMAIL_TIMEZONE_LABEL", "").strip(),
            sync_interval_seconds=env_int("VOICEMAIL_PORTAL_SYNC_INTERVAL", 60, minimum=5),
            page_limit=env_int("VOICEMAIL_PORTAL_PAGE_LIMIT", 250, minimum=1),
            auto_delete_short_seconds=env_int("VOICEMAIL_PORTAL_AUTO_DELETE_SHORT_SECONDS", 5, minimum=0),
            delete_comment_user_extensions=env_csv_values("VOICEMAIL_PORTAL_DELETE_COMMENT_USER_EXTENSIONS"),
            deleted_retention_days=env_int("VOICEMAIL_PORTAL_DELETED_RETENTION_DAYS", 60, minimum=1),
            forward_enabled=env_bool("VOICEMAIL_PORTAL_FORWARD_ENABLED", False),
            forward_email_enabled=env_bool("VOICEMAIL_PORTAL_FORWARD_EMAIL_ENABLED", False),
            smtp_host=os.environ.get("SMTP_HOST", ""),
            smtp_port=env_int("SMTP_PORT", 25, minimum=1),
            smtp_timeout_seconds=env_int("SMTP_TIMEOUT", 30, minimum=1),
            smtp_starttls=env_bool("SMTP_STARTTLS", True),
            from_address=os.environ.get("VOICEMAIL_FROM_ADDRESS", "").strip(),
            from_name=os.environ.get(
                "VOICEMAIL_FROM_NAME",
                "Local Voicemail Transcription",
            ).strip(),
            fallback_recipient=os.environ.get("VOICEMAIL_FALLBACK_RECIPIENT", "").strip(),
            mwi_refresh_enabled=bool(ami_username and ami_secret),
            ami_host=os.environ.get("VOICEMAIL_PORTAL_AMI_HOST", "127.0.0.1").strip() or "127.0.0.1",
            ami_port=env_int("VOICEMAIL_PORTAL_AMI_PORT", 5038, minimum=1),
            ami_username=ami_username,
            ami_secret=ami_secret,
            ami_timeout_seconds=env_float("VOICEMAIL_PORTAL_AMI_TIMEOUT", 3.0, minimum=0.1),
            ami_default_context=os.environ.get("VOICEMAIL_PORTAL_AMI_CONTEXT", "").strip(),
            log_level=os.environ.get("VOICEMAIL_PORTAL_LOG_LEVEL", "INFO").upper(),
            logo_url=os.environ.get("VOICEMAIL_PORTAL_LOGO_URL", "").strip(),
            logo_file=os.environ.get("VOICEMAIL_PORTAL_LOGO_FILE", "").strip(),
            light_logo_url=os.environ.get("VOICEMAIL_PORTAL_LIGHT_LOGO_URL", "").strip(),
            light_logo_file=os.environ.get("VOICEMAIL_PORTAL_LIGHT_LOGO_FILE", "").strip(),
            brand_name=os.environ.get(
                "VOICEMAIL_PORTAL_BRAND_NAME",
                "Local Voicemail Transcription",
            ).strip()
            or "Local Voicemail Transcription",
            brand_tagline=os.environ.get(
                "VOICEMAIL_PORTAL_BRAND_TAGLINE",
                "Sign in to view recordings and transcripts.",
            ).strip()
            or "Sign in to view recordings and transcripts.",
        )


def normalize_user_extensions(raw_value: Any = None) -> tuple[str, ...]:
    values: list[Any] = []
    if isinstance(raw_value, str):
        values.extend(re.split(r"[\s,;]+", raw_value))
    elif isinstance(raw_value, Iterable) and not isinstance(raw_value, (bytes, bytearray, dict)):
        values.extend(raw_value)

    allowed: list[str] = []
    seen: set[str] = set()
    for value in values:
        extension = str(value or "").strip()
        if not re.fullmatch(r"\d{3,6}", extension):
            continue
        if extension in seen:
            continue
        seen.add(extension)
        allowed.append(extension)
    return tuple(allowed)


def normalize_user_allowed_extensions(primary_extension: str, raw_value: Any = None) -> tuple[str, ...]:
    values: list[Any] = []
    primary = str(primary_extension or "").strip()
    if primary and primary != "*":
        values.append(primary)
    values.extend(normalize_user_extensions(raw_value))
    return normalize_user_extensions(values)


@dataclass(frozen=True)
class PortalUser:
    username: str
    extension: str
    password_hash: str
    display_name: str
    is_admin: bool = False
    allowed_extensions: tuple[str, ...] = ()
    excluded_extensions: tuple[str, ...] = ()

    def accessible_extensions(self) -> tuple[str, ...]:
        return normalize_user_allowed_extensions(self.extension, self.allowed_extensions)

    def inaccessible_extensions(self) -> tuple[str, ...]:
        return normalize_user_extensions(self.excluded_extensions)

    def can_access_extension(self, extension: str) -> bool:
        requested = str(extension or "").strip()
        if requested in self.inaccessible_extensions():
            return False
        return self.is_admin or self.extension == "*" or requested in self.accessible_extensions()


class BulkDeleteRequest(BaseModel):
    file_keys: list[str]
    comment: Optional[str] = None


class DeleteVoicemailRequest(BaseModel):
    comment: Optional[str] = None


class ForwardVoicemailRequest(BaseModel):
    target_extension: str


class SaveVoicemailCommentRequest(BaseModel):
    comment: Optional[str] = None


def delete_comment_required_for_user(
    user: PortalUser,
    settings: Optional[Settings] = None,
    voicemail_extension: Optional[str] = None,
) -> bool:
    active_settings = settings or SETTINGS
    protected_extensions = set(active_settings.delete_comment_user_extensions)
    return user.extension in protected_extensions or (
        voicemail_extension is not None and str(voicemail_extension) in protected_extensions
    )


def normalize_delete_comment(comment: Optional[str]) -> Optional[str]:
    if comment is None:
        return None
    trimmed = comment.strip()
    if not trimmed:
        return None
    if len(trimmed) > DELETE_COMMENT_MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Morgan Example must be {DELETE_COMMENT_MAX_CHARS} characters or fewer",
        )
    return trimmed


def validate_delete_comment_for_user(
    user: PortalUser,
    comment: Optional[str],
    settings: Optional[Settings] = None,
    voicemail_extension: Optional[str] = None,
) -> Optional[str]:
    normalized = normalize_delete_comment(comment)
    if delete_comment_required_for_user(user, settings, voicemail_extension) and not normalized:
        raise HTTPException(status_code=400, detail="A comment is required to delete this message.")
    return normalized


SETTINGS = Settings.from_env()
FAVICON_PATH = Path(__file__).resolve().parent / "lvt_assets" / "voicemail-portal-icon.svg"

logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _ami_header_value(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    if "\r" in text or "\n" in text:
        raise ValueError(f"{label} must not contain line breaks")
    return text


def _send_ami_action(sock: socket.socket, headers: dict[str, Any]) -> None:
    payload = "".join(
        f"{key}: {_ami_header_value(value, key)}\r\n"
        for key, value in headers.items()
    )
    sock.sendall((payload + "\r\n").encode("utf-8"))


def _read_ami_message(sock: socket.socket) -> str:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        data = b"".join(chunks)
        if b"\r\n\r\n" in data or b"\n\n" in data:
            break
    return b"".join(chunks).decode("utf-8", errors="replace")


def _ami_response_success(message: str) -> bool:
    for line in message.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "response":
            return value.strip().lower() == "success"
    return False


def send_ami_voicemail_refresh(context: str, mailbox: str, settings: Settings) -> bool:
    try:
        safe_context = _ami_header_value(context, "Context")
        safe_mailbox = _ami_header_value(mailbox, "Mailbox")
        safe_username = _ami_header_value(settings.ami_username, "AMI username")
        safe_secret = _ami_header_value(settings.ami_secret, "AMI secret")
        with socket.create_connection(
            (settings.ami_host, settings.ami_port),
            timeout=settings.ami_timeout_seconds,
        ) as sock:
            sock.settimeout(settings.ami_timeout_seconds)
            try:
                _read_ami_message(sock)
            except socket.timeout:
                pass
            _send_ami_action(
                sock,
                {
                    "Action": "Login",
                    "Username": safe_username,
                    "Secret": safe_secret,
                    "Events": "off",
                },
            )
            login_response = _read_ami_message(sock)
            if not _ami_response_success(login_response):
                raise RuntimeError("AMI login did not succeed")
            _send_ami_action(
                sock,
                {
                    "Action": "VoicemailRefresh",
                    "Context": safe_context,
                    "Mailbox": safe_mailbox,
                },
            )
            _send_ami_action(sock, {"Action": "Logoff"})
        return True
    except Exception as exc:
        logger.warning(
            "AMI MWI refresh failed host=%s port=%s context=%s mailbox=%s error=%s",
            settings.ami_host,
            settings.ami_port,
            context,
            mailbox,
            exc,
        )
        return False


def _path_mailbox_context_parts(path: str, settings: Settings) -> tuple[str, ...]:
    candidate = Path(str(path or ""))
    if not str(candidate):
        return ()
    try:
        return candidate.resolve(strict=False).relative_to(Path(settings.watch_dir).resolve(strict=False)).parts
    except ValueError:
        return candidate.parts


def mailbox_context_from_path(path: str, settings: Settings = SETTINGS) -> Optional[tuple[str, str]]:
    parts = _path_mailbox_context_parts(path, settings)
    for index, part in enumerate(parts):
        if str(part).upper() == "INBOX" and index >= 2:
            context = str(parts[index - 2]).strip()
            mailbox = str(parts[index - 1]).strip()
            if context and mailbox:
                return context, mailbox
    return None


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None


def voicemail_mwi_target_for_row(row: Any, settings: Settings = SETTINGS) -> Optional[tuple[str, str]]:
    for key in ("txt_path", "wav_path"):
        target = mailbox_context_from_path(str(_row_value(row, key) or ""), settings)
        if target:
            return target

    context = settings.ami_default_context.strip()
    mailbox = str(_row_value(row, "mailbox") or _row_value(row, "extension") or "").strip()
    if context and mailbox:
        return context, mailbox
    return None


def refresh_voicemail_mwi_for_row(row: Any, settings: Settings = SETTINGS, reason: str = "") -> bool:
    if not settings.mwi_refresh_enabled:
        return False
    try:
        target = voicemail_mwi_target_for_row(row, settings)
        if not target:
            logger.warning("MWI refresh skipped because mailbox target could not be derived reason=%s", reason)
            return False
        context, mailbox = target
        refreshed = send_ami_voicemail_refresh(context, mailbox, settings)
        if refreshed:
            logger.info("MWI refresh requested context=%s mailbox=%s reason=%s", context, mailbox, reason)
        return refreshed
    except Exception as exc:
        logger.warning("MWI refresh failed reason=%s error=%s", reason, exc)
        return False


def refresh_voicemail_mwi_for_rows(rows: list[Any], settings: Settings = SETTINGS, reason: str = "") -> int:
    if not settings.mwi_refresh_enabled:
        return 0
    refreshed = 0
    seen: set[tuple[str, str]] = set()
    for row in rows:
        try:
            target = voicemail_mwi_target_for_row(row, settings)
        except Exception as exc:
            logger.warning("MWI refresh target derivation failed reason=%s error=%s", reason, exc)
            continue
        if not target or target in seen:
            continue
        seen.add(target)
        context, mailbox = target
        try:
            if send_ami_voicemail_refresh(context, mailbox, settings):
                refreshed += 1
                logger.info("MWI refresh requested context=%s mailbox=%s reason=%s", context, mailbox, reason)
        except Exception as exc:
            logger.warning("MWI refresh failed reason=%s error=%s", reason, exc)
    return refreshed


def app_path(path: str = "") -> str:
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{SETTINGS.base_path}{suffix}" if SETTINGS.base_path else suffix


def favicon_link() -> str:
    href = html.escape(f"{app_path('/brand/favicon')}?v=1", quote=True)
    return f'<link rel="icon" type="image/svg+xml" href="{href}">'


def logo_src(light: bool = False) -> str:
    if light and SETTINGS.light_logo_url:
        return SETTINGS.light_logo_url
    if light and SETTINGS.light_logo_file:
        return app_path("/brand/logo/light")
    if SETTINGS.logo_url:
        return SETTINGS.logo_url
    if SETTINGS.logo_file:
        return app_path("/brand/logo")
    return ""


def logo_img(class_name: str, allow_light_variant: bool = False) -> str:
    src = logo_src()
    if not src:
        return ""
    escaped_class = html.escape(class_name)
    escaped_src = html.escape(src)
    escaped_alt = html.escape(SETTINGS.brand_name, quote=True)
    if not allow_light_variant:
        return f'<img class="{escaped_class}" src="{escaped_src}" alt="{escaped_alt}">'

    light_src = logo_src(light=True)
    if not light_src or light_src == src:
        return f'<img class="{escaped_class} logo-needs-light-plate" src="{escaped_src}" alt="{escaped_alt}">'

    escaped_light_src = html.escape(light_src)
    return (
        f'<img class="{escaped_class} logo-dark" src="{escaped_src}" alt="{escaped_alt}">'
        f'<img class="{escaped_class} logo-light" src="{escaped_light_src}" alt="{escaped_alt}">'
    )


def cookie_path() -> str:
    return SETTINGS.base_path or "/"


class PortalStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._last_sync_monotonic = 0.0
        self._ensure_parent_directory()
        self.ensure_schema()

    def _ensure_parent_directory(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.settings.state_db))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.settings.state_db, timeout=30)
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

    def ensure_schema(self) -> None:
        with self._transaction() as conn:
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voicemail_forward_events (
                    id INTEGER PRIMARY KEY,
                    source_file_key TEXT NOT NULL,
                    target_file_key TEXT NOT NULL,
                    source_extension TEXT NOT NULL,
                    target_extension TEXT NOT NULL,
                    forwarded_by TEXT NOT NULL,
                    forwarded_utc TEXT NOT NULL,
                    email_sent INTEGER NOT NULL DEFAULT 0,
                    email_error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_voicemail_forward_events_source
                ON voicemail_forward_events(source_file_key, forwarded_utc)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_voicemail_forward_events_target
                ON voicemail_forward_events(target_file_key)
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

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def migrate_legacy_key(self, legacy_key: str, file_key: str) -> None:
        if legacy_key == file_key:
            return

        with self._transaction() as conn:
            legacy_row = conn.execute(
                "SELECT file_key FROM voicemail_transcripts WHERE file_key = ?",
                (legacy_key,),
            ).fetchone()
            if not legacy_row:
                return

            existing_row = conn.execute(
                "SELECT file_key FROM voicemail_transcripts WHERE file_key = ?",
                (file_key,),
            ).fetchone()
            if existing_row:
                return

            conn.execute(
                "UPDATE voicemail_transcripts SET file_key = ? WHERE file_key = ?",
                (file_key, legacy_key),
            )

            legacy_voicemail = conn.execute(
                "SELECT file_key FROM voicemails WHERE file_key = ?",
                (legacy_key,),
            ).fetchone()
            existing_voicemail = conn.execute(
                "SELECT file_key FROM voicemails WHERE file_key = ?",
                (file_key,),
            ).fetchone()
            if legacy_voicemail and not existing_voicemail:
                conn.execute(
                    "UPDATE voicemails SET file_key = ? WHERE file_key = ?",
                    (file_key, legacy_key),
                )

        logger.info("Migrated legacy voicemail key legacy_key=%s file_key=%s", legacy_key, file_key)

    def _row_matches_current_message(
        self,
        row: sqlite3.Row,
        msg_name: str,
        info: dict[str, str],
    ) -> bool:
        return (
            str(row["msg_name"] or "") == msg_name
            and optional_int(info.get("origtime")) == row["origtime"]
            and str(row["callerid"] or "").strip() == info.get("callerid", "").strip()
        )

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

    def _copy_auxiliary_rows_to_key(self, conn: sqlite3.Connection, source_key: str, target_key: str) -> None:
        for table in [
            "voicemail_field_verification",
            "asr_runs",
            "asr_span_candidates",
            "transcript_corrections",
        ]:
            if not self._table_has_file_key(conn, table):
                continue
            source_count = conn.execute(f"SELECT count(*) FROM {table} WHERE file_key = ?", (source_key,)).fetchone()[0]
            target_count = conn.execute(f"SELECT count(*) FROM {table} WHERE file_key = ?", (target_key,)).fetchone()[0]
            if source_count and not target_count:
                try:
                    conn.execute(f"UPDATE {table} SET file_key = ? WHERE file_key = ?", (target_key, source_key))
                except sqlite3.IntegrityError:
                    logger.warning(
                        "Could not re-key auxiliary portal rows table=%s source_key=%s target_key=%s",
                        table,
                        source_key,
                        target_key,
                    )

    def _merge_duplicate_message_row(
        self,
        conn: sqlite3.Connection,
        source_key: str,
        target_key: str,
        now: str,
    ) -> None:
        source = conn.execute(
            """
            SELECT transcript, entities_json
            FROM voicemail_transcripts
            WHERE file_key = ?
            """,
            (source_key,),
        ).fetchone()
        target = conn.execute(
            """
            SELECT transcript, entities_json
            FROM voicemail_transcripts
            WHERE file_key = ?
            """,
            (target_key,),
        ).fetchone()
        if source and target and (source["transcript"] or "").strip() and not (target["transcript"] or "").strip():
            conn.execute(
                """
                UPDATE voicemail_transcripts
                SET transcript = ?,
                    entities_json = ?,
                    updated_utc = ?
                WHERE file_key = ?
                """,
                (source["transcript"], source["entities_json"] or "{}", now, target_key),
            )

        source_queue = conn.execute(
            "SELECT * FROM voicemails WHERE file_key = ?",
            (source_key,),
        ).fetchone()
        target_queue = conn.execute(
            "SELECT * FROM voicemails WHERE file_key = ?",
            (target_key,),
        ).fetchone()
        if source_queue and not target_queue:
            conn.execute("UPDATE voicemails SET file_key = ? WHERE file_key = ?", (target_key, source_key))
        elif source_queue and target_queue and source_queue["status"] == "completed" and target_queue["status"] not in {
            "completed",
            "skipped",
            "dead",
        }:
            conn.execute(
                """
                UPDATE voicemails
                SET status = 'completed',
                    emailed_utc = COALESCE(emailed_utc, ?),
                    transcript_chars = COALESCE(transcript_chars, ?),
                    updated_utc = ?,
                    last_error = NULL
                WHERE file_key = ?
                """,
                (source_queue["emailed_utc"], source_queue["transcript_chars"], now, target_key),
            )

        self._copy_auxiliary_rows_to_key(conn, source_key, target_key)

    def _retire_duplicate_message_row(
        self,
        conn: sqlite3.Connection,
        file_key: str,
        now: str,
        reason: str,
    ) -> None:
        conn.execute(
            """
            UPDATE voicemail_transcripts
            SET folder = 'Deleted',
                deleted_utc = COALESCE(deleted_utc, ?),
                deleted_by = COALESCE(deleted_by, ?),
                updated_utc = ?
            WHERE file_key = ?
            """,
            (now, reason, now, file_key),
        )
        conn.execute(
            """
            UPDATE voicemails
            SET status = 'dead',
                updated_utc = ?,
                last_error = ?
            WHERE file_key = ?
              AND status NOT IN ('completed', 'skipped', 'dead')
            """,
            (now, f"portal sync: {reason}", file_key),
        )

    def repair_active_path_duplicates(self) -> int:
        now = utc_now_iso()
        repaired = 0
        with self._transaction() as conn:
            groups = conn.execute(
                """
                SELECT txt_path
                FROM voicemail_transcripts
                WHERE deleted_utc IS NULL
                  AND folder = 'INBOX'
                GROUP BY txt_path
                HAVING count(*) > 1
                """
            ).fetchall()

            for group in groups:
                txt_path = str(group["txt_path"] or "")
                try:
                    info = parse_txt(txt_path)
                except OSError:
                    continue
                extension = extract_extension(txt_path)
                current_key = build_file_key(extension, info, txt_path) if extension else None
                msg_name = os.path.splitext(os.path.basename(txt_path))[0]
                rows = conn.execute(
                    """
                    SELECT t.*,
                           v.status AS queue_status,
                           CASE WHEN t.transcript IS NULL OR trim(t.transcript) = '' THEN 0 ELSE 1 END AS has_transcript,
                           CASE WHEN v.file_key IS NULL THEN 0 ELSE 1 END AS has_queue_row,
                           CASE WHEN v.status = 'completed' THEN 1 ELSE 0 END AS queue_completed
                    FROM voicemail_transcripts t
                    LEFT JOIN voicemails v ON v.file_key = t.file_key
                    WHERE t.deleted_utc IS NULL
                      AND t.folder = 'INBOX'
                      AND t.txt_path = ?
                    """,
                    (txt_path,),
                ).fetchall()
                current_rows = [row for row in rows if self._row_matches_current_message(row, msg_name, info)]
                current_keys = {str(row["file_key"]) for row in current_rows}
                stale_rows = [row for row in rows if str(row["file_key"]) not in current_keys]
                candidates = current_rows or rows
                candidates.sort(
                    key=lambda row: (
                        1 if current_key and row["file_key"] == current_key else 0,
                        row["has_transcript"],
                        row["queue_completed"],
                        row["has_queue_row"],
                        str(row["updated_utc"] or ""),
                    ),
                    reverse=True,
                )
                keep_key = str(candidates[0]["file_key"])
                for row in stale_rows:
                    old_key = str(row["file_key"])
                    if old_key != keep_key:
                        self._retire_duplicate_message_row(conn, old_key, now, "deduped_by_current_inbox_path_reused")
                        repaired += 1
                for row in candidates[1:]:
                    old_key = str(row["file_key"])
                    if old_key == keep_key:
                        continue
                    self._merge_duplicate_message_row(conn, old_key, keep_key, now)
                    self._retire_duplicate_message_row(conn, old_key, now, "deduped_by_current_inbox_path")
                    repaired += 1
        return repaired

    def enqueue_missing_watcher_rows(self) -> int:
        now = utc_now_iso()
        queued = 0
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT t.file_key, t.extension, t.txt_path, t.wav_path
                FROM voicemail_transcripts t
                LEFT JOIN voicemails v ON v.file_key = t.file_key
                WHERE t.deleted_utc IS NULL
                  AND t.folder = 'INBOX'
                  AND v.file_key IS NULL
                  AND (t.transcript IS NULL OR trim(t.transcript) = '')
                """
            ).fetchall()
            for row in rows:
                txt_path = str(row["txt_path"] or "")
                wav_path = str(row["wav_path"] or "")
                if not txt_path or not wav_path or not os.path.exists(txt_path) or not os.path.exists(wav_path):
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO voicemails (
                        file_key, status, extension, txt_path, wav_path,
                        attempts, first_seen_utc, updated_utc, last_error
                    )
                    VALUES (?, 'discovered', ?, ?, ?, 0, ?, ?, NULL)
                    """,
                    (row["file_key"], row["extension"], txt_path, wav_path, now, now),
                )
                queued += conn.execute("SELECT changes()").fetchone()[0]
        return queued

    def upsert_discovered(
        self,
        file_key: str,
        extension: str,
        txt_path: str,
        wav_path: str,
        info: dict[str, str],
    ) -> None:
        now = utc_now_iso()
        mailbox = info.get("origmailbox", extension).strip() or extension
        msg_name = os.path.splitext(os.path.basename(txt_path))[0]

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO voicemail_transcripts (
                    file_key, extension, mailbox, folder, msg_name, txt_path, wav_path,
                    callerid, origtime, origdate, duration, transcript, entities_json,
                    created_utc, updated_utc, deleted_utc, deleted_by, deleted_comment
                )
                VALUES (?, ?, ?, 'INBOX', ?, ?, ?, ?, ?, ?, ?, NULL, '{}', ?, ?, NULL, NULL, NULL)
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
                    now,
                    now,
                ),
            )

    def maybe_sync_filesystem(self) -> None:
        now = time.monotonic()
        if now - self._last_sync_monotonic < self.settings.sync_interval_seconds:
            return
        self._last_sync_monotonic = now
        self.sync_filesystem()

    def sync_filesystem(self) -> int:
        count = 0
        retained_copied = 0
        purged_deleted = self.purge_expired_deleted_voicemails()
        if not os.path.isdir(self.settings.watch_dir):
            logger.warning("Voicemail watch directory does not exist: %s", self.settings.watch_dir)
            return count

        for root, _, files in os.walk(self.settings.watch_dir):
            if "/INBOX" not in normalize_path(root):
                continue
            for name in files:
                if not name.endswith(".txt") or "msg" not in name:
                    continue

                txt_path = os.path.join(root, name)
                if not is_voicemail_txt(txt_path):
                    continue

                try:
                    extension = extract_extension(txt_path)
                    if not extension:
                        continue
                    info = parse_txt(txt_path)
                    file_key = build_file_key(extension, info, txt_path)
                    if not file_key:
                        continue
                    legacy_key = build_legacy_file_key(extension, info, txt_path)
                    if legacy_key:
                        self.migrate_legacy_key(legacy_key, file_key)
                    self.upsert_discovered(file_key, extension, txt_path, matching_wav_path(txt_path), info)
                    retained_copied += copy_message_to_external_retention(file_key, txt_path, self.settings)
                    count += 1
                except OSError as exc:
                    logger.warning("Could not sync voicemail metadata path=%s error=%s", txt_path, exc)
                except HTTPException as exc:
                    logger.warning("Could not retain voicemail files path=%s error=%s", txt_path, exc.detail)

        repaired_duplicates = self.repair_active_path_duplicates()
        auto_deleted = self.auto_delete_short_voicemails()
        queued_missing = self.enqueue_missing_watcher_rows()
        self.mark_missing_files_deleted()
        logger.info(
            (
                "Portal filesystem sync saw %s INBOX voicemail metadata file(s), "
                "deduped_active_paths=%s auto_deleted_short=%s queued_missing_watcher_rows=%s "
                "retained_copied=%s purged_deleted=%s"
            ),
            count,
            repaired_duplicates,
            auto_deleted,
            queued_missing,
            retained_copied,
            purged_deleted,
        )
        return count

    def auto_delete_short_voicemails(self) -> int:
        threshold = self.settings.auto_delete_short_seconds
        if threshold <= 0:
            return 0

        deleted = 0
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM voicemail_transcripts
                WHERE deleted_utc IS NULL
                  AND folder = 'INBOX'
                  AND duration IS NOT NULL
                  AND duration < ?
                ORDER BY COALESCE(origtime, 0) DESC, created_utc DESC
                """,
                (threshold,),
            ).fetchall()

        for row in rows:
            try:
                moved = move_message_to_trash(row, self.settings)
                if not moved:
                    logger.warning(
                        "Short voicemail auto-delete found no movable files file_key=%s duration=%s threshold=%s",
                        row["file_key"],
                        row["duration"],
                        threshold,
                    )
                    continue
                self.mark_deleted(str(row["file_key"]), "portal-auto-short", moved)
                refresh_voicemail_mwi_for_row(row, self.settings, reason="auto-delete short voicemail")
                deleted += 1
                logger.info(
                    "Short voicemail auto-deleted file_key=%s extension=%s duration=%ss threshold=%ss moved_files=%s",
                    row["file_key"],
                    row["extension"],
                    row["duration"],
                    threshold,
                    len(moved),
                )
            except Exception as exc:
                logger.warning(
                    "Short voicemail auto-delete failed file_key=%s duration=%s threshold=%s error=%s",
                    row["file_key"],
                    row["duration"],
                    threshold,
                    exc,
                )
        return deleted

    def mark_missing_files_deleted(self) -> int:
        now = utc_now_iso()
        changed = 0
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT file_key, extension, msg_name, txt_path, wav_path
                FROM voicemail_transcripts
                WHERE deleted_utc IS NULL
                """
            ).fetchall()
            for row in rows:
                txt_path = str(row["txt_path"])
                wav_path = str(row["wav_path"])
                if os.path.exists(txt_path) or os.path.exists(wav_path):
                    continue
                moved_txt_path, moved_wav_path = find_moved_message_paths(row, self.settings)
                if not moved_txt_path or not moved_wav_path:
                    retained_txt_path, retained_wav_path = find_retained_message_paths(row, self.settings)
                    if retained_txt_path and (not moved_txt_path or not moved_wav_path):
                        moved_txt_path = retained_txt_path
                        moved_wav_path = retained_wav_path
                conn.execute(
                    """
                    UPDATE voicemail_transcripts
                    SET folder = 'Deleted',
                        txt_path = COALESCE(?, txt_path),
                        wav_path = COALESCE(?, wav_path),
                        deleted_utc = ?,
                        deleted_by = COALESCE(deleted_by, 'external')
                    WHERE file_key = ? AND deleted_utc IS NULL
                    """,
                    (moved_txt_path, moved_wav_path, now, row["file_key"]),
                )
                changed += 1
        return changed

    def _delete_trash_files_for_row(self, row: sqlite3.Row) -> int:
        trash_root = Path(self.settings.trash_dir).resolve()
        file_key = str(row["file_key"] or "")
        deleted_files = 0
        pruned_dirs: set[Path] = set()

        if file_key:
            for stamp_dir in trash_root.glob("*"):
                key_dir = stamp_dir / file_key
                try:
                    key_dir = safe_under_root(str(key_dir), str(trash_root))
                except HTTPException:
                    continue
                if key_dir.is_dir():
                    for path in key_dir.rglob("*"):
                        if path.is_file():
                            deleted_files += 1
                    shutil.rmtree(key_dir)
                    pruned_dirs.add(key_dir.parent)

        if deleted_files == 0:
            for raw_path in (row["txt_path"], row["wav_path"]):
                if not raw_path:
                    continue
                try:
                    path = safe_under_root(str(raw_path), str(trash_root))
                except HTTPException:
                    logger.warning(
                        "Skipping retention unlink outside trash file_key=%s path=%s",
                        row["file_key"],
                        raw_path,
                    )
                    continue
                parent = path.parent
                stem = path.stem or str(row["msg_name"] or "")
                if not stem:
                    continue
                for candidate in parent.glob(f"{stem}.*"):
                    try:
                        candidate = safe_under_root(str(candidate), str(trash_root))
                    except HTTPException:
                        continue
                    if candidate.is_file():
                        candidate.unlink()
                        deleted_files += 1
                        pruned_dirs.add(candidate.parent)

        for directory in sorted(pruned_dirs, key=lambda value: len(value.parts), reverse=True):
            current = directory
            while current != trash_root:
                try:
                    current.rmdir()
                except OSError:
                    break
                current = current.parent

        return deleted_files

    def purge_expired_deleted_voicemails(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.settings.deleted_retention_days)).isoformat()
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM voicemail_transcripts
                WHERE deleted_utc IS NOT NULL
                  AND deleted_utc < ?
                ORDER BY deleted_utc ASC
                """,
                (cutoff,),
            ).fetchall()

        purged = 0
        deleted_files = 0
        for row in rows:
            file_key = str(row["file_key"])
            try:
                deleted_files += self._delete_trash_files_for_row(row)
            except OSError as exc:
                logger.warning("Could not purge deleted voicemail files file_key=%s error=%s", file_key, exc)
                continue

            with self._transaction() as conn:
                self._delete_auxiliary_rows(conn, file_key)
                conn.execute("DELETE FROM voicemails WHERE file_key = ?", (file_key,))
                conn.execute(
                    "DELETE FROM voicemail_transcripts WHERE file_key = ? AND deleted_utc IS NOT NULL",
                    (file_key,),
                )
            purged += 1

        if purged:
            logger.info(
                "Purged expired deleted voicemails rows=%s files=%s retention_days=%s cutoff=%s",
                purged,
                deleted_files,
                self.settings.deleted_retention_days,
                cutoff,
            )
        return purged

    def list_voicemails(self, user: PortalUser, folder: str = "active") -> list[dict[str, Any]]:
        self.maybe_sync_filesystem()
        params: list[Any] = []
        if folder == "deleted":
            where = "t.deleted_utc IS NOT NULL"
            order_by = "COALESCE(t.deleted_utc, t.updated_utc) DESC, COALESCE(t.origtime, 0) DESC"
        else:
            where = "t.deleted_utc IS NULL"
            order_by = "COALESCE(t.origtime, 0) DESC, t.created_utc DESC"

        if not user.is_admin and user.extension != "*":
            allowed_extensions = user.accessible_extensions()
            if not allowed_extensions:
                return []
            placeholders = ",".join("?" for _ in allowed_extensions)
            where += f" AND t.extension IN ({placeholders})"
            params.extend(allowed_extensions)
        excluded_extensions = user.inaccessible_extensions()
        if excluded_extensions:
            placeholders = ",".join("?" for _ in excluded_extensions)
            where += f" AND t.extension NOT IN ({placeholders})"
            params.extend(excluded_extensions)
        params.append(self.settings.page_limit)

        with self._transaction() as conn:
            rows = conn.execute(
                f"""
                SELECT t.*, v.status AS processing_status
                FROM voicemail_transcripts t
                LEFT JOIN voicemails v ON v.file_key = t.file_key
                WHERE {where}
                ORDER BY {order_by}
                LIMIT ?
                """,
                params,
            ).fetchall()
            verifications_by_key = self._field_verifications_for_keys(
                conn,
                [str(row["file_key"]) for row in rows],
            )
            forwarding_by_key = self._forwarding_events_for_keys(
                conn,
                [str(row["file_key"]) for row in rows],
            )

        return [
            self._row_to_payload(
                row,
                include_transcript=True,
                field_verifications=verifications_by_key.get(str(row["file_key"]), {}),
                forwarding_events=forwarding_by_key.get(str(row["file_key"]), {}),
                user=user,
            )
            for row in rows
        ]

    def _field_verifications_for_keys(
        self,
        conn: sqlite3.Connection,
        file_keys: list[str],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if not file_keys:
            return {}

        placeholders = ",".join("?" for _ in file_keys)
        try:
            rows = conn.execute(
                f"""
                SELECT file_key,
                       field_name,
                       final_value,
                       normalized_value,
                       status,
                       needs_review,
                       review_reasons_json,
                       attribution_json,
                       whisper_json,
                       gemma_json,
                       parakeet_json,
                       clip_json,
                       updated_utc
                FROM voicemail_field_verification
                WHERE file_key IN ({placeholders})
                """,
                file_keys,
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            raise

        result: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            gemma_rows = parse_json_array(row["gemma_json"])
            parakeet_rows = parse_json_array(row["parakeet_json"])
            attribution_rows = parse_json_array(row["attribution_json"])
            clip_rows = parse_json_array(row["clip_json"])
            review_reasons = parse_json_array(row["review_reasons_json"])
            whisper_json = parse_json_object(row["whisper_json"])
            result.setdefault(str(row["file_key"]), {})[str(row["field_name"])] = {
                "field_name": row["field_name"],
                "final_value": row["final_value"],
                "normalized_value": row["normalized_value"],
                "status": row["status"],
                "needs_review": bool(row["needs_review"]),
                "review_reasons": review_reasons,
                "attribution_json": attribution_rows,
                "whisper_json": whisper_json,
                "clip_json": clip_rows,
                "used_gemma": bool(gemma_rows),
                "used_parakeet": bool(parakeet_rows),
                "updated_utc": row["updated_utc"],
            }
        return result

    def _forwarding_events_for_keys(
        self,
        conn: sqlite3.Connection,
        file_keys: list[str],
    ) -> dict[str, dict[str, Any]]:
        if not file_keys:
            return {}

        placeholders = ",".join("?" for _ in file_keys)
        try:
            rows = conn.execute(
                f"""
                SELECT id,
                       source_file_key,
                       target_file_key,
                       source_extension,
                       target_extension,
                       forwarded_by,
                       forwarded_utc,
                       email_sent,
                       email_error
                FROM voicemail_forward_events
                WHERE source_file_key IN ({placeholders})
                   OR target_file_key IN ({placeholders})
                ORDER BY forwarded_utc ASC, id ASC
                """,
                [*file_keys, *file_keys],
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            raise

        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            event = {
                "id": row["id"],
                "source_file_key": row["source_file_key"],
                "target_file_key": row["target_file_key"],
                "source_extension": row["source_extension"],
                "target_extension": row["target_extension"],
                "forwarded_by": row["forwarded_by"],
                "forwarded_utc": row["forwarded_utc"],
                "forwarded_display": format_iso_display(row["forwarded_utc"], self.settings.local_timezone),
                "email_sent": bool(row["email_sent"]),
                "email_error": row["email_error"] or "",
            }
            result.setdefault(str(row["source_file_key"]), {}).setdefault("forwarded_to", []).append(event)
            result.setdefault(str(row["target_file_key"]), {})["forwarded_from"] = event
        return result

    def get_voicemail(self, file_key: str, user: PortalUser, include_deleted: bool = False) -> sqlite3.Row:
        if not EMAIL_SAFE_FILE_KEY_RE.match(file_key):
            raise HTTPException(status_code=404, detail="Voicemail not found")

        deleted_filter = "" if include_deleted else "AND deleted_utc IS NULL"
        with self._transaction() as conn:
            row = conn.execute(
                f"""
                SELECT *
                FROM voicemail_transcripts
                WHERE file_key = ? {deleted_filter}
                """,
                (file_key,),
            ).fetchone()

        if row is None or not user.can_access_extension(str(row["extension"])):
            raise HTTPException(status_code=404, detail="Voicemail not found")
        return row

    def mark_deleted(
        self,
        file_key: str,
        username: str,
        moved_paths: Optional[list[str]] = None,
        deleted_comment: Optional[str] = None,
    ) -> None:
        now = utc_now_iso()
        moved_paths = moved_paths or []
        moved_txt_path = next((path for path in moved_paths if Path(path).suffix.lower() == ".txt"), None)
        moved_wav_path = next((path for path in moved_paths if Path(path).suffix.lower() == ".wav"), None)
        deleted_comment = normalize_delete_comment(deleted_comment)
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE voicemail_transcripts
                SET folder = 'Deleted',
                    txt_path = COALESCE(?, txt_path),
                    wav_path = COALESCE(?, wav_path),
                    deleted_utc = ?,
                    deleted_by = ?,
                    deleted_comment = ?
                WHERE file_key = ?
                """,
                (moved_txt_path, moved_wav_path, now, username, deleted_comment, file_key),
            )
            conn.execute(
                """
                UPDATE voicemails
                SET status = 'skipped', updated_utc = ?, last_error = ?
                WHERE file_key = ? AND status NOT IN ('completed', 'skipped', 'dead')
                """,
                (now, f"Deleted from voicemail portal by {username}", file_key),
            )
        try:
            remove_external_retention_for_key(file_key, self.settings)
        except OSError as exc:
            logger.warning("Could not remove retained copy after portal delete file_key=%s error=%s", file_key, exc)

    def save_comment(self, file_key: str, user: PortalUser, comment: Optional[str]) -> Optional[str]:
        row = self.get_voicemail(file_key, user)
        normalized_comment = normalize_delete_comment(comment)
        now = utc_now_iso()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE voicemail_transcripts
                SET deleted_comment = ?,
                    updated_utc = ?
                WHERE file_key = ?
                  AND deleted_utc IS NULL
                """,
                (normalized_comment, now, row["file_key"]),
            )
        return normalized_comment

    def mark_restored(
        self,
        old_file_key: str,
        new_file_key: str,
        txt_path: str,
        wav_path: str,
        msg_name: str,
    ) -> None:
        now = utc_now_iso()
        with self._transaction() as conn:
            if old_file_key != new_file_key:
                existing = conn.execute(
                    "SELECT file_key FROM voicemail_transcripts WHERE file_key = ?",
                    (new_file_key,),
                ).fetchone()
                if existing:
                    raise HTTPException(status_code=409, detail="Restored voicemail key already exists")

            conn.execute(
                """
                UPDATE voicemail_transcripts
                SET file_key = ?,
                    folder = 'INBOX',
                    msg_name = ?,
                    txt_path = ?,
                    wav_path = ?,
                    updated_utc = ?,
                    deleted_utc = NULL,
                    deleted_by = NULL,
                    deleted_comment = NULL
                WHERE file_key = ?
                """,
                (new_file_key, msg_name, txt_path, wav_path, now, old_file_key),
            )

            existing_voicemail = conn.execute(
                "SELECT file_key FROM voicemails WHERE file_key = ?",
                (new_file_key,),
            ).fetchone()
            if old_file_key != new_file_key and existing_voicemail:
                conn.execute("DELETE FROM voicemails WHERE file_key = ?", (old_file_key,))
                should_update_voicemail_row = False
            else:
                should_update_voicemail_row = True

            if should_update_voicemail_row:
                conn.execute(
                    """
                    UPDATE voicemails
                    SET file_key = ?,
                        txt_path = ?,
                        wav_path = ?,
                        updated_utc = ?
                    WHERE file_key = ?
                    """,
                    (new_file_key, txt_path, wav_path, now, old_file_key),
                )
        try:
            remove_external_retention_for_key(old_file_key, self.settings)
        except OSError as exc:
            logger.warning("Could not remove retained copy after restore file_key=%s error=%s", old_file_key, exc)

    def create_forwarded_copy_record(
        self,
        source_row: sqlite3.Row,
        copied_message: dict[str, Any],
        target_extension: str,
        forwarded_by: str,
    ) -> None:
        now = utc_now_iso()
        source_transcript = str(source_row["transcript"] or "")
        source_entities_json = str(source_row["entities_json"] or "{}")
        target_file_key = str(copied_message["file_key"])
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT file_key FROM voicemail_transcripts WHERE file_key = ?",
                (target_file_key,),
            ).fetchone()
            if existing:
                raise HTTPException(status_code=409, detail="Forwarded voicemail key already exists")

            conn.execute(
                """
                INSERT INTO voicemail_transcripts (
                    file_key, extension, mailbox, folder, msg_name, txt_path, wav_path,
                    callerid, origtime, origdate, duration, transcript, entities_json,
                    created_utc, updated_utc, deleted_utc, deleted_by, deleted_comment
                )
                VALUES (?, ?, ?, 'INBOX', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    target_file_key,
                    target_extension,
                    target_extension,
                    str(copied_message["msg_name"]),
                    str(copied_message["txt_path"]),
                    str(copied_message["wav_path"]),
                    source_row["callerid"],
                    source_row["origtime"],
                    source_row["origdate"],
                    source_row["duration"],
                    source_transcript,
                    source_entities_json,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO voicemails (
                    file_key, status, extension, txt_path, wav_path,
                    attempts, first_seen_utc, updated_utc, last_error,
                    emailed_utc, transcript_chars
                )
                VALUES (?, 'completed', ?, ?, ?, 1, ?, ?, NULL, NULL, ?)
                """,
                (
                    target_file_key,
                    target_extension,
                    str(copied_message["txt_path"]),
                    str(copied_message["wav_path"]),
                    now,
                    now,
                    len(source_transcript),
                ),
            )
            conn.execute(
                """
                INSERT INTO voicemail_forward_events (
                    source_file_key, target_file_key, source_extension, target_extension,
                    forwarded_by, forwarded_utc, email_sent, email_error
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, '')
                """,
                (
                    source_row["file_key"],
                    target_file_key,
                    source_row["extension"],
                    target_extension,
                    forwarded_by,
                    now,
                ),
            )

    def mark_forward_email_result(
        self,
        target_file_key: str,
        email_sent: bool,
        email_error: str = "",
    ) -> None:
        now = utc_now_iso()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE voicemail_forward_events
                SET email_sent = ?,
                    email_error = ?
                WHERE target_file_key = ?
                """,
                (1 if email_sent else 0, email_error[:1000], target_file_key),
            )
            if email_sent:
                conn.execute(
                    """
                    UPDATE voicemails
                    SET emailed_utc = COALESCE(emailed_utc, ?),
                        updated_utc = ?
                    WHERE file_key = ?
                    """,
                    (now, now, target_file_key),
                )

    def _row_to_payload(
        self,
        row: sqlite3.Row,
        include_transcript: bool,
        field_verifications: Optional[dict[str, dict[str, Any]]] = None,
        forwarding_events: Optional[dict[str, Any]] = None,
        user: Optional[PortalUser] = None,
    ) -> dict[str, Any]:
        transcript = row["transcript"] or ""
        try:
            entities = json.loads(row["entities_json"] or "{}")
        except json.JSONDecodeError:
            entities = {}
        if not isinstance(entities, dict):
            entities = {}

        callerid = row["callerid"] or "Unknown"
        entities["callback_matches_caller_id"] = callback_matches_caller_id(
            callerid,
            entities.get("callback_number"),
        )
        word_timestamps = normalize_word_timestamps(entities.get("_word_timestamps"))

        wav_path = str(row["wav_path"] or "")
        txt_path = str(row["txt_path"] or "")
        return {
            "file_key": row["file_key"],
            "extension": row["extension"],
            "mailbox": row["mailbox"] or row["extension"],
            "folder": row["folder"] or "INBOX",
            "callerid": format_callerid_display(callerid),
            "callerid_number_digits": caller_number_digits_from_callerid(callerid),
            "origtime": row["origtime"],
            "origdate": row["origdate"] or "",
            "display_date": format_display_date(row["origtime"], row["origdate"] or "", SETTINGS.local_timezone),
            "duration": row["duration"],
            "duration_display": format_duration(row["duration"]),
            "has_audio": bool(wav_path and os.path.exists(wav_path)),
            "has_metadata": bool(txt_path and os.path.exists(txt_path)),
            "has_transcript": bool(transcript.strip()),
            "transcript": transcript if include_transcript else "",
            "word_timestamps": word_timestamps if include_transcript else [],
            "entities": entities,
            "field_verifications": field_verifications or {},
            "forwarded_to": list((forwarding_events or {}).get("forwarded_to") or []),
            "forwarded_from": (forwarding_events or {}).get("forwarded_from"),
            "processing_status": row["processing_status"] if "processing_status" in row.keys() else None,
            "deleted_utc": row["deleted_utc"] if "deleted_utc" in row.keys() else None,
            "deleted_by": row["deleted_by"] if "deleted_by" in row.keys() else None,
            "deleted_comment": row["deleted_comment"] if "deleted_comment" in row.keys() else None,
            "delete_comment_required": (
                delete_comment_required_for_user(user, self.settings, row["extension"]) if user else False
            ),
            "deleted_display": format_iso_display(
                row["deleted_utc"] if "deleted_utc" in row.keys() else None,
                SETTINGS.local_timezone,
            ),
        }


STORE: Optional[PortalStore] = None


def get_store() -> PortalStore:
    global STORE
    if STORE is None:
        STORE = PortalStore(SETTINGS)
    return STORE


def load_manual_users(settings: Settings = SETTINGS) -> dict[str, PortalUser]:
    try:
        with open(settings.users_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        if settings.auto_users:
            return {}
        raise RuntimeError(f"Portal users file not found: {settings.users_file}")

    if not isinstance(payload, list):
        raise RuntimeError("Portal users file must contain a JSON array")

    users: dict[str, PortalUser] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        username = str(item.get("username", "")).strip()
        extension = str(item.get("extension", "")).strip()
        password_hash_value = str(item.get("password_hash", "")).strip()
        if not username or not extension or not password_hash_value:
            continue
        users[username] = PortalUser(
            username=username,
            extension=extension,
            password_hash=password_hash_value,
            display_name=str(item.get("display_name", username)).strip() or username,
            is_admin=bool(item.get("is_admin", False)),
            allowed_extensions=normalize_user_allowed_extensions(extension, item.get("allowed_extensions")),
            excluded_extensions=normalize_user_extensions(item.get("excluded_extensions")),
        )

    if not users and not settings.auto_users:
        raise RuntimeError("Portal users file did not contain any valid users")
    return users


def voicemail_config_paths(settings: Settings = SETTINGS) -> list[str]:
    paths: list[str] = []
    for candidate in [settings.voicemail_config, *glob.glob(settings.voicemail_config_glob)]:
        if candidate and candidate not in paths:
            paths.append(candidate)
    return paths


def clean_mailbox_name(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip(" ,;")


def discover_mailbox_directory_extensions(settings: Settings = SETTINGS) -> set[str]:
    extensions: set[str] = set()
    if not os.path.isdir(settings.watch_dir):
        return extensions

    for root, dirs, _files in os.walk(settings.watch_dir):
        normalized = normalize_path(root)
        for match in re.finditer(r"/(?P<extension>\d{3,6})(?:/|$)", normalized):
            if "INBOX" in dirs or "/INBOX" in normalized:
                extensions.add(match.group("extension"))

    return extensions


def discover_mailbox_directory_names(settings: Settings = SETTINGS) -> dict[str, str]:
    return {extension: "" for extension in discover_mailbox_directory_extensions(settings)}


def discover_mailbox_config_names(settings: Settings = SETTINGS) -> dict[str, str]:
    mailboxes: dict[str, str] = {}

    for config_path in voicemail_config_paths(settings):
        try:
            with open(config_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    match = re.match(r"^\s*(?P<extension>\d{3,6})\s*=>\s*(?P<body>.*)$", line)
                    if not match:
                        continue

                    extension = match.group("extension")
                    parts = [part.strip() for part in match.group("body").split(",")]
                    display_name = clean_mailbox_name(parts[1]) if len(parts) >= 2 else ""
                    if not display_name and extension not in mailboxes:
                        display_name = ""
                    mailboxes[extension] = display_name
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Could not read voicemail config %s: %s", config_path, exc)

    return mailboxes


def discover_mailboxes(settings: Settings = SETTINGS) -> list[dict[str, str]]:
    names = discover_mailbox_directory_names(settings)
    names.update(discover_mailbox_config_names(settings))

    def sort_key(item: tuple[str, str]) -> tuple[int, str]:
        extension, _display_name = item
        try:
            return int(extension), extension
        except ValueError:
            return 999999, extension

    return [
        {"extension": extension, "display_name": display_name}
        for extension, display_name in sorted(names.items(), key=sort_key)
    ]


def discover_mailbox_extensions(settings: Settings = SETTINGS) -> set[str]:
    return {mailbox["extension"] for mailbox in discover_mailboxes(settings)}


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
    raise ValueError("No valid email recipient and fallback is invalid")


def get_email(extension: str, settings: Settings = SETTINGS) -> list[str]:
    recipients: list[str] = []

    for config_path in voicemail_config_paths(settings):
        try:
            with open(config_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    match = re.match(rf"^\s*{re.escape(extension)}\s*=>\s*(?P<body>.*)$", line)
                    if not match:
                        continue

                    parts = [part.strip() for part in match.group("body").split(",")]
                    if len(parts) >= 3:
                        candidate_fields = [parts[2]]
                        candidate_fields.extend(part for part in parts[3:] if "@" in part and "=" not in part)
                        raw = ";".join(candidate_fields)
                        recipients = [addr.strip() for addr in re.split(r"[|;,]", raw) if addr.strip()]
                    return validate_recipients(recipients, settings.fallback_recipient)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Could not read voicemail config %s: %s", config_path, exc)

    return validate_recipients(recipients, settings.fallback_recipient)


def included_or_default(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "Not Included"


def build_forwarded_email_body(
    source_row: sqlite3.Row,
    target_row: sqlite3.Row,
    forwarded_by: str,
    settings: Settings = SETTINGS,
) -> str:
    try:
        entities = json.loads(target_row["entities_json"] or "{}")
    except json.JSONDecodeError:
        entities = {}
    if not isinstance(entities, dict):
        entities = {}

    transcript = str(target_row["transcript"] or "")
    source_extension = str(source_row["extension"])
    target_extension = str(target_row["extension"])
    source_mailbox = str(source_row["mailbox"] or source_extension)
    caller_name, caller_number = parse_callerid_for_email(str(source_row["callerid"] or "Unknown"))
    callback_number = entities.get("callback_number")
    date_str = format_email_date(
        str(source_row["origdate"] or ""),
        settings.local_timezone,
        settings.date_timezone_label,
    )
    separator = "-" * 40
    return (
        f"Forwarded from Ext {source_extension} to Ext {target_extension} by {forwarded_by}.\n\n"
        "Voicemail Transcript\n"
        f"{separator}\n"
        f"Extension: {source_mailbox}\n"
        f"Caller ID: {format_caller_id_for_email(caller_name)}\n"
        f"Caller Number: {caller_number}\n"
        f"Duration: {format_duration(source_row['duration'])}\n"
        f"Date: {date_str}\n\n"
        f"Name: {included_or_default(entities.get('name'))}\n"
        f"DOB: {included_or_default(entities.get('dob'))}\n"
        f"Callback Number: {included_or_default(callback_number)}\n"
        f"Callback Matches Caller ID: {callback_match_status_for_email(caller_number, callback_number)}\n"
        f"Fax Number: {included_or_default(entities.get('fax_number'))}\n\n"
        f"{transcript}\n\n"
        f"{separator}\n"
        "This transcript was generated by AI.\n"
        "Accuracy may vary based on audio quality.\n"
        "Please verify all clinical details.\n"
    )


def send_forwarded_voicemail_email(
    source_row: sqlite3.Row,
    target_row: sqlite3.Row,
    forwarded_by: str,
    settings: Settings = SETTINGS,
) -> dict[str, Any]:
    recipients = get_email(str(target_row["extension"]), settings)
    callerid = sanitize_header(str(source_row["callerid"] or "Unknown"))
    source_extension = str(source_row["extension"])
    target_extension = str(target_row["extension"])
    msg = EmailMessage()
    msg["Subject"] = f"Forwarded Voicemail from {callerid}"
    msg["From"] = f"{sanitize_header(settings.from_name)} <{settings.from_address}>"
    msg["To"] = ", ".join(recipients)
    msg["Message-ID"] = f"<voicemail-forward-{target_row['file_key']}@local.invalid>"
    msg.set_content(build_forwarded_email_body(source_row, target_row, forwarded_by, settings))

    wav_path = safe_under_root(str(target_row["wav_path"]), settings.watch_dir)
    with open(wav_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="audio",
            subtype="wav",
            filename=os.path.basename(wav_path),
        )

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=settings.smtp_timeout_seconds) as smtp:
        smtp.ehlo()
        if settings.smtp_starttls:
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
        smtp.send_message(msg, from_addr=settings.from_address, to_addrs=recipients)

    logger.info(
        "Forwarded voicemail email sent source_key=%s target_key=%s source_ext=%s target_ext=%s recipient_count=%s",
        source_row["file_key"],
        target_row["file_key"],
        source_extension,
        target_extension,
        len(recipients),
    )
    return {"email_sent": True, "recipient_count": len(recipients), "email_error": ""}


def get_portal_user(username: str, settings: Settings = SETTINGS) -> Optional[PortalUser]:
    users = load_manual_users(settings)
    manual_user = users.get(username)
    if manual_user is not None:
        return manual_user

    if not settings.auto_users or not settings.shared_password_hash:
        return None
    if not re.fullmatch(r"\d{3,6}", username):
        return None
    if username not in discover_mailbox_extensions(settings):
        return None

    return PortalUser(
        username=username,
        extension=username,
        password_hash=settings.shared_password_hash,
        display_name=f"Extension {username}",
        is_admin=False,
        allowed_extensions=(username,),
    )


def sign_session(payload: dict[str, Any], settings: Settings = SETTINGS) -> str:
    encoded_payload = b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = hmac.new(settings.session_secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256)
    return f"{encoded_payload}.{b64url_encode(signature.digest())}"


def read_session(token: str, settings: Settings = SETTINGS) -> dict[str, Any]:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        expected = hmac.new(settings.session_secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256)
        if not hmac.compare_digest(b64url_encode(expected.digest()), encoded_signature):
            raise ValueError("bad signature")
        payload = json.loads(b64url_decode(encoded_payload))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        return payload
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required") from exc


def current_user(request: Request) -> PortalUser:
    token = request.cookies.get(SESSION_COOKIE, "")
    payload = read_session(token)
    username = str(payload.get("sub", ""))
    user = get_portal_user(username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def login_rate_limit_key(request: Request, username: str) -> str:
    client_host = getattr(getattr(request, "client", None), "host", "") or "unknown"
    return f"{client_host}:{username.strip().lower()}"


def login_rate_limited(key: str, settings: Settings = SETTINGS) -> bool:
    attempts = settings.login_rate_limit_attempts
    if attempts <= 0:
        return False
    now = time.time()
    cutoff = now - settings.login_rate_limit_window_seconds
    with LOGIN_FAILURES_LOCK:
        failures = [item for item in LOGIN_FAILURES.get(key, []) if item >= cutoff]
        LOGIN_FAILURES[key] = failures
        return len(failures) >= attempts


def record_login_failure(key: str, settings: Settings = SETTINGS) -> None:
    if settings.login_rate_limit_attempts <= 0:
        return
    now = time.time()
    cutoff = now - settings.login_rate_limit_window_seconds
    with LOGIN_FAILURES_LOCK:
        LOGIN_FAILURES[key] = [item for item in LOGIN_FAILURES.get(key, []) if item >= cutoff]
        LOGIN_FAILURES[key].append(now)


def clear_login_failures(key: str) -> None:
    with LOGIN_FAILURES_LOCK:
        LOGIN_FAILURES.pop(key, None)


def current_csrf(request: Request) -> str:
    token = request.cookies.get(SESSION_COOKIE, "")
    payload = read_session(token)
    return str(payload.get("csrf", ""))


def require_csrf(request: Request) -> None:
    expected = current_csrf(request)
    supplied = request.headers.get(CSRF_HEADER, "")
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")


def safe_under_root(path: str, root: str) -> Path:
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Voicemail not found") from exc
    return resolved


def safe_under_roots(path: str, roots: Iterable[str]) -> Path:
    resolved = Path(path).resolve()
    for root in roots:
        root_resolved = Path(root).resolve()
        try:
            resolved.relative_to(root_resolved)
            return resolved
        except ValueError:
            continue
    raise HTTPException(status_code=404, detail="Voicemail not found")


def external_retention_root(settings: Settings = SETTINGS) -> Path:
    return Path(settings.trash_dir).resolve() / EXTERNAL_RETENTION_DIR_NAME


def external_retention_key_dir(file_key: str, settings: Settings = SETTINGS) -> Path:
    if not EMAIL_SAFE_FILE_KEY_RE.match(str(file_key)):
        raise HTTPException(status_code=404, detail="Voicemail not found")
    return safe_under_root(str(external_retention_root(settings) / str(file_key)), settings.trash_dir)


def external_retention_message_dir(
    file_key: str,
    txt_path: str,
    settings: Settings = SETTINGS,
) -> Path:
    source_txt = safe_under_root(str(txt_path), settings.watch_dir)
    relative_parent = source_txt.parent.relative_to(Path(settings.watch_dir).resolve())
    return safe_under_root(str(external_retention_key_dir(file_key, settings) / relative_parent), settings.trash_dir)


def file_copy_needs_refresh(source: Path, destination: Path) -> bool:
    if not destination.exists():
        return True
    source_stat = source.stat()
    destination_stat = destination.stat()
    return (
        source_stat.st_size != destination_stat.st_size
        or abs(source_stat.st_mtime - destination_stat.st_mtime) > 0.001
    )


def copy_message_to_external_retention(
    file_key: str,
    txt_path: str,
    settings: Settings = SETTINGS,
) -> int:
    source_txt = safe_under_root(str(txt_path), settings.watch_dir)
    if not source_txt.exists() or not source_txt.is_file():
        return 0

    destination_dir = external_retention_message_dir(file_key, str(source_txt), settings)
    copied = 0
    for raw_source in source_txt.parent.glob(f"{source_txt.stem}.*"):
        try:
            source = safe_under_root(str(raw_source), settings.watch_dir)
            if not source.is_file():
                continue
            destination = safe_under_root(str(destination_dir / source.name), settings.trash_dir)
            if not file_copy_needs_refresh(source, destination):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(destination))
            copied += 1
        except (OSError, HTTPException) as exc:
            detail = getattr(exc, "detail", exc)
            logger.warning(
                "Could not retain voicemail file copy file_key=%s source=%s error=%s",
                file_key,
                raw_source,
                detail,
            )
    return copied


def find_retained_message_paths(row: sqlite3.Row, settings: Settings = SETTINGS) -> tuple[Optional[str], Optional[str]]:
    try:
        file_key = str(row["file_key"])
        original_txt = str(row["txt_path"] or "")
        msg_stem = Path(original_txt).stem or str(row["msg_name"] or "")
        if not msg_stem:
            return None, None

        retention_dir = external_retention_message_dir(file_key, original_txt, settings)
        candidate_txt = safe_under_root(str(retention_dir / f"{msg_stem}.txt"), settings.trash_dir)
        if not candidate_txt.exists() or not candidate_txt.is_file():
            return None, None

        info = parse_txt(str(candidate_txt))
        candidate_key = build_file_key(str(row["extension"]), info, str(candidate_txt))
        if candidate_key != file_key:
            return None, None

        candidate_wav = candidate_txt.with_suffix(".wav")
        wav_path = str(candidate_wav) if candidate_wav.exists() and candidate_wav.is_file() else None
        return str(candidate_txt), wav_path
    except (OSError, HTTPException) as exc:
        detail = getattr(exc, "detail", exc)
        logger.debug("Could not locate retained voicemail files file_key=%s error=%s", row["file_key"], detail)

    return None, None


def remove_external_retention_for_key(file_key: str, settings: Settings = SETTINGS) -> int:
    try:
        key_dir = external_retention_key_dir(file_key, settings)
    except HTTPException:
        return 0
    if not key_dir.exists():
        return 0

    deleted_files = sum(1 for path in key_dir.rglob("*") if path.is_file())
    shutil.rmtree(key_dir)

    retention_root = external_retention_root(settings)
    current = key_dir.parent
    while current != retention_root.parent:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
    return deleted_files


def find_moved_message_paths(row: sqlite3.Row, settings: Settings = SETTINGS) -> tuple[Optional[str], Optional[str]]:
    try:
        original_txt = Path(str(row["txt_path"]))
        msg_stem = original_txt.stem or str(row["msg_name"])
        extension_dir = original_txt.parent.parent
        if not msg_stem or not extension_dir.exists():
            return None, None

        for candidate_txt in extension_dir.glob(f"*/{msg_stem}.txt"):
            candidate_txt = safe_under_root(str(candidate_txt), settings.watch_dir)
            if candidate_txt.parent.name.upper() == "INBOX":
                continue

            info = parse_txt(str(candidate_txt))
            candidate_key = build_file_key(str(row["extension"]), info, str(candidate_txt))
            if candidate_key != row["file_key"]:
                continue

            candidate_wav = candidate_txt.with_suffix(".wav")
            wav_path = str(candidate_wav) if candidate_wav.exists() else None
            return str(candidate_txt), wav_path
    except (OSError, HTTPException) as exc:
        logger.debug("Could not locate moved voicemail files file_key=%s error=%s", row["file_key"], exc)

    return None, None


def find_mailbox_inbox_dir(extension: str, settings: Settings = SETTINGS) -> Optional[Path]:
    for root, dirs, _files in os.walk(settings.watch_dir):
        root_path = Path(root)
        if root_path.name == extension and "INBOX" in dirs:
            return root_path / "INBOX"
        if root_path.name.upper() == "INBOX" and root_path.parent.name == extension:
            return root_path
    return None


def restore_source_txt_path(row: sqlite3.Row, settings: Settings = SETTINGS) -> Path:
    txt_path = str(row["txt_path"] or "")
    if txt_path:
        try:
            candidate = safe_under_roots(txt_path, [settings.watch_dir, settings.trash_dir])
            if candidate.exists() and candidate.is_file():
                return candidate
        except HTTPException:
            pass

    moved_txt_path, _moved_wav_path = find_moved_message_paths(row, settings)
    if moved_txt_path:
        candidate = safe_under_root(moved_txt_path, settings.watch_dir)
        if candidate.exists() and candidate.is_file():
            return candidate

    raise HTTPException(status_code=404, detail="Deleted voicemail files were not found")


def restore_destination_inbox(row: sqlite3.Row, source_txt: Path, settings: Settings = SETTINGS) -> Path:
    watch_dir = Path(settings.watch_dir).resolve()
    trash_dir = Path(settings.trash_dir).resolve()

    try:
        source_txt.relative_to(watch_dir)
        if source_txt.parent.name.upper() == "INBOX":
            candidate = source_txt.parent
        else:
            candidate = source_txt.parent.parent / "INBOX"
        return safe_under_root(str(candidate), settings.watch_dir)
    except ValueError:
        pass

    try:
        relative_to_trash = source_txt.relative_to(trash_dir)
        parts = relative_to_trash.parts
        file_key = str(row["file_key"])
        if file_key in parts:
            key_index = parts.index(file_key)
            original_relative_path = Path(*parts[key_index + 1 :])
            candidate = watch_dir / original_relative_path.parent
            if candidate.name.upper() != "INBOX":
                candidate = candidate.parent / "INBOX"
            return safe_under_root(str(candidate), settings.watch_dir)
    except (ValueError, IndexError):
        pass

    fallback = find_mailbox_inbox_dir(str(row["extension"]), settings)
    if fallback is None:
        raise HTTPException(status_code=404, detail="Mailbox INBOX folder was not found")
    return safe_under_root(str(fallback), settings.watch_dir)


def available_restore_stem(inbox_dir: Path, preferred_stem: str) -> str:
    if not any(inbox_dir.glob(f"{preferred_stem}.*")):
        return preferred_stem

    used_numbers: set[int] = set()
    for path in inbox_dir.glob("msg*.*"):
        match = MSG_STEM_RE.match(path.stem)
        if match:
            used_numbers.add(int(match.group("number")))

    for number in range(10000):
        candidate = f"msg{number:04d}"
        if number not in used_numbers and not any(inbox_dir.glob(f"{candidate}.*")):
            return candidate

    raise HTTPException(status_code=409, detail="No available message slot in mailbox INBOX")


def available_forward_stem(inbox_dir: Path) -> str:
    used_numbers: set[int] = set()
    for path in inbox_dir.glob("msg*.*"):
        match = MSG_STEM_RE.match(path.stem)
        if match:
            used_numbers.add(int(match.group("number")))

    for number in range(10000):
        candidate = f"msg{number:04d}"
        if number not in used_numbers and not any(inbox_dir.glob(f"{candidate}.*")):
            return candidate

    raise HTTPException(status_code=409, detail="No available message slot in mailbox INBOX")


def rewrite_voicemail_metadata_value(txt_path: Path, key: str, value: str) -> None:
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    updated: list[str] = []
    replaced = False
    key_prefix = f"{key}="
    for line in lines:
        if line.lstrip().startswith(key_prefix):
            newline = "\n" if line.endswith("\n") else ""
            updated.append(f"{key}={value}{newline}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        if updated and not updated[-1].endswith("\n"):
            updated[-1] += "\n"
        updated.append(f"{key}={value}\n")
    txt_path.write_text("".join(updated), encoding="utf-8")


def forward_destination_inbox(
    source_txt: Path,
    target_extension: str,
    settings: Settings = SETTINGS,
) -> Path:
    existing = find_mailbox_inbox_dir(target_extension, settings)
    if existing is not None:
        return safe_under_root(str(existing), settings.watch_dir)

    watch_dir = Path(settings.watch_dir).resolve()
    try:
        relative_source = source_txt.resolve().relative_to(watch_dir)
        parts = relative_source.parts
        if len(parts) >= 4 and parts[-1].lower().endswith(".txt") and parts[-2].upper() == "INBOX":
            candidate = watch_dir / Path(*parts[:-3]) / target_extension / "INBOX"
            return safe_under_root(str(candidate), settings.watch_dir)
    except ValueError:
        pass

    raise HTTPException(status_code=404, detail="Target mailbox INBOX folder was not found")


def remove_copied_message_files(paths: Iterable[str]) -> None:
    for path in paths:
        try:
            candidate = Path(path)
            if candidate.exists() and candidate.is_file():
                candidate.unlink()
        except OSError as exc:
            logger.warning("Could not remove failed forwarded voicemail copy path=%s error=%s", path, exc)


def copy_message_to_mailbox(
    row: sqlite3.Row,
    target_extension: str,
    settings: Settings = SETTINGS,
) -> dict[str, Any]:
    source_txt = safe_under_root(str(row["txt_path"]), settings.watch_dir)
    if not source_txt.exists() or not source_txt.is_file():
        raise HTTPException(status_code=404, detail="Source voicemail metadata file was not found")

    source_parent = source_txt.parent
    source_stem = source_txt.stem
    source_files = sorted(
        safe_under_root(str(path), settings.watch_dir)
        for path in source_parent.glob(f"{source_stem}.*")
        if path.is_file()
    )
    if not source_files:
        raise HTTPException(status_code=404, detail="Source voicemail files were not found")

    inbox_dir = forward_destination_inbox(source_txt, target_extension, settings)
    inbox_dir.mkdir(parents=True, exist_ok=True)
    target_stem = available_forward_stem(inbox_dir)

    planned_copies: list[tuple[Path, Path]] = []
    for source in source_files:
        destination = inbox_dir / f"{target_stem}{source.suffix}"
        if destination.exists():
            raise HTTPException(status_code=409, detail=f"Forward target already exists: {destination.name}")
        planned_copies.append((source, destination))

    copied_paths: list[str] = []
    try:
        for source, destination in planned_copies:
            shutil.copy2(str(source), str(destination))
            copied_paths.append(str(destination))
        copied_txt = inbox_dir / f"{target_stem}.txt"
        if not copied_txt.exists():
            raise HTTPException(status_code=404, detail="Forwarded voicemail metadata file was not created")
        rewrite_voicemail_metadata_value(copied_txt, "origmailbox", target_extension)
        info = parse_txt(str(copied_txt))
        target_file_key = build_file_key(target_extension, info, str(copied_txt))
        if not target_file_key:
            raise HTTPException(status_code=409, detail="Forwarded voicemail did not have a valid identity")
        copied_wav = inbox_dir / f"{target_stem}.wav"
        return {
            "file_key": target_file_key,
            "txt_path": str(copied_txt),
            "wav_path": str(copied_wav),
            "msg_name": target_stem,
            "copied_files": copied_paths,
        }
    except Exception:
        remove_copied_message_files(copied_paths)
        raise


def iter_file_range(path: Path, start: int, end: int, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def stream_audio_file(path: Path, request: Request) -> StreamingResponse:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    file_size = path.stat().st_size
    if file_size <= 0:
        raise HTTPException(status_code=404, detail="Audio file is empty")

    media_type = mimetypes.guess_type(str(path))[0] or "audio/wav"
    range_header = request.headers.get("range")
    start = 0
    end = file_size - 1
    status_code = status.HTTP_200_OK
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Cache-Control": "private, no-store",
    }

    if range_header:
        match = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
        if not match:
            raise HTTPException(status_code=416, detail="Invalid Range header")

        start_raw, end_raw = match.groups()
        if start_raw == "" and end_raw == "":
            raise HTTPException(status_code=416, detail="Invalid Range header")

        if start_raw == "":
            suffix_length = int(end_raw)
            start = max(0, file_size - suffix_length)
        else:
            start = int(start_raw)
            if end_raw:
                end = min(file_size - 1, int(end_raw))

        if start > end or start >= file_size:
            headers["Content-Range"] = f"bytes */{file_size}"
            raise HTTPException(status_code=416, detail="Requested range not satisfiable", headers=headers)

        status_code = status.HTTP_206_PARTIAL_CONTENT
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(end - start + 1)

    return StreamingResponse(
        iter_file_range(path, start, end),
        status_code=status_code,
        media_type=media_type,
        headers=headers,
    )


def move_message_to_trash(row: sqlite3.Row, settings: Settings = SETTINGS) -> list[str]:
    txt_path = safe_under_root(str(row["txt_path"]), settings.watch_dir)
    stem = txt_path.stem
    parent = txt_path.parent
    if not parent.exists():
        return []

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    relative_parent = parent.relative_to(Path(settings.watch_dir).resolve())
    destination_dir = Path(settings.trash_dir).resolve() / stamp / str(row["file_key"]) / relative_parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    for source in parent.glob(f"{stem}.*"):
        source = safe_under_root(str(source), settings.watch_dir)
        if not source.exists() or not source.is_file():
            continue
        destination = destination_dir / source.name
        if destination.exists():
            destination = destination_dir / f"{source.stem}-{int(time.time())}{source.suffix}"
        shutil.move(str(source), str(destination))
        moved.append(str(destination))

    return moved


def restore_message_to_inbox(row: sqlite3.Row, settings: Settings = SETTINGS) -> dict[str, Any]:
    if not row["deleted_utc"]:
        raise HTTPException(status_code=409, detail="Voicemail is already active")

    source_txt = restore_source_txt_path(row, settings)
    source_stem = source_txt.stem
    source_parent = source_txt.parent
    source_files = sorted(
        safe_under_roots(str(path), [settings.watch_dir, settings.trash_dir])
        for path in source_parent.glob(f"{source_stem}.*")
        if path.is_file()
    )
    if not source_files:
        raise HTTPException(status_code=404, detail="Deleted voicemail files were not found")

    inbox_dir = restore_destination_inbox(row, source_txt, settings)
    inbox_dir.mkdir(parents=True, exist_ok=True)
    restore_stem = available_restore_stem(inbox_dir, source_stem)

    planned_moves: list[tuple[Path, Path]] = []
    for source in source_files:
        destination = inbox_dir / f"{restore_stem}{source.suffix}"
        if destination.exists():
            raise HTTPException(status_code=409, detail=f"Restore target already exists: {destination.name}")
        planned_moves.append((source, destination))

    moved_pairs: list[tuple[Path, Path]] = []
    try:
        for source, destination in planned_moves:
            shutil.move(str(source), str(destination))
            moved_pairs.append((source, destination))
    except Exception:
        for original, restored in reversed(moved_pairs):
            if restored.exists() and not original.exists():
                try:
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(restored), str(original))
                except Exception as rollback_exc:
                    logger.warning(
                        "Could not roll back failed restore file_key=%s path=%s error=%s",
                        row["file_key"],
                        restored,
                        rollback_exc,
                    )
        raise

    restored_txt = inbox_dir / f"{restore_stem}.txt"
    restored_wav = inbox_dir / f"{restore_stem}.wav"
    if not restored_txt.exists():
        raise HTTPException(status_code=404, detail="Restored voicemail metadata file was not found")

    info = parse_txt(str(restored_txt))
    new_file_key = build_file_key(str(row["extension"]), info, str(restored_txt))
    if not new_file_key:
        raise HTTPException(status_code=409, detail="Restored voicemail did not have a valid identity")

    return {
        "file_key": new_file_key,
        "txt_path": str(restored_txt),
        "wav_path": str(restored_wav),
        "msg_name": restore_stem,
        "moved_files": [str(destination) for _source, destination in moved_pairs],
    }


def validate_startup() -> None:
    if len(SETTINGS.session_secret) < 32:
        raise RuntimeError("VOICEMAIL_PORTAL_SESSION_SECRET must be set to at least 32 characters")
    if not os.path.isdir(SETTINGS.watch_dir):
        raise RuntimeError(f"Voicemail watch directory does not exist: {SETTINGS.watch_dir}")
    if SETTINGS.auto_users and not SETTINGS.shared_password_hash:
        raise RuntimeError("VOICEMAIL_PORTAL_AUTO_USERS is true but VOICEMAIL_PORTAL_SHARED_PASSWORD_HASH is not set")
    os.makedirs(SETTINGS.trash_dir, exist_ok=True)
    ZoneInfo(SETTINGS.local_timezone)
    load_manual_users()
    if SETTINGS.auto_users:
        discovered_count = len(discover_mailbox_extensions())
        logger.info("Auto user discovery found %s voicemail mailbox extension(s)", discovered_count)
    store = get_store()
    store.ensure_schema()
    store.sync_filesystem()


def login_page(error: str = "") -> HTMLResponse:
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
    login_action = html.escape(app_path("/login"))
    login_logo = logo_img("login-logo")
    favicon_html = favicon_link()
    brand_name = html.escape(SETTINGS.brand_name)
    brand_tagline = html.escape(SETTINGS.brand_tagline)
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {favicon_html}
  <title>{brand_name}</title>
  <style>
    :root {{
      color-scheme: dark;
      --ink: #f6f1e7;
      --muted: #c9c1b4;
      --line: #5b4a1f;
      --panel: #242424;
      --soft: #2a2a2a;
      --bg: #1a1a1a;
      --accent: #fec030;
      --accent-strong: #fbc12f;
      --danger: #ff6b57;
      --shadow: rgba(0, 0, 0, 0.45);
    }}
    .login-logo {{
      display: block;
      max-width: 240px;
      max-height: 80px;
      width: auto;
      height: auto;
      object-fit: contain;
      margin: 0 auto 20px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: var(--bg);
      color: var(--ink);
      font-family: 'Inter', 'Segoe UI', Arial, Helvetica, sans-serif;
    }}
    main {{
      width: min(420px, calc(100vw - 32px));
      background: var(--panel);
      border: 1px solid var(--line);
      border-top: 4px solid var(--accent);
      border-radius: 10px;
      padding: 28px;
      box-shadow: 0 20px 50px var(--shadow);
    }}
    h1 {{
      margin: 0 0 8px;
      font-family: 'Outfit', 'Inter', 'Segoe UI', Arial, Helvetica, sans-serif;
      font-size: 28px;
      letter-spacing: 0;
      text-align: center;
    }}
    p {{ margin: 0 0 24px; color: var(--muted); text-align: center; }}
    label {{ display: block; margin: 14px 0 6px; font-weight: 700; }}
    input {{
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 16px;
      background: var(--soft);
      color: var(--ink);
    }}
    input:focus {{
      border-color: var(--accent);
      outline: 2px solid rgba(254, 192, 48, 0.28);
      outline-offset: 1px;
    }}
    button {{
      width: 100%;
      min-height: 42px;
      margin-top: 20px;
      border: 1px solid var(--accent);
      border-radius: 8px;
      background: var(--accent);
      color: #000000;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-strong); }}
    .error {{
      border: 1px solid rgba(255, 107, 87, 0.42);
      background: rgba(255, 107, 87, 0.12);
      color: var(--danger);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 16px;
    }}
  </style>
</head>
<body>
  <main>
    {login_logo}
    <h1>{brand_name}</h1>
    <p>{brand_tagline}</p>
    {error_html}
    <form method="post" action="{login_action}">
      <label for="username">Username</label>
      <input id="username" name="username" autocomplete="username" required>
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">Sign In</button>
    </form>
  </main>
</body>
</html>"""
    )


def portal_page(user: PortalUser, csrf_token: str) -> HTMLResponse:
    user_label = html.escape(user.display_name)
    mobile_user_label = "Administrator" if user.is_admin else user_label
    csrf_json = json.dumps(csrf_token)
    base_path_json = json.dumps(SETTINGS.base_path)
    is_admin_json = json.dumps(user.is_admin)
    forwarding_enabled_json = json.dumps(SETTINGS.forward_enabled)
    forwarding_hidden = "" if SETTINGS.forward_enabled else " hidden"
    header_logo = logo_img("header-logo", allow_light_variant=True)
    favicon_html = favicon_link()
    workflow_link_html = ""
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {favicon_html}
  <title>Voicemails</title>
  <style>
    :root {{
      color-scheme: dark;
      --ink: #f6f1e7;
      --muted: #c9c1b4;
      --line: #5b4a1f;
      --bg: #1a1a1a;
      --panel: #242424;
      --soft: #2a2a2a;
      --header-bg: #1a1a1a;
      --audio-bg: #303030;
      --accent: #fec030;
      --accent-strong: #fbc12f;
      --accent-weak: #332812;
      --danger: #ff6b57;
      --shadow: rgba(0, 0, 0, 0.42);
    }}
    :root[data-theme="light"] {{
      color-scheme: light;
      --ink: #1a1a1a;
      --muted: #5f594d;
      --line: #e3d4ad;
      --bg: #f6f1e7;
      --panel: #fffaf0;
      --soft: #ffffff;
      --header-bg: #fffaf0;
      --audio-bg: #eee6d6;
      --accent: #fec030;
      --accent-strong: #d99a00;
      --accent-weak: #fff0c2;
      --danger: #b42318;
      --shadow: rgba(26, 26, 26, 0.14);
    }}
    :root[data-theme="dark"] {{
      color-scheme: dark;
      --ink: #f6f1e7;
      --muted: #c9c1b4;
      --line: #5b4a1f;
      --bg: #1a1a1a;
      --panel: #242424;
      --soft: #2a2a2a;
      --header-bg: #1a1a1a;
      --audio-bg: #303030;
      --accent: #fec030;
      --accent-strong: #fbc12f;
      --accent-weak: #332812;
      --danger: #ff6b57;
      --shadow: rgba(0, 0, 0, 0.42);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: 'Inter', 'Segoe UI', Arial, Helvetica, sans-serif;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 22px;
      background: var(--header-bg);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    .brandbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }}
    .header-logo {{
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      max-width: 230px;
      max-height: 58px;
      width: auto;
      height: auto;
      object-fit: contain;
      pointer-events: none;
    }}
    .logo-light {{ display: none; }}
    :root[data-theme="light"] .logo-dark {{ display: none; }}
    :root[data-theme="light"] .logo-light {{ display: block; }}
    :root[data-theme="light"] .header-logo.logo-needs-light-plate {{
      background: #1a1a1a;
      border: 1px solid #5b4a1f;
      border-radius: 8px;
      padding: 6px 10px;
      box-shadow: 0 8px 22px rgba(0, 0, 0, 0.2);
    }}
    h1 {{
      margin: 0;
      font-family: 'Outfit', 'Inter', 'Segoe UI', Arial, Helvetica, sans-serif;
      font-size: 24px;
      letter-spacing: 0;
      color: var(--ink);
    }}
    .userbar {{ display: flex; gap: 12px; align-items: center; color: var(--muted); }}
    .user-label-mobile {{ display: none; }}
    .userbar a.nav-link {{
      min-height: 36px;
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      font-weight: 700;
      padding: 0 12px;
      text-decoration: none;
    }}
    .userbar button {{
      background: var(--panel);
      color: var(--ink);
      border-color: var(--line);
    }}
    .userbar #themeBtn {{
      background: var(--accent);
      color: #000000;
      border-color: var(--accent);
    }}
    .userbar #themeBtn.theme-toggle {{
      width: 40px;
      min-width: 40px;
      height: 40px;
      min-height: 40px;
      display: inline-grid;
      place-items: center;
      padding: 0;
    }}
    .theme-toggle svg {{
      width: 20px;
      height: 20px;
      stroke: currentColor;
    }}
    button {{
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--accent);
      color: #000000;
      cursor: pointer;
      font-weight: 700;
      padding: 0 12px;
    }}
    button:hover {{ border-color: var(--accent-strong); filter: brightness(1.04); }}
    button.primary {{ background: var(--accent); border-color: var(--accent); color: #000000; }}
    button.danger {{ color: var(--danger); }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.45;
    }}
    .icon-btn {{
      width: 36px;
      min-width: 36px;
      height: 36px;
      min-height: 36px;
      display: inline-grid;
      place-items: center;
      padding: 0;
      border-radius: 6px;
      background: var(--soft);
    }}
    .icon-btn.danger {{
      background: var(--soft);
      color: var(--danger);
    }}
    .icon-btn svg {{
      width: 18px;
      height: 18px;
      stroke: currentColor;
    }}
    .icon-btn.large {{
      width: 58px;
      min-width: 58px;
      height: 58px;
      min-height: 58px;
    }}
    .icon-btn.large svg {{
      width: 22px;
      height: 22px;
    }}
    .icon-btn.play {{ color: var(--accent); }}
    .icon-btn.forward {{ color: var(--accent); }}
    .icon-btn.restore {{ color: var(--accent); }}
    .menu-btn {{
      width: 40px;
      min-width: 40px;
      height: 40px;
      min-height: 40px;
      display: inline-grid;
      place-items: center;
      padding: 0;
      border-color: transparent;
      background: transparent;
      color: var(--accent);
    }}
    .menu-btn svg {{
      width: 22px;
      height: 22px;
      stroke: currentColor;
    }}
    #menuBtn svg {{
      width: 28px;
      height: 28px;
    }}
    #directoryBtn svg {{
      width: 25px;
      height: 25px;
    }}
    #directoryBtn .directory-book-shape,
    #directoryBtn .directory-x-line {{
      transform-box: fill-box;
      transform-origin: center;
      transition: transform 160ms ease, opacity 120ms ease;
    }}
    #directoryBtn .directory-x-line {{
      opacity: 0;
    }}
    #directoryBtn[aria-expanded="true"] .directory-book-shape {{
      opacity: 0;
      transform: scale(0.84);
    }}
    #directoryBtn[aria-expanded="true"] .directory-x-line {{
      opacity: 1;
    }}
    #menuBtn .menu-line {{
      transform-box: fill-box;
      transform-origin: center;
      transition: transform 160ms ease, opacity 120ms ease;
    }}
    #menuBtn[aria-expanded="true"] .menu-line-top {{
      transform: translateY(5px) rotate(45deg);
    }}
    #menuBtn[aria-expanded="true"] .menu-line-middle {{
      opacity: 0;
    }}
    #menuBtn[aria-expanded="true"] .menu-line-bottom {{
      transform: translateY(-5px) rotate(-45deg);
    }}
    .extension-menu {{
      position: fixed;
      top: 72px;
      left: 18px;
      width: min(340px, calc(100vw - 36px));
      max-height: calc(100vh - 92px);
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: 0 18px 44px var(--shadow);
      z-index: 6;
      padding: 8px;
    }}
    .extension-menu[hidden] {{ display: none; }}
    .directory-menu {{
      width: min(390px, calc(100vw - 36px));
    }}
    .menu-search-wrap {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--panel);
      padding-bottom: 8px;
    }}
    .menu-search-wrap input {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 15px;
      background: var(--soft);
      color: var(--ink);
    }}
    .menu-search-wrap input:focus {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
    .directory-entry {{
      width: 100%;
      min-height: 42px;
      display: flex;
      align-items: center;
      border: 1px solid transparent;
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      padding: 8px 10px;
      font-weight: 700;
      text-align: left;
      cursor: default;
      user-select: text;
    }}
    .directory-entry:hover {{ background: rgba(254, 192, 48, 0.08); }}
    .directory-entry.directory-action {{
      cursor: pointer;
      user-select: none;
    }}
    .directory-empty {{
      color: var(--muted);
      padding: 14px 10px;
      font-size: 14px;
    }}
    .extension-option {{
      width: 100%;
      min-height: 46px;
      display: block;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      padding: 8px 10px;
      font-weight: 700;
    }}
    .extension-option.active {{ background: var(--accent-weak); color: var(--accent); border-color: var(--line); }}
    .extension-folder-option {{
      margin-top: 8px;
      border-color: var(--line);
      color: var(--accent);
    }}
    .extension-folder-option.active {{
      color: var(--accent);
    }}
    .extension-folder-option.deleted-folder {{
      color: var(--danger);
    }}
    .extension-option span {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 400;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(320px, 460px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      min-height: calc(100vh - 69px);
    }}
    .list, .detail {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      min-width: 0;
      box-shadow: 0 12px 30px var(--shadow);
    }}
    .toolbar {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }}
    .folder-tabs {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .folder-tab {{
      min-height: 34px;
      background: var(--soft);
      color: var(--ink);
      border-color: var(--line);
    }}
    .folder-tab.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #000000;
    }}
    .toolbar input {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 15px;
      background: var(--soft);
      color: var(--ink);
    }}
    .toolbar input:focus {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
    .bulkbar {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .bulkbar[hidden] {{ display: none; }}
    .bulk-check {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
    }}
    .bulk-check input,
    .bulk-select input {{
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
      cursor: pointer;
    }}
    .bulkbar button {{
      min-height: 32px;
      padding: 0 10px;
      background: var(--soft);
      border-color: var(--line);
      color: var(--ink);
      font-size: 13px;
    }}
    .bulkbar button.danger {{ color: var(--danger); }}
    .bulkbar button:disabled {{
      cursor: not-allowed;
      opacity: 0.5;
    }}
    .bulk-count {{
      margin-left: auto;
      color: var(--muted);
      font-size: 13px;
    }}
    .items {{ max-height: calc(100vh - 142px); overflow: auto; }}
    .item {{
      width: 100%;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: stretch;
      gap: 8px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      background: var(--panel);
    }}
    .item.has-bulk {{ grid-template-columns: 34px minmax(0, 1fr) auto; }}
    .bulk-select {{
      display: grid;
      place-items: start center;
      padding: 15px 0 0 8px;
    }}
    .item.active {{
      background: var(--accent-weak);
      box-shadow: inset 4px 0 0 var(--accent);
    }}
    .item-main:hover {{ background: rgba(254, 192, 48, 0.08); }}
    .item-main {{
      min-width: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
      color: var(--ink);
      text-align: left;
      padding: 12px;
      font-weight: 400;
    }}
    .item-actions {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding-right: 10px;
    }}
    .item-actions .icon-btn {{ background: var(--soft); }}
    .item-title {{ display: flex; justify-content: space-between; gap: 12px; font-weight: 700; }}
    .item-meta {{ margin-top: 4px; color: var(--muted); font-size: 13px; }}
    .item-preview {{ margin-top: 8px; color: var(--ink); font-size: 14px; line-height: 1.35; }}
    .detail {{ padding: 18px; overflow: auto; }}
    .detail-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }}
    .detail-title {{ min-width: 0; }}
    .detail-title h2 {{
      margin: 0;
      overflow-wrap: anywhere;
      font-family: 'Outfit', 'Inter', 'Segoe UI', Arial, Helvetica, sans-serif;
      letter-spacing: 0;
      line-height: 1.12;
    }}
    .caller-title-text {{
      display: block;
      min-width: 0;
    }}
    .caller-copy-row {{
      display: flex;
      justify-content: flex-start;
      align-items: center;
      min-height: 28px;
      margin: 8px 0 8px;
    }}
    .detail-actions {{
      display: flex;
      align-items: flex-start;
      gap: 8px;
      flex-shrink: 0;
    }}
    .mailbox-badge {{
      min-width: 86px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 10px;
      line-height: 1.15;
      font-size: 16px;
      text-align: center;
      min-height: 58px;
      display: grid;
      align-content: center;
      background: var(--soft);
    }}
    .mailbox-badge span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 2px;
    }}
    #reviewProgressSlot:empty {{ display: none; }}
    .review-progress {{
      min-width: 86px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 10px;
      font-size: 16px;
      text-align: center;
      min-height: 58px;
      display: grid;
      align-content: center;
      background: var(--soft);
      color: var(--ink);
      font-weight: 800;
    }}
    .review-progress-fraction {{
      display: inline-grid;
      justify-items: center;
      align-items: center;
      gap: 2px;
      line-height: 1;
    }}
    .review-progress-current {{
      color: var(--accent);
      font-weight: 900;
      font-size: 18px;
    }}
    .review-progress-divider {{
      display: block;
      width: 36px;
      height: 2px;
      border-radius: 999px;
      background: var(--ink);
    }}
    .review-progress-total {{
      display: block;
      color: var(--ink);
      font-weight: 900;
      font-size: 16px;
    }}
    .deleted-banner {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 11px;
      margin: 12px 0;
      color: var(--ink);
      background: var(--accent-weak);
      font-weight: 700;
    }}
    .forwarding-notice {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 11px;
      margin: 12px 0;
      color: var(--ink);
      background: var(--soft);
      font-weight: 700;
      line-height: 1.35;
    }}
    .forwarding-notice ul {{
      margin: 6px 0 0 18px;
      padding: 0;
    }}
    .forward-summary {{
      margin-top: 4px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
    }}
    .forward-picker {{
      position: fixed;
      inset: 0;
      z-index: 12;
      display: grid;
      place-items: center;
      background: rgba(0, 0, 0, 0.48);
      padding: 18px;
    }}
    .forward-picker[hidden] {{ display: none; }}
    .forward-picker-panel {{
      width: min(430px, 100%);
      max-height: min(640px, calc(100vh - 36px));
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      box-shadow: 0 18px 44px var(--shadow);
      padding: 14px;
    }}
    .forward-picker-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .forward-picker-header h3 {{
      margin: 0;
    }}
    .forward-picker-actions {{
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 12px;
    }}
    .empty {{ color: var(--muted); padding: 24px; }}
    .fields {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 10px 0 12px;
      max-width: 1120px;
    }}
    .field {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 9px;
      min-width: 0;
      font-size: 15px;
      line-height: 1.2;
      background: var(--soft);
    }}
    .field span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 2px;
    }}
    .field-value-line {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .field-value-text {{
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    .segment-play, .copy-btn {{
      width: 28px;
      min-width: 28px;
      height: 28px;
      min-height: 28px;
      display: inline-grid;
      place-items: center;
      padding: 0;
      flex: 0 0 auto;
      color: var(--accent);
      background: transparent;
      border-color: var(--line);
    }}
    .segment-play svg {{
      width: 15px;
      height: 15px;
      stroke: currentColor;
    }}
    .copy-btn svg {{
      width: 16px;
      height: 16px;
      stroke: currentColor;
    }}
    .copy-btn.copied {{
      color: var(--accent);
      border-color: var(--accent);
    }}
    .field-extra {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.2;
    }}
    .field-extra strong {{ color: var(--ink); }}
    .field-extra strong.match-yes {{ color: var(--accent); font-weight: 700; }}
    .field-extra strong.review-needed {{ color: var(--danger); }}
    h3 {{
      font-family: 'Outfit', 'Inter', 'Segoe UI', Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }}
    h3 {{ margin-bottom: 8px; }}
    audio {{ width: 100%; margin: 10px 0 16px; background: var(--audio-bg); border-radius: 999px; }}
    .audio-tools {{ margin: 10px 0 16px; }}
    .audio-tools audio {{ margin: 0 0 8px; }}
    .speed-controls {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .speed-btn {{
      min-height: 28px;
      padding: 0 8px;
      background: var(--soft);
      border-color: var(--line);
      color: var(--ink);
      font-size: 12px;
    }}
    .speed-btn.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #000000;
    }}
    pre, .transcript-box {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 8px;
      background: var(--soft);
      line-height: 1.45;
      font-family: 'Inter', 'Segoe UI', Arial, Helvetica, sans-serif;
      font-size: 15px;
    }}
    .transcript-box {{
      position: relative;
      padding-right: 54px;
      min-height: 58px;
    }}
    .transcript-copy {{
      position: absolute;
      top: 10px;
      right: 10px;
      background: var(--soft);
    }}
    .transcript-word {{
      appearance: none;
      display: inline;
      min-height: 0;
      border: 0;
      border-radius: 0;
      padding: 0;
      margin: 0;
      background: transparent;
      color: inherit;
      font: inherit;
      font-weight: inherit;
      line-height: inherit;
      text-align: inherit;
      vertical-align: baseline;
      cursor: pointer;
    }}
    .transcript-word:hover, .transcript-word:focus-visible {{
      background: transparent;
      color: var(--accent);
      filter: none;
      outline: 0;
    }}
    .transcript-word.current {{
      color: var(--accent);
    }}
    .transcript-disclaimer {{
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
      line-height: 1.35;
      padding: 0 2px;
    }}
    .toast-host {{
      position: fixed;
      right: 18px;
      bottom: var(--toast-bottom, 96px);
      z-index: 10;
      display: grid;
      gap: 8px;
      max-width: min(360px, calc(100vw - 36px));
    }}
    .toast {{
      display: flex;
      align-items: center;
      gap: 12px;
      justify-content: space-between;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      box-shadow: 0 18px 44px var(--shadow);
      padding: 10px 12px;
      font-weight: 700;
    }}
    .toast button {{
      min-height: 30px;
      padding: 0 10px;
      background: var(--accent);
      border-color: var(--accent);
      color: #000000;
    }}
    .delete-comment-panel {{
      margin-top: 16px;
    }}
    .delete-comment-panel label {{
      display: block;
      margin-bottom: 8px;
      color: var(--ink);
      font-weight: 800;
      font-family: 'Outfit', 'Inter', 'Segoe UI', Arial, Helvetica, sans-serif;
      font-size: 22px;
      line-height: 1.2;
    }}
    .delete-comment-panel textarea {{
      width: 100%;
      min-height: 96px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      resize: vertical;
      background: var(--soft);
      color: var(--ink);
      font: inherit;
      line-height: 1.45;
    }}
    .delete-comment-panel textarea:focus {{
      border-color: var(--accent);
      outline: 2px solid rgba(254, 192, 48, 0.28);
      outline-offset: 1px;
    }}
    .delete-comment-help {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }}
    .delete-comment-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      min-height: 22px;
      margin-top: 6px;
      font-size: 12px;
    }}
    .delete-comment-error {{
      color: var(--danger);
      font-weight: 800;
    }}
    .delete-comment-count {{
      color: var(--muted);
      margin-left: auto;
      white-space: nowrap;
    }}
    .delete-comment-side {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: auto;
    }}
    .delete-comment-save {{
      min-height: 28px;
      padding: 0 10px;
      background: var(--accent);
      color: #000000;
      border-color: var(--accent);
      font-size: 12px;
    }}
    .delete-comment-save.saved {{
      background: var(--accent);
      border-color: var(--accent-strong);
      color: #000000;
    }}
    @media (max-width: 1120px) {{
      header {{ flex-wrap: wrap; }}
      .header-logo {{
        position: static;
        transform: none;
        flex-basis: 100%;
        margin: 4px auto 0;
        max-width: 210px;
        max-height: 56px;
        order: 3;
      }}
    }}
    @media (max-width: 820px) {{
      body {{ font-size: 16px; }}
      button, input, textarea, select {{
        font-family: inherit;
        font-size: 16px;
      }}
      h1 {{ font-size: 24px; }}
      .extension-option span {{ font-size: 12px; }}
      .bulk-check,
      .bulkbar button,
      .bulk-count,
      .item-meta {{ font-size: 13px; }}
      .directory-empty,
      .item-preview {{ font-size: 14px; }}
      .menu-search-wrap input,
      .toolbar input {{ font-size: 15px; }}
      .detail {{ font-size: 18px; }}
      .detail button,
      .detail input,
      .detail textarea,
      .detail select {{ font-size: 18px; }}
      .detail .mailbox-badge span {{ font-size: 12.375px; }}
      .detail .field span {{ font-size: 13.6125px; }}
      .detail .speed-controls,
      .detail .speed-btn,
      .detail .delete-comment-meta,
      .detail .delete-comment-save {{ font-size: 13.5px; }}
      .detail .field-extra {{ font-size: 14.85px; }}
      .detail .forward-summary,
      .detail .transcript-disclaimer,
      .detail .delete-comment-help {{ font-size: 14.625px; }}
      .detail .field,
      .detail pre,
      .detail .transcript-box {{ font-size: 18.5625px; }}
      .detail .mailbox-badge,
      .detail .review-progress,
      .detail .review-progress-total {{ font-size: 18px; }}
      .detail .review-progress-current {{ font-size: 20.25px; }}
      .detail .delete-comment-panel label {{ font-size: 24.75px; }}
      main {{
        grid-template-columns: 1fr;
        padding: 10px;
        padding-bottom: calc(10px + env(safe-area-inset-bottom));
      }}
      .items {{ max-height: none; }}
      .fields {{ grid-template-columns: 1fr; }}
      .detail-header {{ flex-direction: column; }}
      header {{ align-items: flex-start; flex-direction: column; }}
      .userbar {{
        flex-wrap: nowrap;
        gap: 4px;
        max-width: 100%;
        width: 100%;
      }}
      .userbar .user-label {{
        flex: 1 1 auto;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}
      .userbar .user-label-full {{ display: none; }}
      .userbar .user-label-mobile {{ display: inline; }}
      .userbar button,
      .userbar a.nav-link {{
        flex: 0 0 auto;
        font-size: 13px;
        padding: 0 6px;
        white-space: nowrap;
      }}
      .detail-actions {{
        flex-wrap: wrap;
        max-width: 100%;
        width: 100%;
      }}
      button,
      .userbar a.nav-link,
      .menu-search-wrap input,
      .toolbar input,
      .delete-comment-panel textarea {{ min-height: 44px; }}
      .icon-btn,
      .menu-btn,
      .userbar #themeBtn.theme-toggle,
      .segment-play,
      .copy-btn,
      .speed-btn,
      .folder-tab,
      .bulkbar button,
      .toast button,
      .delete-comment-save {{
        min-width: 44px;
        min-height: 44px;
      }}
      .toast-host {{
        bottom: calc(var(--toast-bottom, 96px) + env(safe-area-inset-bottom));
      }}
      .header-logo {{
        align-self: center;
        max-width: 190px;
        order: -1;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="brandbar">
      <button id="menuBtn" class="menu-btn" type="button" title="Filter by extension" aria-label="Filter by extension" aria-expanded="false">
        <svg viewBox="0 0 24 24" fill="none" stroke-width="2.4" stroke-linecap="round" aria-hidden="true">
          <path class="menu-line menu-line-top" d="M4 7h16"></path>
          <path class="menu-line menu-line-middle" d="M4 12h16"></path>
          <path class="menu-line menu-line-bottom" d="M4 17h16"></path>
        </svg>
      </button>
      <button id="directoryBtn" class="menu-btn" type="button" title="Directory" aria-label="Directory" aria-expanded="false"{forwarding_hidden}>
        <svg class="address-book-icon" viewBox="0 0 24 24" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <g class="directory-book-shape">
            <path d="M7 4.5h10a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-11a2 2 0 0 1 2-2z"></path>
            <path d="M9 4.5v15"></path>
            <path d="M3.8 8h2.4"></path>
            <path d="M3.8 12h2.4"></path>
            <path d="M3.8 16h2.4"></path>
            <circle cx="13.5" cy="10" r="1.9"></circle>
            <path d="M10.8 15.5c.5-1.5 1.5-2.3 2.7-2.3s2.2.8 2.7 2.3"></path>
          </g>
          <path class="directory-x-line directory-x-line-first" d="M7 7l10 10" stroke-width="2.4"></path>
          <path class="directory-x-line directory-x-line-second" d="M17 7 7 17" stroke-width="2.4"></path>
        </svg>
      </button>
      <h1>Voicemails</h1>
    </div>
    {header_logo}
    <div class="userbar">
      <span class="user-label" aria-label="{user_label}">
        <span class="user-label-full">{user_label}</span>
        <span class="user-label-mobile">{mobile_user_label}</span>
      </span>
      {workflow_link_html}
      <button id="themeBtn" class="theme-toggle" type="button" title="Switch to light mode" aria-label="Switch to light mode" aria-pressed="true">
        <svg viewBox="0 0 24 24" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="4"></circle>
          <path d="M12 2v2"></path>
          <path d="M12 20v2"></path>
          <path d="m4.93 4.93 1.41 1.41"></path>
          <path d="m17.66 17.66 1.41 1.41"></path>
          <path d="M2 12h2"></path>
          <path d="M20 12h2"></path>
          <path d="m6.34 17.66-1.41 1.41"></path>
          <path d="m19.07 4.93-1.41 1.41"></path>
        </svg>
      </button>
      <button id="refreshBtn" type="button">Refresh</button>
      <button id="logoutBtn" type="button">Sign Out</button>
    </div>
  </header>
  <nav id="extensionMenu" class="extension-menu" hidden></nav>
  <nav id="directoryMenu" class="extension-menu directory-menu" hidden>
    <div class="menu-search-wrap directory-search-wrap">
      <input id="directorySearch" type="search" placeholder="Search directory" autocomplete="off">
    </div>
    <div id="directoryItems" class="directory-items"><div class="directory-empty">Loading directory...</div></div>
  </nav>
  <div id="forwardPicker" class="forward-picker" hidden>
    <div class="forward-picker-panel" role="dialog" aria-modal="true" aria-labelledby="forwardPickerTitle">
      <div class="forward-picker-header">
        <h3 id="forwardPickerTitle">Forward voicemail</h3>
        <button id="forwardCloseBtn" class="icon-btn" type="button" title="Close" aria-label="Close">×</button>
      </div>
      <div class="menu-search-wrap">
        <input id="forwardSearch" type="search" placeholder="Search target mailbox" autocomplete="off">
      </div>
      <div id="forwardOptions" class="directory-items"><div class="directory-empty">Loading directory...</div></div>
      <div class="forward-picker-actions">
        <button id="forwardCancelBtn" type="button">Cancel</button>
      </div>
    </div>
  </div>
  <div id="toastHost" class="toast-host" aria-live="polite"></div>
  <main>
    <section class="list">
      <div class="toolbar">
        <div id="folderTabs" class="folder-tabs" hidden></div>
        <input id="search" placeholder="Search extension, caller, transcript, callback, date">
        <div id="bulkBar" class="bulkbar" hidden>
          <label class="bulk-check"><input id="selectAllBox" type="checkbox"> Select all</label>
          <button id="bulkDeleteBtn" class="danger" type="button" disabled>Delete Selected</button>
          <button id="clearSelectionBtn" type="button">Clear</button>
          <span id="bulkCount" class="bulk-count"></span>
        </div>
      </div>
      <div id="items" class="items"><div class="empty">Loading...</div></div>
    </section>
    <section id="detail" class="detail">
      <div class="empty">Select a voicemail.</div>
    </section>
  </main>
  <script>
    const basePath = {base_path_json};
    const csrfToken = {csrf_json};
    const isAdmin = {is_admin_json};
    const forwardingEnabled = {forwarding_enabled_json};
    let voicemails = [];
    let extensions = [];
    let directoryEntries = [];
    let extensionMenuSearchTerm = "";
    let directorySearchTerm = "";
    let forwardSearchTerm = "";
    let selectedExtension = "";
    let currentFolder = "active";
    let selectedKey = null;
    let forwardingKey = null;
    let selectedBulkKeys = new Set();
    let deleteCommentDrafts = new Map();
    let pendingDeleteKeys = new Set();
    let queuedUndoKeys = new Set();
    let playingKey = null;
    let loadingVoicemails = false;
    let activeSegmentEnd = null;
    let activeSegmentButton = null;
    let highlightedTranscriptWord = null;
    const refreshIntervalMs = 30000;
    const themeStorageKey = "voicemailPortalOciTheme";
    const playbackRateStorageKey = "voicemailPortalPlaybackRate";
    const playbackRates = [0.8, 1, 1.25, 1.5];
    let selectedPlaybackRate = 1;
    let undoToastTimer = null;
    let undoToastKey = null;

    function text(value) {{
      return value == null || value === "" ? "Not Included" : String(value);
    }}

    function escapeHtml(value) {{
      return String(value || "").replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[char]));
    }}

    function preview(item) {{
      const transcript = item.transcript || "";
      if (!transcript.trim()) return "Transcript pending or unavailable.";
      return transcript.length > 150 ? transcript.slice(0, 150) + "..." : transcript;
    }}

    function normalizeTranscriptWord(value) {{
      return String(value || "")
        .replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "")
        .toLowerCase();
    }}

    function digitsOnly(value) {{
      return String(value || "").replace(/\\D/g, "");
    }}

    function canonicalPhoneDigits(value) {{
      const digits = digitsOnly(value);
      return digits.length === 11 && digits.startsWith("1") ? digits.slice(1) : digits;
    }}

    function canonicalDateDigits(value) {{
      const digits = digitsOnly(value);
      if (digits.length === 8) return digits;
      if (digits.length === 7) {{
        return `0${{digits}}`;
      }}
      if (digits.length === 6) {{
        const yy = Number(digits.slice(4));
        const century = yy > 30 ? "19" : "20";
        return `${{digits.slice(0, 4)}}${{century}}${{digits.slice(4)}}`;
      }}
      return digits;
    }}

    function formatSeekTime(seconds) {{
      const value = Math.max(0, Math.floor(Number(seconds) || 0));
      return `${{Math.floor(value / 60)}}:${{String(value % 60).padStart(2, "0")}}`;
    }}

    function normalizePlaybackRate(value) {{
      const numeric = Number(value);
      return playbackRates.includes(numeric) ? numeric : 1;
    }}

    function playbackRateLabel(rate) {{
      return `${{rate}}x`;
    }}

    function initPlaybackRate() {{
      selectedPlaybackRate = normalizePlaybackRate(localStorage.getItem(playbackRateStorageKey));
      localStorage.setItem(playbackRateStorageKey, String(selectedPlaybackRate));
    }}

    function applyPlaybackRate(audio) {{
      if (!audio) return;
      audio.playbackRate = selectedPlaybackRate;
    }}

    function syncPlaybackRateButtons() {{
      document.querySelectorAll("#detail [data-rate]").forEach(button => {{
        const rate = normalizePlaybackRate(button.dataset.rate);
        const active = rate === selectedPlaybackRate;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
      }});
    }}

    function setPlaybackRate(value) {{
      selectedPlaybackRate = normalizePlaybackRate(value);
      localStorage.setItem(playbackRateStorageKey, String(selectedPlaybackRate));
      const audio = document.querySelector("#detail audio");
      if (audio) {{
        applyPlaybackRate(audio);
        updateTranscriptHighlight(audio.currentTime);
      }}
      syncPlaybackRateButtons();
    }}

    function playbackSpeedControlsHtml() {{
      return '<div class="speed-controls" aria-label="Playback speed"><span>Playback Speed</span>' + playbackRates.map(rate => {{
        const label = playbackRateLabel(rate);
        const active = rate === selectedPlaybackRate ? " active" : "";
        const pressed = rate === selectedPlaybackRate ? "true" : "false";
        return '<button class="speed-btn' + active + '" data-rate="' + rate + '" type="button" aria-pressed="' + pressed + '">' + label + '</button>';
      }}).join("") + '</div>';
    }}

    function timedTranscriptHtml(item) {{
      const transcript = item.transcript || "Transcript pending or unavailable.";
      const timings = Array.isArray(item.word_timestamps) ? item.word_timestamps : [];
      if (!item.has_audio || !timings.length) {{
        return escapeHtml(transcript);
      }}

      const parts = transcript.match(/\\s+|\\S+/g) || [];
      let timingIndex = 0;
      return parts.map(part => {{
        if (/^\\s+$/.test(part)) {{
          return escapeHtml(part);
        }}

        const normalizedPart = normalizeTranscriptWord(part);
        if (!normalizedPart) {{
          return escapeHtml(part);
        }}

        const matchedIndex = timingIndex;
        const matched = timings[timingIndex] || null;
        timingIndex += 1;
        if (!matched) {{
          return escapeHtml(part);
        }}

        const start = Number(matched.start);
        if (!Number.isFinite(start)) {{
          return escapeHtml(part);
        }}
        const end = Number(matched.end);
        const endAttr = Number.isFinite(end) ? ` data-end="${{end}}"` : "";

        return `<button class="transcript-word" type="button" data-word-index="${{matchedIndex}}" data-seek="${{start}}"${{endAttr}} title="Play from ${{formatSeekTime(start)}}">${{escapeHtml(part)}}</button>`;
      }}).join("");
    }}

    function playIcon() {{
      return '<svg viewBox="0 0 24 24" fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="8 5 19 12 8 19 8 5"></polygon></svg>';
    }}

    function pauseIcon() {{
      return '<svg viewBox="0 0 24 24" fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 5v14"></path><path d="M16 5v14"></path></svg>';
    }}

    function copyIcon() {{
      return '<svg viewBox="0 0 20 20" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="8" y="4" width="7" height="9" rx="1.5"></rect><path d="M8 7H7a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h3.5a2 2 0 0 0 2-2v-1"></path></svg>';
    }}

    function checkIcon() {{
      return '<svg viewBox="0 0 24 24" fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"></path></svg>';
    }}

    function trashIcon() {{
      return '<svg viewBox="0 0 24 24" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"></path><path d="M8 6V4h8v2"></path><path d="M19 6l-1 14H6L5 6"></path><path d="M10 11v5"></path><path d="M14 11v5"></path></svg>';
    }}

    function restoreIcon() {{
      return '<svg viewBox="0 0 24 24" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7v6h6"></path><path d="M21 17a9 9 0 0 0-15-6.7L3 13"></path></svg>';
    }}

    function forwardIcon() {{
      return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 18c1.4-5.1 4.9-7.8 10-7.8h.6V6.5L21 12l-6.4 5.5v-3.7h-.9c-3.7 0-6.7 1.2-9.7 4.2Z"></path></svg>';
    }}

    function captureFocus() {{
      const element = document.activeElement;
      if (!element || element === document.body) return null;
      if (element.id) return {{ type: "id", value: element.id }};
      const dataKeys = ["open", "play", "delete", "restore", "forward", "seek", "folder", "extension", "rate"];
      for (const key of dataKeys) {{
        if (element.dataset && element.dataset[key]) {{
          return {{ type: "data", key, value: element.dataset[key] }};
        }}
      }}
      return null;
    }}

    function restoreFocus(snapshot) {{
      if (!snapshot) return;
      let element = null;
      if (snapshot.type === "id") {{
        element = document.getElementById(snapshot.value);
      }} else if (snapshot.type === "data") {{
        element = Array.from(document.querySelectorAll(`[data-${{snapshot.key}}]`))
          .find(candidate => candidate.dataset && candidate.dataset[snapshot.key] === snapshot.value);
      }}
      if (element && typeof element.focus === "function") {{
        try {{
          element.focus({{ preventScroll: true }});
        }} catch (_error) {{
          element.focus();
        }}
      }}
    }}

    function sunIcon() {{
      return '<svg viewBox="0 0 24 24" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path><path d="M2 12h2"></path><path d="M20 12h2"></path><path d="m6.34 17.66-1.41 1.41"></path><path d="m19.07 4.93-1.41 1.41"></path></svg>';
    }}

    function moonIcon() {{
      return '<svg viewBox="0 0 24 24" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.99 13.06A8.5 8.5 0 1 1 10.94 3.01 6.8 6.8 0 0 0 20.99 13.06Z"></path></svg>';
    }}

    function applyTheme(theme) {{
      const normalized = theme === "dark" ? "dark" : "light";
      document.documentElement.dataset.theme = normalized;
      localStorage.setItem(themeStorageKey, normalized);
      const button = document.getElementById("themeBtn");
      if (button) {{
        const label = normalized === "dark" ? "Switch to light mode" : "Switch to dark mode";
        button.innerHTML = normalized === "dark" ? sunIcon() : moonIcon();
        button.title = label;
        button.setAttribute("aria-label", label);
        button.setAttribute("aria-pressed", normalized === "dark" ? "true" : "false");
      }}
    }}

    function initTheme() {{
      const saved = localStorage.getItem(themeStorageKey);
      applyTheme(saved || "dark");
    }}

    function toggleTheme() {{
      applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    }}

    function callbackMatchHtml(value) {{
      const normalized = String(value || "").trim().toLowerCase();
      if (!normalized || normalized === "callback not included") {{
        return "<strong>Callback Not Included</strong>";
      }}
      if (normalized.startsWith("no")) {{
        return '<strong class="review-needed">No - Needs Review</strong>';
      }}
      if (normalized === "yes") {{
        return '<strong class="match-yes">Yes</strong>';
      }}
      return `<strong>${{escapeHtml(value)}}</strong>`;
    }}

    function fieldVerification(item, fieldName) {{
      const verifications = item && item.field_verifications ? item.field_verifications : {{}};
      return verifications[fieldName] || null;
    }}

    function verificationStatusHtml(item, fieldName) {{
      const row = fieldVerification(item, fieldName);
      if (!row) return "";

      const status = String(row.status || "").trim();
      const normalized = status.toLowerCase();
      const whisperJson = row.whisper_json && typeof row.whisper_json === "object" ? row.whisper_json : {{}};
      const statusLabels = {{
        verified: row.used_parakeet && whisperJson.agreement_source === "entity" ? "Whisper + Parakeet agree" : row.used_parakeet ? "Verified by Parakeet" : "Verified",
        whisper_caller_id_verified: "Whisper + Caller ID agree",
        parakeet_override: "Parakeet override",
        whisper_span_fallback: "Whisper fallback",
        gemma_final: "Gemma extracted",
        caller_id_spelling_corrected: "Caller ID spelling corrected",
        ambiguous: "Needs review",
        legacy_fallback: "Original preserved",
        not_included: "Not included"
      }};
      const label = statusLabels[normalized] || status || "Verification recorded";
      const needsReview = Boolean(row.needs_review) || normalized === "ambiguous" || normalized === "parakeet_override";
      const cls = needsReview ? "review-needed" : (["verified", "whisper_caller_id_verified"].includes(normalized) ? "match-yes" : "");
      const classAttr = cls ? ` class="${{cls}}"` : "";
      const sources = [];
      if (row.used_gemma) sources.push("Gemma");
      if (row.used_parakeet) sources.push("Parakeet");
      const sourceText = sources.length ? ` (${{sources.join(" + ")}})` : "";
      const reasons = Array.isArray(row.review_reasons) ? row.review_reasons.filter(Boolean).join(", ") : "";
      const titleAttr = reasons ? ` title="${{escapeHtml(reasons)}}"` : "";
      return `<small class="field-extra"${{titleAttr}}>Status: <strong${{classAttr}}>${{escapeHtml(label)}}</strong>${{escapeHtml(sourceText)}}</small>`;
    }}

    function phoneConfidenceHtml(item, fieldName, value) {{
      if (text(value) === "Not Included") return "";

      const row = fieldVerification(item, fieldName);
      if (!row) return "";

      const status = String(row.status || "").trim().toLowerCase();
      if (status === "not_included") return "";

      const reasons = Array.isArray(row.review_reasons) ? row.review_reasons.filter(Boolean).join(", ") : "";
      const titleAttr = reasons ? ` title="${{escapeHtml(reasons)}}"` : "";

      if (status === "parakeet_override") {{
        return `<small class="field-extra"${{titleAttr}}><strong class="review-needed">Medium Confidence - Please Review</strong></small>`;
      }}

      if (status === "verified" || status === "whisper_caller_id_verified") {{
        return `<small class="field-extra"${{titleAttr}}><strong class="match-yes">High Confidence</strong></small>`;
      }}

      return "";
    }}

    function dobConfidenceHtml(item, value) {{
      if (text(value) === "Not Included") return "";

      const row = fieldVerification(item, "dob");
      if (!row) return "";

      const status = String(row.status || "").trim().toLowerCase();
      if (status === "not_included") return "";

      const reasons = Array.isArray(row.review_reasons) ? row.review_reasons.filter(Boolean).join(", ") : "";
      const titleAttr = reasons ? ` title="${{escapeHtml(reasons)}}"` : "";
      const needsReview = Boolean(row.needs_review) || ["ambiguous", "legacy_fallback"].includes(status);

      if (needsReview) {{
        return `<small class="field-extra"${{titleAttr}}><strong class="review-needed">Medium Confidence - Please Review</strong></small>`;
      }}

      if (status === "gemma_final" || status === "verified") {{
        return `<small class="field-extra"${{titleAttr}}><strong class="match-yes">High Confidence</strong></small>`;
      }}

      return "";
    }}

    function copyButtonHtml(value, label) {{
      const copyValue = text(value);
      return `<button class="copy-btn" data-copy-value="${{escapeHtml(copyValue)}}" type="button" title="Copy ${{escapeHtml(label)}}" aria-label="Copy ${{escapeHtml(label)}}">${{copyIcon()}}</button>`;
    }}

    function callerIdCopyButtonHtml(item) {{
      const digits = String((item && item.callerid_number_digits) || "");
      if (!digits) return "";
      return `<button class="copy-btn caller-copy" data-copy-value="${{escapeHtml(digits)}}" type="button" title="Copy caller ID number" aria-label="Copy caller ID number">${{copyIcon()}}</button>`;
    }}

    function plainFieldHtml(value, label) {{
      const display = text(value);
      return `<div class="field-value-line"><div class="field-value-text">${{escapeHtml(display)}}</div>${{copyButtonHtml(display, label)}}</div>`;
    }}

    function expandedSpokenSegmentEnd(timings, endIndex, segmentStart, segmentEnd, tailSeconds, minimumSeconds, maxNextGapSeconds) {{
      const nextTiming = timings
        .slice(endIndex + 1)
        .find(next => Number.isFinite(Number(next && next.start)));
      const nextStart = nextTiming ? Number(nextTiming.start) : NaN;
      const boundaryEnd = Number.isFinite(nextStart) && nextStart > segmentEnd && nextStart - segmentEnd < maxNextGapSeconds
        ? Math.max(segmentEnd, nextStart - 0.05)
        : segmentEnd;
      return Math.max(segmentStart + minimumSeconds, boundaryEnd + tailSeconds);
    }}

    function findNumberSegment(item, numberValue) {{
      const targetDigits = canonicalPhoneDigits(numberValue);
      if (!targetDigits || targetDigits.length < 7 || !Array.isArray(item.word_timestamps)) {{
        return null;
      }}

      const timings = item.word_timestamps;
      for (let startIndex = 0; startIndex < timings.length; startIndex += 1) {{
        let combined = "";
        let segmentStart = null;
        let segmentEnd = null;
        const maxEnd = Math.min(timings.length, startIndex + 8);

        for (let endIndex = startIndex; endIndex < maxEnd; endIndex += 1) {{
          const timing = timings[endIndex] || {{}};
          const digits = digitsOnly(timing.word);
          if (!digits) {{
            if (combined) break;
            continue;
          }}

          const start = Number(timing.start);
          const end = Number(timing.end);
          if (!Number.isFinite(start)) break;
          if (segmentStart === null) segmentStart = start;
          if (Number.isFinite(end)) segmentEnd = end;
          combined += digits;

          const candidateDigits = canonicalPhoneDigits(combined);
          const exactMatch = candidateDigits === targetDigits;
          const embeddedMatch = targetDigits.length >= 7 && candidateDigits.includes(targetDigits);
          if ((exactMatch || embeddedMatch) && segmentStart !== null && segmentEnd !== null) {{
            return {{
              start: Math.max(0, segmentStart - 0.15),
              end: expandedSpokenSegmentEnd(
                timings,
                endIndex,
                segmentStart,
                segmentEnd,
                0.6,
                Math.max(1.8, Math.min(candidateDigits.length, 12) * 0.28),
                5
              )
            }};
          }}

          if (
            candidateDigits.length > targetDigits.length + 5 ||
            (candidateDigits && !targetDigits.startsWith(candidateDigits) && !candidateDigits.startsWith(targetDigits))
          ) {{
            break;
          }}
        }}
      }}

      return null;
    }}

    function finiteSegment(startValue, endValue) {{
      const start = Number(startValue);
      const end = Number(endValue);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {{
        return null;
      }}
      return {{
        start: Math.max(0, start),
        end
      }};
    }}

    function verificationTimingSegment(item, fieldName) {{
      const row = fieldVerification(item, fieldName);
      if (!row) return null;

      const attributions = Array.isArray(row.attribution_json) ? row.attribution_json : [];
      for (const attribution of attributions) {{
        const segment = finiteSegment(attribution && attribution.start, attribution && attribution.end);
        if (segment) return segment;
      }}

      const clips = Array.isArray(row.clip_json) ? row.clip_json : [];
      for (const clip of clips) {{
        const segment = finiteSegment(clip && clip.start, clip && clip.end);
        if (segment) return segment;
      }}

      return null;
    }}

    function numberFieldHtml(item, value, label) {{
      const display = text(value);
      const fieldName = label === "fax" ? "fax_number" : "callback_number";
      const segment = findNumberSegment(item, value) || verificationTimingSegment(item, fieldName);
      const button = segment
        ? `<button class="segment-play" data-number-segment type="button" data-segment-start="${{segment.start}}" data-segment-end="${{segment.end}}" data-segment-label="${{escapeHtml(label)}}" title="Play ${{escapeHtml(label)}} number" aria-label="Play ${{escapeHtml(label)}} number">${{playIcon()}}</button>`
        : "";
      return `<div class="field-value-line"><div class="field-value-text">${{escapeHtml(display)}}</div>${{button}}${{copyButtonHtml(display, label)}}</div>`;
    }}

    function findDobSegment(item, dobValue) {{
      const targetDigits = canonicalDateDigits(dobValue);
      if (!targetDigits || targetDigits.length < 6 || !Array.isArray(item.word_timestamps)) {{
        return null;
      }}

      const timings = item.word_timestamps;
      for (let startIndex = 0; startIndex < timings.length; startIndex += 1) {{
        let combined = "";
        let segmentStart = null;
        let segmentEnd = null;
        const maxEnd = Math.min(timings.length, startIndex + 8);

        for (let endIndex = startIndex; endIndex < maxEnd; endIndex += 1) {{
          const timing = timings[endIndex] || {{}};
          const digits = digitsOnly(timing.word);
          if (!digits) {{
            if (combined) break;
            continue;
          }}

          const start = Number(timing.start);
          const end = Number(timing.end);
          if (!Number.isFinite(start)) break;
          if (segmentStart === null) segmentStart = start;
          if (Number.isFinite(end)) segmentEnd = end;
          combined += digits;

          const candidateDigits = canonicalDateDigits(combined);
          if (candidateDigits === targetDigits && segmentStart !== null && segmentEnd !== null) {{
            return {{
              start: Math.max(0, segmentStart - 0.15),
              end: expandedSpokenSegmentEnd(
                timings,
                endIndex,
                segmentStart,
                segmentEnd,
                1.25,
                Math.max(2.4, targetDigits.length * 0.3),
                5
              )
            }};
          }}

          if (
            candidateDigits.length > targetDigits.length + 2 ||
            (candidateDigits && !targetDigits.startsWith(candidateDigits) && !candidateDigits.startsWith(targetDigits))
          ) {{
            break;
          }}
        }}
      }}

      return null;
    }}

    function dobFieldHtml(item, value) {{
      const display = text(value);
      const segment = findDobSegment(item, value);
      const button = segment
        ? `<button class="segment-play" data-number-segment type="button" data-segment-start="${{segment.start}}" data-segment-end="${{segment.end}}" data-segment-label="DOB" title="Play DOB" aria-label="Play DOB">${{playIcon()}}</button>`
        : "";
      return `<div class="field-value-line"><div class="field-value-text">${{escapeHtml(display)}}</div>${{button}}${{copyButtonHtml(display, "DOB")}}</div>`;
    }}

    function extensionLabel(extension) {{
      if (!extension) return isAdmin ? "All Extensions" : "All Assigned Mailboxes";
      const match = extensions.find(item => item.extension === extension);
      if (!match || !match.display_name) return `Ext ${{extension}}`;
      return `Ext ${{extension}} - ${{match.display_name}}`;
    }}

    function toggleExtensionMenu(forceOpen = null) {{
      const menu = document.getElementById("extensionMenu");
      const button = document.getElementById("menuBtn");
      const willOpen = forceOpen === null ? menu.hidden : forceOpen;
      if (willOpen) toggleDirectoryMenu(false);
      menu.hidden = !willOpen;
      button.setAttribute("aria-expanded", willOpen ? "true" : "false");
      if (willOpen && isAdmin) {{
        const search = document.getElementById("extensionMenuSearch");
        if (search) window.requestAnimationFrame(() => search.focus());
      }}
    }}

    function directoryEntryLabel(item) {{
      const name = String((item && item.display_name) || "").trim();
      if (name) return `${{name}} - Ext ${{item.extension}}`;
      return `Ext ${{item.extension}}`;
    }}

    function forwardingEventTime(event) {{
      return (event && (event.forwarded_display || event.forwarded_utc)) || "";
    }}

    function forwardedToSummary(item) {{
      const events = Array.isArray(item && item.forwarded_to) ? item.forwarded_to : [];
      if (!events.length) return "";
      const first = events[0];
      const suffix = events.length > 1 ? ` and ${{events.length - 1}} more` : "";
      return `Forwarded to Ext ${{escapeHtml(first.target_extension || "")}}${{suffix}}`;
    }}

    function forwardingNoticeHtml(item, compact = false) {{
      if (!item) return "";
      const forwardedFrom = item.forwarded_from || null;
      const forwardedTo = Array.isArray(item.forwarded_to) ? item.forwarded_to : [];
      if (compact) {{
        if (forwardedFrom) {{
          return `<div class="forward-summary">Forwarded from Ext ${{escapeHtml(forwardedFrom.source_extension || "")}}</div>`;
        }}
        const summary = forwardedToSummary(item);
        return summary ? `<div class="forward-summary">${{summary}}</div>` : "";
      }}
      const rows = [];
      if (forwardedFrom) {{
        rows.push(`Forwarded from Ext ${{escapeHtml(forwardedFrom.source_extension || "")}} by ${{escapeHtml(forwardedFrom.forwarded_by || "unknown")}} ${{escapeHtml(forwardingEventTime(forwardedFrom))}}`);
      }}
      forwardedTo.forEach(event => {{
        rows.push(`Forwarded to Ext ${{escapeHtml(event.target_extension || "")}} by ${{escapeHtml(event.forwarded_by || "unknown")}} ${{escapeHtml(forwardingEventTime(event))}}`);
      }});
      if (!rows.length) return "";
      if (rows.length === 1) {{
        return `<div class="forwarding-notice">${{rows[0]}}</div>`;
      }}
      return `<div class="forwarding-notice"><strong>Forwarded</strong><ul>${{rows.map(row => `<li>${{row}}</li>`).join("")}}</ul></div>`;
    }}

    function directoryMatchesSearch(item) {{
      const term = directorySearchTerm.trim().toLowerCase();
      if (!term) return true;
      const haystack = `${{(item && item.display_name) || ""}} ${{(item && item.extension) || ""}}`.toLowerCase();
      return haystack.includes(term);
    }}

    function numericExtensionSortValue(item) {{
      const value = Number((item && item.extension) || "");
      return Number.isFinite(value) ? value : 999999;
    }}

    function sortedDirectoryEntries() {{
      const filtered = directoryEntries.filter(directoryMatchesSearch);
      const namedEntries = filtered
        .filter(item => String((item && item.display_name) || "").trim())
        .sort((a, b) => {{
          const byName = String(a.display_name || "").localeCompare(String(b.display_name || ""), undefined, {{ sensitivity: "base" }});
          return byName || numericExtensionSortValue(a) - numericExtensionSortValue(b);
        }});
      const unnamedEntries = filtered
        .filter(item => !String((item && item.display_name) || "").trim())
        .sort((a, b) => numericExtensionSortValue(a) - numericExtensionSortValue(b));
      return [...namedEntries, ...unnamedEntries];
    }}

    function renderDirectoryMenu() {{
      const items = document.getElementById("directoryItems");
      if (!items) return;
      const entries = sortedDirectoryEntries();
      if (!entries.length) {{
        items.innerHTML = '<div class="directory-empty">No directory entries found.</div>';
        return;
      }}
      items.innerHTML = entries
        .map(item => {{
          const label = escapeHtml(directoryEntryLabel(item));
          if (isAdmin) {{
            return `<button class="directory-entry directory-action" type="button" data-directory-extension="${{escapeHtml(item.extension)}}">${{label}}</button>`;
          }}
          return `<div class="directory-entry" role="listitem">${{label}}</div>`;
        }})
        .join("");
    }}

    function targetableForwardEntries(item) {{
      const sourceExtension = String((item && item.extension) || "");
      const term = forwardSearchTerm.trim().toLowerCase();
      return directoryEntries
        .filter(entry => String((entry && entry.extension) || "") !== sourceExtension)
        .filter(entry => {{
          if (!term) return true;
          const haystack = `${{(entry && entry.display_name) || ""}} ${{(entry && entry.extension) || ""}}`.toLowerCase();
          return haystack.includes(term);
        }})
        .sort((a, b) => {{
          const aHasName = Boolean(String(a.display_name || "").trim());
          const bHasName = Boolean(String(b.display_name || "").trim());
          if (aHasName !== bHasName) return aHasName ? -1 : 1;
          const byName = String(a.display_name || "").localeCompare(String(b.display_name || ""), undefined, {{ sensitivity: "base" }});
          return byName || numericExtensionSortValue(a) - numericExtensionSortValue(b);
        }});
    }}

    function forwardTargetLabel(targetExtension) {{
      const extension = String(targetExtension || "");
      const entry = directoryEntries.find(candidate => String((candidate && candidate.extension) || "") === extension);
      return entry ? directoryEntryLabel(entry) : `Ext ${{extension}}`;
    }}

    function renderForwardPicker() {{
      const options = document.getElementById("forwardOptions");
      if (!options) return;
      const item = voicemailByKey(forwardingKey);
      if (!item) {{
        options.innerHTML = '<div class="directory-empty">Select a voicemail first.</div>';
        return;
      }}
      const entries = targetableForwardEntries(item);
      if (!entries.length) {{
        options.innerHTML = '<div class="directory-empty">No target mailboxes found.</div>';
        return;
      }}
      options.innerHTML = entries.map(entry => {{
        const label = escapeHtml(directoryEntryLabel(entry));
        return `<button class="directory-entry" type="button" data-forward-target="${{escapeHtml(entry.extension)}}">${{label}}</button>`;
      }}).join("");
      options.querySelectorAll("[data-forward-target]").forEach(button => {{
        button.addEventListener("click", () => {{
          const key = forwardingKey;
          const target = button.dataset.forwardTarget || "";
          closeForwardPicker();
          forwardVoicemail(key, target);
        }});
      }});
    }}

    function openForwardPicker(key) {{
      const item = voicemailByKey(key);
      if (!item || item.deleted_utc) return;
      forwardingKey = key;
      forwardSearchTerm = "";
      const picker = document.getElementById("forwardPicker");
      const search = document.getElementById("forwardSearch");
      if (search) search.value = "";
      renderForwardPicker();
      if (picker) picker.hidden = false;
      if (search) window.requestAnimationFrame(() => search.focus());
    }}

    function closeForwardPicker() {{
      const picker = document.getElementById("forwardPicker");
      if (picker) picker.hidden = true;
      forwardingKey = null;
    }}

    function toggleDirectoryMenu(forceOpen = null) {{
      const menu = document.getElementById("directoryMenu");
      const button = document.getElementById("directoryBtn");
      const willOpen = forceOpen === null ? menu.hidden : forceOpen;
      if (willOpen) {{
        toggleExtensionMenu(false);
        renderDirectoryMenu();
      }}
      menu.hidden = !willOpen;
      button.setAttribute("aria-expanded", willOpen ? "true" : "false");
      if (willOpen) {{
        const search = document.getElementById("directorySearch");
        if (search) window.requestAnimationFrame(() => search.focus());
      }}
    }}

    function extensionMenuMatchesSearch(item) {{
      const term = extensionMenuSearchTerm.trim().toLowerCase();
      if (!term) return true;
      const haystack = `${{(item && item.display_name) || ""}} ${{(item && item.extension) || ""}}`.toLowerCase();
      return haystack.includes(term);
    }}

    function folderLabel() {{
      return currentFolder === "deleted" ? "Deleted" : "Inbox";
    }}

    function renderFolderTabs() {{
      const tabs = document.getElementById("folderTabs");
      if (!tabs || !isAdmin) {{
        return;
      }}
      tabs.hidden = false;
      tabs.innerHTML = `
        <button class="folder-tab ${{currentFolder === "active" ? "active" : ""}}" data-folder="active" type="button">Inbox</button>
        <button class="folder-tab ${{currentFolder === "deleted" ? "active" : ""}}" data-folder="deleted" type="button">Deleted</button>
      `;
      tabs.querySelectorAll("[data-folder]").forEach(button => {{
        button.addEventListener("click", () => {{
          const nextFolder = button.dataset.folder || "active";
          if (nextFolder === currentFolder) return;
          currentFolder = nextFolder;
          selectedKey = null;
          selectedBulkKeys.clear();
          playingKey = null;
          document.getElementById("detail").innerHTML = `<div class="empty">Loading ${{escapeHtml(folderLabel())}}.</div>`;
          loadVoicemails();
        }});
      }});
    }}

    function renderExtensionMenu() {{
      const menu = document.getElementById("extensionMenu");
      const allOption = selectedExtension === "" ? "active" : "";
      const regularFolderTarget = currentFolder === "deleted" ? "active" : "deleted";
      const regularFolderLabel = currentFolder === "deleted" ? "Inbox" : "Deleted";
      const regularFolderHint = currentFolder === "deleted" ? "Return to active voicemails" : "View and restore deleted voicemails";
      const extensionRows = isAdmin ? extensions.filter(extensionMenuMatchesSearch) : extensions;
      const searchHtml = isAdmin
        ? `<div class="menu-search-wrap"><input id="extensionMenuSearch" type="search" placeholder="Search extensions" autocomplete="off" value="${{escapeHtml(extensionMenuSearchTerm)}}"></div>`
        : "";
      const emptySearchHtml = isAdmin && !extensionRows.length
        ? '<div class="directory-empty">No extensions found.</div>'
        : "";
      const allExtensionLabel = isAdmin ? "All Extensions" : "All Assigned Mailboxes";
      const folderOptionHtml = isAdmin ? "" : `<button class="extension-option extension-folder-option ${{currentFolder === "deleted" ? "active" : "deleted-folder"}}" data-user-folder="${{regularFolderTarget}}" type="button">${{regularFolderLabel}}<span>${{regularFolderHint}}</span></button>`;
      const options = [
        searchHtml,
        ...(currentFolder === "deleted" && folderOptionHtml ? [folderOptionHtml] : []),
        `<button class="extension-option ${{allOption}}" data-extension="" type="button">${{allExtensionLabel}}<span>${{voicemails.length}} voicemail${{voicemails.length === 1 ? "" : "s"}}</span></button>`,
        ...extensionRows.map(item => {{
          const count = voicemails.filter(vm => vm.extension === item.extension).length;
          const active = item.extension === selectedExtension ? "active" : "";
          const name = item.display_name ? escapeHtml(item.display_name) : "No mailbox name found";
          return `<button class="extension-option ${{active}}" data-extension="${{escapeHtml(item.extension)}}" type="button">Ext ${{escapeHtml(item.extension)}}<span>${{name}} | ${{count}} voicemail${{count === 1 ? "" : "s"}}</span></button>`;
        }}),
        emptySearchHtml,
        ...(currentFolder !== "deleted" && folderOptionHtml ? [folderOptionHtml] : [])
      ];
      menu.innerHTML = options.join("");
      const extensionSearch = document.getElementById("extensionMenuSearch");
      if (extensionSearch) {{
        extensionSearch.addEventListener("input", event => {{
          extensionMenuSearchTerm = event.target.value || "";
          renderExtensionMenu();
          window.requestAnimationFrame(() => {{
            const nextSearch = document.getElementById("extensionMenuSearch");
            if (!nextSearch) return;
            nextSearch.focus();
            const valueLength = nextSearch.value.length;
            nextSearch.setSelectionRange(valueLength, valueLength);
          }});
        }});
      }}
      menu.querySelectorAll("[data-extension]").forEach(button => {{
        button.addEventListener("click", () => {{
          selectExtensionFilter(button.dataset.extension || "");
        }});
      }});
      menu.querySelectorAll("[data-user-folder]").forEach(button => {{
        button.addEventListener("click", () => {{
          const nextFolder = button.dataset.userFolder || "active";
          if (nextFolder === currentFolder) return;
          currentFolder = nextFolder;
          selectedKey = null;
          selectedBulkKeys.clear();
          playingKey = null;
          document.getElementById("detail").innerHTML = `<div class="empty">Loading ${{escapeHtml(folderLabel())}}.</div>`;
          toggleExtensionMenu(false);
          loadVoicemails();
        }});
      }});
    }}

    function selectExtensionFilter(extension) {{
      selectedExtension = String(extension || "");
      selectedKey = null;
      selectedBulkKeys.clear();
      toggleExtensionMenu(false);
      toggleDirectoryMenu(false);
      renderExtensionMenu();
      renderList();
      if (!selectFirstVisibleVoicemail()) {{
        document.getElementById("detail").innerHTML = `<div class="empty">No voicemails found for ${{escapeHtml(extensionLabel(selectedExtension))}}.</div>`;
      }}
    }}

    function filteredItems() {{
      const query = document.getElementById("search").value.trim().toLowerCase();
      let items = voicemails;
      if (selectedExtension) {{
        items = items.filter(item => item.extension === selectedExtension);
      }}
      if (!query) return items;
      return items.filter(item => {{
        const entities = item.entities || {{}};
        const haystack = [
          item.extension,
          item.mailbox,
          item.callerid,
          item.display_date,
          item.transcript,
          entities.name,
          entities.dob,
          entities.callback_number,
          entities.fax_number
        ].join(" ").toLowerCase();
        return haystack.includes(query);
      }});
    }}

    function visibleBulkItems() {{
      return currentFolder === "deleted" ? [] : filteredItems();
    }}

    function pruneBulkSelection() {{
      const available = new Set(voicemails.map(item => item.file_key));
      selectedBulkKeys.forEach(key => {{
        if (!available.has(key)) {{
          selectedBulkKeys.delete(key);
        }}
      }});
    }}

    function updateBulkBar() {{
      const bulkBar = document.getElementById("bulkBar");
      const selectAllBox = document.getElementById("selectAllBox");
      const bulkDeleteBtn = document.getElementById("bulkDeleteBtn");
      const clearSelectionBtn = document.getElementById("clearSelectionBtn");
      const bulkCount = document.getElementById("bulkCount");
      if (!bulkBar || !selectAllBox || !bulkDeleteBtn || !clearSelectionBtn || !bulkCount) return;

      if (currentFolder === "deleted") {{
        bulkBar.hidden = true;
        selectAllBox.checked = false;
        selectAllBox.indeterminate = false;
        return;
      }}

      bulkBar.hidden = false;
      const visibleItems = visibleBulkItems();
      const selectedVisibleCount = visibleItems.filter(item => selectedBulkKeys.has(item.file_key)).length;
      const selectedCount = selectedBulkKeys.size;
      selectAllBox.disabled = visibleItems.length === 0;
      selectAllBox.checked = visibleItems.length > 0 && selectedVisibleCount === visibleItems.length;
      selectAllBox.indeterminate = selectedVisibleCount > 0 && selectedVisibleCount < visibleItems.length;
      bulkDeleteBtn.disabled = selectedCount === 0;
      clearSelectionBtn.disabled = selectedCount === 0;
      bulkCount.textContent = selectedCount ? `${{selectedCount}} selected` : "";
    }}

    function selectAllVisible(checked) {{
      visibleBulkItems().forEach(item => {{
        if (checked) {{
          selectedBulkKeys.add(item.file_key);
        }} else {{
          selectedBulkKeys.delete(item.file_key);
        }}
      }});
      renderList();
    }}

    function clearBulkSelection() {{
      selectedBulkKeys.clear();
      renderList();
    }}

    function renderList() {{
      const root = document.getElementById("items");
      const items = filteredItems();
      if (!items.length) {{
        root.innerHTML = `<div class="empty">No ${{currentFolder === "deleted" ? "deleted " : ""}}voicemails found.</div>`;
        updateBulkBar();
        refreshReviewProgress();
        return;
      }}
      root.innerHTML = items.map(item => `
        <div class="item ${{currentFolder === "deleted" ? "" : "has-bulk"}} ${{item.file_key === selectedKey ? "active" : ""}}" data-key="${{item.file_key}}">
          ${{currentFolder === "deleted" ? "" : `<label class="bulk-select" title="Select voicemail" aria-label="Select voicemail"><input data-select="${{item.file_key}}" type="checkbox" ${{selectedBulkKeys.has(item.file_key) ? "checked" : ""}}></label>`}}
          <button class="item-main" data-open="${{item.file_key}}" type="button">
            <div class="item-title">
              <span>${{escapeHtml(item.callerid || "Unknown")}}</span>
              <span>${{escapeHtml(item.duration_display || "")}}</span>
            </div>
            <div class="item-meta">${{escapeHtml(item.display_date)}} | Ext ${{escapeHtml(item.extension)}}</div>
            ${{item.deleted_utc ? `<div class="item-meta">Deleted ${{escapeHtml(item.deleted_display || item.deleted_utc)}} by ${{escapeHtml(item.deleted_by || "unknown")}}</div>` : ""}}
            ${{item.deleted_comment ? `<div class="item-meta">Comment: ${{escapeHtml(item.deleted_comment)}}</div>` : ""}}
            ${{forwardingNoticeHtml(item, true)}}
            <div class="item-preview">${{escapeHtml(preview(item))}}</div>
          </button>
          <div class="item-actions">
            <button class="icon-btn play" data-play="${{item.file_key}}" type="button" title="${{playingKey === item.file_key ? "Pause voicemail" : "Play voicemail"}}" aria-label="${{playingKey === item.file_key ? "Pause voicemail" : "Play voicemail"}}" ${{item.has_audio ? "" : "disabled"}}>${{playingKey === item.file_key ? pauseIcon() : playIcon()}}</button>
            ${{currentFolder === "deleted" ? `<button class="icon-btn restore" data-restore="${{item.file_key}}" type="button" title="Restore voicemail" aria-label="Restore voicemail">${{restoreIcon()}}</button>` : `${{forwardingEnabled ? `<button class="icon-btn forward" data-forward="${{item.file_key}}" type="button" title="Forward voicemail" aria-label="Forward voicemail">${{forwardIcon()}}</button>` : ""}}<button class="icon-btn danger" data-delete="${{item.file_key}}" type="button" title="Delete voicemail" aria-label="Delete voicemail">${{trashIcon()}}</button>`}}
          </div>
        </div>
      `).join("");
      root.querySelectorAll("[data-open]").forEach(button => {{
        button.addEventListener("click", () => selectVoicemail(button.dataset.open));
      }});
      root.querySelectorAll("[data-select]").forEach(input => {{
        input.addEventListener("click", event => event.stopPropagation());
        input.addEventListener("change", () => {{
          if (input.checked) {{
            selectedBulkKeys.add(input.dataset.select);
          }} else {{
            selectedBulkKeys.delete(input.dataset.select);
          }}
          updateBulkBar();
        }});
      }});
      root.querySelectorAll("[data-play]").forEach(button => {{
        button.addEventListener("click", event => {{
          event.stopPropagation();
          togglePlayback(button.dataset.play);
        }});
      }});
      root.querySelectorAll("[data-delete]").forEach(button => {{
        button.addEventListener("click", event => {{
          event.stopPropagation();
          deleteVoicemail(button.dataset.delete, {{ confirmDelete: false, advanceAfterDelete: true, showUndo: true }});
        }});
      }});
      root.querySelectorAll("[data-forward]").forEach(button => {{
        button.addEventListener("click", event => {{
          event.stopPropagation();
          openForwardPicker(button.dataset.forward);
        }});
      }});
      root.querySelectorAll("[data-restore]").forEach(button => {{
        button.addEventListener("click", event => {{
          event.stopPropagation();
          restoreVoicemail(button.dataset.restore);
        }});
      }});
      updateBulkBar();
      refreshReviewProgress();
    }}

    function scrollRowIntoListView(row) {{
      const list = document.getElementById("items");
      if (!list || !row) return;

      const rowRect = row.getBoundingClientRect();
      const listRect = list.getBoundingClientRect();
      const desiredBottomComfort = Math.min(120, Math.max(64, list.clientHeight * 0.22));
      const maxBottomComfort = Math.max(16, list.clientHeight - row.offsetHeight - 16);
      const bottomComfort = Math.min(desiredBottomComfort, maxBottomComfort);
      const topLimit = listRect.top + 8;
      const bottomLimit = listRect.bottom - bottomComfort;

      if (rowRect.top < topLimit) {{
        list.scrollTop -= topLimit - rowRect.top;
      }} else if (rowRect.bottom > bottomLimit) {{
        list.scrollTop += rowRect.bottom - bottomLimit;
      }}
    }}

    function moveSelection(delta, autoplay = false) {{
      const items = filteredItems();
      if (!items.length) return;

      const currentIndex = items.findIndex(item => item.file_key === selectedKey);
      let nextIndex = currentIndex === -1 ? (delta > 0 ? 0 : items.length - 1) : currentIndex + delta;
      nextIndex = Math.max(0, Math.min(items.length - 1, nextIndex));

      const next = items[nextIndex];
      if (!next || next.file_key === selectedKey) return;
      selectVoicemail(next.file_key, autoplay);

      requestAnimationFrame(() => {{
        const row = document.querySelector(`[data-key="${{next.file_key}}"]`);
        if (row) {{
          scrollRowIntoListView(row);
        }}
      }});
    }}

    function firstVisibleVoicemail() {{
      return filteredItems()[0] || null;
    }}

    function selectFirstVisibleVoicemail() {{
      const first = firstVisibleVoicemail();
      if (!first) return false;
      selectVoicemail(first.file_key);
      return true;
    }}

    function optimisticallyRemoveVoicemails(keys) {{
      const keySet = new Set((keys || []).filter(Boolean));
      const snapshots = [];
      keySet.forEach(key => {{
        if (pendingDeleteKeys.has(key)) return;
        const index = voicemails.findIndex(item => item.file_key === key);
        if (index === -1) return;
        const wasPlaying = playingKey === key;
        snapshots.push({{
          key,
          item: voicemails[index],
          index,
          wasSelected: selectedKey === key,
          wasPlaying,
          wasBulkSelected: selectedBulkKeys.has(key),
          hadCommentDraft: deleteCommentDrafts.has(key),
          commentDraft: deleteCommentDrafts.get(key)
        }});
        pendingDeleteKeys.add(key);
        selectedBulkKeys.delete(key);
        deleteCommentDrafts.delete(key);
        if (wasPlaying) {{
          const audio = selectedAudio();
          if (audio) audio.pause();
          playingKey = null;
          activeSegmentEnd = null;
          resetActiveSegmentButton();
          clearTranscriptHighlight();
        }}
      }});

      if (!snapshots.length) return [];

      snapshots
        .slice()
        .sort((left, right) => right.index - left.index)
        .forEach(snapshot => voicemails.splice(snapshot.index, 1));

      if (snapshots.some(snapshot => snapshot.wasSelected)) {{
        selectedKey = null;
        document.getElementById("detail").innerHTML = '<div class="empty">Select a voicemail.</div>';
      }}

      renderExtensionMenu();
      renderList();
      return snapshots;
    }}

    function restoreOptimisticVoicemails(snapshots) {{
      if (!Array.isArray(snapshots) || !snapshots.length) return;

      const ordered = snapshots.slice().sort((left, right) => left.index - right.index);
      ordered.forEach(snapshot => {{
        pendingDeleteKeys.delete(snapshot.key);
        if (!snapshot.item || voicemails.some(item => item.file_key === snapshot.key)) return;
        const insertAt = Math.min(Math.max(snapshot.index, 0), voicemails.length);
        voicemails.splice(insertAt, 0, snapshot.item);
        if (snapshot.wasBulkSelected) {{
          selectedBulkKeys.add(snapshot.key);
        }} else {{
          selectedBulkKeys.delete(snapshot.key);
        }}
        if (snapshot.hadCommentDraft) {{
          deleteCommentDrafts.set(snapshot.key, snapshot.commentDraft);
        }} else {{
          deleteCommentDrafts.delete(snapshot.key);
        }}
      }});

      const selectedSnapshot = ordered.find(
        snapshot => snapshot.wasSelected && voicemails.some(item => item.file_key === snapshot.key)
      );
      renderExtensionMenu();
      renderList();
      if (selectedSnapshot) {{
        selectVoicemail(selectedSnapshot.key, selectedSnapshot.wasPlaying);
      }} else if (!selectedKey && !selectFirstVisibleVoicemail()) {{
        document.getElementById("detail").innerHTML = '<div class="empty">Select a voicemail.</div>';
      }}
    }}

    function selectedAudio() {{
      return document.querySelector("#detail audio");
    }}

    function toggleSelectedPlayback() {{
      if (!selectedKey) {{
        const first = firstVisibleVoicemail();
        if (first) selectVoicemail(first.file_key, true);
        return;
      }}
      togglePlayback(selectedKey);
    }}

    function seekSelectedAudio(deltaSeconds) {{
      const audio = selectedAudio();
      if (!audio) return;
      const duration = Number.isFinite(audio.duration) ? audio.duration : Infinity;
      audio.currentTime = Math.max(0, Math.min(duration, audio.currentTime + deltaSeconds));
      updateTranscriptHighlight(audio.currentTime);
    }}

    function nextVisibleKeyAfter(key) {{
      const items = filteredItems();
      if (items.length <= 1) return null;
      const index = items.findIndex(item => item.file_key === key);
      if (index === -1) return items[0].file_key;
      const nextIndex = index < items.length - 1 ? index + 1 : index - 1;
      return items[nextIndex] ? items[nextIndex].file_key : null;
    }}

    function reviewProgressHtml() {{
      if (!selectedKey) return "";
      const items = filteredItems();
      const index = items.findIndex(item => item.file_key === selectedKey);
      if (index === -1) return "";
      return `<div class="review-progress" title="Position in current view"><span class="review-progress-fraction"><strong class="review-progress-current">${{index + 1}}</strong><span class="review-progress-divider" aria-hidden="true"></span><span class="review-progress-total">${{items.length}}</span></span></div>`;
    }}

    function refreshReviewProgress() {{
      const slot = document.getElementById("reviewProgressSlot");
      if (slot) {{
        slot.innerHTML = reviewProgressHtml();
      }}
    }}

    function setPlayingKey(key) {{
      playingKey = key || null;
      renderList();
    }}

    function bindAudioState(audio, key) {{
      if (!audio) return;
      applyPlaybackRate(audio);
      audio.addEventListener("play", () => {{
        applyPlaybackRate(audio);
        updateTranscriptHighlight(audio.currentTime);
        setPlayingKey(key);
      }});
      audio.addEventListener("timeupdate", () => {{
        updateTranscriptHighlight(audio.currentTime);
        enforceSegmentEnd(audio);
      }});
      audio.addEventListener("seeked", () => updateTranscriptHighlight(audio.currentTime));
      audio.addEventListener("ratechange", () => updateTranscriptHighlight(audio.currentTime));
      audio.addEventListener("pause", () => {{
        if (activeSegmentEnd !== null && audio.currentTime < activeSegmentEnd - 0.05) {{
          activeSegmentEnd = null;
        }}
        resetActiveSegmentButton();
        if (playingKey === key) setPlayingKey(null);
      }});
      audio.addEventListener("ended", () => {{
        activeSegmentEnd = null;
        resetActiveSegmentButton();
        clearTranscriptHighlight();
        if (playingKey === key) setPlayingKey(null);
      }});
    }}

    function togglePlayback(key) {{
      if (selectedKey !== key) {{
        selectVoicemail(key, true);
        return;
      }}

      const audio = document.querySelector("#detail audio");
      if (!audio) {{
        selectVoicemail(key, true);
        return;
      }}

      if (audio.paused) {{
        activeSegmentEnd = null;
        resetActiveSegmentButton();
        audio.play().catch(() => {{}});
      }} else {{
        audio.pause();
      }}
    }}

    function playFromTranscriptWord(seconds) {{
      const audio = document.querySelector("#detail audio");
      if (!audio) return;
      const start = Math.max(0, Number(seconds) || 0);
      activeSegmentEnd = null;
      resetActiveSegmentButton();
      audio.currentTime = start;
      audio.play().catch(() => {{}});
    }}

    function setSegmentButtonState(button, isPlaying) {{
      if (!button) return;
      const label = button.dataset.segmentLabel || "number";
      button.innerHTML = isPlaying ? pauseIcon() : playIcon();
      button.title = `${{isPlaying ? "Pause" : "Play"}} ${{label}} number`;
      button.setAttribute("aria-label", `${{isPlaying ? "Pause" : "Play"}} ${{label}} number`);
      button.classList.toggle("playing", isPlaying);
    }}

    function resetActiveSegmentButton() {{
      if (activeSegmentButton) {{
        setSegmentButtonState(activeSegmentButton, false);
        activeSegmentButton = null;
      }}
    }}

    function playNumberSegment(startSeconds, endSeconds, button) {{
      const audio = document.querySelector("#detail audio");
      if (!audio) return;
      const start = Math.max(0, Number(startSeconds) || 0);
      const end = Number(endSeconds);
      if (!Number.isFinite(end) || end <= start) return;

      if (activeSegmentButton === button && !audio.paused) {{
        activeSegmentEnd = null;
        resetActiveSegmentButton();
        audio.pause();
        return;
      }}

      resetActiveSegmentButton();
      activeSegmentEnd = end;
      activeSegmentButton = button || null;
      setSegmentButtonState(activeSegmentButton, true);
      audio.currentTime = start;
      audio.play().catch(() => {{
        activeSegmentEnd = null;
        resetActiveSegmentButton();
      }});
    }}

    async function copyText(value, button) {{
      const textToCopy = String(value || "");
      try {{
        if (navigator.clipboard && window.isSecureContext) {{
          await navigator.clipboard.writeText(textToCopy);
        }} else {{
          const textarea = document.createElement("textarea");
          textarea.value = textToCopy;
          textarea.setAttribute("readonly", "");
          textarea.style.position = "fixed";
          textarea.style.left = "-9999px";
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand("copy");
          textarea.remove();
        }}
        showCopyFeedback(button);
      }} catch (_error) {{
        alert("Copy failed.");
      }}
    }}

    function showCopyFeedback(button) {{
      if (!button) return;
      const originalHtml = button.innerHTML;
      const originalTitle = button.title;
      const originalLabel = button.getAttribute("aria-label");
      button.innerHTML = checkIcon();
      button.title = "Copied";
      button.setAttribute("aria-label", "Copied");
      button.classList.add("copied");
      window.setTimeout(() => {{
        button.innerHTML = originalHtml;
        button.title = originalTitle;
        if (originalLabel) {{
          button.setAttribute("aria-label", originalLabel);
        }}
        button.classList.remove("copied");
      }}, 1000);
    }}

    function clearTranscriptHighlight() {{
      if (highlightedTranscriptWord) {{
        highlightedTranscriptWord.classList.remove("current");
        highlightedTranscriptWord = null;
      }}
    }}

    function updateTranscriptHighlight(seconds) {{
      const words = Array.from(document.querySelectorAll("#detail .transcript-word[data-seek]"));
      if (!words.length) {{
        clearTranscriptHighlight();
        return;
      }}

      let activeWord = null;
      for (let index = 0; index < words.length; index += 1) {{
        const word = words[index];
        const start = Number(word.dataset.seek);
        const explicitEnd = Number(word.dataset.end);
        const nextStart = index + 1 < words.length ? Number(words[index + 1].dataset.seek) : NaN;
        const end = Number.isFinite(explicitEnd)
          ? explicitEnd + 0.08
          : (Number.isFinite(nextStart) ? nextStart : start + 0.65);
        if (Number.isFinite(start) && seconds >= start && seconds <= end) {{
          activeWord = word;
          break;
        }}
      }}

      if (activeWord === highlightedTranscriptWord) return;
      clearTranscriptHighlight();
      if (activeWord) {{
        activeWord.classList.add("current");
        highlightedTranscriptWord = activeWord;
      }}
    }}

    function enforceSegmentEnd(audio) {{
      if (activeSegmentEnd === null) return;
      if (audio.currentTime >= activeSegmentEnd) {{
        activeSegmentEnd = null;
        resetActiveSegmentButton();
        audio.pause();
      }}
    }}

    function detailHasFocus() {{
      const detail = document.getElementById("detail");
      return Boolean(detail && detail.contains(document.activeElement));
    }}

    function selectedAudioIsActive() {{
      const audio = document.querySelector("#detail audio");
      return Boolean(audio && (!audio.paused || audio.currentTime > 0));
    }}

    function deleteCommentPanelHtml(item) {{
      if (item.deleted_utc && !item.deleted_comment) return "";
      const draft = deleteCommentDrafts.has(item.file_key)
        ? deleteCommentDrafts.get(item.file_key)
        : (item.deleted_comment || "");
      const readOnlyAttrs = item.deleted_utc ? "readonly" : "";
      const saveControls = item.deleted_utc
        ? ""
        : `
            <div class="delete-comment-side">
              <span id="deleteCommentCount" class="delete-comment-count">${{draft.length}}/{DELETE_COMMENT_MAX_CHARS}</span>
              <button id="deleteCommentSave" class="delete-comment-save" type="button">Save</button>
            </div>
          `;
      return `
        <div class="delete-comment-panel" id="deleteCommentPanel">
          <label for="deleteCommentInline">Comment</label>
          <textarea id="deleteCommentInline" maxlength="{DELETE_COMMENT_MAX_CHARS}" rows="4" ${{readOnlyAttrs}}>${{escapeHtml(draft)}}</textarea>
          <div class="delete-comment-meta">
            <span id="deleteCommentError" class="delete-comment-error" aria-live="polite"></span>
            ${{saveControls}}
          </div>
        </div>
      `;
    }}

    function bindDeleteCommentInput(item) {{
      const input = document.getElementById("deleteCommentInline");
      const count = document.getElementById("deleteCommentCount");
      const error = document.getElementById("deleteCommentError");
      const saveButton = document.getElementById("deleteCommentSave");
      if (!input || !count || !error || !saveButton || item.deleted_utc) return;
      const update = () => {{
        deleteCommentDrafts.set(item.file_key, input.value);
        count.textContent = `${{input.value.length}}/{DELETE_COMMENT_MAX_CHARS}`;
        saveButton.classList.remove("saved");
        saveButton.textContent = "Save";
        if (input.value.trim()) error.textContent = "";
      }};
      input.addEventListener("input", update);
      saveButton.addEventListener("click", () => saveDeleteComment(item.file_key));
      update();
    }}

    function selectVoicemail(key, autoplay = false) {{
      selectedKey = key;
      activeSegmentEnd = null;
      resetActiveSegmentButton();
      clearTranscriptHighlight();
      renderList();
      const item = voicemails.find(entry => entry.file_key === key);
      if (!item) return;
      const entities = item.entities || {{}};
      const audioHtml = item.has_audio
        ? `<div class="audio-tools"><audio controls preload="metadata" src="${{basePath}}/api/voicemails/${{item.file_key}}/audio"></audio>${{playbackSpeedControlsHtml()}}</div>`
        : '<div class="empty">Audio file is not available on disk.</div>';
      const deletedBannerHtml = item.deleted_utc
        ? `<div class="deleted-banner">Deleted ${{escapeHtml(item.deleted_display || item.deleted_utc)}} by ${{escapeHtml(item.deleted_by || "unknown")}}</div>`
        : "";
      const deleteButtonHtml = item.deleted_utc
        ? `<button class="icon-btn large restore" id="restoreBtn" type="button" title="Restore voicemail" aria-label="Restore voicemail">${{restoreIcon()}}</button>`
        : `<button class="icon-btn large danger" id="deleteBtn" type="button" title="Delete voicemail" aria-label="Delete voicemail">${{trashIcon()}}</button>`;
      const forwardButtonHtml = item.deleted_utc || !forwardingEnabled
        ? ""
        : `<button class="icon-btn forward large" id="forwardBtn" type="button" title="Forward voicemail" aria-label="Forward voicemail">${{forwardIcon()}}</button>`;
      document.getElementById("detail").innerHTML = `
        <div class="detail-header">
          <div class="detail-title">
            <h2><span class="caller-title-text">${{escapeHtml(item.callerid || "Unknown")}}</span></h2>
            <div class="caller-copy-row">${{callerIdCopyButtonHtml(item)}}</div>
            <div>${{escapeHtml(item.display_date)}} | Duration ${{escapeHtml(item.duration_display || "Unknown")}}</div>
          </div>
          <div class="detail-actions">
            <div id="reviewProgressSlot">${{reviewProgressHtml()}}</div>
            <div class="mailbox-badge"><span>Mailbox</span>${{escapeHtml(item.mailbox || item.extension)}}</div>
            ${{forwardButtonHtml}}
            ${{deleteButtonHtml}}
          </div>
        </div>
        ${{deletedBannerHtml}}
        ${{forwardingNoticeHtml(item)}}
        ${{item.deleted_utc ? deleteCommentPanelHtml(item) : ""}}
        ${{audioHtml}}
        <div class="fields">
          <div class="field"><span>Name</span>${{plainFieldHtml(entities.name, "name")}}</div>
          <div class="field"><span>DOB</span>${{dobFieldHtml(item, entities.dob)}}${{dobConfidenceHtml(item, entities.dob)}}</div>
          <div class="field"><span>Callback</span>${{numberFieldHtml(item, entities.callback_number, "callback")}}${{phoneConfidenceHtml(item, "callback_number", entities.callback_number)}}<small class="field-extra">Callback Matches Caller ID: ${{callbackMatchHtml(entities.callback_matches_caller_id)}}</small></div>
          <div class="field"><span>Fax</span>${{numberFieldHtml(item, entities.fax_number, "fax")}}${{phoneConfidenceHtml(item, "fax_number", entities.fax_number)}}</div>
        </div>
        <h3>Transcript</h3>
        <div class="transcript-box"><button class="copy-btn transcript-copy" id="copyTranscriptBtn" type="button" title="Copy transcript" aria-label="Copy transcript">${{copyIcon()}}</button>${{timedTranscriptHtml(item)}}</div>
        <div class="transcript-disclaimer">This transcript was generated by AI.<br>Accuracy may vary based on audio quality.<br>Please verify all clinical details.</div>
        ${{item.deleted_utc ? "" : deleteCommentPanelHtml(item)}}
      `;
      const deleteButton = document.getElementById("deleteBtn");
      if (deleteButton) {{
        deleteButton.addEventListener("click", () => deleteVoicemail(item.file_key, {{ confirmDelete: false, advanceAfterDelete: true, showUndo: true }}));
      }}
      const forwardButton = document.getElementById("forwardBtn");
      if (forwardButton) {{
        forwardButton.addEventListener("click", () => openForwardPicker(item.file_key));
      }}
      const restoreButton = document.getElementById("restoreBtn");
      if (restoreButton) {{
        restoreButton.addEventListener("click", () => restoreVoicemail(item.file_key));
      }}
      document.querySelectorAll("#detail [data-seek]").forEach(button => {{
        button.addEventListener("click", () => playFromTranscriptWord(button.dataset.seek));
      }});
      document.querySelectorAll("#detail [data-number-segment]").forEach(button => {{
        button.addEventListener("click", () => playNumberSegment(button.dataset.segmentStart, button.dataset.segmentEnd, button));
      }});
      document.querySelectorAll("#detail [data-copy-value]").forEach(button => {{
        button.addEventListener("click", () => copyText(button.dataset.copyValue || "", button));
      }});
      document.querySelectorAll("#detail [data-rate]").forEach(button => {{
        button.addEventListener("click", () => setPlaybackRate(button.dataset.rate));
      }});
      const copyTranscriptButton = document.getElementById("copyTranscriptBtn");
      if (copyTranscriptButton) {{
        copyTranscriptButton.addEventListener("click", () => copyText(item.transcript || "", copyTranscriptButton));
      }}
      bindDeleteCommentInput(item);
      const audio = document.querySelector("#detail audio");
      bindAudioState(audio, item.file_key);
      syncPlaybackRateButtons();
      if (autoplay) {{
        if (audio) {{
          audio.play().catch(() => {{}});
        }}
      }}
    }}

    async function loadVoicemails(options = {{}}) {{
      if (loadingVoicemails) return;
      loadingVoicemails = true;
      const silent = Boolean(options.silent);
      const focusSnapshot = silent ? captureFocus() : null;
      const selectedBefore = selectedKey;
      const mayRefreshDetail = Boolean(options.refreshSelectedDetail) || (!detailHasFocus() && !selectedAudioIsActive());

      try {{
        renderFolderTabs();
        const folderParam = currentFolder === "deleted" ? "?folder=deleted" : "";
        let response = await fetch(`${{basePath}}/api/voicemails${{folderParam}}`, {{ credentials: "same-origin" }});
        if (response.status === 401) {{
          location.href = `${{basePath}}/login`;
          return;
        }}
        if (response.status === 403) {{
          currentFolder = "active";
          renderFolderTabs();
          response = await fetch(`${{basePath}}/api/voicemails`, {{ credentials: "same-origin" }});
          if (response.status === 401) {{
            location.href = `${{basePath}}/login`;
            return;
          }}
        }}
        if (!response.ok) {{
          return;
        }}
        const loadedVoicemails = await response.json();
        voicemails = loadedVoicemails.filter(item => !pendingDeleteKeys.has(item.file_key));
        pruneBulkSelection();
        if (selectedKey && !voicemails.some(item => item.file_key === selectedKey)) {{
          selectedKey = null;
          document.getElementById("detail").innerHTML = `<div class="empty">Select a ${{currentFolder === "deleted" ? "deleted " : ""}}voicemail.</div>`;
        }}
        renderFolderTabs();
        renderExtensionMenu();
        renderList();
        if (!selectedKey && !silent) {{
          selectFirstVisibleVoicemail();
        }} else if (selectedKey && selectedKey === selectedBefore && mayRefreshDetail) {{
          selectVoicemail(selectedKey);
        }}
      }} finally {{
        loadingVoicemails = false;
        if (silent) restoreFocus(focusSnapshot);
      }}
    }}

    function pollVoicemails() {{
      if (document.hidden) return;
      loadVoicemails({{ silent: true }});
    }}

    async function loadExtensions() {{
      const response = await fetch(`${{basePath}}/api/extensions`, {{ credentials: "same-origin" }});
      if (response.ok) {{
        extensions = await response.json();
        renderExtensionMenu();
      }}
    }}

    async function loadDirectory() {{
      if (!forwardingEnabled) {{
        directoryEntries = [];
        return;
      }}
      const response = await fetch(`${{basePath}}/api/directory`, {{ credentials: "same-origin" }});
      if (response.ok) {{
        directoryEntries = await response.json();
        renderDirectoryMenu();
      }}
    }}

    function hideUndoToast() {{
      if (undoToastTimer) {{
        window.clearTimeout(undoToastTimer);
        undoToastTimer = null;
      }}
      undoToastKey = null;
      const host = document.getElementById("toastHost");
      if (host) host.innerHTML = "";
    }}

    function positionUndoToastAboveControls() {{
      const host = document.getElementById("toastHost");
      if (!host) return;
      let bottom = 96;
      const saveButton = document.getElementById("deleteCommentSave");
      if (saveButton) {{
        const rect = saveButton.getBoundingClientRect();
        const visible = rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < window.innerHeight;
        if (visible) {{
          bottom = Math.max(bottom, Math.ceil(window.innerHeight - rect.top + 16));
        }}
      }}
      host.style.setProperty("--toast-bottom", `${{bottom}}px`);
    }}

    function showUndoToast(key) {{
      const host = document.getElementById("toastHost");
      if (!host || !key) return;
      hideUndoToast();
      undoToastKey = key;
      host.innerHTML = '<div class="toast" role="status"><span>Voicemail deleted.</span><button id="undoDeleteBtn" type="button">Undo</button></div>';
      positionUndoToastAboveControls();
      const button = document.getElementById("undoDeleteBtn");
      if (button) {{
        button.addEventListener("click", () => {{
          const restoreKey = undoToastKey;
          if (!restoreKey) return;
          if (pendingDeleteKeys.has(restoreKey)) {{
            queuedUndoKeys.add(restoreKey);
            if (undoToastTimer) {{
              window.clearTimeout(undoToastTimer);
              undoToastTimer = null;
            }}
            button.disabled = true;
            button.textContent = "Restoring";
            return;
          }}
          hideUndoToast();
          restoreDeletedVoicemail(restoreKey, {{ confirmRestore: false, requireDeletedFolder: false }});
        }});
      }}
      undoToastTimer = window.setTimeout(hideUndoToast, 8000);
    }}

    function showStatusToast(message) {{
      const host = document.getElementById("toastHost");
      if (!host) return;
      hideUndoToast();
      host.innerHTML = `<div class="toast" role="status"><span>${{escapeHtml(message)}}</span></div>`;
      undoToastTimer = window.setTimeout(hideUndoToast, 8000);
    }}

    function voicemailByKey(key) {{
      return voicemails.find(item => item.file_key === key) || null;
    }}

    function showDeleteCommentRequired(key, message = "A comment is required to delete this message.") {{
      const item = voicemailByKey(key);
      if (!item) return;
      if (selectedKey !== key) {{
        selectVoicemail(key, false);
      }}
      window.requestAnimationFrame(() => {{
        const input = document.getElementById("deleteCommentInline");
        const error = document.getElementById("deleteCommentError");
        if (error) error.textContent = message;
        if (input) {{
          input.focus();
          input.scrollIntoView({{ behavior: "smooth", block: "center" }});
        }}
      }});
    }}

    function deleteCommentForKey(key, forceRequired = false) {{
      const item = voicemailByKey(key);
      if (!item) {{
        return {{ ok: true, comment: null }};
      }}
      const inlineInput = selectedKey === key ? document.getElementById("deleteCommentInline") : null;
      const rawComment = inlineInput ? inlineInput.value : (deleteCommentDrafts.get(key) || "");
      const comment = rawComment.trim();
      const required = forceRequired || Boolean(item.delete_comment_required);
      if (required && !comment) {{
        showDeleteCommentRequired(key);
        return {{ ok: false, comment: null }};
      }}
      if (comment.length > {DELETE_COMMENT_MAX_CHARS}) {{
        showDeleteCommentRequired(key, "Morgan Example must be {DELETE_COMMENT_MAX_CHARS} characters or fewer.");
        return {{ ok: false, comment: null }};
      }}
      deleteCommentDrafts.set(key, rawComment);
      return {{ ok: true, comment: comment || null }};
    }}

    async function saveDeleteComment(key) {{
      const input = selectedKey === key ? document.getElementById("deleteCommentInline") : null;
      const error = document.getElementById("deleteCommentError");
      const saveButton = document.getElementById("deleteCommentSave");
      const rawComment = input ? input.value : (deleteCommentDrafts.get(key) || "");
      const comment = rawComment.trim();
      if (comment.length > {DELETE_COMMENT_MAX_CHARS}) {{
        if (error) error.textContent = "Morgan Example must be {DELETE_COMMENT_MAX_CHARS} characters or fewer.";
        if (input) input.focus();
        return;
      }}

      if (saveButton) {{
        saveButton.disabled = true;
        saveButton.textContent = "Saving";
      }}

      const response = await fetch(`${{basePath}}/api/voicemails/${{key}}/comment`, {{
        method: "POST",
        credentials: "same-origin",
        headers: {{
          "Content-Type": "application/json",
          "X-CSRF-Token": csrfToken
        }},
        body: JSON.stringify({{ comment }})
      }});

      let payload = {{}};
      try {{
        payload = await response.json();
      }} catch (_error) {{}}

      if (!response.ok) {{
        if (error) error.textContent = payload.detail || "Comment save failed.";
        if (saveButton) {{
          saveButton.disabled = false;
          saveButton.textContent = "Save";
        }}
        return;
      }}

      const item = voicemailByKey(key);
      if (item) item.deleted_comment = payload.comment || null;
      deleteCommentDrafts.set(key, rawComment);
      if (error) error.textContent = "";
      if (saveButton) {{
        saveButton.disabled = false;
        saveButton.textContent = "Saved";
        saveButton.classList.add("saved");
      }}
    }}

    async function forwardVoicemail(key, targetExtension) {{
      if (!key || !targetExtension) return;
      const item = voicemailByKey(key);
      if (!item || item.deleted_utc) return;
      const targetLabel = forwardTargetLabel(targetExtension);
      if (!confirm(`Forward this voicemail to ${{targetLabel}}? The original will stay in this mailbox.`)) return;
      let response;
      try {{
        response = await fetch(`${{basePath}}/api/voicemails/${{key}}/forward`, {{
          method: "POST",
          credentials: "same-origin",
          headers: {{
            "content-type": "application/json",
            "x-csrf-token": csrfToken
          }},
          body: JSON.stringify({{ target_extension: targetExtension }})
        }});
      }} catch (_error) {{
        alert("Forward failed.");
        return;
      }}

      let payload = {{}};
      try {{
        payload = await response.json();
      }} catch (_error) {{
        payload = {{}};
      }}

      if (!response.ok || !payload.ok) {{
        alert(payload.detail || "Forward failed.");
        return;
      }}

      const emailSuffix = payload.email_sent
        ? " Email sent."
        : (payload.email_error ? ` Email warning: ${{payload.email_error}}` : "");
      showStatusToast(`Forwarded to Ext ${{payload.target_extension}}.${{emailSuffix}}`);
      await loadVoicemails({{ silent: true, refreshSelectedDetail: true }});
      if (selectedKey) {{
        selectVoicemail(selectedKey);
      }}
    }}

    async function deleteVoicemail(key, options = {{}}) {{
      if (currentFolder === "deleted") return;
      const confirmDelete = options.confirmDelete !== false;
      const advanceAfterDelete = Boolean(options.advanceAfterDelete);
      const showUndo = options.showUndo !== false;
      const nextKey = advanceAfterDelete ? nextVisibleKeyAfter(key) : null;
      if (confirmDelete && !confirm("Delete this voicemail from the mailbox?")) return;
      const commentResult = deleteCommentForKey(key);
      if (!commentResult.ok) return;
      const comment = commentResult.comment;
      const optimisticSnapshots = optimisticallyRemoveVoicemails([key]);
      if (!optimisticSnapshots.length) return;
      if (nextKey && voicemails.some(item => item.file_key === nextKey)) {{
        selectVoicemail(nextKey, false);
      }} else if (!selectedKey && !selectFirstVisibleVoicemail()) {{
        document.getElementById("detail").innerHTML = '<div class="empty">Select a voicemail.</div>';
      }}
      if (showUndo) showUndoToast(key);

      let response;
      try {{
        response = await fetch(`${{basePath}}/api/voicemails/${{key}}/delete`, {{
          method: "POST",
          credentials: "same-origin",
          headers: {{
            "Content-Type": "application/json",
            "X-CSRF-Token": csrfToken
          }},
          body: JSON.stringify({{ comment }})
        }});
      }} catch (_error) {{
        restoreOptimisticVoicemails(optimisticSnapshots);
        queuedUndoKeys.delete(key);
        if (undoToastKey === key) hideUndoToast();
        alert("Delete failed.");
        return;
      }}
      if (!response.ok) {{
        let payload = {{}};
        try {{
          payload = await response.json();
        }} catch (_error) {{}}
        restoreOptimisticVoicemails(optimisticSnapshots);
        queuedUndoKeys.delete(key);
        if (undoToastKey === key) hideUndoToast();
        alert(payload.detail || "Delete failed.");
        return;
      }}
      const undoQueued = queuedUndoKeys.has(key);
      optimisticSnapshots.forEach(snapshot => pendingDeleteKeys.delete(snapshot.key));
      queuedUndoKeys.delete(key);
      if (undoQueued) {{
        hideUndoToast();
        restoreDeletedVoicemail(key, {{ confirmRestore: false, requireDeletedFolder: false }});
      }} else {{
        void loadVoicemails({{ silent: true }});
      }}
    }}

    async function deleteSelectedVoicemails() {{
      if (currentFolder === "deleted") return;
      const keys = Array.from(selectedBulkKeys);
      if (!keys.length) return;
      const label = keys.length === 1 ? "voicemail" : "voicemails";
      if (!confirm(`Delete ${{keys.length}} selected ${{label}} from the mailbox?`)) return;
      const requiresComment = keys.some(key => {{
        const item = voicemailByKey(key);
        return item && item.delete_comment_required;
      }});
      const firstRequiredKey = keys.find(key => {{
        const item = voicemailByKey(key);
        return item && item.delete_comment_required;
      }});
      const commentKey = selectedKey && selectedBulkKeys.has(selectedKey) ? selectedKey : (firstRequiredKey || keys[0]);
      const commentResult = deleteCommentForKey(commentKey, requiresComment);
      if (!commentResult.ok) return;
      const comment = commentResult.comment;

      const response = await fetch(`${{basePath}}/api/voicemails/bulk-delete`, {{
        method: "POST",
        credentials: "same-origin",
        headers: {{
          "Content-Type": "application/json",
          "X-CSRF-Token": csrfToken
        }},
        body: JSON.stringify({{ file_keys: keys, comment }})
      }});

      let payload = {{}};
      try {{
        payload = await response.json();
      }} catch (_error) {{}}

      if (!response.ok && !payload.deleted_count) {{
        alert(payload.detail || "Bulk delete failed.");
        return;
      }}

      const deletedKeys = new Set(payload.deleted || []);
      deletedKeys.forEach(key => {{
        selectedBulkKeys.delete(key);
        deleteCommentDrafts.delete(key);
      }});
      if (selectedKey && deletedKeys.has(selectedKey)) {{
        selectedKey = null;
        document.getElementById("detail").innerHTML = '<div class="empty">Select a voicemail.</div>';
      }}

      await loadVoicemails();

      const errorCount = Array.isArray(payload.errors) ? payload.errors.length : 0;
      if (errorCount) {{
        alert(`Deleted ${{payload.deleted_count || 0}} selected voicemail${{payload.deleted_count === 1 ? "" : "s"}}. ${{errorCount}} could not be deleted.`);
      }}
    }}

    async function restoreDeletedVoicemail(key, options = {{}}) {{
      const requireDeletedFolder = options.requireDeletedFolder !== false;
      const confirmRestore = options.confirmRestore !== false;
      if (requireDeletedFolder && currentFolder !== "deleted") return;
      if (confirmRestore && !confirm("Restore this voicemail to the mailbox INBOX?")) return;
      const restoreSnapshots = currentFolder === "deleted" ? optimisticallyRemoveVoicemails([key]) : [];
      let response;
      try {{
        response = await fetch(`${{basePath}}/api/voicemails/${{key}}/restore`, {{
          method: "POST",
          credentials: "same-origin",
          headers: {{ "X-CSRF-Token": csrfToken }}
        }});
      }} catch (_error) {{
        restoreOptimisticVoicemails(restoreSnapshots);
        alert("Restore failed.");
        return;
      }}
      if (!response.ok) {{
        let message = "Restore failed.";
        try {{
          const payload = await response.json();
          if (payload.detail) message = payload.detail;
        }} catch (_error) {{}}
        restoreOptimisticVoicemails(restoreSnapshots);
        alert(message);
        return;
      }}
      const payload = await response.json();
      restoreSnapshots.forEach(snapshot => pendingDeleteKeys.delete(snapshot.key));
      currentFolder = "active";
      selectedKey = payload.file_key || null;
      document.getElementById("detail").innerHTML = '<div class="empty">Restored to Inbox.</div>';
      await loadVoicemails();
      if (selectedKey) {{
        selectVoicemail(selectedKey);
      }}
    }}

    async function restoreVoicemail(key) {{
      return restoreDeletedVoicemail(key, {{ confirmRestore: true, requireDeletedFolder: true }});
    }}

    async function logout() {{
      await fetch(`${{basePath}}/logout`, {{
        method: "POST",
        credentials: "same-origin",
        headers: {{ "X-CSRF-Token": csrfToken }}
      }});
      location.href = `${{basePath}}/login`;
    }}

    document.getElementById("search").addEventListener("input", renderList);
    document.getElementById("selectAllBox").addEventListener("change", event => selectAllVisible(event.target.checked));
    document.getElementById("bulkDeleteBtn").addEventListener("click", deleteSelectedVoicemails);
    document.getElementById("clearSelectionBtn").addEventListener("click", clearBulkSelection);
    document.getElementById("menuBtn").addEventListener("click", () => toggleExtensionMenu());
    document.getElementById("directoryBtn").addEventListener("click", () => toggleDirectoryMenu());
    document.getElementById("directorySearch").addEventListener("input", event => {{
      directorySearchTerm = event.target.value || "";
      renderDirectoryMenu();
    }});
    document.getElementById("directoryItems").addEventListener("click", event => {{
      if (!isAdmin) return;
      const button = event.target.closest("[data-directory-extension]");
      if (!button) return;
      selectExtensionFilter(button.dataset.directoryExtension || "");
    }});
    document.getElementById("forwardSearch").addEventListener("input", event => {{
      forwardSearchTerm = event.target.value || "";
      renderForwardPicker();
    }});
    document.getElementById("forwardCancelBtn").addEventListener("click", closeForwardPicker);
    document.getElementById("forwardCloseBtn").addEventListener("click", closeForwardPicker);
    document.getElementById("forwardPicker").addEventListener("click", event => {{
      if (event.target && event.target.id === "forwardPicker") {{
        closeForwardPicker();
      }}
    }});
    document.getElementById("themeBtn").addEventListener("click", toggleTheme);
    document.getElementById("refreshBtn").addEventListener("click", () => loadVoicemails({{ refreshSelectedDetail: true }}));
    document.getElementById("logoutBtn").addEventListener("click", logout);
    document.addEventListener("click", event => {{
      const menu = document.getElementById("extensionMenu");
      const button = document.getElementById("menuBtn");
      const directoryMenu = document.getElementById("directoryMenu");
      const directoryButton = document.getElementById("directoryBtn");
      if (!menu.hidden && !menu.contains(event.target) && !button.contains(event.target)) {{
        toggleExtensionMenu(false);
      }}
      if (!directoryMenu.hidden && !directoryMenu.contains(event.target) && !directoryButton.contains(event.target)) {{
        toggleDirectoryMenu(false);
      }}
    }});
    document.addEventListener("keydown", event => {{
      if (event.key === "Escape") {{
        closeForwardPicker();
        return;
      }}
      const tagName = event.target && event.target.tagName ? event.target.tagName.toLowerCase() : "";
      if (tagName === "input" || tagName === "textarea" || event.target.isContentEditable) {{
        return;
      }}
      if (event.key === "ArrowDown") {{
        event.preventDefault();
        moveSelection(1);
      }} else if (event.key === "ArrowUp") {{
        event.preventDefault();
        moveSelection(-1);
      }} else if (event.key === " " || event.code === "Space") {{
        event.preventDefault();
        toggleSelectedPlayback();
      }} else if (event.key.toLowerCase() === "j") {{
        event.preventDefault();
        seekSelectedAudio(-5);
      }} else if (event.key.toLowerCase() === "l") {{
        event.preventDefault();
        seekSelectedAudio(5);
      }} else if (event.key.toLowerCase() === "n") {{
        event.preventDefault();
        moveSelection(1);
      }} else if (event.key.toLowerCase() === "d") {{
        event.preventDefault();
        if (selectedKey && currentFolder !== "deleted") {{
          deleteVoicemail(selectedKey, {{ confirmDelete: false, advanceAfterDelete: true, showUndo: true }});
        }}
      }}
    }});
    initTheme();
    initPlaybackRate();
    loadExtensions();
    loadDirectory();
    loadVoicemails();
    setInterval(pollVoicemails, refreshIntervalMs);
  </script>
</body>
</html>"""
    )


app = FastAPI(
    title=SETTINGS.brand_name,
    version=APP_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
INTEGRATION_API_CONFIG = IntegrationAPIConfig.from_env()
mount_api_v1(
    app,
    INTEGRATION_API_CONFIG,
    SQLiteIntegrationBackend(Path(SETTINGS.state_db)),
)

@app.on_event("startup")
def on_startup() -> None:
    validate_startup()
    if INTEGRATION_API_CONFIG.enabled:
        load_service_principals(INTEGRATION_API_CONFIG.token_file)
    logger.info(
        "Voicemail portal ready state_db=%s watch_dir=%s trash_dir=%s",
        SETTINGS.state_db,
        SETTINGS.watch_dir,
        SETTINGS.trash_dir,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "watch_dir": SETTINGS.watch_dir,
    }


@app.get("/brand/favicon")
def brand_favicon() -> Response:
    if not FAVICON_PATH.is_file():
        raise HTTPException(status_code=404, detail="Favicon file was not found")
    return FileResponse(str(FAVICON_PATH), media_type="image/svg+xml")


@app.get("/brand/logo")
def brand_logo() -> Response:
    if not SETTINGS.logo_file:
        raise HTTPException(status_code=404, detail="Logo file is not configured")

    path = Path(SETTINGS.logo_file)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Logo file was not found")

    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(str(path), media_type=media_type)


@app.get("/brand/logo/light")
def brand_light_logo() -> Response:
    if not SETTINGS.light_logo_file:
        raise HTTPException(status_code=404, detail="Light logo file is not configured")

    path = Path(SETTINGS.light_logo_file)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Light logo file was not found")

    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(str(path), media_type=media_type)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    try:
        current_user(request)
    except HTTPException:
        return RedirectResponse(app_path("/login"), status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(app_path("/voicemails"), status_code=status.HTTP_303_SEE_OTHER)


@app.get("/login", response_class=HTMLResponse)
def get_login(request: Request) -> Response:
    try:
        current_user(request)
        return RedirectResponse(app_path("/voicemails"), status_code=status.HTTP_303_SEE_OTHER)
    except HTTPException:
        return login_page()


@app.post("/login")
async def post_login(request: Request) -> Response:
    body = (await request.body()).decode("utf-8", errors="replace")
    form = parse_qs(body, keep_blank_values=True)
    username = form.get("username", [""])[0].strip()
    password = form.get("password", [""])[0]
    rate_key = login_rate_limit_key(request, username)

    if login_rate_limited(rate_key):
        return login_page("Too many login attempts. Please wait and try again.")

    try:
        user = get_portal_user(username)
    except RuntimeError as exc:
        logger.error("Could not load portal users: %s", exc)
        return login_page("Portal users are not configured.")

    if user is None or not verify_password(password, user.password_hash):
        record_login_failure(rate_key)
        return login_page("Invalid username or password.")

    clear_login_failures(rate_key)
    csrf_token = secrets.token_urlsafe(24)
    session_payload = {
        "sub": user.username,
        "exp": int(time.time()) + SETTINGS.session_ttl_seconds,
        "csrf": csrf_token,
    }
    response = RedirectResponse(app_path("/voicemails"), status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        SESSION_COOKIE,
        sign_session(session_payload),
        max_age=SETTINGS.session_ttl_seconds,
        httponly=True,
        secure=SETTINGS.cookie_secure,
        samesite="lax",
        path=cookie_path(),
    )
    return response


@app.post("/logout")
def logout(request: Request) -> Response:
    require_csrf(request)
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path=cookie_path())
    return response


@app.get("/voicemails", response_class=HTMLResponse)
def voicemails_page(request: Request) -> Response:
    try:
        user = current_user(request)
        csrf_token = current_csrf(request)
    except HTTPException:
        return RedirectResponse(app_path("/login"), status_code=status.HTTP_303_SEE_OTHER)
    return portal_page(user, csrf_token)


@app.get("/api/voicemails")
def list_voicemails(request: Request) -> JSONResponse:
    user = current_user(request)
    folder = request.query_params.get("folder", "active").strip().lower()
    if folder not in {"active", "deleted"}:
        folder = "active"
    return JSONResponse(get_store().list_voicemails(user, folder=folder))


@app.get("/api/extensions")
def list_extensions(request: Request) -> JSONResponse:
    user = current_user(request)
    mailboxes = discover_mailboxes()
    excluded_set = set(user.inaccessible_extensions())
    if excluded_set:
        mailboxes = [
            mailbox
            for mailbox in mailboxes
            if str(mailbox.get("extension", "")) not in excluded_set
        ]
    if not user.is_admin and user.extension != "*":
        allowed_extensions = user.accessible_extensions()
        if excluded_set:
            allowed_extensions = tuple(
                extension
                for extension in allowed_extensions
                if extension not in excluded_set
            )
        allowed_set = set(allowed_extensions)
        mailboxes = [
            mailbox
            for mailbox in mailboxes
            if mailbox["extension"] in allowed_set
        ]
        found = {str(mailbox["extension"]) for mailbox in mailboxes}
        for extension in allowed_extensions:
            if extension not in found:
                display_name = user.display_name if extension == user.extension else ""
                mailboxes.append({"extension": extension, "display_name": display_name})
        mailboxes.sort(key=lambda mailbox: 0 if str(mailbox["extension"]) == user.extension else 1)
    return JSONResponse(mailboxes)


@app.get("/api/directory")
def list_directory(request: Request) -> JSONResponse:
    if not SETTINGS.forward_enabled:
        raise HTTPException(status_code=403, detail="Voicemail forwarding is disabled.")
    current_user(request)
    return JSONResponse(discover_mailboxes())


@app.get("/api/voicemails/{file_key}/audio")
def get_audio(file_key: str, request: Request) -> StreamingResponse:
    user = current_user(request)
    row = get_store().get_voicemail(file_key, user, include_deleted=True)
    audio_path = safe_under_roots(str(row["wav_path"]), [SETTINGS.watch_dir, SETTINGS.trash_dir])
    return stream_audio_file(audio_path, request)


@app.post("/api/voicemails/{file_key}/restore")
def restore_voicemail(file_key: str, request: Request) -> JSONResponse:
    require_csrf(request)
    user = current_user(request)

    store = get_store()
    row = store.get_voicemail(file_key, user, include_deleted=True)
    restored = restore_message_to_inbox(row, store.settings)
    new_file_key = str(restored["file_key"])
    store.mark_restored(
        file_key,
        new_file_key,
        str(restored["txt_path"]),
        str(restored["wav_path"]),
        str(restored["msg_name"]),
    )
    refresh_voicemail_mwi_for_row(
        {
            "file_key": new_file_key,
            "extension": row["extension"],
            "mailbox": row["mailbox"] if "mailbox" in row.keys() else row["extension"],
            "txt_path": str(restored["txt_path"]),
            "wav_path": str(restored["wav_path"]),
        },
        store.settings,
        reason="portal restore",
    )
    logger.info(
        "Voicemail restored old_file_key=%s new_file_key=%s extension=%s user=%s moved_files=%s",
        file_key,
        new_file_key,
        row["extension"],
        user.username,
        len(restored["moved_files"]),
    )
    return JSONResponse(
        {
            "ok": True,
            "file_key": new_file_key,
            "moved_files": len(restored["moved_files"]),
        }
    )


@app.post("/api/voicemails/{file_key}/forward")
def forward_voicemail(
    file_key: str,
    payload: ForwardVoicemailRequest,
    request: Request,
) -> JSONResponse:
    if not SETTINGS.forward_enabled:
        raise HTTPException(status_code=403, detail="Voicemail forwarding is disabled.")
    require_csrf(request)
    user = current_user(request)
    store = get_store()
    row = store.get_voicemail(file_key, user)

    target_extension = str(payload.target_extension or "").strip()
    if not re.fullmatch(r"\d{3,6}", target_extension):
        raise HTTPException(status_code=400, detail="Choose a valid target mailbox.")
    if target_extension == str(row["extension"]):
        raise HTTPException(status_code=400, detail="Choose a different target mailbox.")
    if target_extension not in discover_mailbox_extensions(store.settings):
        raise HTTPException(status_code=404, detail="Target mailbox was not found.")

    copied_message = copy_message_to_mailbox(row, target_extension, store.settings)
    try:
        store.create_forwarded_copy_record(row, copied_message, target_extension, user.username)
    except Exception:
        remove_copied_message_files(copied_message.get("copied_files", []))
        raise

    target_lookup_user = PortalUser("portal-forward", "*", "", "Portal Forward", True)
    target_row = store.get_voicemail(str(copied_message["file_key"]), target_lookup_user)
    refresh_voicemail_mwi_for_row(target_row, store.settings, reason="portal forward")

    email_result = {"email_sent": False, "recipient_count": 0, "email_error": ""}
    if store.settings.forward_email_enabled:
        try:
            email_result = send_forwarded_voicemail_email(row, target_row, user.username, store.settings)
        except Exception as exc:
            email_result = {
                "email_sent": False,
                "recipient_count": 0,
                "email_error": str(exc)[:1000],
            }
            logger.warning(
                "Forwarded voicemail email failed source_key=%s target_key=%s source_ext=%s target_ext=%s error=%s",
                row["file_key"],
                target_row["file_key"],
                row["extension"],
                target_extension,
                exc,
            )

    store.mark_forward_email_result(
        str(copied_message["file_key"]),
        bool(email_result.get("email_sent")),
        str(email_result.get("email_error") or ""),
    )
    logger.info(
        "Voicemail forwarded source_key=%s target_key=%s source_ext=%s target_ext=%s user=%s copied_files=%s email_sent=%s",
        file_key,
        copied_message["file_key"],
        row["extension"],
        target_extension,
        user.username,
        len(copied_message.get("copied_files", [])),
        bool(email_result.get("email_sent")),
    )
    return JSONResponse(
        {
            "ok": True,
            "file_key": copied_message["file_key"],
            "target_extension": target_extension,
            "copied_files": len(copied_message.get("copied_files", [])),
            "email_sent": bool(email_result.get("email_sent")),
            "email_recipient_count": int(email_result.get("recipient_count") or 0),
            "email_error": str(email_result.get("email_error") or ""),
        }
    )


def _delete_voicemail_for_user(
    file_key: str,
    user: PortalUser,
    comment: Optional[str],
    bulk: bool = False,
) -> tuple[sqlite3.Row, list[str]]:
    store = get_store()
    row = store.get_voicemail(file_key, user)
    deleted_comment = validate_delete_comment_for_user(user, comment, store.settings, row["extension"])
    moved = move_message_to_trash(row, store.settings)
    store.mark_deleted(file_key, user.username, moved, deleted_comment)
    refresh_voicemail_mwi_for_row(row, store.settings, reason="portal delete")
    logger.info(
        "Voicemail deleted file_key=%s extension=%s user=%s moved_files=%s bulk=%s comment=%s",
        file_key,
        row["extension"],
        user.username,
        len(moved),
        bulk,
        bool(deleted_comment),
    )
    return row, moved


@app.post("/api/voicemails/{file_key}/delete")
def delete_voicemail_with_comment(
    file_key: str,
    payload: DeleteVoicemailRequest,
    request: Request,
) -> JSONResponse:
    require_csrf(request)
    user = current_user(request)
    _row, moved = _delete_voicemail_for_user(file_key, user, payload.comment)
    return JSONResponse({"ok": True, "moved_files": len(moved)})


@app.post("/api/voicemails/{file_key}/comment")
def save_voicemail_comment(
    file_key: str,
    payload: SaveVoicemailCommentRequest,
    request: Request,
) -> JSONResponse:
    require_csrf(request)
    user = current_user(request)
    comment = get_store().save_comment(file_key, user, payload.comment)
    return JSONResponse({"ok": True, "comment": comment})


@app.post("/api/voicemails/bulk-delete")
def bulk_delete_voicemails(payload: BulkDeleteRequest, request: Request) -> JSONResponse:
    require_csrf(request)
    user = current_user(request)
    store = get_store()
    file_keys = []
    seen: set[str] = set()
    for file_key in payload.file_keys:
        if file_key in seen:
            continue
        seen.add(file_key)
        file_keys.append(file_key)

    if not file_keys:
        raise HTTPException(status_code=400, detail="No voicemails selected")
    if len(file_keys) > 500:
        raise HTTPException(status_code=400, detail="Select 500 or fewer voicemails at a time")

    deleted_comment = normalize_delete_comment(payload.comment)
    deleted: list[str] = []
    errors: list[dict[str, Any]] = []
    moved_count = 0
    rows_by_key: dict[str, sqlite3.Row] = {}
    rows_to_refresh_mwi: list[sqlite3.Row] = []

    for file_key in file_keys:
        try:
            row = store.get_voicemail(file_key, user)
            if delete_comment_required_for_user(user, store.settings, row["extension"]) and not deleted_comment:
                raise HTTPException(status_code=400, detail="A comment is required to delete this message.")
            rows_by_key[file_key] = row
        except HTTPException as exc:
            errors.append(
                {
                    "file_key": file_key,
                    "status_code": exc.status_code,
                    "detail": str(exc.detail),
                }
            )
        except Exception:
            logger.exception("Bulk voicemail delete preflight failed file_key=%s user=%s", file_key, user.username)
            errors.append({"file_key": file_key, "status_code": 500, "detail": "Delete failed"})

    comment_required_errors = [
        error
        for error in errors
        if error["status_code"] == 400 and error["detail"] == "A comment is required to delete this message."
    ]
    if comment_required_errors:
        return JSONResponse(
            {
                "ok": False,
                "deleted": [],
                "deleted_count": 0,
                "moved_files": 0,
                "errors": comment_required_errors,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    for file_key in rows_by_key:
        try:
            row = rows_by_key[file_key]
            moved = move_message_to_trash(row, store.settings)
            store.mark_deleted(file_key, user.username, moved, deleted_comment)
            deleted.append(file_key)
            rows_to_refresh_mwi.append(row)
            moved_count += len(moved)
            logger.info(
                "Voicemail deleted file_key=%s extension=%s user=%s moved_files=%s bulk=true comment=%s",
                file_key,
                row["extension"],
                user.username,
                len(moved),
                bool(deleted_comment),
            )
        except HTTPException as exc:
            errors.append(
                {
                    "file_key": file_key,
                    "status_code": exc.status_code,
                    "detail": str(exc.detail),
                }
            )
        except Exception:
            logger.exception("Bulk voicemail delete failed file_key=%s user=%s", file_key, user.username)
            errors.append({"file_key": file_key, "status_code": 500, "detail": "Delete failed"})

    refresh_voicemail_mwi_for_rows(rows_to_refresh_mwi, store.settings, reason="portal bulk delete")

    return JSONResponse(
        {
            "ok": not errors,
            "deleted": deleted,
            "deleted_count": len(deleted),
            "moved_files": moved_count,
            "errors": errors,
        },
        status_code=status.HTTP_200_OK if deleted else status.HTTP_400_BAD_REQUEST,
    )


@app.delete("/api/voicemails/{file_key}")
def delete_voicemail(file_key: str, request: Request) -> JSONResponse:
    require_csrf(request)
    user = current_user(request)
    _row, moved = _delete_voicemail_for_user(file_key, user, None)
    return JSONResponse({"ok": True, "moved_files": len(moved)})


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "hash-password":
        return print_password_hash(argv)

    if uvicorn is None:
        raise RuntimeError("Missing required dependency: uvicorn")
    uvicorn.run(
        "voicemail_portal:app",
        host=SETTINGS.host,
        port=SETTINGS.port,
        log_level=SETTINGS.log_level.lower(),
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
