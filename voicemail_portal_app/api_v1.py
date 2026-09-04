"""Stable, token-authenticated integration API for normalized voicemail data."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, Protocol

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


READ_SCOPE = "voicemail:read"
_ALLOWED_SCOPES = {READ_SCOPE, "mailbox:metadata"}
_EXTENSION_RE = re.compile(r"^[0-9]{1,20}$")
_FILE_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,255}$")
_TOKEN_DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")
_MAX_TOKEN_FILE_BYTES = 64 * 1024


class IntegrationBackend(Protocol):
    def list_voicemails(
        self, *, mailboxes: tuple[str, ...], limit: int, offset: int
    ) -> tuple[list[dict[str, object]], int]: ...

    def get_voicemail(
        self, file_key: str, *, mailboxes: tuple[str, ...]
    ) -> dict[str, object] | None: ...

    def get_mailbox(
        self, extension: str, *, mailboxes: tuple[str, ...], limit: int, offset: int
    ) -> tuple[list[dict[str, object]], int] | None: ...


class SQLiteIntegrationBackend:
    """Read normalized portal records without selecting internal host paths."""

    def __init__(self, state_db: Path) -> None:
        self.state_db = Path(state_db)

    def _connect(self) -> sqlite3.Connection:
        if not self.state_db.is_file():
            raise RuntimeError("Voicemail state database is unavailable")
        uri = self.state_db.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def list_voicemails(
        self, *, mailboxes: tuple[str, ...], limit: int, offset: int
    ) -> tuple[list[dict[str, object]], int]:
        return self._query(mailboxes=mailboxes, limit=limit, offset=offset)

    def get_voicemail(
        self, file_key: str, *, mailboxes: tuple[str, ...]
    ) -> dict[str, object] | None:
        rows, _total = self._query(
            mailboxes=mailboxes,
            limit=1,
            offset=0,
            file_key=file_key,
        )
        return rows[0] if rows else None

    def get_mailbox(
        self, extension: str, *, mailboxes: tuple[str, ...], limit: int, offset: int
    ) -> tuple[list[dict[str, object]], int] | None:
        if mailboxes and extension not in mailboxes:
            return None
        return self._query(
            mailboxes=(extension,),
            limit=limit,
            offset=offset,
        )

    def _query(
        self,
        *,
        mailboxes: tuple[str, ...],
        limit: int,
        offset: int,
        file_key: str | None = None,
    ) -> tuple[list[dict[str, object]], int]:
        clauses = ["t.deleted_utc IS NULL"]
        params: list[object] = []
        if mailboxes:
            clauses.append("t.extension IN (" + ",".join("?" for _ in mailboxes) + ")")
            params.extend(mailboxes)
        if file_key is not None:
            clauses.append("t.file_key = ?")
            params.append(file_key)
        where = " AND ".join(clauses)
        with self._connect() as connection:
            count_row = connection.execute(
                f"SELECT count(*) FROM voicemail_transcripts t WHERE {where}", params
            ).fetchone()
            rows = connection.execute(
                f"""
                SELECT t.file_key,
                       t.extension,
                       t.mailbox,
                       t.folder,
                       t.callerid,
                       t.origtime,
                       t.origdate,
                       t.duration,
                       t.transcript,
                       t.entities_json,
                       v.status AS processing_status
                FROM voicemail_transcripts t
                LEFT JOIN voicemails v ON v.file_key = t.file_key
                WHERE {where}
                ORDER BY COALESCE(t.origtime, 0) DESC, t.file_key DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
            keys = [str(row["file_key"]) for row in rows]
            verifications = self._verifications(connection, keys)
        normalized: list[dict[str, object]] = []
        for row in rows:
            try:
                entities = json.loads(row["entities_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                entities = {}
            normalized.append(
                {
                    "file_key": row["file_key"],
                    "extension": row["extension"],
                    "mailbox": row["mailbox"],
                    "folder": row["folder"],
                    "callerid": row["callerid"],
                    "origtime": row["origtime"],
                    "origdate": row["origdate"],
                    "duration": row["duration"],
                    "transcript": row["transcript"],
                    "entities": entities if isinstance(entities, dict) else {},
                    "field_verifications": verifications.get(str(row["file_key"]), {}),
                    "processing_status": row["processing_status"],
                }
            )
        return normalized, int(count_row[0] if count_row else 0)

    @staticmethod
    def _verifications(
        connection: sqlite3.Connection, file_keys: list[str]
    ) -> dict[str, dict[str, dict[str, object]]]:
        if not file_keys:
            return {}
        placeholders = ",".join("?" for _ in file_keys)
        try:
            rows = connection.execute(
                f"""
                SELECT file_key, field_name, final_value, status, needs_review
                FROM voicemail_field_verification
                WHERE file_key IN ({placeholders})
                """,
                file_keys,
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            raise
        result: dict[str, dict[str, dict[str, object]]] = {}
        for row in rows:
            confidence = 1.0 if row["status"] == "verified" and not row["needs_review"] else 0.5
            result.setdefault(str(row["file_key"]), {})[str(row["field_name"])] = {
                "final_value": row["final_value"],
                "status": row["status"],
                "confidence": confidence,
            }
        return result


@dataclass(frozen=True)
class IntegrationAPIConfig:
    enabled: bool = False
    token_file: Path = Path("/etc/local-voicemail-transcription/api-tokens.json")
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    max_page_size: int = 100

    def __post_init__(self) -> None:
        if self.rate_limit_requests < 1 or self.rate_limit_requests > 10_000:
            raise ValueError("Integration API request limit must be between 1 and 10000")
        if self.rate_limit_window_seconds < 1 or self.rate_limit_window_seconds > 3600:
            raise ValueError("Integration API rate window must be between 1 and 3600 seconds")
        if self.max_page_size < 1 or self.max_page_size > 100:
            raise ValueError("Integration API maximum page size must be between 1 and 100")

    @classmethod
    def from_env(cls, values: Mapping[str, str] | None = None) -> "IntegrationAPIConfig":
        env = os.environ if values is None else values

        def boolean(key: str, default: bool = False) -> bool:
            raw = str(env.get(key, "true" if default else "false")).strip().lower()
            if raw in {"1", "true", "yes", "on"}:
                return True
            if raw in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{key} must be a boolean")

        def integer(key: str, default: int) -> int:
            try:
                return int(str(env.get(key, default)).strip())
            except ValueError as exc:
                raise ValueError(f"{key} must be an integer") from exc

        enabled = boolean("LVT_INTEGRATION_API_ENABLED")
        token_file_raw = str(
            env.get(
                "LVT_INTEGRATION_API_TOKEN_FILE",
                "/etc/local-voicemail-transcription/api-tokens.json",
            )
        ).strip()
        token_file = Path(token_file_raw)
        if enabled and not token_file_raw.startswith("/"):
            raise ValueError("LVT_INTEGRATION_API_TOKEN_FILE must be absolute")
        return cls(
            enabled=enabled,
            token_file=token_file,
            rate_limit_requests=integer("LVT_INTEGRATION_API_RATE_LIMIT", 60),
            rate_limit_window_seconds=integer("LVT_INTEGRATION_API_RATE_WINDOW", 60),
            max_page_size=integer("LVT_INTEGRATION_API_MAX_PAGE_SIZE", 100),
        )


@dataclass(frozen=True)
class ServicePrincipal:
    name: str
    token_digest: str
    scopes: tuple[str, ...]
    mailboxes: tuple[str, ...]
    expires_utc: datetime | None = None
    rate_limit_requests: int | None = None


def hash_service_token(token: str) -> str:
    value = token.strip()
    if len(value) < 32 or len(value) > 512 or any(character.isspace() for character in value):
        raise ValueError("Service token must be a bounded non-whitespace secret")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_service_principals(
    path: Path,
    *,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[ServicePrincipal, ...]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError("Integration API token file is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_TOKEN_FILE_BYTES:
        raise RuntimeError("Integration API token file is unsafe")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) not in {0o600, 0o640}:
        raise RuntimeError("Integration API token file must have mode 0600 or 0640")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Integration API token file is malformed") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1 or not isinstance(payload.get("tokens"), list):
        raise RuntimeError("Integration API token file schema is invalid")
    principals: list[ServicePrincipal] = []
    names: set[str] = set()
    digests: set[str] = set()
    for raw in payload["tokens"]:
        if not isinstance(raw, dict) or raw.get("revoked") is True:
            continue
        name = str(raw.get("name", "")).strip()
        digest = str(raw.get("token_sha256", "")).strip().lower()
        scopes = tuple(sorted(set(raw.get("scopes", [])))) if isinstance(raw.get("scopes"), list) else ()
        mailboxes = tuple(sorted(set(raw.get("mailboxes", [])))) if isinstance(raw.get("mailboxes"), list) else ()
        expires = _optional_utc_timestamp(raw.get("expires_utc"))
        rate_limit = raw.get("rate_limit_requests")
        if (
            not name
            or len(name) > 64
            or not _TOKEN_DIGEST_RE.fullmatch(digest)
            or not scopes
            or not set(scopes).issubset(_ALLOWED_SCOPES)
            or any(not isinstance(value, str) for value in (*scopes, *mailboxes))
            or any(value != "*" and not _EXTENSION_RE.fullmatch(value) for value in mailboxes)
            or name in names
            or digest in digests
            or (
                rate_limit is not None
                and (
                    not isinstance(rate_limit, int)
                    or isinstance(rate_limit, bool)
                    or not 1 <= rate_limit <= 10_000
                )
            )
        ):
            raise RuntimeError("Integration API token entry is invalid")
        names.add(name)
        digests.add(digest)
        if expires is not None and expires <= _as_utc(wall_clock()):
            continue
        principals.append(
            ServicePrincipal(
                name,
                digest,
                scopes,
                () if "*" in mailboxes else mailboxes,
                expires,
                rate_limit,
            )
        )
    return tuple(principals)


def _optional_utc_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError("Integration API token entry is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeError("Integration API token entry is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise RuntimeError("Integration API token entry is invalid")
    return parsed.astimezone(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise RuntimeError("Integration API wall clock must be timezone-aware")
    return value.astimezone(UTC)


class _RateLimiter:
    def __init__(self, maximum: int, window_seconds: int, clock: Callable[[], float]) -> None:
        self.maximum = maximum
        self.window_seconds = window_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._requests: dict[str, list[float]] = {}

    def allow(self, identity: str, maximum: int | None = None) -> bool:
        effective_maximum = self.maximum if maximum is None else maximum
        now = self.clock()
        cutoff = now - self.window_seconds
        with self._lock:
            recent = [stamp for stamp in self._requests.get(identity, ()) if stamp > cutoff]
            if len(recent) >= effective_maximum:
                self._requests[identity] = recent
                return False
            recent.append(now)
            self._requests[identity] = recent
            return True


def build_api_v1_app(
    config: IntegrationAPIConfig,
    backend: IntegrationBackend,
    *,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> FastAPI:
    if not config.enabled:
        return FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    app = FastAPI(
        title="Local Voicemail Transcription Integration API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    bearer = HTTPBearer(auto_error=False)
    limiter = _RateLimiter(
        config.rate_limit_requests,
        config.rate_limit_window_seconds,
        clock,
    )

    def authenticate(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> ServicePrincipal:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
        try:
            supplied = hash_service_token(credentials.credentials)
            principals = load_service_principals(config.token_file, wall_clock=wall_clock)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service token") from None
        except RuntimeError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service authentication unavailable") from None
        principal = next(
            (
                candidate
                for candidate in principals
                if hmac.compare_digest(candidate.token_digest, supplied)
            ),
            None,
        )
        if principal is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service token")
        if not limiter.allow(principal.token_digest, principal.rate_limit_requests):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
        return principal

    def require_read(principal: ServicePrincipal = Depends(authenticate)) -> ServicePrincipal:
        if READ_SCOPE not in principal.scopes:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient scope")
        return principal

    @app.get("/voicemails")
    def list_voicemails(
        limit: int = Query(default=50, ge=1, le=config.max_page_size),
        offset: int = Query(default=0, ge=0, le=10_000),
        principal: ServicePrincipal = Depends(require_read),
    ) -> dict[str, object]:
        rows, total = backend.list_voicemails(
            mailboxes=principal.mailboxes, limit=limit, offset=offset
        )
        return {
            "items": [_normalized_voicemail(row) for row in rows],
            "pagination": {"limit": limit, "offset": offset, "total": max(0, int(total))},
        }

    @app.get("/voicemails/{file_key}")
    def get_voicemail(
        file_key: str,
        principal: ServicePrincipal = Depends(require_read),
    ) -> dict[str, object]:
        if not _FILE_KEY_RE.fullmatch(file_key):
            raise HTTPException(status_code=404, detail="Voicemail not found")
        row = backend.get_voicemail(file_key, mailboxes=principal.mailboxes)
        if row is None:
            raise HTTPException(status_code=404, detail="Voicemail not found")
        return _normalized_voicemail(row)

    @app.get("/mailboxes/{extension}")
    def get_mailbox(
        extension: str,
        limit: int = Query(default=50, ge=1, le=config.max_page_size),
        offset: int = Query(default=0, ge=0, le=10_000),
        principal: ServicePrincipal = Depends(require_read),
    ) -> dict[str, object]:
        if not _EXTENSION_RE.fullmatch(extension) or (
            principal.mailboxes and extension not in principal.mailboxes
        ):
            raise HTTPException(status_code=404, detail="Mailbox not found")
        result = backend.get_mailbox(
            extension,
            mailboxes=principal.mailboxes,
            limit=limit,
            offset=offset,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Mailbox not found")
        rows, total = result
        return {
            "extension": extension,
            "items": [_normalized_voicemail(row) for row in rows],
            "pagination": {"limit": limit, "offset": offset, "total": max(0, int(total))},
        }

    return app


def mount_api_v1(
    parent: FastAPI,
    config: IntegrationAPIConfig,
    backend: IntegrationBackend,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Mount the isolated versioned surface only after explicit enablement."""

    if not config.enabled:
        return False
    parent.mount("/api/v1", build_api_v1_app(config, backend, clock=clock))
    return True


def _normalized_voicemail(row: Mapping[str, object]) -> dict[str, object]:
    entities = row.get("entities") if isinstance(row.get("entities"), dict) else {}
    raw_verification = (
        row.get("field_verifications")
        if isinstance(row.get("field_verifications"), dict)
        else {}
    )
    verification: dict[str, dict[str, object]] = {}
    for field_name, raw in raw_verification.items():
        if not isinstance(field_name, str) or not isinstance(raw, Mapping):
            continue
        verification[field_name] = {
            "value": raw.get("final_value"),
            "status": raw.get("status"),
            "confidence": raw.get("confidence"),
        }
    return {
        "file_key": str(row.get("file_key", "")),
        "extension": str(row.get("extension", "")),
        "mailbox": str(row.get("mailbox") or row.get("extension") or ""),
        "folder": str(row.get("folder") or "INBOX"),
        "caller_id": str(row.get("callerid") or ""),
        "received_at": row.get("origtime"),
        "received_display": str(row.get("origdate") or ""),
        "duration_seconds": row.get("duration"),
        "transcript": str(row.get("transcript") or ""),
        "extracted_fields": entities,
        "verification": verification,
        "processing_status": row.get("processing_status"),
    }
