from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from voicemail_portal_app.api_v1 import (
    IntegrationAPIConfig,
    SQLiteIntegrationBackend,
    build_api_v1_app,
    hash_service_token,
    mount_api_v1,
)


TOKEN = "lvt_test_" + "a" * 40
WALL_NOW = datetime(2026, 8, 8, 20, 0, tzinfo=UTC)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.row = {
            "file_key": "safe-key-100",
            "extension": "100",
            "mailbox": "100",
            "folder": "INBOX",
            "callerid": "Synthetic Caller <2025550142>",
            "origtime": 1_700_000_000,
            "origdate": "2023-11-14 22:13:20",
            "duration": 12,
            "transcript": "Synthetic message.",
            "entities": {"callback_number": "2025550142"},
            "field_verifications": {
                "callback_number": {
                    "final_value": "2025550142",
                    "status": "verified",
                    "confidence": 0.98,
                    "clip_json": [{"path": "/secret/clip.wav"}],
                }
            },
            "processing_status": "completed",
            "txt_path": "/var/spool/asterisk/voicemail/100/INBOX/msg0001.txt",
            "wav_path": "/var/spool/asterisk/voicemail/100/INBOX/msg0001.wav",
            "last_error": "token=must-not-leak",
        }

    def list_voicemails(
        self, *, mailboxes: tuple[str, ...], limit: int, offset: int
    ) -> tuple[list[dict[str, object]], int]:
        self.calls.append(("list", mailboxes, limit, offset))
        rows = [self.row] if not mailboxes or "100" in mailboxes else []
        return rows[offset : offset + limit], len(rows)

    def get_voicemail(
        self, file_key: str, *, mailboxes: tuple[str, ...]
    ) -> dict[str, object] | None:
        self.calls.append(("get", file_key, mailboxes))
        if file_key != self.row["file_key"] or (mailboxes and "100" not in mailboxes):
            return None
        return self.row

    def get_mailbox(
        self, extension: str, *, mailboxes: tuple[str, ...], limit: int, offset: int
    ) -> tuple[list[dict[str, object]], int] | None:
        self.calls.append(("mailbox", extension, mailboxes, limit, offset))
        if mailboxes and extension not in mailboxes:
            return None
        rows = [self.row] if extension == "100" else []
        return rows[offset : offset + limit], len(rows)


def _token_file(
    tmp_path: Path,
    *,
    scopes: tuple[str, ...] = ("voicemail:read",),
    expires_utc: str | None = None,
    token_rate_limit: int | None = None,
) -> Path:
    path = tmp_path / "api-tokens.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "tokens": [
                    ({
                        "name": "synthetic-integration",
                        "token_sha256": hash_service_token(TOKEN),
                        "scopes": list(scopes),
                        "mailboxes": ["100"],
                        "revoked": False,
                    }
                    | ({"expires_utc": expires_utc} if expires_utc else {})
                    | ({"rate_limit_requests": token_rate_limit} if token_rate_limit else {}))
                ],
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _client(
    tmp_path: Path,
    backend: FakeBackend,
    *,
    enabled: bool = True,
    scopes: tuple[str, ...] = ("voicemail:read",),
    rate_limit: int = 60,
    expires_utc: str | None = None,
    token_rate_limit: int | None = None,
) -> TestClient:
    config = IntegrationAPIConfig(
        enabled=enabled,
        token_file=_token_file(
            tmp_path,
            scopes=scopes,
            expires_utc=expires_utc,
            token_rate_limit=token_rate_limit,
        ),
        rate_limit_requests=rate_limit,
        rate_limit_window_seconds=60,
        max_page_size=100,
    )
    return TestClient(build_api_v1_app(config, backend, wall_clock=lambda: WALL_NOW))


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_api_is_disabled_by_default_and_has_no_openapi_surface(tmp_path: Path) -> None:
    """Catches an integration listener becoming available without explicit enablement."""
    app = build_api_v1_app(
        IntegrationAPIConfig(enabled=False, token_file=tmp_path / "missing.json"),
        FakeBackend(),
    )
    client = TestClient(app)
    assert client.get("/voicemails").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_api_environment_defaults_off_and_rejects_ambiguous_boolean() -> None:
    """Catches a missing or misspelled enable flag accidentally activating the API."""
    assert not IntegrationAPIConfig.from_env({}).enabled
    enabled = IntegrationAPIConfig.from_env(
        {
            "LVT_INTEGRATION_API_ENABLED": "true",
            "LVT_INTEGRATION_API_TOKEN_FILE": "/protected/tokens.json",
            "LVT_INTEGRATION_API_RATE_LIMIT": "25",
            "LVT_INTEGRATION_API_RATE_WINDOW": "30",
            "LVT_INTEGRATION_API_MAX_PAGE_SIZE": "75",
        }
    )
    assert enabled.enabled
    assert enabled.token_file == Path("/protected/tokens.json")
    assert enabled.rate_limit_requests == 25
    assert enabled.max_page_size == 75

    try:
        IntegrationAPIConfig.from_env({"LVT_INTEGRATION_API_ENABLED": "maybe"})
    except ValueError as exc:
        assert "boolean" in str(exc)
    else:
        raise AssertionError("ambiguous API enable value was accepted")


def test_service_tokens_are_hashed_scoped_and_mailbox_limited(tmp_path: Path) -> None:
    """Catches plaintext token storage, missing scope enforcement, or mailbox overreach."""
    backend = FakeBackend()
    client = _client(tmp_path, backend)

    assert client.get("/voicemails").status_code == 401
    assert client.get("/voicemails", headers={"Authorization": "Bearer wrong"}).status_code == 401

    response = client.get("/voicemails?limit=10&offset=0", headers=_headers())
    assert response.status_code == 200
    assert backend.calls == [("list", ("100",), 10, 0)]
    assert response.json()["pagination"] == {"limit": 10, "offset": 0, "total": 1}

    denied = _client(tmp_path, backend, scopes=("mailbox:metadata",))
    assert denied.get("/voicemails", headers=_headers()).status_code == 403


def test_expired_token_has_same_public_failure_as_unknown_token(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        FakeBackend(),
        expires_utc="2026-08-07T20:00:00Z",
    )

    expired = client.get("/voicemails", headers=_headers())
    unknown = client.get(
        "/voicemails",
        headers={"Authorization": f"Bearer lvt_unknown_{'x' * 40}"},
    )

    assert expired.status_code == unknown.status_code == 401
    assert expired.json() == unknown.json() == {"detail": "Invalid service token"}


def test_future_expiration_remains_valid(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        FakeBackend(),
        expires_utc="2026-08-09T20:00:00Z",
    )

    assert client.get("/voicemails", headers=_headers()).status_code == 200


def test_per_token_rate_limit_overrides_global_count(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeBackend(), rate_limit=60, token_rate_limit=2)

    statuses = [client.get("/voicemails", headers=_headers()).status_code for _ in range(3)]

    assert statuses == [200, 200, 429]


def test_api_response_excludes_internal_paths_errors_and_verification_artifacts(tmp_path: Path) -> None:
    """Catches server paths, secret-bearing errors, or internal clip metadata escaping v1."""
    client = _client(tmp_path, FakeBackend())

    response = client.get("/voicemails/safe-key-100", headers=_headers())

    assert response.status_code == 200
    payload = response.json()
    serialized = json.dumps(payload)
    assert payload["file_key"] == "safe-key-100"
    assert payload["verification"]["callback_number"] == {
        "value": "2025550142",
        "status": "verified",
        "confidence": 0.98,
    }
    assert "/var/" not in serialized
    assert "/secret/" not in serialized
    assert "must-not-leak" not in serialized
    assert "txt_path" not in serialized
    assert "wav_path" not in serialized


def test_mailbox_scope_uses_not_found_to_prevent_enumeration(tmp_path: Path) -> None:
    """Catches an integration token discovering mailboxes outside its assigned scope."""
    client = _client(tmp_path, FakeBackend())
    assert client.get("/mailboxes/100", headers=_headers()).status_code == 200
    assert client.get("/mailboxes/200", headers=_headers()).status_code == 404
    assert client.get("/voicemails/other-key", headers=_headers()).status_code == 404


def test_pagination_is_bounded_and_rate_limit_is_per_token(tmp_path: Path) -> None:
    """Catches unbounded database reads or a token bypassing its request budget."""
    client = _client(tmp_path, FakeBackend(), rate_limit=2)
    assert client.get("/voicemails?limit=101", headers=_headers()).status_code == 422
    assert client.get("/voicemails?limit=1", headers=_headers()).status_code == 200
    assert client.get("/voicemails?limit=1", headers=_headers()).status_code == 429


def test_openapi_exists_only_on_the_enabled_versioned_app(tmp_path: Path) -> None:
    """Catches the stable integration schema being omitted when the API is enabled."""
    client = _client(tmp_path, FakeBackend())
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/voicemails",
        "/voicemails/{file_key}",
        "/mailboxes/{extension}",
    }


def test_versioned_app_mount_does_not_enable_parent_openapi(tmp_path: Path) -> None:
    """Catches v1 routes leaking into the browser-session API or its unversioned schema."""
    parent = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    config = IntegrationAPIConfig(enabled=True, token_file=_token_file(tmp_path))

    assert mount_api_v1(parent, config, FakeBackend())
    client = TestClient(parent)
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/api/v1/openapi.json").status_code == 200
    assert client.get("/api/v1/voicemails", headers=_headers()).status_code == 200

    disabled_parent = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    assert not mount_api_v1(
        disabled_parent,
        IntegrationAPIConfig(enabled=False, token_file=tmp_path / "missing"),
        FakeBackend(),
    )
    assert TestClient(disabled_parent).get("/api/v1/voicemails").status_code == 404


def test_sqlite_backend_enforces_mailbox_scope_in_the_query(tmp_path: Path) -> None:
    """Catches filtering after retrieval or selecting internal filesystem columns."""
    database = tmp_path / "state.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE voicemail_transcripts (
                file_key TEXT PRIMARY KEY,
                extension TEXT NOT NULL,
                mailbox TEXT,
                folder TEXT,
                callerid TEXT,
                origtime INTEGER,
                origdate TEXT,
                duration INTEGER,
                transcript TEXT,
                entities_json TEXT,
                deleted_utc TEXT
            );
            CREATE TABLE voicemails (file_key TEXT PRIMARY KEY, status TEXT);
            CREATE TABLE voicemail_field_verification (
                file_key TEXT,
                field_name TEXT,
                final_value TEXT,
                status TEXT,
                needs_review INTEGER
            );
            """
        )
        for extension in ("100", "200"):
            connection.execute(
                "INSERT INTO voicemail_transcripts VALUES (?, ?, ?, 'INBOX', '', 1, '', 4, ?, '{}', NULL)",
                (f"key-{extension}", extension, extension, f"message-{extension}"),
            )
            connection.execute(
                "INSERT INTO voicemails VALUES (?, 'completed')", (f"key-{extension}",)
            )
        connection.execute(
            "INSERT INTO voicemail_field_verification VALUES ('key-100', 'callback_number', '2025550142', 'verified', 0)"
        )

    backend = SQLiteIntegrationBackend(database)
    rows, total = backend.list_voicemails(mailboxes=("100",), limit=50, offset=0)

    assert total == 1
    assert [row["file_key"] for row in rows] == ["key-100"]
    assert rows[0]["field_verifications"]["callback_number"] == {
        "final_value": "2025550142",
        "status": "verified",
        "confidence": 1.0,
    }
    assert not ({"txt_path", "wav_path", "last_error"} & rows[0].keys())
    assert backend.get_voicemail("key-200", mailboxes=("100",)) is None
