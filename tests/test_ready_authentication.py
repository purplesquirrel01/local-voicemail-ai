from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

import litert_chat_web
import parakeet_server
import whisper_server


@pytest.mark.parametrize("service", ("whisper", "parakeet", "gemma"))
def test_authenticated_readiness_rejects_missing_and_invalid_credentials(
    service: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "readiness-secret-value"
    if service == "whisper":
        monkeypatch.setattr(
            whisper_server,
            "SETTINGS",
            replace(whisper_server.SETTINGS, require_auth=True, api_key=secret, min_tmp_free_bytes=0),
        )
        monkeypatch.setattr(whisper_server, "whisper_model", object())
        app = whisper_server.app
    elif service == "parakeet":
        monkeypatch.setattr(parakeet_server, "PARAKEET_REQUIRE_AUTH", True)
        monkeypatch.setattr(parakeet_server, "PARAKEET_API_KEY", secret)
        monkeypatch.setattr(parakeet_server, "MODEL", object())
        app = parakeet_server.app
    else:
        monkeypatch.setattr(litert_chat_web, "LITERT_REQUIRE_AUTH", True)
        monkeypatch.setattr(litert_chat_web, "LITERT_API_KEY", secret)
        monkeypatch.setattr(litert_chat_web, "LITERT_ORCHESTRATOR_ONLY", True)
        app = litert_chat_web.app

    client = TestClient(app)
    assert client.get("/ready").status_code == 401
    assert client.get("/ready", headers={"Authorization": "Bearer invalid"}).status_code == 401
    assert client.get("/ready", headers={"Authorization": f"Bearer {secret}"}).status_code == 200
