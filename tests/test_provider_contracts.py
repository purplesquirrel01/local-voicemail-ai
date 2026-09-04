from __future__ import annotations

from dataclasses import dataclass

import pytest

from voicemail_watcher.providers import (
    ExtractionProvider,
    ProviderConfigError,
    TranscriptionProvider,
    VerificationProvider,
    build_provider_registry,
)


def _local_env() -> dict[str, str]:
    return {
        "WHISPER_URL": "http://127.0.0.1:8765/transcribe/voicemail",
        "GEMMA_BASE_URL": "http://127.0.0.1:8787",
        "PARAKEET_VERIFICATION_URL": "http://127.0.0.1:8766/transcribe",
    }


def test_local_provider_registry_keeps_the_proven_stack_as_default() -> None:
    """Catches an advanced provider becoming the automatic voicemail path."""
    registry = build_provider_registry(_local_env())
    assert registry.transcription.name == "faster-whisper"
    assert registry.extraction.name == "litert-gemma"
    assert registry.verification.name == "parakeet"
    assert isinstance(registry.transcription, TranscriptionProvider)
    assert isinstance(registry.extraction, ExtractionProvider)
    assert isinstance(registry.verification, VerificationProvider)
    assert registry.external_phi_warning is None


def test_external_http_provider_requires_explicit_phi_egress_acknowledgement() -> None:
    """Catches voicemail content being routed off-network from an edited environment."""
    env = {
        **_local_env(),
        "LVT_TRANSCRIPTION_PROVIDER": "http",
        "LVT_TRANSCRIPTION_PROVIDER_URL": "https://speech.example.invalid/v1/transcribe",
    }
    with pytest.raises(ProviderConfigError, match="could leave the local network"):
        build_provider_registry(env)

    accepted = build_provider_registry(
        {**env, "LVT_ALLOW_EXTERNAL_PHI_PROVIDER": "true"}
    )
    assert accepted.transcription.name == "http"
    assert "voicemail could leave the local network" in accepted.external_phi_warning


@pytest.mark.parametrize(
    "url",
    (
        "file:///etc/passwd",
        "http://user:pass@192.0.2.2:9000/transcribe",
        "http://192.0.2.2:9000/transcribe?token=secret",
        "http://192.0.2.2:9000/transcribe#fragment",
    ),
)
def test_http_provider_urls_reject_non_http_and_secret_bearing_forms(url: str) -> None:
    """Catches provider configuration putting credentials in config or shell-visible URLs."""
    with pytest.raises(ProviderConfigError):
        build_provider_registry(
            {
                **_local_env(),
                "LVT_EXTRACTION_PROVIDER": "http",
                "LVT_EXTRACTION_PROVIDER_URL": url,
            }
        )


def test_provider_timeouts_are_bounded() -> None:
    """Catches an advanced endpoint hanging the watcher indefinitely."""
    with pytest.raises(ProviderConfigError, match="timeout"):
        build_provider_registry(
            {**_local_env(), "LVT_PROVIDER_TIMEOUT_SECONDS": "0"}
        )
    with pytest.raises(ProviderConfigError, match="timeout"):
        build_provider_registry(
            {**_local_env(), "LVT_PROVIDER_TIMEOUT_SECONDS": "601"}
        )


@dataclass
class _Response:
    status_code: int


class _Session:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.calls: list[tuple[str, float]] = []

    def get(self, url: str, *, headers: dict[str, str] | None, timeout: float) -> _Response:
        assert headers is None or "Authorization" in headers
        self.calls.append((url, timeout))
        return _Response(self.status_code)


def test_provider_health_checks_are_bounded_and_do_not_expose_tokens() -> None:
    """Catches health checks using payload endpoints or leaking bearer secrets in repr."""
    registry = build_provider_registry(
        {**_local_env(), "WHISPER_API_KEY": "synthetic-provider-secret"}
    )
    session = _Session(200)
    assert registry.transcription.healthy(session=session)
    assert session.calls == [("http://127.0.0.1:8765/ready", 3.0)]
    assert "synthetic-provider-secret" not in repr(registry)
