"""Validated provider contracts for the voicemail AI pipeline."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import requests


class ProviderConfigError(ValueError):
    pass


class _Secret:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return "<redacted>"


@runtime_checkable
class TranscriptionProvider(Protocol):
    name: str

    def transcribe(self, audio_path: Path, request_id: str) -> Mapping[str, Any]: ...

    def healthy(self, *, session: Any = requests) -> bool: ...


@runtime_checkable
class ExtractionProvider(Protocol):
    name: str

    def extract(self, payload: Mapping[str, Any], request_id: str) -> Mapping[str, Any]: ...

    def healthy(self, *, session: Any = requests) -> bool: ...


@runtime_checkable
class VerificationProvider(Protocol):
    name: str

    def verify(self, audio_path: Path, request_id: str) -> Mapping[str, Any]: ...

    def healthy(self, *, session: Any = requests) -> bool: ...


@dataclass(frozen=True)
class _HTTPProvider:
    name: str
    url: str
    ready_url: str
    timeout_seconds: float
    token: _Secret

    def _headers(self, request_id: str = "") -> dict[str, str] | None:
        headers: dict[str, str] = {}
        if self.token.value:
            headers["Authorization"] = f"Bearer {self.token.value}"
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers or None

    def healthy(self, *, session: Any = requests) -> bool:
        try:
            response = session.get(
                self.ready_url,
                headers=self._headers(),
                timeout=min(3.0, self.timeout_seconds),
            )
            return int(response.status_code) == 200
        except Exception:
            return False


@dataclass(frozen=True)
class HTTPTranscriptionProvider(_HTTPProvider):
    def transcribe(self, audio_path: Path, request_id: str) -> Mapping[str, Any]:
        with Path(audio_path).open("rb") as audio:
            response = requests.post(
                self.url,
                files={"file": (Path(audio_path).name, audio, "audio/wav")},
                headers=self._headers(request_id),
                timeout=(min(10.0, self.timeout_seconds), self.timeout_seconds),
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise RuntimeError("Transcription provider returned an invalid response")
        return payload


@dataclass(frozen=True)
class HTTPExtractionProvider(_HTTPProvider):
    def extract(self, payload: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
        response = requests.post(
            self.url,
            json=dict(payload),
            headers=self._headers(request_id),
            timeout=(min(10.0, self.timeout_seconds), self.timeout_seconds),
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("Extraction provider returned an invalid response")
        return result


@dataclass(frozen=True)
class HTTPVerificationProvider(_HTTPProvider):
    def verify(self, audio_path: Path, request_id: str) -> Mapping[str, Any]:
        with Path(audio_path).open("rb") as audio:
            response = requests.post(
                self.url,
                files={"file": (Path(audio_path).name, audio, "audio/wav")},
                headers=self._headers(request_id),
                timeout=(min(10.0, self.timeout_seconds), self.timeout_seconds),
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Verification provider returned an invalid response")
        return payload


@dataclass(frozen=True)
class ProviderRegistry:
    transcription: TranscriptionProvider
    extraction: ExtractionProvider
    verification: VerificationProvider
    external_phi_warning: str | None = None


def build_provider_registry(values: Mapping[str, str] | None = None) -> ProviderRegistry:
    env = os.environ if values is None else values
    timeout = _timeout(env.get("LVT_PROVIDER_TIMEOUT_SECONDS", "240"))
    allow_external = _boolean(env.get("LVT_ALLOW_EXTERNAL_PHI_PROVIDER", "false"))

    transcription_name = str(env.get("LVT_TRANSCRIPTION_PROVIDER", "faster-whisper")).strip().lower()
    extraction_name = str(env.get("LVT_EXTRACTION_PROVIDER", "litert-gemma")).strip().lower()
    verification_name = str(env.get("LVT_VERIFICATION_PROVIDER", "parakeet")).strip().lower()
    if transcription_name not in {"faster-whisper", "http"}:
        raise ProviderConfigError("Unsupported transcription provider")
    if extraction_name not in {"litert-gemma", "http"}:
        raise ProviderConfigError("Unsupported extraction provider")
    if verification_name not in {"parakeet", "http"}:
        raise ProviderConfigError("Unsupported verification provider")

    specs = (
        (
            "transcription",
            transcription_name,
            env.get("LVT_TRANSCRIPTION_PROVIDER_URL") or env.get("WHISPER_URL", "http://127.0.0.1:8765/transcribe/voicemail"),
            env.get("LVT_TRANSCRIPTION_PROVIDER_TOKEN") or env.get("WHISPER_API_KEY", ""),
        ),
        (
            "extraction",
            extraction_name,
            env.get("LVT_EXTRACTION_PROVIDER_URL") or env.get("GEMMA_BASE_URL", "http://127.0.0.1:8787"),
            env.get("LVT_EXTRACTION_PROVIDER_TOKEN") or env.get("GEMMA_API_KEY", ""),
        ),
        (
            "verification",
            verification_name,
            env.get("LVT_VERIFICATION_PROVIDER_URL") or env.get("PARAKEET_VERIFICATION_URL", "http://127.0.0.1:8766/transcribe"),
            env.get("LVT_VERIFICATION_PROVIDER_TOKEN") or env.get("PARAKEET_API_KEY", ""),
        ),
    )
    external = False
    configured: dict[str, _HTTPProvider] = {}
    for kind, name, raw_url, raw_token in specs:
        url, is_external = _provider_url(str(raw_url), kind)
        external = external or is_external
        ready = _ready_url(url)
        cls = {
            "transcription": HTTPTranscriptionProvider,
            "extraction": HTTPExtractionProvider,
            "verification": HTTPVerificationProvider,
        }[kind]
        configured[kind] = cls(name, url, ready, timeout, _Secret(str(raw_token).strip()))
    if external and not allow_external:
        raise ProviderConfigError(
            "An HTTP provider could leave the local network; set LVT_ALLOW_EXTERNAL_PHI_PROVIDER=true only after authorization"
        )
    warning = (
        "WARNING: voicemail could leave the local network through an advanced HTTP provider"
        if external
        else None
    )
    return ProviderRegistry(
        transcription=configured["transcription"],
        extraction=configured["extraction"],
        verification=configured["verification"],
        external_phi_warning=warning,
    )


def _provider_url(raw: str, label: str) -> tuple[str, bool]:
    value = raw.strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderConfigError(f"{label} provider URL must be credential-free HTTP(S)")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ProviderConfigError(f"{label} provider URL has an invalid port") from exc
    if port is not None and (port < 1 or port > 65535):
        raise ProviderConfigError(f"{label} provider URL has an invalid port")
    hostname = parsed.hostname
    try:
        address = ipaddress.ip_address(hostname)
        is_external = not (address.is_private or address.is_loopback)
    except ValueError:
        is_external = hostname not in {"localhost"} and not hostname.endswith(".local")
    normalized = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", "", ""))
    return normalized, is_external


def _ready_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/ready", "", ""))


def _timeout(raw: object) -> float:
    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise ProviderConfigError("Provider timeout must be numeric") from exc
    if value < 1 or value > 600:
        raise ProviderConfigError("Provider timeout must be between 1 and 600 seconds")
    return value


def _boolean(raw: object) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ProviderConfigError("Provider acknowledgement must be a boolean")
