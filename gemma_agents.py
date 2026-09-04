from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from json_utils import parse_json_strict_or_repair


INPUT_JSON_DELIMITER = "\n\nInput JSON:\n"
TextGenerator = Callable[..., str]
PromptBuilder = Callable[[dict[str, Any]], str]
ConstraintBuilder = Callable[[str, dict[str, Any]], dict[str, Any] | None]


class _JsonHttpResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = int(status_code)
        self.body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP worker returned status {self.status_code}")

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def _stdlib_post(url: str, *, json: dict[str, Any], timeout: int) -> _JsonHttpResponse:
    body = __import__("json").dumps(json, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LITERT_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _JsonHttpResponse(int(response.status), response.read())
    except urllib.error.HTTPError as exc:
        return _JsonHttpResponse(int(exc.code), exc.read())


def _response_payload_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return str(payload or "")
    for key in ("response", "text", "content"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


class HttpAgentTextGenerator:
    """Text generator that forwards focused prompts to per-agent HTTP workers."""

    def __init__(
        self,
        agent_urls: dict[str, str],
        *,
        timeout_seconds: int = 420,
        per_agent_timeout_seconds: dict[str, int] | None = None,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.agent_urls = {str(key): str(value) for key, value in agent_urls.items() if value}
        self.timeout_seconds = int(timeout_seconds)
        self.per_agent_timeout_seconds: dict[str, int] = {}
        for key, value in (per_agent_timeout_seconds or {}).items():
            try:
                timeout = int(value)
            except (TypeError, ValueError):
                continue
            if timeout > 0:
                self.per_agent_timeout_seconds[str(key)] = timeout
        self.post = post or _stdlib_post
        self.last_constraint_modes: dict[str, str] = {}

    def __call__(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        agent_name: str = "agent",
        constraint_schema: dict[str, Any] | None = None,
        constraint_name: str | None = None,
    ) -> str:
        url = self.agent_urls.get(agent_name)
        if not url:
            raise RuntimeError(f"no HTTP worker URL configured for agent {agent_name!r}")
        body = {
            "message": prompt,
            "history": [],
            "show_thinking": False,
            "max_output_tokens": int(max_output_tokens),
        }
        if constraint_schema is not None:
            body["constraint_schema"] = constraint_schema
            body["constraint_name"] = str(constraint_name or f"{agent_name}_compact_json")
        timeout = self.per_agent_timeout_seconds.get(agent_name, self.timeout_seconds)
        response = self.post(
            url,
            json=body,
            timeout=timeout,
        )
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        payload = response.json()
        mode = "unsupported_fallback" if constraint_schema is not None else "disabled"
        if isinstance(payload, dict) and isinstance(payload.get("constraint_mode"), str):
            mode = payload["constraint_mode"]
        self.last_constraint_modes[agent_name] = mode
        return _response_payload_text(payload)


class CachedAgent:
    """Focused Gemma agent wrapper with a static instruction prefix."""

    def __init__(
        self,
        name: str,
        static_prompt: str,
        max_output_tokens: int,
        expected_fields: list[str],
        *,
        text_generator: TextGenerator | None = None,
        prompt_builder: PromptBuilder | None = None,
        constraint_builder: ConstraintBuilder | None = None,
        use_rewind_lock: bool = True,
    ) -> None:
        self.name = name
        self.static_prompt = static_prompt.strip()
        self.max_output_tokens = int(max_output_tokens)
        self.expected_fields = list(expected_fields)
        self.text_generator = text_generator
        self.prompt_builder = prompt_builder
        self.constraint_builder = constraint_builder
        self.lock = threading.Lock() if use_rewind_lock else _NullLock()
        self.metrics: dict[str, Any] = {
            "runs": 0,
            "last_duration_ms": 0,
            "last_output_chars": 0,
            "last_constraint_mode": "disabled",
        }

    @classmethod
    def from_prompt_file(
        cls,
        name: str,
        prompt_path: str | Path,
        max_output_tokens: int,
        expected_fields: list[str],
        *,
        text_generator: TextGenerator | None = None,
        constraint_builder: ConstraintBuilder | None = None,
    ) -> "CachedAgent":
        prompt = Path(prompt_path).read_text(encoding="utf-8").strip()
        return cls(
            name=name,
            static_prompt=prompt,
            max_output_tokens=max_output_tokens,
            expected_fields=expected_fields,
            text_generator=text_generator,
            constraint_builder=constraint_builder,
        )

    def build_prompt(self, dynamic_payload: dict[str, Any]) -> str:
        dynamic_json = json.dumps(dynamic_payload, ensure_ascii=True, separators=(",", ":"))
        static_prompt = self.prompt_builder(dynamic_payload).strip() if self.prompt_builder else self.static_prompt
        return f"{static_prompt}{INPUT_JSON_DELIMITER}{dynamic_json}"

    def run(self, dynamic_payload: dict[str, Any]) -> dict[str, Any]:
        if self.text_generator is None:
            raise RuntimeError(f"CachedAgent {self.name!r} has no text generator")

        prompt = self.build_prompt(dynamic_payload)
        constraint_schema = None
        constraint_name = None
        if self.constraint_builder is not None:
            constraint_schema = self.constraint_builder(self.name, dynamic_payload)
            if constraint_schema is not None:
                constraint_name = f"{self.name}_compact_json"
        start = time.perf_counter()
        with self.lock:
            if constraint_schema is None:
                raw = self.text_generator(
                    prompt,
                    max_output_tokens=self.max_output_tokens,
                    agent_name=self.name,
                )
            else:
                raw = self.text_generator(
                    prompt,
                    max_output_tokens=self.max_output_tokens,
                    agent_name=self.name,
                    constraint_schema=constraint_schema,
                    constraint_name=constraint_name,
                )
        duration_ms = int((time.perf_counter() - start) * 1000)
        constraint_mode = "json_schema" if constraint_schema is not None else "disabled"
        generator_modes = getattr(self.text_generator, "last_constraint_modes", None)
        if isinstance(generator_modes, dict) and isinstance(generator_modes.get(self.name), str):
            constraint_mode = generator_modes[self.name]
        self.metrics["runs"] = int(self.metrics.get("runs") or 0) + 1
        self.metrics["last_duration_ms"] = duration_ms
        self.metrics["last_output_chars"] = len(str(raw or ""))
        self.metrics["last_constraint_mode"] = constraint_mode
        return parse_json_strict_or_repair(str(raw or ""), self.expected_fields)


class _NullLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None
