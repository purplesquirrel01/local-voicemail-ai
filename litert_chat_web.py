#!/usr/bin/env python3
import ctypes
import inspect
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from extraction_orchestrator import (
    build_default_agents,
    caller_id_correction_agent_enabled,
    candidate_agent_e2b_scout_enabled,
    candidate_agent_execution_mode,
    candidate_agent_health,
    candidate_agent_mode,
    candidate_agent_topology,
    candidate_agent_selection,
    extract_with_candidate_agents,
    fallback_to_legacy_enabled,
    parse_extraction_request_from_prompt,
    record_legacy_fallback,
    spelling_correction_agent_enabled,
)
from final_resolver import FINAL_SCHEMA_KEYS, empty_final_json
from gemma_agents import HttpAgentTextGenerator
from json_utils import parse_json_strict_or_repair

try:
    import litert_lm
except Exception:  # pragma: no cover - optional model runtime
    litert_lm = None  # type: ignore[assignment]

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from litert_lm.conversation import Conversation
except Exception:  # pragma: no cover - depends on installed LiteRT-LM wheel
    Conversation = None  # type: ignore[assignment]

try:
    from litert_lm.utils import _sampler_config_to_params
except Exception:  # pragma: no cover - depends on installed LiteRT-LM wheel
    _sampler_config_to_params = None  # type: ignore[assignment]


MAX_NUM_TOKENS = int(os.environ.get("LITERT_MAX_NUM_TOKENS", "8192"))
SERVER_MAX_TOKENS = int(os.environ.get("LITERT_SERVER_MAX_TOKENS", "1024"))
CLIENT_MAX_TOKENS = int(os.environ.get("LITERT_CLIENT_MAX_TOKENS", str(MAX_NUM_TOKENS)))
TEMPERATURE = float(os.environ.get("LITERT_TEMPERATURE", "0.0"))
TOP_K = int(os.environ.get("LITERT_TOP_K", "1"))
TOP_P = float(os.environ.get("LITERT_TOP_P", "1.0"))
EXAMPLE = int(os.environ.get("LITERT_EXAMPLE", "0"))
MODEL_PATH = Path(os.environ.get("LITERT_MODEL_PATH", "./gemma-4-E4B-it.litertlm"))
CACHE_DIR = Path(os.environ.get("LITERT_CACHE_DIR", "./litert-cache"))
LITERT_API_KEY = os.environ.get("LITERT_API_KEY", "").strip()
LITERT_REQUIRE_AUTH = os.environ.get(
    "LITERT_REQUIRE_AUTH",
    "true" if LITERT_API_KEY else "false",
).strip().lower() in {"1", "true", "yes", "y", "on"}
LITERT_MAX_INPUT_CHARS = int(os.environ.get("LITERT_MAX_INPUT_CHARS", "12000"))
LITERT_ORCHESTRATOR_ONLY = os.environ.get("LITERT_ORCHESTRATOR_ONLY", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}


SYSTEM_PROMPT = """You are a helpful local assistant running privately on the user's server.

Rules:
- Be concise and practical.
- For medical voicemail transcript review, do not invent facts.
- Preserve phone numbers exactly.
- If asked for JSON, return valid JSON only.
"""

REASONING_PROMPT = """You are a helpful local assistant running privately on the user's server.

You must return valid JSON only using this exact schema:
{
  "reasoning_summary": "",
  "answer": "",
  "confidence": 0.0
}

Rules for reasoning_summary:
- Provide a concise user-visible explanation of how you approached the answer.
- Do not reveal hidden chain-of-thought.
- Do not write a long private scratchpad.
- Keep it short, practical, and audit-friendly.
- For voicemail or healthcare text, mention only facts present in the input.

Rules for answer:
- Give the final response to the user.
- Be concise and practical.
- Preserve phone numbers exactly.
- Do not invent facts.
- Do not include markdown unless the user specifically asks for markdown.

Rules for confidence:
- Use a number from 0.0 to 1.0.
- Use lower confidence when the input is ambiguous or incomplete.
"""


app = FastAPI(title="LiteRT Gemma Chat")
logger = logging.getLogger("litert_chat_web")

engine: Any = None
engine_lock = threading.Lock()
engine_loaded_at: float | None = None
SESSION_MODE = "unknown"
request_lock = threading.Lock()
active_requests = 0
completed_requests = 0
candidate_agents_lock = threading.Lock()
candidate_agents_cache: dict[str, Any] | None = None
last_generation_constraint_mode = "disabled"
last_generation_constraint_name = ""
last_generation_constraint_supported = False


def request_stats() -> dict[str, int]:
    with request_lock:
        return {
            "in_flight": active_requests,
            "active_requests": active_requests,
            "completed_requests": completed_requests,
        }


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def shadow_legacy_max_output_tokens() -> int:
    return env_int(
        "CANDIDATE_SHADOW_LEGACY_MAX_OUTPUT_TOKENS",
        min(SERVER_MAX_TOKENS, 1024),
        minimum=64,
    )


def candidate_agent_worker_urls() -> dict[str, str]:
    urls = {
        "candidate_scout": os.environ.get(
            "CANDIDATE_AGENT_E2B_SCOUT_URL",
            "http://127.0.0.1:8794/api/chat" if candidate_agent_e2b_scout_enabled() else "",
        ).strip(),
        "numbers": os.environ.get("CANDIDATE_AGENT_NUMBERS_URL", "").strip(),
        "identity": os.environ.get("CANDIDATE_AGENT_IDENTITY_URL", "").strip(),
        "name": os.environ.get("CANDIDATE_AGENT_NAME_URL", "").strip(),
        "subject_name": os.environ.get("CANDIDATE_AGENT_SUBJECT_NAME_URL", "").strip(),
        "caller_name_fallback": os.environ.get("CANDIDATE_AGENT_CALLER_NAME_FALLBACK_URL", "").strip(),
        "name_correction": os.environ.get("CANDIDATE_AGENT_NAME_CORRECTION_URL", "").strip(),
        "spelling_correction": os.environ.get("CANDIDATE_AGENT_SPELLING_CORRECTION_URL", "").strip(),
        "caller_id_correction": os.environ.get("CANDIDATE_AGENT_CALLER_ID_CORRECTION_URL", "").strip(),
        "dob": os.environ.get("CANDIDATE_AGENT_DOB_URL", "").strip(),
    }
    if not spelling_correction_agent_enabled():
        urls["spelling_correction"] = ""
    if not caller_id_correction_agent_enabled():
        urls["caller_id_correction"] = ""
    return {name: url for name, url in urls.items() if url}


def candidate_agent_http_timeout_seconds() -> int:
    return env_int("CANDIDATE_AGENT_HTTP_TIMEOUT_SECONDS", 420, minimum=1)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def litert_constrained_decoding_enabled() -> bool:
    return env_bool("LITERT_CONSTRAINED_DECODING", False)


def litert_constrained_decoding_required() -> bool:
    return env_bool("LITERT_CONSTRAINED_DECODING_REQUIRE", False)


def mark_request_started() -> None:
    global active_requests
    with request_lock:
        active_requests += 1


def mark_request_finished() -> None:
    global active_requests, completed_requests
    with request_lock:
        active_requests = max(0, active_requests - 1)
        completed_requests += 1


def authorize_request(authorization: str | None = None, x_api_key: str | None = None) -> None:
    if not LITERT_REQUIRE_AUTH:
        return
    if not LITERT_API_KEY:
        raise HTTPException(status_code=500, detail="LiteRT API auth is required but LITERT_API_KEY is not set")
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    supplied = (x_api_key or bearer or "").strip()
    if not supplied or not secrets.compare_digest(supplied, LITERT_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def validate_request_size(message: str) -> None:
    if len(message or "") > LITERT_MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail="Message is too large")


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)
    show_thinking: bool = False
    max_output_tokens: int | None = None
    constraint_schema: dict[str, Any] | None = None
    constraint_name: str | None = None


class GenerateRequest(BaseModel):
    model: str | None = None
    prompt: str
    stream: bool = False
    format: Any = None
    options: dict[str, Any] = Field(default_factory=dict)


def make_sampler_config() -> Any:
    if litert_lm is None:
        return None
    try:
        return litert_lm.SamplerConfig(
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            example=EXAMPLE,
        )
    except Exception:
        return None


def to_litert_message(role: str, text: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }


def response_text(response: Any) -> str:
    if response is None:
        return ""

    if isinstance(response, str):
        return response

    texts = getattr(response, "texts", None)
    if isinstance(texts, list):
        return "".join(str(item) for item in texts)

    if isinstance(response, dict):
        content = response.get("content", [])
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            content = [content]
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)

    return str(response or "")


def _method_accepts_keyword(method: Any, keywords: set[str]) -> bool:
    try:
        signature = inspect.signature(method)
    except Exception:
        return False
    return any(keyword in signature.parameters for keyword in keywords)


def litert_constrained_decoding_supported(conversation: Any | None = None) -> bool:
    if conversation is None:
        return False
    send_message = getattr(conversation, "send_message_async", None)
    if not callable(send_message):
        return False
    return _method_accepts_keyword(send_message, {"decoding_constraint", "constraint", "constraint_schema"})


def _json_schema_constraint(schema: dict[str, Any], name: str | None) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": str(name or "agent_compact_json"),
        "schema": schema,
    }


def send_message_async_with_constraints(conversation: Any, req: ChatRequest):
    global last_generation_constraint_mode, last_generation_constraint_name, last_generation_constraint_supported

    schema = req.constraint_schema if isinstance(req.constraint_schema, dict) else None
    last_generation_constraint_name = str(req.constraint_name or "")
    last_generation_constraint_supported = litert_constrained_decoding_supported(conversation)
    if schema is None or not litert_constrained_decoding_enabled():
        last_generation_constraint_mode = "disabled"
        return conversation.send_message_async(req.message)

    send_message = conversation.send_message_async
    if not last_generation_constraint_supported:
        if litert_constrained_decoding_required():
            last_generation_constraint_mode = "unsupported_required"
            raise RuntimeError(
                "LiteRT constrained decoding was required but this runtime does not expose "
                "a supported JSON Schema constraint API"
            )
        last_generation_constraint_mode = "unsupported_fallback"
        return send_message(req.message)

    constraint = _json_schema_constraint(schema, req.constraint_name)
    try:
        last_generation_constraint_mode = "json_schema"
        if _method_accepts_keyword(send_message, {"decoding_constraint"}):
            return send_message(req.message, decoding_constraint=constraint)
        if _method_accepts_keyword(send_message, {"constraint"}):
            return send_message(req.message, constraint=constraint)
        if _method_accepts_keyword(send_message, {"constraint_schema"}):
            return send_message(req.message, constraint_schema=schema)
    except TypeError as exc:
        if litert_constrained_decoding_required():
            last_generation_constraint_mode = "unsupported_required"
            raise RuntimeError("LiteRT constrained decoding failed through the exposed runtime API") from exc

    if litert_constrained_decoding_required():
        last_generation_constraint_mode = "unsupported_required"
        raise RuntimeError("LiteRT constrained decoding was required but no supported call path succeeded")
    last_generation_constraint_mode = "unsupported_fallback"
    return send_message(req.message)


def create_session_signature() -> str | None:
    if engine is None or not hasattr(engine, "create_session"):
        return None
    try:
        return str(inspect.signature(engine.create_session))
    except Exception:
        return "(*args, **kwargs)"


def create_conversation_signature() -> str | None:
    if engine is None or not hasattr(engine, "create_conversation"):
        return None
    try:
        return str(inspect.signature(engine.create_conversation))
    except Exception:
        return "(*args, **kwargs)"


def c_has_max_output_setter() -> bool:
    if engine is not None:
        lib = getattr(engine, "_lib", None)
        if lib is not None and hasattr(lib, "litert_lm_session_config_set_max_output_tokens"):
            return True

    try:
        from litert_lm._ffi import _get_lib

        return hasattr(_get_lib(), "litert_lm_session_config_set_max_output_tokens")
    except Exception:
        return False


def build_conversation_messages(req: ChatRequest) -> list[dict[str, Any]]:
    # Watcher requests are already complete extraction prompts and normally use
    # history=[] / show_thinking=false. Leave those prompts untouched.
    if not req.history and not req.show_thinking:
        return []

    system_prompt = REASONING_PROMPT if req.show_thinking else SYSTEM_PROMPT
    messages = [to_litert_message("system", system_prompt)]
    for item in req.history[-20:]:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "")
        if role in {"user", "assistant"} and content:
            messages.append(to_litert_message(role, content))
    return messages


def create_conversation_ffi(
    messages: list[dict[str, Any]],
    extra_context: dict[str, Any],
    sampler_config: Any,
    max_output_tokens: int,
):
    if Conversation is None:
        raise RuntimeError("litert_lm.conversation.Conversation is unavailable")

    lib = getattr(engine, "_lib", None)
    engine_ptr = getattr(engine, "_engine_ptr", None)
    if lib is None or engine_ptr is None:
        raise RuntimeError("LiteRT engine internals are unavailable for conversation FFI fallback")

    set_max_output = getattr(lib, "litert_lm_session_config_set_max_output_tokens", None)
    if not callable(set_max_output):
        raise RuntimeError("LiteRT C library does not expose litert_lm_session_config_set_max_output_tokens")

    try:
        lib.litert_lm_session_config_create.restype = ctypes.c_void_p
        lib.litert_lm_session_config_delete.argtypes = [ctypes.c_void_p]
        set_max_output.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.litert_lm_conversation_config_create.restype = ctypes.c_void_p
        lib.litert_lm_conversation_config_delete.argtypes = [ctypes.c_void_p]
        lib.litert_lm_conversation_config_set_session_config.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.litert_lm_conversation_config_set_messages.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.litert_lm_conversation_config_set_extra_context.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.litert_lm_conversation_create.restype = ctypes.c_void_p
        lib.litert_lm_conversation_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    except Exception:
        pass

    session_config = lib.litert_lm_session_config_create()
    if not session_config:
        raise RuntimeError("Failed to create LiteRT-LM session config")

    conv_config = None
    try:
        if sampler_config is not None and _sampler_config_to_params is not None:
            try:
                params = _sampler_config_to_params(sampler_config)
                lib.litert_lm_session_config_set_sampler_params(session_config, ctypes.byref(params))
            except Exception:
                pass

        set_max_output(session_config, int(max_output_tokens))

        conv_config = lib.litert_lm_conversation_config_create()
        if not conv_config:
            raise RuntimeError("Failed to create LiteRT-LM conversation config")

        lib.litert_lm_conversation_config_set_session_config(conv_config, session_config)
        lib.litert_lm_session_config_delete(session_config)
        session_config = None

        if messages:
            lib.litert_lm_conversation_config_set_messages(conv_config, json.dumps(messages).encode("utf-8"))
        if extra_context:
            lib.litert_lm_conversation_config_set_extra_context(
                conv_config,
                json.dumps(extra_context).encode("utf-8"),
            )

        conv_ptr = lib.litert_lm_conversation_create(engine_ptr, conv_config)
    finally:
        if conv_config:
            lib.litert_lm_conversation_config_delete(conv_config)
        if session_config:
            lib.litert_lm_session_config_delete(session_config)

    if not conv_ptr:
        raise RuntimeError("Failed to create LiteRT-LM conversation")

    return Conversation(
        lib,
        conv_ptr,
        engine=engine,
        messages=messages,
        tools=[],
        tools_map={},
        tool_event_handler=None,
        automatic_tool_calling=True,
        extra_context=extra_context,
        sampler_config=sampler_config,
    )


def create_conversation_with_output_limit(
    messages: list[dict[str, Any]],
    *,
    max_output_tokens: int | None = None,
):
    global SESSION_MODE

    output_limit = int(max_output_tokens or SERVER_MAX_TOKENS)
    extra_context = {
        "client_max_tokens": CLIENT_MAX_TOKENS,
        "server_max_tokens": output_limit,
    }
    sampler_config = make_sampler_config()

    try:
        conversation = engine.create_conversation(
            messages=messages,
            extra_context=extra_context,
            sampler_config=sampler_config,
            max_output_tokens=output_limit,
        )
        SESSION_MODE = "conversation_native_max_output_tokens"
        return conversation
    except TypeError:
        pass

    conversation = create_conversation_ffi(messages, extra_context, sampler_config, output_limit)
    SESSION_MODE = "conversation_ffi_max_output_tokens"
    return conversation


def generate_text(req: ChatRequest, *, max_output_tokens: int | None = None) -> str:
    messages = build_conversation_messages(req)
    with create_conversation_with_output_limit(messages, max_output_tokens=max_output_tokens) as conversation:
        parts = []
        for chunk in send_message_async_with_constraints(conversation, req):
            parts.append(response_text(chunk))
    return "".join(parts).strip()


def run_generate_text_locked(req: ChatRequest, *, max_output_tokens: int | None = None) -> str:
    with engine_lock:
        return generate_text(req, max_output_tokens=max_output_tokens)


def generate_agent_text(
    prompt: str,
    *,
    max_output_tokens: int,
    agent_name: str = "agent",
    constraint_schema: dict[str, Any] | None = None,
    constraint_name: str | None = None,
) -> str:
    text = run_generate_text_locked(
        ChatRequest(
            message=prompt,
            history=[],
            show_thinking=False,
            constraint_schema=constraint_schema,
            constraint_name=constraint_name,
        ),
        max_output_tokens=max_output_tokens,
    )
    modes = getattr(generate_agent_text, "last_constraint_modes", None)
    if not isinstance(modes, dict):
        modes = {}
        setattr(generate_agent_text, "last_constraint_modes", modes)
    modes[agent_name] = last_generation_constraint_mode
    print(
        "LiteRT focused agent generated "
        f"agent={agent_name} "
        f"response_chars={len(text)} "
        f"prompt_chars={len(prompt)} "
        f"max_output_tokens={max_output_tokens} "
        f"max_num_tokens={MAX_NUM_TOKENS} "
        f"constraint_mode={last_generation_constraint_mode} "
        f"api={SESSION_MODE}",
        flush=True,
    )
    return text


def get_candidate_agents() -> dict[str, Any]:
    global candidate_agents_cache

    with candidate_agents_lock:
        if candidate_agents_cache is None:
            if candidate_agent_execution_mode() == "parallel_http":
                text_generator = HttpAgentTextGenerator(
                    candidate_agent_worker_urls(),
                    timeout_seconds=candidate_agent_http_timeout_seconds(),
                    per_agent_timeout_seconds={
                        "candidate_scout": env_int("CANDIDATE_AGENT_E2B_SCOUT_TIMEOUT_SECONDS", 90, minimum=1),
                    },
                )
            else:
                text_generator = generate_agent_text
            candidate_agents_cache = build_default_agents(text_generator=text_generator)
        return candidate_agents_cache


def candidate_agents_loaded_names() -> list[str]:
    if candidate_agents_cache is not None:
        return sorted(candidate_agents_cache.keys())

    prompt_dir = Path(__file__).resolve().parent / "prompts"
    topology = candidate_agent_topology()
    if topology == "custom":
        capability_names = {
            "numbers": ("numbers",),
            "name": ("name",),
            "dob": ("dob",),
            "subject_fallback": ("subject_name", "caller_name_fallback"),
            "spelling_correction": ("spelling_correction",),
            "caller_id_correction": ("caller_id_correction",),
        }
        return sorted(
            name
            for capability in candidate_agent_selection()
            for name in capability_names[capability]
            if (prompt_dir / f"{name}_agent.md").exists()
        )
    if topology == "scout_subject_general_fallback":
        names = [
            agent_name
            for agent_name, filename in (
                ("candidate_scout", "candidate_scout_agent.md"),
                ("numbers", "numbers_agent.md"),
                ("dob", "dob_agent.md"),
                ("subject_name", "subject_name_agent.md"),
                ("name", "name_agent.md"),
                ("caller_name_fallback", "caller_name_fallback_agent.md"),
            )
            if (prompt_dir / filename).exists()
        ]
        return sorted(names)

    names = []
    if (prompt_dir / "numbers_agent.md").exists():
        names.append("numbers")
    if topology == "numbers_only":
        return names
    if candidate_agent_e2b_scout_enabled() and (prompt_dir / "candidate_scout_agent.md").exists():
        names.append("candidate_scout")
    if topology in {
        "split_identity",
        "split_identity_correction",
        "split_identity_dual_correction",
        "split_identity_subject_fallback_dual_correction",
    }:
        if topology == "split_identity_subject_fallback_dual_correction":
            if (prompt_dir / "subject_name_agent.md").exists():
                names.append("subject_name")
            if (prompt_dir / "caller_name_fallback_agent.md").exists():
                names.append("caller_name_fallback")
        elif (prompt_dir / "name_agent.md").exists():
            names.append("name")
        if topology == "split_identity_correction" and (prompt_dir / "name_correction_agent.md").exists():
            names.append("name_correction")
        if topology in {
            "split_identity_dual_correction",
            "split_identity_subject_fallback_dual_correction",
        }:
            if spelling_correction_agent_enabled() and (prompt_dir / "spelling_correction_agent.md").exists():
                names.append("spelling_correction")
            if caller_id_correction_agent_enabled() and (prompt_dir / "caller_id_correction_agent.md").exists():
                names.append("caller_id_correction")
        if (prompt_dir / "dob_agent.md").exists():
            names.append("dob")
    elif (prompt_dir / "identity_agent.md").exists():
        names.append("identity")
    return names


def parse_legacy_text_for_shadow(legacy_text: str) -> dict[str, Any]:
    try:
        return parse_json_strict_or_repair(legacy_text, FINAL_SCHEMA_KEYS)
    except Exception:
        return empty_final_json()


def generate_candidate_or_shadow_text(
    req: ChatRequest,
    input_payload: dict[str, Any],
    trace_sink=None,
) -> str:
    mode = candidate_agent_mode()
    if mode == "shadow_candidate_agents":
        legacy_text = run_generate_text_locked(req, max_output_tokens=shadow_legacy_max_output_tokens())
        legacy_obj = parse_legacy_text_for_shadow(legacy_text)
        try:
            extract_with_candidate_agents(
                input_payload,
                legacy_extractor=lambda _request: legacy_obj,
                agents=get_candidate_agents(),
                mode="shadow_candidate_agents",
                fallback_to_legacy=False,
                trace_sink=trace_sink,
            )
        except Exception as exc:
            logger.warning("Candidate-agent shadow route failed error=%s", exc)
        return legacy_text

    if mode == "candidate_agents":
        try:
            final = extract_with_candidate_agents(
                input_payload,
                agents=get_candidate_agents(),
                mode="candidate_agents",
                fallback_to_legacy=False,
                trace_sink=trace_sink,
            )
            return json.dumps(final, ensure_ascii=True)
        except Exception as exc:
            logger.warning("Candidate-agent active route failed error=%s", exc)
            if fallback_to_legacy_enabled():
                record_legacy_fallback()
                if LITERT_ORCHESTRATOR_ONLY:
                    raise RuntimeError("legacy fallback is unavailable in orchestrator-only mode") from exc
                return run_generate_text_locked(req)
            raise

    return run_generate_text_locked(req)


def request_max_output_tokens(req: ChatRequest) -> int | None:
    value = getattr(req, "max_output_tokens", None)
    if value is None:
        return None
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return None


def extract_json_object(text: str) -> dict:
    if not text:
        raise ValueError("empty response")
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    in_string = False
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("incomplete JSON object")


@app.on_event("startup")
def startup() -> None:
    global engine, engine_loaded_at, SESSION_MODE

    if LITERT_ORCHESTRATOR_ONLY:
        engine = None
        engine_loaded_at = None
        SESSION_MODE = "orchestrator_only"
        print("LiteRT orchestrator-only startup; model load skipped", flush=True)
        return

    if litert_lm is None:
        raise RuntimeError("Missing optional dependency: litert_lm")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH.resolve()}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    load_start = time.perf_counter()
    engine = litert_lm.Engine(
        str(MODEL_PATH),
        backend=litert_lm.Backend.CPU,
        max_num_tokens=MAX_NUM_TOKENS,
        cache_dir=str(CACHE_DIR),
    )
    engine_loaded_at = time.perf_counter() - load_start
    print(
        "LiteRT model loaded in "
        f"{engine_loaded_at:.2f} seconds "
        f"max_num_tokens={MAX_NUM_TOKENS} "
        f"max_output_tokens={SERVER_MAX_TOKENS} "
        f"c_has_max_output_setter={c_has_max_output_setter()}",
        flush=True,
    )


@app.on_event("shutdown")
def shutdown() -> None:
    global engine
    if engine is not None:
        try:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        finally:
            engine = None


@app.get("/health")
def health() -> dict:
    model_loaded = engine is not None
    payload = {
        "status": "ok" if model_loaded or LITERT_ORCHESTRATOR_ONLY else "loading",
        "model_path": str(MODEL_PATH),
        "backend": "orchestrator" if LITERT_ORCHESTRATOR_ONLY else "CPU",
        "orchestrator_only": LITERT_ORCHESTRATOR_ONLY,
        "model_loaded": model_loaded,
        "candidate_agent_worker_urls": candidate_agent_worker_urls(),
        "engine_loaded_seconds": engine_loaded_at,
        "max_num_tokens": MAX_NUM_TOKENS,
        "max_output_tokens": SERVER_MAX_TOKENS,
        "client_max_tokens_compat": CLIENT_MAX_TOKENS,
        "session_mode_last_used": SESSION_MODE,
        "create_session_signature": create_session_signature(),
        "create_conversation_signature": create_conversation_signature(),
        "c_has_max_output_setter": c_has_max_output_setter(),
        "auth_required": LITERT_REQUIRE_AUTH,
        "litert_lm_available": litert_lm is not None,
        "litert_constrained_decoding_enabled": litert_constrained_decoding_enabled(),
        "litert_constrained_decoding_supported": last_generation_constraint_supported,
        "litert_constrained_decoding_required": litert_constrained_decoding_required(),
        "last_generation_constraint_mode": last_generation_constraint_mode,
        "last_generation_constraint_name": last_generation_constraint_name,
        "sampler": {
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "example": EXAMPLE,
        },
        "load": request_stats(),
    }
    payload.update(
        candidate_agent_health(
            agents_loaded=candidate_agents_loaded_names(),
        )
    )
    return payload


@app.get("/ready")
def ready(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict:
    authorize_request(authorization, x_api_key)
    if LITERT_ORCHESTRATOR_ONLY:
        return {
            "status": "ready",
            "orchestrator_only": True,
            "model_loaded": False,
            "load": request_stats(),
            "candidate_agent_worker_urls": candidate_agent_worker_urls(),
        }
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine is not loaded yet.")
    return {
        "status": "ready",
        "model_path": str(MODEL_PATH),
        "load": request_stats(),
        "max_num_tokens": MAX_NUM_TOKENS,
        "max_output_tokens": SERVER_MAX_TOKENS,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LiteRT Gemma Chat</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #111827; color: #f9fafb; margin: 0; }
    .wrap { max-width: 980px; margin: 0 auto; padding: 24px; }
    #chat { background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 18px; min-height: 520px; max-height: 68vh; overflow-y: auto; }
    .msg { margin: 12px 0; padding: 12px 14px; border-radius: 10px; white-space: pre-wrap; line-height: 1.45; overflow-wrap: anywhere; }
    .user { background: #2563eb; margin-left: 80px; }
    .assistant { background: #374151; margin-right: 80px; }
    .thinking { background: #3f3f46; margin-right: 120px; border-left: 4px solid #f59e0b; }
    .error { background: #7f1d1d; margin-right: 80px; }
    .meta { font-size: 12px; opacity: .75; margin-bottom: 4px; font-weight: 700; }
    .controls { display: flex; align-items: center; gap: 14px; margin: 12px 0 8px; color: #d1d5db; font-size: 14px; }
    .bar { display: flex; gap: 10px; margin-top: 10px; }
    textarea { flex: 1; min-height: 76px; border-radius: 10px; border: 1px solid #4b5563; background: #030712; color: #f9fafb; padding: 12px; font-size: 15px; resize: vertical; }
    button { border: 0; border-radius: 10px; padding: 0 20px; background: #10b981; color: #052e1f; font-weight: 800; cursor: pointer; min-width: 90px; }
    button.secondary { background: #4b5563; color: #f9fafb; }
    button:disabled { opacity: .5; cursor: wait; }
    .small { margin-top: 10px; color: #9ca3af; font-size: 13px; }
    .confidence { color: #fbbf24; font-size: 12px; margin-top: 6px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>LiteRT Gemma Chat</h1>
    <div class="small">Running locally on LiteRT Gemma.</div>
    <div id="chat"></div>
    <div class="controls">
      <label><input type="checkbox" id="showThinking" /> Show reasoning summary</label>
      <button class="secondary" onclick="clearChat()" type="button">Clear</button>
    </div>
    <div class="bar">
      <textarea id="input" placeholder="Type a message. Shift+Enter for newline. Enter to send."></textarea>
      <button id="send" onclick="sendMessage()" type="button">Send</button>
    </div>
  </div>
<script>
let history = [];
function addMessage(role, text, reasoningSummary="", confidence=null) {
  const chat = document.getElementById("chat");
  if (role === "assistant" && reasoningSummary && reasoningSummary.trim()) {
    const thinkDiv = document.createElement("div");
    thinkDiv.className = "msg thinking";
    thinkDiv.innerHTML = '<div class="meta">Reasoning summary</div>';
    const thinkBody = document.createElement("div");
    thinkBody.textContent = reasoningSummary;
    thinkDiv.appendChild(thinkBody);
    if (confidence !== null && confidence !== undefined && confidence !== "") {
      const conf = document.createElement("div");
      conf.className = "confidence";
      conf.textContent = "Confidence: " + confidence;
      thinkDiv.appendChild(conf);
    }
    chat.appendChild(thinkDiv);
  }
  const div = document.createElement("div");
  div.className = "msg " + role;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "user" ? "You" : role === "error" ? "Error" : "Gemma";
  const body = document.createElement("div");
  body.textContent = text;
  div.appendChild(meta);
  div.appendChild(body);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}
function clearChat() {
  history = [];
  document.getElementById("chat").innerHTML = "";
  document.getElementById("input").focus();
}
async function sendMessage() {
  const input = document.getElementById("input");
  const button = document.getElementById("send");
  const showThinking = document.getElementById("showThinking").checked;
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  button.disabled = true;
  addMessage("user", message);
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message, history, show_thinking: showThinking})
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const answer = data.response || data.content || data.text || "";
    addMessage("assistant", answer, data.reasoning_summary || "", data.confidence);
    history.push({role: "user", content: message});
    history.push({role: "assistant", content: answer});
    if (history.length > 20) history = history.slice(history.length - 20);
  } catch (err) {
    addMessage("error", err.message);
  } finally {
    button.disabled = false;
    input.focus();
  }
}
document.getElementById("input").addEventListener("keydown", function(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
window.onload = function() { document.getElementById("input").focus(); };
</script>
</body>
</html>
"""


@app.post("/api/chat")
def chat(
    req: ChatRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict:
    authorize_request(authorization, x_api_key)

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty.")
    validate_request_size(req.message)

    mark_request_started()
    try:
        extraction_payload = None
        if not req.history and not req.show_thinking and candidate_agent_mode() != "legacy":
            extraction_payload = parse_extraction_request_from_prompt(req.message)

        if LITERT_ORCHESTRATOR_ONLY:
            if extraction_payload is None:
                raise HTTPException(
                    status_code=503,
                    detail="Orchestrator-only mode only supports extraction prompts.",
                )
        elif engine is None:
            raise HTTPException(status_code=503, detail="Model engine is not loaded yet.")

        if extraction_payload is not None:
            text = generate_candidate_or_shadow_text(req, extraction_payload)
        else:
            text = run_generate_text_locked(req, max_output_tokens=request_max_output_tokens(req))

        print(
            "LiteRT generated "
            f"response_chars={len(text)} "
            f"prompt_chars={len(req.message)} "
            f"max_output_tokens={request_max_output_tokens(req) or SERVER_MAX_TOKENS} "
            f"max_num_tokens={MAX_NUM_TOKENS} "
            f"api={SESSION_MODE}",
            flush=True,
        )

        if req.show_thinking:
            try:
                parsed = extract_json_object(text)
                reasoning_summary = str(parsed.get("reasoning_summary", "")).strip()
                answer = str(parsed.get("answer", "")).strip() or text
                confidence = parsed.get("confidence", "")
                response = {
                    "reasoning_summary": reasoning_summary,
                    "response": answer,
                    "message": {"role": "assistant", "content": answer},
                    "content": answer,
                    "text": answer,
                    "confidence": confidence,
                    "raw": text,
                }
                if req.constraint_schema is not None:
                    response["constraint_mode"] = last_generation_constraint_mode
                return response
            except Exception:
                pass

        response = {
            "reasoning_summary": "",
            "response": text,
            "message": {"role": "assistant", "content": text},
            "content": text,
            "text": text,
            "confidence": "",
            "raw": text,
        }
        if req.constraint_schema is not None:
            response["constraint_mode"] = last_generation_constraint_mode
        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        mark_request_finished()


@app.post("/api/generate")
def generate_ollama_compat(
    req: GenerateRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict:
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported")
    response = chat(
        ChatRequest(message=req.prompt, history=[], show_thinking=False),
        authorization=authorization,
        x_api_key=x_api_key,
    )
    text = str(response.get("response") or response.get("text") or "")
    return {
        "model": req.model or "litert",
        "response": text,
        "done": True,
        "context": [],
    }
