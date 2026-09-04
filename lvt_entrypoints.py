from __future__ import annotations

import os
import sys


def portal_main() -> int:
    from voicemail_portal import main

    return main(sys.argv)


def _run_uvicorn(target: str, host_env: str, port_env: str, default_port: int) -> int:
    import uvicorn

    host = os.environ.get(host_env, "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get(port_env, str(default_port)))
    uvicorn.run(target, host=host, port=port)
    return 0


def whisper_main() -> int:
    return _run_uvicorn("whisper_server:app", "WHISPER_HOST", "WHISPER_PORT", 8765)


def parakeet_main() -> int:
    return _run_uvicorn("parakeet_server:app", "PARAKEET_HOST", "PARAKEET_PORT", 8766)


def gemma_main() -> int:
    return _run_uvicorn("litert_chat_web:app", "LITERT_HOST", "LITERT_PORT", 8787)
