# Reference configuration

Application settings come from environment variables. The examples below are
for understanding and synthetic development, not complete deployment recipes.
No installer or environment-file loader is included. Do not connect a development
run to a production spool or database.

| Area | Representative settings | Source |
| --- | --- | --- |
| Watcher | `VOICEMAIL_WATCH_DIR`, `VOICEMAIL_STATE_DB`, `WHISPER_URL` | [watcher.py](../watcher.py) |
| Gemma extraction | `GEMMA_BASE_URL`, `GEMMA_API_KEY`, `GEMMA_EXTRACT_MODE` | [watcher.py](../watcher.py), [orchestrator](../extraction_orchestrator.py) |
| Number verification | `PARAKEET_VERIFICATION_URL`, `PARAKEET_API_KEY` | [watcher.py](../watcher.py) |
| Portal | `VOICEMAIL_PORTAL_USERS_FILE`, `VOICEMAIL_PORTAL_SESSION_SECRET`, `VOICEMAIL_PORTAL_HOST` | [voicemail_portal.py](../voicemail_portal.py) |
| Whisper service | `WHISPER_MODEL`, `WHISPER_LOCAL_FILES_ONLY`, `WHISPER_REQUIRE_AUTH` | [whisper_server.py](../whisper_server.py) |
| Parakeet service | `PARAKEET_MODEL_NAME`, `PARAKEET_REQUIRE_AUTH` | [parakeet_server.py](../parakeet_server.py) |
| Gemma service | `LITERT_MODEL_PATH`, `LITERT_REQUIRE_AUTH`, `LITERT_ORCHESTRATOR_ONLY` | [litert_chat_web.py](../litert_chat_web.py) |

[application.env.example](examples/application.env.example) illustrates loopback
endpoints and relative synthetic paths. Its secret fields are intentionally empty
and it is not a ready-to-run configuration. A complete inference setup also needs
local models, authentication, matching worker endpoints, and the intended agent
mode. Tests configure these boundaries with fixtures rather than starting services.

[voicemail_portal_users.example.json](../voicemail_portal_users.example.json)
shows a fictional mailbox-limited user. It contains a placeholder, not a usable
password hash. Runtime users and secrets belong outside source control.

The source package exposes `lvt-watcher`, `lvt-portal`, `lvt-whisper-api`,
`lvt-parakeet-api`, and `lvt-gemma-api`. These launch applications; they do not
install services. Inspect their configuration before use. The AI runtime extra
is optional and separate from the small model-free development installation.
