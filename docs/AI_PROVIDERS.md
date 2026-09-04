# AI provider contracts

The [provider interfaces](../voicemail_watcher/providers.py) separate primary
transcription (`faster-whisper`), semantic extraction (`litert-gemma`), and number
verification (`parakeet`). The application calls local HTTP services and validates
response structure, bounded timeouts, and readiness. Tokens are kept out of URLs
and redacted from provider representations.

The configurable HTTP adapter rejects endpoints outside loopback/private/local
address categories unless `LVT_ALLOW_EXTERNAL_PHI_PROVIDER` explicitly allows
them. That check is not a network firewall or a compliance guarantee. Keep
inference local and review transport and authentication before handling any
sensitive data. The repository does not install or configure providers.

The [provider tests](../tests/test_provider_contracts.py) exercise payloads,
timeouts, health failures, configuration rejection, and secret redaction with
mocked requests. [Service readiness tests](../tests/test_ready_authentication.py)
exercise authentication without loading weights.
