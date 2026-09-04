# Application architecture

Start with the processing diagram and source map in the [README](../README.md).

The watcher observes Asterisk voicemail metadata and matching audio, creates a
stable message identity, and persists processing state in SQLite. Its worker
pipeline handles retries, missing files, duplicate discoveries, and terminal
states. The shared spool and database modules isolate file naming and storage
details from the model interfaces.

Whisper provides the primary transcript and ASR evidence. The extraction
orchestrator combines deterministic candidates with constrained Gemma responses.
Parakeet supplies a second source of number evidence. Verification normalizes
values, validates response schemas, resolves agreement, and flags uncertain or
conflicting fields. Raw evidence remains distinct from provisional display text.

The portal reads the shared state and presents searchable, mailbox-authorized
messages, audio, review flags, and supported message actions. Its HTML, CSS, and
JavaScript are embedded in `voicemail_portal.py`; `voicemail_portal_app/` exposes
focused interfaces and the optional read-only integration API.

Services can communicate through HTTP when PBX-facing processing and inference
are separated. This source does not provision hosts, networks, service accounts,
TLS, models, or operating-system services. Those remain deployment responsibilities.

See [configuration](CONFIGURATION.md), [provider contracts](AI_PROVIDERS.md),
[security](SECURITY_AND_PHI.md), and [validation](VALIDATION.md).
