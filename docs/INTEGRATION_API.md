# Read-only integration API

The optional `/api/v1` application exposes normalized voicemail data separately
from the staff-facing portal. It is disabled by default. This reference includes
the API and its synthetic tests, without administrative token-provisioning tools.

The [implementation](../voicemail_portal_app/api_v1.py) validates bearer-token
digests, expiration, scopes, mailbox access, and per-token request limits. Its
SQLite backend opens the database read-only and applies mailbox restrictions in
the query. Responses omit internal paths, raw errors, and model evidence.
Missing or out-of-scope messages return indistinguishable not-found responses.

The [API tests](../tests/test_api_v1.py) demonstrate enabled/disabled behavior,
authentication, pagination, rate limiting, and mailbox isolation using temporary
synthetic state and in-process clients. They do not need a listener or real token.

Runtime activation requires separately managed settings and a protected token
file following the implementation's schema. No usable token or production token
file is included. See [security](SECURITY_AND_PHI.md).
