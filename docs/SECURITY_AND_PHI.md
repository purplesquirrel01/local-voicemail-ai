# Security and sensitive data

This reference implementation makes no HIPAA compliance or certification claim.
Local hosting is an architectural choice, not proof of regulatory compliance.

The source includes mailbox authorization, hashed passwords, signed sessions,
CSRF protection, scoped integration tokens, request limits, and provider response
validation. Email, forwarding, and raw model-response logging default off.
The standard service entry points default to loopback addresses.

Audio, transcripts, extracted fields, verification records, databases, logs,
configuration, tokens, and backups can all contain sensitive information.
Development and bug reports must use fictional data. Never submit runtime files
or production screenshots to this repository or a hosted issue tracker.

Independent deployment work must address TLS, secret storage, host and mailbox
access, retention, encryption, monitoring, incident response, and operational
review. Private HTTP is not encrypted. Inference services and the optional API
must be authenticated before they handle sensitive data. See
[provider contracts](AI_PROVIDERS.md) and [API boundaries](INTEGRATION_API.md).

Model output can be wrong. Review flags and access to audio support human
judgment; they do not guarantee correctness. Read the [reporting policy](../SECURITY.md)
for handling a suspected vulnerability without disclosing private information.
