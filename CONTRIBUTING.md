# Contributing

Keep changes focused on the reference application's ingestion, model contracts,
deterministic verification, storage, and human-review workflow. Explain the
behavioral change and preserve meaningful regression coverage.

Use fictional names, reserved telephone numbers, example domains, and generated
audio where necessary. Do not submit runtime state, production-derived scenarios,
credentials, model weights, or screenshots from a real deployment.

Use the development setup in the [README](README.md), then run every gate in
[the validation guide](docs/VALIDATION.md). A passing fixture suite checks software
behavior; it is not evidence of model accuracy or production suitability.

Contributions remain under Apache-2.0. Report security issues privately as
described in [SECURITY.md](SECURITY.md).
