# Validation

The [CI workflow](../.github/workflows/ci.yml) uses Python 3.11 and the `dev`
extra. Run it in a disposable Linux workspace, with the virtual environment
outside the source tree or named `.venv`. Python 3.11.8 or newer provides the
tar extraction filter used by the package audit.

After `python -m pip install -e '.[dev]'`, the blocking commands are:

```bash
python -m compileall -q -x '/(\.git|\.venv|build|dist)/' .
python -m pytest -q
ruff check .
python -m tools.public_release_audit .
python -m tools.audit_secrets .
python -m tools.validate_documentation .
python -m build
python -m tools.audit_release_artifacts dist/*.whl dist/*.tar.gz
```

To verify installation, choose a nonexistent environment path outside the source:

```bash
python -m tools.check_wheel dist/*.whl --venv /tmp/lvt-portfolio-wheel-check
```

CI uses its own temporary directory for that environment. The checker installs
the wheel with the portal dependencies, runs `pip check`, imports application
modules outside the source tree, loads all five console entry points, checks
packaged prompts/assets, and verifies service dispatch with the server mocked.
It does not start a listener or watcher. It refuses to reuse an existing environment.

## What the suite exercises

- Extraction, schema validation, normalization, ambiguous evidence, caller-ID
  correction, name spelling, callback verification, and final field resolution.
- Watcher state, retries, duplicate records, missing files, and SQLite behavior.
- Portal authorization, authentication, sessions, CSRF, API response filtering,
  mailbox isolation, scoped integration tokens, and rate limits.
- Provider request/response contracts and authenticated service readiness.
- Source privacy checks, negative secret controls, documentation links, and
  package metadata/resources.

Pytest collects the unittest classes as well as pytest-style API and integration
tests. Report its total once; do not add a separate unittest count. Fixtures use
temporary state, fictional text, synthetic audio, and mocked model responses.
No live PBX, root permissions, model weights, or external listener are required.

The tests validate application behavior, not clinical suitability or model
accuracy. No transcription benchmark is claimed. The screenshot is a separately
recreated synthetic demonstration; see [its provenance](images/README.md).
