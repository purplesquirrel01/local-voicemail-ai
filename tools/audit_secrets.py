from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence


IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".superpowers",
    ".venv",
    ".worktrees",
    "__pycache__",
    "build",
    "dist",
}
IGNORED_PREFIXES = (
    ".build-tmp",
    ".d6-",
    ".pytest-",
    ".test-tmp",
    ".tmp",
    "local-test-release-",
)
CONFIG_SUFFIXES = {".conf", ".env", ".ini", ".json", ".toml", ".yaml", ".yml"}
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?m)^(?:LVT_API_KEY|LVT_PORTAL_SESSION_SECRET|LVT_MODEL_DOWNLOAD_TOKEN)="
    r"(?:[A-Fa-f0-9]{24,}|[A-Za-z0-9_+/=-]{32,})$"
)
PRIVATE_KEY_MARKERS = (
    "-----BEGIN " + "PRIVATE KEY-----",
    "-----BEGIN OPENSSH " + "PRIVATE KEY-----",
    "-----BEGIN RSA " + "PRIVATE KEY-----",
    "-----BEGIN EC " + "PRIVATE KEY-----",
    "-----BEGIN DSA " + "PRIVATE KEY-----",
    "-----BEGIN ENCRYPTED " + "PRIVATE KEY-----",
)
PROVIDER_TOKEN = re.compile(
    r"\b(?:gh" + r"[pousr]_[A-Za-z0-9]{30,}|github_" + r"pat_[A-Za-z0-9_]{30,}"
    r"|AK" + r"IA[A-Z0-9]{16}|sk" + r"-[A-Za-z0-9_-]{32,})\b"
)


@dataclass(frozen=True)
class SecretFinding:
    path: Path
    kind: str


def scan_for_secrets(root: Path) -> list[SecretFinding]:
    root = root.resolve()
    findings: list[SecretFinding] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if set(relative.parts) & IGNORED_PARTS or any(
            part.startswith(IGNORED_PREFIXES) for part in relative.parts
        ):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if any(marker in text for marker in PRIVATE_KEY_MARKERS):
            findings.append(SecretFinding(relative, "private-key"))
        if PROVIDER_TOKEN.search(text):
            findings.append(SecretFinding(relative, "provider-token"))
        if path.suffix.lower() in CONFIG_SUFFIXES and CREDENTIAL_ASSIGNMENT.search(text):
            findings.append(SecretFinding(relative, "credential-assignment"))
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit committed LVT source for credential and private-key material")
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args(argv)
    findings = scan_for_secrets(args.root)
    for finding in findings:
        print(f"{finding.path}: {finding.kind}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
