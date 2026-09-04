#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str


EXCLUDED_PATH_PARTS = {
    "iphone",
    "workflow_insights",
    "switchboard",
    "voicemail_demo",
    ".deployment-artifacts",
}
EXCLUDED_NAMES = {"problem.txt"}
IGNORED_PARTS = {
    ".git",
    ".venv",
    ".tmp-wheel-verify",
    "__pycache__",
    "build",
    "dist",
    ".pytest_cache",
}
IGNORED_PREFIXES = (
    ".build-tmp",
    ".d6-",
    ".pytest-",
    ".test-tmp",
    ".tmp",
    "local-test-release-",
)
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".env",
    ".example",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".service",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PHONE_LIKE_RE = re.compile(
    r"(?<![A-Za-z0-9])\(?([2-9]\d{2})\)?[- .]?(\d{3})[- .]?(\d{4})(?![A-Za-z0-9])"
)
FORBIDDEN_SUFFIXES = {
    ".db", ".sqlite", ".sqlite3", ".wav", ".gsm", ".ulaw", ".mp3", ".m4a",
    ".flac", ".ogg", ".log", ".bak", ".backup", ".nemo", ".litertlm", ".gguf",
    ".safetensors", ".onnx", ".pt", ".pth", ".pem", ".key", ".p12", ".pfx", ".crt",
}
FORBIDDEN_DATA_DIRECTORIES = {"data", "voicemails", "transcripts", "logs", "backups", "models"}
IPV4_LITERAL = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
EMAIL_LITERAL = re.compile(r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[A-Za-z]{2,}(?![\w-])")
DOCUMENTATION_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",
))


def _contact_findings(relative: Path, number: int, line: str) -> list[Finding]:
    findings = []
    for match in EMAIL_LITERAL.finditer(line):
        domain = match.group().rsplit("@", 1)[1].lower()
        if domain != "example.invalid":
            findings.append(Finding(relative, number, "email literal is outside example.invalid"))
    for match in IPV4_LITERAL.finditer(line):
        try:
            address = ipaddress.ip_address(match.group())
        except ValueError:
            continue
        # Wildcard bindings and loopback are functional settings, not site addresses.
        if address.is_loopback or address.is_unspecified or str(address) == "255.255.255.255":
            continue
        if not any(address in network for network in DOCUMENTATION_NETWORKS):
            findings.append(Finding(relative, number, "network literal is outside documentation networks"))
    return findings
SYNTHETIC_PROVENANCE_DOCUMENT = "SYNTHETIC_DATA_PROVENANCE.md"
SYNTHETIC_PROVENANCE_VALUES = {
    "generated",
    "hand-authored",
}


def _synthetic_fixture_findings(root: Path) -> list[Finding]:
    eval_root = root / "eval"
    fixtures = sorted(eval_root.glob("*.jsonl")) if eval_root.is_dir() else []
    if not fixtures:
        return []

    findings: list[Finding] = []
    provenance_document = eval_root / SYNTHETIC_PROVENANCE_DOCUMENT
    if not provenance_document.is_file():
        findings.append(
            Finding(
                eval_root.relative_to(root),
                0,
                "synthetic fixture provenance document is missing",
            )
        )

    for path in fixtures:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                findings.append(
                    Finding(
                        path.relative_to(root),
                        line_number,
                        "synthetic fixture JSON is invalid",
                    )
                )
                continue
            metadata = record.get("metadata") if isinstance(record, dict) else None
            provenance = metadata.get("provenance") if isinstance(metadata, dict) else None
            if (
                not isinstance(metadata, dict)
                or metadata.get("synthetic") is not True
                or provenance not in SYNTHETIC_PROVENANCE_VALUES
            ):
                findings.append(
                    Finding(
                        path.relative_to(root),
                        line_number,
                        "synthetic fixture provenance marker is missing",
                    )
                )
    return findings


def scan_tree(root: Path) -> list[Finding]:
    findings = _synthetic_fixture_findings(root)
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & IGNORED_PARTS or any(
            part.startswith(IGNORED_PREFIXES) for part in lowered_parts
        ):
            continue
        if path.name.lower() in EXCLUDED_NAMES or lowered_parts & EXCLUDED_PATH_PARTS:
            findings.append(Finding(relative, 0, "excluded path is present"))
            if path.is_dir():
                continue
        if path.is_symlink():
            findings.append(Finding(relative, 0, "symbolic link is not permitted in public source"))
            continue
        if not path.is_file():
            continue
        if (
            path.suffix.lower() in FORBIDDEN_SUFFIXES
            or lowered_parts & FORBIDDEN_DATA_DIRECTORIES
            or (path.name.startswith(".env") and not path.name.endswith(".env.example"))
            or (path.suffix == ".env" and not path.name.endswith(".env.example"))
        ):
            findings.append(Finding(relative, 0, "private or generated data file is not permitted"))
        if path.suffix.lower() == ".png":
            # Images require separate visual and provenance review.
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            findings.extend(_contact_findings(relative, line_number, line))
            for match in PHONE_LIKE_RE.finditer(line):
                exchange = match.group(2)
                subscriber = int(match.group(3))
                if exchange != "555" or not 100 <= subscriber <= 199:
                    findings.append(
                        Finding(
                            relative,
                            line_number,
                            "phone-like literal is outside the fictional 555-0100 through 555-0199 range",
                        )
                    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a public LVT source tree")
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    findings = scan_tree(args.root.resolve())
    for finding in findings:
        location = f"{finding.path}:{finding.line}" if finding.line else str(finding.path)
        print(f"{location}: {finding.message}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
