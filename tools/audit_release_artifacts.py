"""Inspect Python wheel/sdist contents without installing or executing them."""
from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import tempfile
import zipfile

from tools.audit_secrets import scan_for_secrets
from tools.public_release_audit import EXCLUDED_NAMES, EXCLUDED_PATH_PARTS, Finding, scan_tree

FORBIDDEN_ARTIFACT_SUFFIXES = {'.bin', '.litertlm', '.log', '.nemo', '.onnx', '.pt', '.pth',
                              '.safetensors', '.sqlite', '.sqlite3', '.wav'}


def _safe_member_path(name: str) -> Path:
    path = Path(name.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe archive member path')
    return path


def audit_archive(archive: Path) -> list[Finding]:
    archive = archive.resolve()
    with tempfile.TemporaryDirectory(prefix='lvt-artifact-audit-') as tmp:
        destination = Path(tmp)
        findings: list[Finding] = []
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
                for name in names:
                    _safe_member_path(name)
                handle.extractall(destination)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive, 'r:*') as handle:
                members = handle.getmembers()
                names = [member.name for member in members]
                for member in members:
                    _safe_member_path(member.name)
                    if not (member.isfile() or member.isdir()):
                        findings.append(Finding(archive, 0, 'archive link or special file is present'))
                if not findings:
                    handle.extractall(destination, filter='data')
        else:
            raise ValueError('unsupported package archive')
        for name in names:
            relative = _safe_member_path(name)
            if relative.name.lower() in EXCLUDED_NAMES or {p.lower() for p in relative.parts} & EXCLUDED_PATH_PARTS:
                findings.append(Finding(archive / relative, 0, 'excluded path is present'))
            if relative.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
                findings.append(Finding(archive / relative, 0, 'forbidden generated/model data is present'))
        findings.extend(scan_tree(destination))
        findings.extend(Finding(item.path, 0, item.kind) for item in scan_for_secrets(destination))
        return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path, nargs='+')
    findings = [finding for archive in parser.parse_args().archives for finding in audit_archive(archive)]
    for finding in findings:
        print(f'{finding.path}: {finding.message}')
    print(f'Package artifact audit: {len(findings)} findings')
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
