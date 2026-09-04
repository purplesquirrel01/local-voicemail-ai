"""Check local Markdown links and anchors across the portfolio source tree."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import unquote

MARKDOWN_LINK = re.compile(r'\[[^]]+\]\(([^)]+)\)')
HEADING = re.compile(r'(?m)^#{1,6}\s+(.+?)\s*$')
IGNORED = {'.git', '.venv', 'build', 'dist', '__pycache__', '.pytest_cache', '.ruff_cache'}


def _slug(value: str) -> str:
    value = re.sub(r'`([^`]*)`', r'\1', value).strip().lower()
    return re.sub(r'\s', '-', re.sub(r'[^\w\- ]', '', value))


def validate_links(root: Path) -> list[str]:
    root = root.resolve()
    findings = []
    for path in sorted(root.rglob('*.md')):
        if set(path.relative_to(root).parts) & IGNORED:
            continue
        for target in MARKDOWN_LINK.findall(path.read_text(encoding='utf-8')):
            target = target.strip().strip('<>')
            if re.match(r'^(?:https?|mailto):', target):
                continue
            relative, _, anchor = unquote(target).partition('#')
            candidate = (path.parent / relative).resolve() if relative else path
            if not candidate.is_relative_to(root) or not candidate.exists():
                findings.append(f'{path.relative_to(root)}: missing or external local link {target}')
            elif anchor:
                anchors = {_slug(m.group(1)) for m in HEADING.finditer(candidate.read_text(encoding='utf-8'))}
                if anchor not in anchors:
                    findings.append(f'{path.relative_to(root)}: missing anchor {target}')
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path('.'))
    findings = validate_links(parser.parse_args().root)
    for finding in findings:
        print(finding)
    print(f'Documentation links: {len(findings)} findings')
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
