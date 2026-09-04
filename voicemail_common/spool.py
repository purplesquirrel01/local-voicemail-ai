"""Asterisk voicemail spool helpers."""

from __future__ import annotations

import os
import re
from typing import Optional

INBOX_EXT_RE = re.compile(r"/(?P<extension>\d{3,6})/INBOX/")


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def extract_extension(path: str) -> Optional[str]:
    match = INBOX_EXT_RE.search(normalize_path(path))
    if not match:
        return None
    return match.group("extension")


def is_voicemail_txt(path: str) -> bool:
    normalized = normalize_path(path)
    return (
        normalized.endswith(".txt")
        and "/INBOX/" in normalized
        and "msg" in os.path.basename(normalized)
        and extract_extension(normalized) is not None
    )


def parse_txt(txt_path: str) -> dict[str, str]:
    info: dict[str, str] = {}
    with open(txt_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if "=" in line and not line.startswith(";"):
                key, _, value = line.partition("=")
                info[key.strip()] = value.strip()
    return info


def matching_wav_path(txt_path: str) -> str:
    return os.path.splitext(txt_path)[0] + ".wav"
