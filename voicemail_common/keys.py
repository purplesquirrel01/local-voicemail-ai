"""Stable voicemail message identity helpers."""

from __future__ import annotations

import hashlib
import os
from typing import Optional


def metadata_file_hash(txt_path: str) -> str:
    digest = hashlib.sha256()
    with open(txt_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_legacy_file_key(extension: str, info: dict[str, str], txt_path: str) -> Optional[str]:
    origtime = info.get("origtime", "").strip()
    if not origtime:
        return None

    mailbox = info.get("origmailbox", extension).strip()
    callerid = info.get("callerid", "").strip()
    msg_name = os.path.splitext(os.path.basename(txt_path))[0]
    material = "\0".join([extension, mailbox, origtime, callerid, msg_name])
    return hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()[:32]


def build_file_key(extension: str, info: dict[str, str], txt_path: str) -> Optional[str]:
    legacy_material = build_legacy_file_key(extension, info, txt_path)
    if not legacy_material:
        return None

    metadata_hash = metadata_file_hash(txt_path)
    material = "\0".join([legacy_material, metadata_hash])
    return hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()[:32]
