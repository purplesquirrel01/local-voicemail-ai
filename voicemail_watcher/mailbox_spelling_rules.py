"""Mailbox-specific final-output spelling correction helpers."""

from __future__ import annotations

from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import Any


logger = logging.getLogger("voicemail_watcher")

Rule = dict[str, str]
RulesByMailbox = dict[str, list[Rule]]

ENTITY_FIELDS = ("name", "dob", "callback_number", "fax_number")


def _clean_rule(item: Any) -> Rule | None:
    if not isinstance(item, dict):
        return None
    source = str(item.get("from") or "").strip()
    replacement = str(item.get("to") or "").strip()
    if not source or not replacement:
        return None
    return {"from": source, "to": replacement}


def _clean_rules(payload: Any) -> RulesByMailbox:
    if not isinstance(payload, dict):
        return {}
    rules: RulesByMailbox = {}
    for mailbox, items in payload.items():
        mailbox_key = str(mailbox or "").strip()
        if not mailbox_key or not isinstance(items, list):
            continue
        clean_items = [rule for rule in (_clean_rule(item) for item in items) if rule]
        if clean_items:
            rules[mailbox_key] = clean_items
    return rules


@lru_cache(maxsize=16)
def load_mailbox_spelling_rules(path: str) -> RulesByMailbox:
    """Load mailbox spelling rules, returning an empty map on any config issue."""

    path_text = str(path or "").strip()
    if not path_text:
        return {}
    config_path = Path(path_text)
    try:
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as handle:
            return _clean_rules(json.load(handle))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Mailbox spelling rules config ignored path=%s error=%s", config_path, type(exc).__name__)
        return {}


def _whole_word_pattern(source: str) -> re.Pattern[str]:
    escaped = re.escape(source)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _apply_rules_to_text(text: str, rules: list[Rule]) -> tuple[str, int]:
    corrected = str(text or "")
    total = 0
    for rule in rules:
        pattern = _whole_word_pattern(rule["from"])
        corrected, count = pattern.subn(rule["to"], corrected)
        total += count
    return corrected, total


def apply_mailbox_spelling_rules(
    mailbox: str,
    transcript: str,
    entities: dict[str, Any],
    rules_by_mailbox: RulesByMailbox,
    *,
    enabled: bool = True,
) -> tuple[str, dict[str, Any], int]:
    """Apply final-output mailbox spelling rules to transcript and simple entity fields."""

    if not enabled:
        return transcript, dict(entities or {}), 0

    mailbox_key = str(mailbox or "").strip()
    rules = rules_by_mailbox.get(mailbox_key) if isinstance(rules_by_mailbox, dict) else None
    if not rules:
        return transcript, dict(entities or {}), 0

    corrected_transcript, replacement_count = _apply_rules_to_text(transcript, rules)
    corrected_entities = dict(entities or {})
    for field in ENTITY_FIELDS:
        value = corrected_entities.get(field)
        if not isinstance(value, str) or not value:
            continue
        corrected_value, count = _apply_rules_to_text(value, rules)
        corrected_entities[field] = corrected_value
        replacement_count += count

    return corrected_transcript, corrected_entities, replacement_count
