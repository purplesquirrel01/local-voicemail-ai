"""Portal mailbox discovery compatibility surface."""

from __future__ import annotations

from voicemail_portal import (
    clean_mailbox_name,
    discover_mailbox_config_names,
    discover_mailbox_directory_extensions,
    discover_mailbox_directory_names,
    discover_mailbox_extensions,
    discover_mailboxes,
    voicemail_config_paths,
)

__all__ = [
    "clean_mailbox_name",
    "discover_mailbox_config_names",
    "discover_mailbox_directory_extensions",
    "discover_mailbox_directory_names",
    "discover_mailbox_extensions",
    "discover_mailboxes",
    "voicemail_config_paths",
]
