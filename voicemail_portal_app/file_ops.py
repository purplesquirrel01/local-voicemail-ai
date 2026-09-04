"""Portal voicemail file operation compatibility surface."""

from __future__ import annotations

from voicemail_portal import (
    available_forward_stem,
    available_restore_stem,
    copy_message_to_mailbox,
    find_mailbox_inbox_dir,
    find_moved_message_paths,
    forward_destination_inbox,
    move_message_to_trash,
    remove_copied_message_files,
    restore_destination_inbox,
    restore_message_to_inbox,
    restore_source_txt_path,
    rewrite_voicemail_metadata_value,
    safe_under_root,
    safe_under_roots,
)

__all__ = [
    "available_forward_stem",
    "available_restore_stem",
    "copy_message_to_mailbox",
    "find_mailbox_inbox_dir",
    "find_moved_message_paths",
    "forward_destination_inbox",
    "move_message_to_trash",
    "remove_copied_message_files",
    "restore_destination_inbox",
    "restore_message_to_inbox",
    "restore_source_txt_path",
    "rewrite_voicemail_metadata_value",
    "safe_under_root",
    "safe_under_roots",
]
