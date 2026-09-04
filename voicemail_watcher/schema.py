"""Watcher schema/status compatibility surface."""

from __future__ import annotations

from watcher import (
    STATUS_COMPLETED,
    STATUS_DEAD,
    STATUS_DISCOVERED,
    STATUS_PROCESSING,
    STATUS_RETRY,
    STATUS_SKIPPED,
    TERMINAL_STATUSES,
)

__all__ = [
    "STATUS_COMPLETED",
    "STATUS_DEAD",
    "STATUS_DISCOVERED",
    "STATUS_PROCESSING",
    "STATUS_RETRY",
    "STATUS_SKIPPED",
    "TERMINAL_STATUSES",
]
