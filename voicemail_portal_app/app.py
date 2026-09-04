"""Portal FastAPI application compatibility surface."""

from __future__ import annotations

from typing import Any

from voicemail_portal import Settings, app, main


def create_app(settings: Settings | None = None, store: Any | None = None):
    """Return the existing portal app without changing route registration."""
    if settings is not None or store is not None:
        raise NotImplementedError("custom portal app injection requires the route extraction phase")
    return app


__all__ = ["Settings", "app", "create_app", "main"]
