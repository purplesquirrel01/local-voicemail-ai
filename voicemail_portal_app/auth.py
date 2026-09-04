"""Portal authentication and session compatibility surface."""

from __future__ import annotations

from voicemail_portal import (
    clear_login_failures,
    current_csrf,
    current_user,
    get_portal_user,
    hash_password,
    login_rate_limit_key,
    login_rate_limited,
    read_session,
    record_login_failure,
    require_csrf,
    sign_session,
    verify_password,
)

__all__ = [
    "clear_login_failures",
    "current_csrf",
    "current_user",
    "get_portal_user",
    "hash_password",
    "login_rate_limit_key",
    "login_rate_limited",
    "read_session",
    "record_login_failure",
    "require_csrf",
    "sign_session",
    "verify_password",
]
