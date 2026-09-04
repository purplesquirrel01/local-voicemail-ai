"""Portal route handler compatibility surface."""

from __future__ import annotations

from voicemail_portal import (
    ForwardVoicemailRequest,
    brand_light_logo,
    brand_logo,
    bulk_delete_voicemails,
    delete_voicemail,
    delete_voicemail_with_comment,
    forward_voicemail,
    get_audio,
    get_login,
    health,
    index,
    list_extensions,
    list_voicemails,
    logout,
    on_startup,
    post_login,
    restore_voicemail,
    save_voicemail_comment,
    voicemails_page,
)

__all__ = [
    "ForwardVoicemailRequest",
    "brand_light_logo",
    "brand_logo",
    "bulk_delete_voicemails",
    "delete_voicemail",
    "delete_voicemail_with_comment",
    "forward_voicemail",
    "get_audio",
    "get_login",
    "health",
    "index",
    "list_extensions",
    "list_voicemails",
    "logout",
    "on_startup",
    "post_login",
    "restore_voicemail",
    "save_voicemail_comment",
    "voicemails_page",
]
