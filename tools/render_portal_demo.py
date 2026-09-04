"""Render the real portal template using only hand-authored synthetic responses."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse
import wave


def render(output: Path) -> None:
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory(prefix="lvt-synthetic-portal-") as temporary:
        root = Path(temporary)
        for key, name in {
            "VOICEMAIL_STATE_DB": "state.sqlite3",
            "VOICEMAIL_WATCH_DIR": "spool",
            "VOICEMAIL_CONFIG": "voicemail.conf",
            "VOICEMAIL_PORTAL_USERS_FILE": "users.json",
            "VOICEMAIL_PORTAL_TRASH_DIR": "trash",
        }.items():
            os.environ[key] = str(root / name)
        os.environ["VOICEMAIL_PORTAL_BASE_PATH"] = ""
        os.environ["VOICEMAIL_PORTAL_BRAND_NAME"] = "Local Voicemail AI"
        os.environ["VOICEMAIL_PORTAL_FORWARD_ENABLED"] = "false"
        os.environ["VOICEMAIL_PORTAL_FORWARD_EMAIL_ENABLED"] = "false"
        import voicemail_portal as portal

        user = portal.PortalUser("synthetic-reviewer", "9901", "", "Avery Example", False)
        html = portal.portal_page(user, "synthetic-demo-csrf").body
        message = {
            "file_key": "synthetic-demo-001", "extension": "9901", "mailbox": "9901",
            "callerid": '"Bailey Sample" <2025550142>',
            "display_date": "Sun, Jan 02 2000, 10:00 AM", "origtime": 946807200,
            "duration": 15, "duration_display": "0:15", "has_audio": True,
            "transcript": "This is Bailey Sample calling for Jordan Sample. "
            "Their date of birth is February 3, 1970. Please call 202-555-0142 "
            "to arrange an appointment. Thank you.",
            "entities": {"name": "Jordan Sample", "dob": "02/03/1970",
                         "callback_number": "202-555-0142", "fax_number": "Not Included",
                         "callback_matches_caller_id": "Yes"},
            "field_verifications": {
                "callback_number": {"status": "verified", "used_parakeet": True},
                "dob": {"status": "ambiguous", "needs_review": True,
                        "review_reasons": ["Illustrative synthetic review state"]},
            }, "deleted_utc": None,
        }
        sound = io.BytesIO()
        with wave.open(sound, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b"\x00\x00" * 8000 * 15)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)

                def respond(route):
                    path = urlparse(route.request.url).path
                    if path == "/api/voicemails":
                        route.fulfill(content_type="application/json", body=json.dumps([message]))
                    elif path == "/api/extensions":
                        route.fulfill(content_type="application/json", body=json.dumps(["9901"]))
                    elif path.endswith("/audio"):
                        route.fulfill(content_type="audio/wav", body=sound.getvalue())
                    elif path == "/":
                        route.fulfill(content_type="text/html", body=html)
                    else:
                        route.fulfill(status=404, body="Synthetic demo: no external resources")

                page.route("**/*", respond)
                page.goto("http://portal.example.invalid/", wait_until="networkidle")
                page.locator("#detail").get_by_text("Jordan Sample", exact=True).wait_for()
                output.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(output), full_page=True)
            finally:
                browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/images/portal-demo.png"))
    args = parser.parse_args()
    render(args.output)
    print("Synthetic portal screenshot rendered; no test service was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
