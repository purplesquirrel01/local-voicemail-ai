"""Display formatting helpers shared by voicemail services."""

from __future__ import annotations

import re
from typing import Any, Optional


def optional_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def format_duration(seconds: Any, *, empty_on_invalid: bool = True) -> str:
    value = optional_int(seconds)
    if value is None:
        return "" if empty_on_invalid else str(seconds)
    return f"{value // 60}:{value % 60:02d}"


def format_phone_number(value: Any) -> Optional[str]:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
