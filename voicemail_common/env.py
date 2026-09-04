"""Environment parsing helpers shared by watcher and portal."""

from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number, got {raw!r}") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def parse_service_urls(primary_url: str, raw_urls: str = "", label: str = "service") -> tuple[str, ...]:
    source = raw_urls.strip() or primary_url.strip()
    if not source:
        return ()

    values: list[str] = []
    for item in source.replace("\n", ",").split(","):
        value = item.strip()
        if not value:
            continue
        if value in values:
            continue
        values.append(value)

    if not values and primary_url.strip():
        values.append(primary_url.strip())
    if not values:
        raise ValueError(f"At least one {label} URL is required")
    return tuple(values)
