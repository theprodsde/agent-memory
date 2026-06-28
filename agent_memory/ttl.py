from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_TTL_PATTERN = re.compile(r"^(\d+)([dhms])$", re.IGNORECASE)


def parse_ttl(ttl: str | int | float | None) -> datetime | None:
    """
    Parse TTL into an absolute expiry timestamp (UTC).

    Supports: 30d, 7d, 24h, 60m, 3600s, or raw seconds as int/float.
    """
    if ttl is None:
        return None

    now = datetime.now(timezone.utc)

    if isinstance(ttl, (int, float)):
        return now + timedelta(seconds=float(ttl))

    ttl = ttl.strip().lower()
    match = _TTL_PATTERN.match(ttl)
    if not match:
        raise ValueError(f"Invalid TTL format: {ttl!r}. Use e.g. 30d, 24h, 60m, 3600s.")

    amount = int(match.group(1))
    unit = match.group(2)

    if unit == "d":
        delta = timedelta(days=amount)
    elif unit == "h":
        delta = timedelta(hours=amount)
    elif unit == "m":
        delta = timedelta(minutes=amount)
    else:
        delta = timedelta(seconds=amount)

    return now + delta


def is_expired(expires_at: datetime | None) -> bool:
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires_at
