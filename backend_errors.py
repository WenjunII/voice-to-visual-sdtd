from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


class RetryableTranscriptionError(RuntimeError):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def retry_after_seconds(headers, default=None, now=None):
    value = headers.get("Retry-After") if headers is not None else None
    if not value:
        return default
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (retry_at - current).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return default


def exponential_backoff(attempt, base_seconds=1.0, max_seconds=10.0):
    return min(max_seconds, base_seconds * (2 ** max(0, attempt)))
