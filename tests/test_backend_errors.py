import unittest
from datetime import datetime, timezone

from backend_errors import exponential_backoff, retry_after_seconds


class BackendErrorTests(unittest.TestCase):
    def test_reads_numeric_retry_after(self):
        self.assertEqual(retry_after_seconds({"Retry-After": "2.5"}), 2.5)

    def test_reads_http_date_retry_after(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        headers = {"Retry-After": "Thu, 01 Jan 2026 00:00:03 GMT"}

        self.assertEqual(retry_after_seconds(headers, now=now), 3.0)

    def test_uses_default_for_invalid_retry_after(self):
        self.assertEqual(retry_after_seconds({"Retry-After": "later"}, default=4), 4)

    def test_exponential_backoff_is_capped(self):
        self.assertEqual(exponential_backoff(0, base_seconds=1, max_seconds=5), 1)
        self.assertEqual(exponential_backoff(3, base_seconds=1, max_seconds=5), 5)
        self.assertEqual(exponential_backoff(100000, base_seconds=1, max_seconds=5), 5)


if __name__ == "__main__":
    unittest.main()
