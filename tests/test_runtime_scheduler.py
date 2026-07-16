import unittest

import numpy as np

from runtime_scheduler import RealtimeJobScheduler
from streaming_core import AudioSegment


def make_segment(segment_id, version, is_final=False):
    return AudioSegment(
        segment_id=segment_id,
        version=version,
        samples=np.array([version], dtype=np.int16),
        is_final=is_final,
    )


class RealtimeJobSchedulerTests(unittest.TestCase):
    def test_keeps_only_the_newest_partial(self):
        scheduler = RealtimeJobScheduler()
        scheduler.submit_partial(make_segment(1, 1), now=1.0)
        scheduler.submit_partial(make_segment(1, 2), now=1.1)

        job = scheduler.next_job(now=1.2)
        metrics = scheduler.metrics()

        self.assertEqual(job.segment.version, 2)
        self.assertEqual(metrics.replaced_partials, 1)
        self.assertEqual(metrics.queue_depth, 0)

    def test_does_not_requeue_an_unchanged_partial(self):
        scheduler = RealtimeJobScheduler()
        segment = make_segment(1, 1)
        scheduler.submit_partial(segment, now=1.0)
        scheduler.next_job(now=1.0)

        duplicate = scheduler.submit_partial(segment, now=1.1)

        self.assertIsNone(duplicate)
        self.assertIsNone(scheduler.next_job(now=1.1))

    def test_prioritizes_final_audio_and_removes_its_partial(self):
        scheduler = RealtimeJobScheduler()
        scheduler.submit_partial(make_segment(1, 2), now=1.0)
        scheduler.submit_final(make_segment(1, 3, is_final=True), now=1.1)
        scheduler.submit_partial(make_segment(2, 1), now=1.2)

        first = scheduler.next_job(now=1.3)
        second = scheduler.next_job(now=1.3)

        self.assertTrue(first.is_final)
        self.assertEqual(first.segment.segment_id, 1)
        self.assertFalse(second.is_final)
        self.assertEqual(second.segment.segment_id, 2)

    def test_retried_final_waits_until_ready(self):
        scheduler = RealtimeJobScheduler()
        job = scheduler.submit_final(make_segment(3, 4, is_final=True), now=1.0)
        job = scheduler.next_job(now=1.0)

        self.assertTrue(scheduler.retry_final(job, now=1.1, delay_seconds=2.0))
        self.assertIsNone(scheduler.next_job(now=3.0))
        retried = scheduler.next_job(now=3.1)

        self.assertEqual(retried.attempts, 1)
        self.assertEqual(scheduler.metrics().retries, 1)

    def test_drops_stale_work_and_reports_capacity_drops(self):
        scheduler = RealtimeJobScheduler(
            max_final_jobs=1,
            partial_max_age_seconds=1.0,
            final_max_age_seconds=1.0,
        )
        scheduler.submit_final(make_segment(1, 1, is_final=True), now=1.0)
        self.assertIsNone(scheduler.submit_final(make_segment(2, 1, is_final=True), now=1.1))
        scheduler.submit_partial(make_segment(3, 1), now=1.1)

        self.assertIsNone(scheduler.next_job(now=2.2))
        metrics = scheduler.metrics()
        self.assertEqual(metrics.dropped_finals, 1)
        self.assertEqual(metrics.dropped_stale, 2)


if __name__ == "__main__":
    unittest.main()
