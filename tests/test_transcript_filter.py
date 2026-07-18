import unittest

from transcript_filter import is_probable_whisper_hallucination


class TranscriptFilterTests(unittest.TestCase):
    def test_filters_standalone_whisper_hallucinations(self):
        self.assertTrue(is_probable_whisper_hallucination(" Thank you for watching! "))
        self.assertTrue(is_probable_whisper_hallucination("SUBTITLE"))

    def test_keeps_real_sentences_containing_trigger_words(self):
        self.assertFalse(
            is_probable_whisper_hallucination(
                "Thank you for bringing warm light into the room"
            )
        )
        self.assertFalse(
            is_probable_whisper_hallucination(
                "A hand reaches toward the subscribe button"
            )
        )

    def test_keeps_empty_transcripts_as_non_hallucinations(self):
        self.assertFalse(is_probable_whisper_hallucination(""))


if __name__ == "__main__":
    unittest.main()
