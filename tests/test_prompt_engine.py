import unittest

from prompt_engine import (
    ConservativeUtf8Tokenizer,
    PromptBudgeter,
    RollingSceneMemory,
)


class WhitespaceTokenizer:
    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def encode(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        tokens = []
        for word in text.split():
            tokens.extend([word] * self.multiplier)
        if add_special_tokens:
            tokens = ["<start>"] + tokens + ["<end>"]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [token for token in token_ids if token not in {"<start>", "<end>"}]
        if self.multiplier > 1:
            tokens = tokens[::self.multiplier]
        return " ".join(tokens)


class RollingSceneMemoryTests(unittest.TestCase):
    def test_merges_segment_overlap_without_repeating_words(self):
        memory = RollingSceneMemory(max_words=20, max_age_seconds=30)
        first = memory.update(1, "we walk through Central Park in autumn", is_final=True, now=1)
        second = memory.update(2, "Central Park in autumn and see golden leaves", now=2)

        self.assertEqual(first, "we walk through Central Park in autumn")
        self.assertEqual(
            second,
            "we walk through Central Park in autumn and see golden leaves",
        )

    def test_partial_hypothesis_replaces_the_same_segment(self):
        memory = RollingSceneMemory(max_words=20, max_age_seconds=30)
        memory.update(1, "a red car", now=1)

        updated = memory.update(1, "a red car drives downtown", now=2)

        self.assertEqual(updated, "a red car drives downtown")

    def test_keeps_newest_words_and_expires_old_scene(self):
        memory = RollingSceneMemory(max_words=4, max_age_seconds=5)
        trimmed = memory.update(1, "one two three four five six", is_final=True, now=1)
        expired = memory.update(2, "new scene", now=7)

        self.assertEqual(trimmed, "three four five six")
        self.assertEqual(expired, "new scene")


class PromptBudgeterTests(unittest.TestCase):
    def test_selects_compact_variant_and_keeps_newest_transcript(self):
        budgeter = PromptBudgeter(
            [WhitespaceTokenizer()],
            max_tokens=10,
            min_transcript_tokens=4,
        )
        variants = [
            ("full", lambda text: f"one two three four five six seven eight {text}"),
            ("compact", lambda text: f"scene {text} cinematic"),
        ]

        result = budgeter.fit(variants, "old detail then newer golden autumn leaves")

        self.assertEqual(result.variant, "compact")
        self.assertLessEqual(result.token_count, 10)
        self.assertTrue(result.text.endswith("newer golden autumn leaves cinematic"))
        self.assertTrue(result.transcript_trimmed)
        self.assertEqual(result.budget_mode, "exact")

    def test_enforces_the_most_restrictive_tokenizer(self):
        budgeter = PromptBudgeter(
            [WhitespaceTokenizer(), WhitespaceTokenizer(multiplier=2)],
            max_tokens=12,
            min_transcript_tokens=2,
        )
        variants = [("compact", lambda text: f"scene {text}")]

        result = budgeter.fit(variants, "one two three four five six")

        self.assertLessEqual(result.token_count, 12)
        self.assertIn("five six", result.text)

    def test_conservative_fallback_bounds_utf8_without_model_files(self):
        budgeter = PromptBudgeter(
            [ConservativeUtf8Tokenizer()],
            max_tokens=40,
            min_transcript_tokens=8,
            mode="fallback",
        )
        variants = [
            ("compact", lambda text: f"scene {text} cinematic"),
        ]

        result = budgeter.fit(
            variants,
            "旧场景变成霓虹森林!!! café",
        )

        self.assertEqual(result.budget_mode, "fallback")
        self.assertLessEqual(result.token_count, 40)
        self.assertEqual(
            result.token_count,
            len(result.text.encode("utf-8")) + 2,
        )
        self.assertTrue(result.text)

    def test_conservative_tokenizer_round_trips_unicode(self):
        tokenizer = ConservativeUtf8Tokenizer()
        text = "画面 café — neon"

        token_ids = tokenizer.encode(text, add_special_tokens=False)

        self.assertEqual(tokenizer.decode(token_ids), text)
        self.assertEqual(len(tokenizer.encode(text)), len(text.encode("utf-8")) + 2)

    def test_rejects_unknown_budget_modes(self):
        with self.assertRaisesRegex(ValueError, "exact or fallback"):
            PromptBudgeter([WhitespaceTokenizer()], mode="approximate")

    def test_budget_must_allow_clip_boundary_tokens(self):
        with self.assertRaisesRegex(ValueError, "boundary tokens"):
            PromptBudgeter([WhitespaceTokenizer()], max_tokens=1)


if __name__ == "__main__":
    unittest.main()
