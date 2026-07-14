import re
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetedPrompt:
    text: str
    token_count: int
    variant: str
    transcript_trimmed: bool


class RollingSceneMemory:
    """Merge finalized and partial transcripts while removing segment overlap."""

    def __init__(self, max_words=36, max_age_seconds=20.0):
        if max_words < 1:
            raise ValueError("max_words must be positive")
        self.max_words = max_words
        self.max_age_seconds = max_age_seconds
        self._committed_words = []
        self._current_words = []
        self._current_segment_id = None
        self._last_update_at = None

    def update(self, segment_id, text, is_final=False, now=None):
        now = time.monotonic() if now is None else now
        if self._is_expired(now):
            self.reset()

        words = _split_words(text)
        if self._current_segment_id is not None and segment_id != self._current_segment_id:
            self._committed_words = self._trim(
                _merge_with_overlap(self._committed_words, self._current_words)
            )

        self._current_segment_id = segment_id
        self._current_words = words
        combined = self._trim(_merge_with_overlap(self._committed_words, self._current_words))

        if is_final:
            self._committed_words = combined
            self._current_words = []
            self._current_segment_id = None

        self._last_update_at = now
        return " ".join(combined).strip()

    def reset(self):
        self._committed_words = []
        self._current_words = []
        self._current_segment_id = None
        self._last_update_at = None

    def _is_expired(self, now):
        return (
            self.max_age_seconds > 0
            and self._last_update_at is not None
            and now - self._last_update_at > self.max_age_seconds
        )

    def _trim(self, words):
        return list(words[-self.max_words:])


class PromptBudgeter:
    """Fit prompts within every configured tokenizer's context window."""

    def __init__(self, tokenizers, max_tokens=77, min_transcript_tokens=20):
        if not tokenizers:
            raise ValueError("at least one tokenizer is required")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.tokenizers = list(tokenizers)
        self.max_tokens = max_tokens
        self.min_transcript_tokens = max(0, min_transcript_tokens)

    def token_count(self, text):
        return max(len(tokenizer.encode(text, truncation=False)) for tokenizer in self.tokenizers)

    def fit(self, variants, transcript):
        if not variants:
            raise ValueError("at least one prompt variant is required")

        transcript = (transcript or "").strip()
        measured = []
        for name, render in variants:
            full_prompt = _clean_prompt(render(transcript))
            full_count = self.token_count(full_prompt)
            if full_count <= self.max_tokens:
                return BudgetedPrompt(full_prompt, full_count, name, False)

            static_prompt = _clean_prompt(render(""))
            static_count = self.token_count(static_prompt)
            measured.append((name, render, static_count))
            if static_count <= self.max_tokens - self.min_transcript_tokens:
                return self._fit_transcript(name, render, transcript)

        name, render, _ = min(measured, key=lambda item: item[2])
        return self._fit_transcript(name, render, transcript)

    def _fit_transcript(self, name, render, transcript):
        candidate = self._longest_fitting_suffix(render, transcript)
        prompt = _clean_prompt(render(candidate))
        token_count = self.token_count(prompt)
        if token_count > self.max_tokens:
            prompt = self._truncate_complete_prompt(prompt)
            token_count = self.token_count(prompt)
        return BudgetedPrompt(
            text=prompt,
            token_count=token_count,
            variant=name,
            transcript_trimmed=candidate.strip() != transcript.strip(),
        )

    def _longest_fitting_suffix(self, render, transcript):
        words = _split_words(transcript)
        if len(words) > 1:
            low = 0
            high = len(words)
            while low < high:
                count = (low + high + 1) // 2
                candidate = " ".join(words[-count:])
                if self.token_count(_clean_prompt(render(candidate))) <= self.max_tokens:
                    low = count
                else:
                    high = count - 1
            return " ".join(words[-low:]) if low else ""

        token_ids = self.tokenizers[0].encode(transcript, add_special_tokens=False)
        low = 0
        high = len(token_ids)
        while low < high:
            count = (low + high + 1) // 2
            candidate = self.tokenizers[0].decode(token_ids[-count:], skip_special_tokens=True).strip()
            if self.token_count(_clean_prompt(render(candidate))) <= self.max_tokens:
                low = count
            else:
                high = count - 1
        if not low:
            return ""
        return self.tokenizers[0].decode(token_ids[-low:], skip_special_tokens=True).strip()

    def _truncate_complete_prompt(self, prompt):
        tokenizer = self.tokenizers[0]
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        for end in range(min(len(token_ids), self.max_tokens), 0, -1):
            candidate = tokenizer.decode(token_ids[:end], skip_special_tokens=True).strip()
            if self.token_count(candidate) <= self.max_tokens:
                return candidate
        return ""


def _split_words(text):
    return [word for word in (text or "").strip().split() if word]


def _normalize_word(word):
    normalized = re.sub(r"^\W+|\W+$", "", word, flags=re.UNICODE).casefold()
    return normalized or word.casefold()


def _merge_with_overlap(left, right):
    left = list(left)
    right = list(right)
    max_overlap = min(len(left), len(right))
    for overlap in range(max_overlap, 0, -1):
        left_tail = [_normalize_word(word) for word in left[-overlap:]]
        right_head = [_normalize_word(word) for word in right[:overlap]]
        if left_tail == right_head:
            return left + right[overlap:]
    return left + right


def _clean_prompt(prompt):
    return re.sub(r"\s+([,.])", r"\1", re.sub(r"\s+", " ", prompt)).strip(" ,")
