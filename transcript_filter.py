import re


_EXACT_HALLUCINATIONS = frozenset(
    {
        "like and subscribe",
        "please subscribe",
        "subscribe",
        "subtitle",
        "subtitles",
        "thank you",
        "thank you for watching",
        "thanks for watching",
    }
)


def is_probable_whisper_hallucination(text):
    """Return True for short, standalone phrases Whisper commonly invents in silence."""

    normalized = re.sub(r"[^\w]+", " ", (text or "").casefold(), flags=re.UNICODE)
    normalized = " ".join(normalized.split())
    return normalized in _EXACT_HALLUCINATIONS
