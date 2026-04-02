"""
utils/text_cleaner.py
=====================
Sanskrit text cleaning and quality filtering utilities.
"""

import re
import unicodedata
from typing import Optional

DEVANAGARI_START = "\u0900"
DEVANAGARI_END   = "\u097F"
VEDIC_START = "\u1CD0"
VEDIC_END   = "\u1CFF"
DANDA        = "।"
DOUBLE_DANDA = "॥"


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def keep_devanagari_only(text: str) -> str:
    """Keep Devanagari, Vedic, Sanskrit punctuation, and whitespace."""
    return re.sub(
        r"[^\u0900-\u097F\u1CD0-\u1CFF\u0964\u0965\s]",
        " ",
        text
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_document(text: str, max_chars: int = 50_000) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_danda = max(truncated.rfind(DANDA), truncated.rfind(DOUBLE_DANDA))
    if last_danda > max_chars // 2:
        return truncated[:last_danda + 1].strip()
    return truncated.strip()


def clean_sanskrit_text(text: str, max_chars: int = 50_000) -> str:
    """
    Full cleaning pipeline for a Sanskrit document.
    Steps: ensure string → NFC normalize → strip HTML → strip URLs
           → keep Devanagari only → normalize whitespace → truncate
    """
    if not isinstance(text, str):
        text = str(text)
    text = normalize_unicode(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = keep_devanagari_only(text)
    text = normalize_whitespace(text)
    text = truncate_document(text, max_chars=max_chars)
    return text


def quality_filter(
    text: str,
    min_words: int = 10,
    min_devanagari_ratio: float = 0.6,
) -> bool:
    """
    Returns True if text is high-quality Sanskrit, False if it should be dropped.
    Checks: non-empty, minimum word count, minimum Devanagari ratio.
    """
    if not text or not text.strip():
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    non_space = text.replace(" ", "")
    if len(non_space) == 0:
        return False
    devanagari_count = sum(
        1 for ch in non_space if DEVANAGARI_START <= ch <= DEVANAGARI_END
    )
    return (devanagari_count / len(non_space)) >= min_devanagari_ratio


def clean_and_filter(
    text: str,
    min_words: int = 10,
    min_devanagari_ratio: float = 0.6,
    max_chars: int = 50_000,
) -> Optional[str]:
    """Clean + filter in one call. Returns cleaned text or None."""
    cleaned = clean_sanskrit_text(text, max_chars=max_chars)
    if quality_filter(cleaned, min_words=min_words,
                      min_devanagari_ratio=min_devanagari_ratio):
        return cleaned
    return None


def text_stats(text: str) -> dict:
    words    = text.split()
    non_space = [c for c in text if c != " "]
    devanagari = [c for c in text if DEVANAGARI_START <= c <= DEVANAGARI_END]
    return {
        "char_count"        : len(text),
        "word_count"        : len(words),
        "devanagari_chars"  : len(devanagari),
        "devanagari_ratio"  : len(devanagari) / max(len(non_space), 1),
        "danda_count"       : text.count(DANDA),
        "double_danda_count": text.count(DOUBLE_DANDA),
    }


if __name__ == "__main__":
    sample = "<html> नमस्ते। अहं संस्कृतम् अधीयामि। Visit https://example.com।।"
    cleaned = clean_sanskrit_text(sample)
    print("Cleaned :", cleaned)
    print("Passes  :", quality_filter(cleaned))
    print("Stats   :", text_stats(cleaned))