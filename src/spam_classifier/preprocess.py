"""Offline Unicode normalization, privacy masking, and regex tokenization."""

from __future__ import annotations

import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from spam_classifier.config import MAX_INPUT_CHARS

_EMAIL_RE = re.compile(
    r"(?iu)\b[\w.!#$%&'*+/=?^`{|}~-]{1,64}@[\w.-]{1,253}\.[^\W\d_]{2,63}\b"
)
_URL_RE = re.compile(
    r"(?iu)\b(?:https?://|www\.)[^\s<>()]+|"
    r"\b(?:[\w](?:[\w-]{0,61}[\w])?\.)+"
    r"(?:xn--[a-z0-9-]{2,59}|[^\W\d_]{2,63})"
    r"(?::\d{1,5})?(?:/[^\s<>()]*)?(?![\w-]|\.[\w])"
)
_CURRENCY_RE = re.compile(
    r"(?i)(?<!\w)(?:[$€£¥₹]|USD|INR|GBP|EUR|Rs\.?)[ \t]*\d[\d,]*(?:\.\d+)?"
)
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s()./-]{5,}\d(?!\w)")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_TOKEN_RE = re.compile(r"(?i)<(?:url|phone|curr|num)>|[a-z]+(?:'[a-z]+)?|\d+")


def validate_message(text: str) -> str:
    """Validate and sanitize inference text before running any regex features."""
    if not isinstance(text, str):
        raise TypeError("Message must be a string.")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"Message is too long; maximum is {MAX_INPUT_CHARS} characters.")

    normalized = unicodedata.normalize("NFKC", text)
    cleaned: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category in {"Cc", "Cf", "Cs"}:
            if char in "\t\n\r":
                cleaned.append(" ")
            continue
        cleaned.append(char)
    normalized = "".join(cleaned)
    if len(normalized) > MAX_INPUT_CHARS:
        raise ValueError(f"Message is too long; maximum is {MAX_INPUT_CHARS} characters.")
    return normalized.strip()


def normalize_text(text: str) -> str:
    """Normalize a message and replace risky/high-signal fields with tokens."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _EMAIL_RE.sub(" <url> ", normalized)
    normalized = _URL_RE.sub(" <url> ", normalized)
    normalized = _CURRENCY_RE.sub(" <curr> ", normalized)
    normalized = _PHONE_RE.sub(" <phone> ", normalized)
    normalized = _NUMBER_RE.sub(" <num> ", normalized)
    return " ".join(normalized.split())


def tokenize_text(text: str) -> list[str]:
    """Tokenize normalized text while retaining placeholder features."""
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def redact_for_log(text: str) -> str:
    """Mask URLs, email addresses, and phone numbers in message logs."""
    # Apply the same control/format-character cleanup used for inference so
    # invisible separators cannot split an address or phone number.
    masked = validate_message(text)
    masked = _EMAIL_RE.sub("<EMAIL>", masked)
    masked = _URL_RE.sub("<URL>", masked)
    masked = _PHONE_RE.sub("<PHONE>", masked)
    return " ".join(masked.split())


def stop_words() -> frozenset[str]:
    """Return scikit-learn's bundled English stop-word set (offline-safe)."""
    return ENGLISH_STOP_WORDS
