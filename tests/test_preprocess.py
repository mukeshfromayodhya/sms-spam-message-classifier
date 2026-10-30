from __future__ import annotations

import time

import pytest

from spam_classifier.preprocess import normalize_text, redact_for_log, tokenize_text, validate_message


def test_normalization_preserves_signal_placeholders() -> None:
    message = "Email me at test@example.org or visit https://example.org; call +1 (415) 555-0199; pay ₹250."
    normalized = normalize_text(message)
    assert "<url>" in normalized
    assert "<phone>" in normalized
    assert "<curr>" in normalized
    assert "<num>" not in normalized  # currency amount is represented by <curr>
    tokens = tokenize_text(normalized)
    assert "<url>" in tokens
    assert "<phone>" in tokens
    assert "<curr>" in tokens


def test_validation_normalizes_and_removes_invisible_controls() -> None:
    cleaned = validate_message("  ＦＲＥＥ\u200b\x00 prize\nnow  ")
    assert cleaned == "FREE prize now"


def test_empty_input_is_validated_as_empty() -> None:
    assert validate_message(" \t\u200b ") == ""


def test_log_redaction_catches_bare_domains_and_invisible_contact_separators() -> None:
    message = (
        "Email Alice\u200b@exämple.com, call +1\u200b (415) 555-0199, "
        "or visit example.com/reset?token=secret123, "
        "例え.テスト/reset?token=idnsecret, "
        "xn--r8jz45g.xn--zckzah/login?key=punysecret."
    )
    masked = redact_for_log(message)
    assert "Alice" not in masked
    assert "exämple.com" not in masked
    assert "415" not in masked
    assert "secret123" not in masked
    assert "例え.テスト" not in masked
    assert "idnsecret" not in masked
    assert "xn--r8jz45g" not in masked
    assert "punysecret" not in masked
    assert masked.count("<EMAIL>") == 1
    assert masked.count("<PHONE>") == 1
    assert masked.count("<URL>") == 3


def test_message_type_and_length_are_checked() -> None:
    with pytest.raises(TypeError):
        validate_message(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="too long"):
        validate_message("a" * 1001)


def test_long_adversarial_input_is_rejected_before_feature_regexes() -> None:
    adversarial = ("+" + "9" * 200_000 + " ") * 4
    started = time.perf_counter()
    with pytest.raises(ValueError, match="too long"):
        validate_message(adversarial)
    assert time.perf_counter() - started < 1.0
