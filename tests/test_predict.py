from __future__ import annotations

import json

import numpy as np
import pytest

from spam_classifier import predict


class RulePipeline:
    classes_ = np.array(["ham", "spam"])

    def predict_proba(self, messages: list[str]) -> np.ndarray:
        return np.array([[0.05, 0.95] if "prize" in text.lower() else [0.95, 0.05] for text in messages])


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict, "load_trusted_metadata", lambda: {"threshold": 0.8})
    monkeypatch.setattr(predict, "load_trusted_pipeline", lambda: RulePipeline())


def test_predict_message_demonstrates_ham_and_spam(stub_model: None) -> None:
    spam = predict.predict_message("Claim your prize now")
    ham = predict.predict_message("Meeting moved to three")
    assert spam == {"label": "spam", "spam_probability": 0.95, "threshold": 0.8}
    assert ham == {"label": "ham", "spam_probability": 0.05, "threshold": 0.8}


def test_empty_input_returns_unknown(stub_model: None) -> None:
    assert predict.predict_message(" \u200b ") == {
        "label": "unknown",
        "spam_probability": 0.0,
        "threshold": 0.8,
    }


def test_bad_input_fails_before_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict, "load_trusted_metadata", lambda: pytest.fail("metadata should not load"))
    with pytest.raises(TypeError):
        predict.predict_message(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="too long"):
        predict.predict_message("x" * 1001)


def test_default_log_redacts_pii_and_raw_mode_is_explicit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(predict, "LOG_DIR", tmp_path)
    result = {"label": "ham", "spam_probability": 0.1, "threshold": 0.8}
    predict._append_log("Reach me at test@example.com or +1 415 555 0199", result, log_raw=False)
    line = json.loads((tmp_path / "sample_predictions.log").read_text(encoding="utf-8").splitlines()[0])
    assert "test@example.com" not in line["message"]
    assert "+1 415 555 0199" not in line["message"]
    assert line["raw_message_enabled"] is False

    predict._append_log("private raw text", result, log_raw=True)
    lines = (tmp_path / "sample_predictions.log").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["message"] == "private raw text"
    assert json.loads(lines[-1])["raw_message_enabled"] is True
