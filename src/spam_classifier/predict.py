"""Checksum-verified single-message prediction and interactive CLI."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from spam_classifier.config import HAM_LABEL, LOG_DIR, SPAM_LABEL
from spam_classifier.preprocess import redact_for_log, validate_message
from spam_classifier.security import load_trusted_metadata, load_trusted_pipeline


def _spam_scores(pipeline: Any, messages: list[str]) -> np.ndarray:
    classes = list(pipeline.classes_)
    if SPAM_LABEL not in classes:
        raise RuntimeError("Loaded model does not contain the spam class.")
    spam_index = classes.index(SPAM_LABEL)
    return pipeline.predict_proba(messages)[:, spam_index]


def predict_message(text: str) -> dict[str, float | str]:
    """Predict one message and return its label, spam probability, and threshold."""
    clean_text = validate_message(text)
    metadata = load_trusted_metadata()
    threshold = float(metadata["threshold"])
    if not clean_text:
        return {"label": "unknown", "spam_probability": 0.0, "threshold": threshold}

    pipeline = load_trusted_pipeline()
    probability = float(_spam_scores(pipeline, [clean_text])[0])
    label = SPAM_LABEL if probability >= threshold else HAM_LABEL
    return {"label": label, "spam_probability": probability, "threshold": threshold}


def _append_log(text: str, result: dict[str, float | str], *, log_raw: bool) -> None:
    """Append a single JSON-safe line, redacting unless explicitly opted in."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    message = text if log_raw else redact_for_log(text)
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message": message,
        "raw_message_enabled": log_raw,
        **result,
    }
    log_path = LOG_DIR / "sample_predictions.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _format_result(result: dict[str, float | str]) -> str:
    return (
        f"Prediction: {result['label']} | "
        f"spam_probability={float(result['spam_probability']):.4f} | "
        f"threshold={float(result['threshold']):.6f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict whether a text message is spam.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Message text to classify.")
    group.add_argument("--interactive", action="store_true", help="Read messages until 'exit'.")
    parser.add_argument(
        "--log-raw",
        action="store_true",
        help="Explicitly allow raw message text in logs; default logs redact contact details.",
    )
    args = parser.parse_args(argv)

    if args.text is not None:
        result = predict_message(args.text)
        print(_format_result(result))
        _append_log(args.text, result, log_raw=args.log_raw)
        return 0

    print("Enter a message to classify. Type 'exit' to quit.")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() == "exit":
            break
        try:
            result = predict_message(text)
        except (TypeError, ValueError) as exc:
            print(f"Input error: {exc}")
            continue
        print(_format_result(result))
        _append_log(text, result, log_raw=args.log_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
