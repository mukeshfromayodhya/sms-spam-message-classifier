"""Dataset loading, exact deduplication, and stratified splitting."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from spam_classifier.config import DATA_PATH, HAM_LABEL, RANDOM_STATE, SPAM_LABEL


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the UCI tab-separated file and drop exact labeled-row duplicates.

    Duplicate removal happens before splitting. Conflicting labels for an
    identical message are removed entirely to prevent the same text leaking
    across splits; the UCI copy currently contains no such conflicts.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download the UCI SMS Spam Collection, "
            "extract SMSSpamCollection, and place it at data/SMSSpamCollection."
        )

    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        label, separator, message = line.partition("\t")
        if not separator or label not in {HAM_LABEL, SPAM_LABEL}:
            raise ValueError(f"Malformed dataset row {line_number}: expected ham/spam and a tab.")
        rows.append((label, message))

    raw_counts = Counter(label for label, _ in rows)
    frame = pd.DataFrame(rows, columns=["label", "message"])
    exact_duplicates = int(frame.duplicated(subset=["label", "message"]).sum())
    frame = frame.drop_duplicates(subset=["label", "message"], keep="first")

    conflicts = frame.groupby("message", sort=False)["label"].nunique()
    conflicting_messages = set(conflicts[conflicts > 1].index)
    if conflicting_messages:
        frame = frame.loc[~frame["message"].isin(conflicting_messages)].copy()
    frame = frame.reset_index(drop=True)

    if frame.empty or set(frame["label"].unique()) != {HAM_LABEL, SPAM_LABEL}:
        raise ValueError("Dataset must contain at least one ham and one spam message.")

    metadata: dict[str, Any] = {
        "path": str(path),
        "raw_rows": len(rows),
        "raw_class_counts": dict(raw_counts),
        "exact_duplicate_rows_removed": exact_duplicates,
        "conflicting_message_texts_removed": len(conflicting_messages),
        "rows_after_deduplication": len(frame),
        "class_counts_after_deduplication": {
            str(label): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
    }
    return frame, metadata


def stratified_split(
    frame: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return disjoint 70/15/15 stratified train, validation, and test frames."""
    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=random_state,
        stratify=frame["label"],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=remainder["label"],
    )
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))  # type: ignore[return-value]
