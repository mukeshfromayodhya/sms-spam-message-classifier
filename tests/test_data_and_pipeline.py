from __future__ import annotations

from pathlib import Path

from spam_classifier.data import load_dataset, stratified_split
from spam_classifier.train import build_pipeline


def test_dataset_deduplicates_before_split(tmp_path: Path) -> None:
    source = tmp_path / "SMSSpamCollection"
    source.write_text(
        "ham\tsee you soon\n"
        "ham\tsee you soon\n"
        "ham\tmeeting at three\n"
        "spam\tclaim your prize\n"
        "spam\tclaim your prize\n"
        "spam\twin cash now\n",
        encoding="utf-8",
    )
    frame, metadata = load_dataset(source)
    assert len(frame) == 4
    assert metadata["exact_duplicate_rows_removed"] == 2
    assert metadata["rows_after_deduplication"] == 4


def test_vectorizer_is_fitted_inside_pipeline_on_training_text_only() -> None:
    training_messages = [
        "ordinary meeting tomorrow",
        "hello friend call me",
        "win free prize now",
        "claim cash prize today",
    ]
    training_labels = ["ham", "ham", "spam", "spam"]
    test_marker = "zxqvtestonlysentinel"
    pipeline = build_pipeline(
        {
            "vectorizer": "tfidf",
            "ngram_range": [1, 1],
            "remove_stop_words": False,
            "model": "MultinomialNB",
            "class_weight": None,
        }
    )
    pipeline.fit(training_messages, training_labels)
    features = set(pipeline.named_steps["vectorizer"].get_feature_names_out())
    assert test_marker not in features
    assert "ordinary" in features
    # Transforming held-out text is allowed; it cannot change the fitted vocabulary.
    pipeline.named_steps["vectorizer"].transform([f"{test_marker} {training_messages[0]}"])
    assert test_marker not in set(pipeline.named_steps["vectorizer"].get_feature_names_out())


def test_stratified_splits_are_disjoint_and_keep_both_labels() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [("ham", f"ham message {index}") for index in range(20)]
        + [("spam", f"spam message {index}") for index in range(20)],
        columns=["label", "message"],
    )
    train, validation, test = stratified_split(frame)
    assert len(train) + len(validation) + len(test) == len(frame)
    assert set(train["message"]).isdisjoint(validation["message"])
    assert set(train["message"]).isdisjoint(test["message"])
    assert set(validation["message"]).isdisjoint(test["message"])
    assert all(set(part["label"]) == {"ham", "spam"} for part in (train, validation, test))
