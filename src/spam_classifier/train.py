"""Train, compare, evaluate, and document the SMS spam classifier."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

from spam_classifier.config import (
    ARTIFACT_DIR,
    DATA_PATH,
    DOCS_DIR,
    HAM_LABEL,
    LOG_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    SPAM_LABEL,
    TARGET_SPAM_PRECISION,
)
from spam_classifier.data import load_dataset, stratified_split
from spam_classifier.preprocess import normalize_text, redact_for_log, stop_words, tokenize_text, validate_message
from spam_classifier.security import (
    METADATA_PATH,
    PIPELINE_PATH,
    VECTORIZER_PATH,
    atomic_joblib_dump,
    atomic_write_artifact_text,
    ensure_safe_artifact_targets,
    write_checksum_manifest,
)

SCORING = {
    "precision_spam": make_scorer(precision_score, pos_label=SPAM_LABEL, zero_division=0),
    "recall_spam": make_scorer(recall_score, pos_label=SPAM_LABEL, zero_division=0),
    "f1_spam": make_scorer(f1_score, pos_label=SPAM_LABEL, zero_division=0),
}

MODERN_DEMO_MESSAGES = [
    {
        "scenario": "UPI account suspension scam",
        "expected_label": SPAM_LABEL,
        "message": "Your UPI account is blocked. Share the OTP now to avoid suspension.",
    },
    {
        "scenario": "KYC renewal link scam",
        "expected_label": SPAM_LABEL,
        "message": "KYC expired today. Verify immediately at www.fake-kyc.example to keep your wallet active.",
    },
    {
        "scenario": "UPI refund fee scam",
        "expected_label": SPAM_LABEL,
        "message": "UPI refund pending. Pay Rs 2 verification fee to receive Rs 10000 now.",
    },
    {
        "scenario": "lottery prize scam",
        "expected_label": SPAM_LABEL,
        "message": "Congratulations! You won a lottery prize. Claim now by replying with your bank details.",
    },
    {
        "scenario": "delivery redelivery scam",
        "expected_label": SPAM_LABEL,
        "message": "Package held. Pay a redelivery fee at https://fake-delivery.example today.",
    },
    {
        "scenario": "ordinary meeting update",
        "expected_label": HAM_LABEL,
        "message": "Meeting moved to 3 pm. See you at the office.",
    },
    {
        "scenario": "ordinary grocery reminder",
        "expected_label": HAM_LABEL,
        "message": "Can you pick up milk on your way home?",
    },
    {
        "scenario": "ordinary account safety notice",
        "expected_label": HAM_LABEL,
        "message": "Your verification code is 000000. Never share this code with anyone.",
    },
]

ADVERSARIAL_DEMO_MESSAGES = [
    ("plain baseline", "Congratulations! You won a free prize, click now"),
    ("punctuated spelling", "Congratulations! You won a f.r.e.e prize, click now"),
    ("digit substitution", "Congratulations! You won a FR33 prize, click now"),
    ("spaced letters", "Congratulations! You won a f r e e prize, click now"),
    ("Cyrillic homoglyphs", "Cоngratulations! Yоu wоn a free prize, click now"),
    ("zero-width character", "Congratulations! You\u200bwon a free prize, click now"),
]


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build a vectorizer/classifier Pipeline from one experiment definition."""
    vectorizer_class = CountVectorizer if config["vectorizer"] == "count" else TfidfVectorizer
    vectorizer_kwargs: dict[str, Any] = {
        "preprocessor": normalize_text,
        "tokenizer": tokenize_text,
        "token_pattern": None,
        "lowercase": False,
        "ngram_range": tuple(config["ngram_range"]),
        "stop_words": list(stop_words()) if config["remove_stop_words"] else None,
        "max_features": 30000,
    }
    if vectorizer_class is TfidfVectorizer:
        vectorizer_kwargs["sublinear_tf"] = True
    vectorizer = vectorizer_class(**vectorizer_kwargs)

    if config["model"] == "MultinomialNB":
        classifier = MultinomialNB(alpha=0.5)
    else:
        classifier = LogisticRegression(
            C=1.0,
            class_weight=config["class_weight"],
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def experiment_grid() -> list[dict[str, Any]]:
    """Create a compact, controlled Count/TF-IDF, n-gram, stop-word, and model grid."""
    configs: list[dict[str, Any]] = []
    for vectorizer in ("count", "tfidf"):
        for ngram_range in ((1, 1), (1, 2)):
            for remove_stop_words in (False, True):
                for model, class_weight in (
                    ("MultinomialNB", None),
                    ("LogisticRegression", None),
                    ("LogisticRegression", "balanced"),
                ):
                    ngram_name = "unigram" if ngram_range == (1, 1) else "unigram_bigram"
                    stop_name = "stop_removed" if remove_stop_words else "stop_kept"
                    weight_name = f"_{class_weight}" if class_weight else ""
                    name = f"{vectorizer}_{ngram_name}_{stop_name}_{model}{weight_name}"
                    configs.append(
                        {
                            "name": name,
                            "vectorizer": vectorizer,
                            "ngram_range": list(ngram_range),
                            "remove_stop_words": remove_stop_words,
                            "model": model,
                            "class_weight": class_weight,
                        }
                    )
    return configs


def cross_validate_config(config: dict[str, Any], x_train: list[str], y_train: list[str]) -> dict[str, Any]:
    """Score one pipeline on stratified folds drawn only from the training split."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    result = cross_validate(
        build_pipeline(config),
        x_train,
        y_train,
        scoring=SCORING,
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    row = dict(config)
    for key in ("precision_spam", "recall_spam", "f1_spam"):
        values = result[f"test_{key}"]
        row[f"{key}_mean"] = float(np.mean(values))
        row[f"{key}_std"] = float(np.std(values, ddof=1))
    return row


def choose_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by training-CV spam precision, breaking ties with spam F1."""
    return max(results, key=lambda row: (row["precision_spam_mean"], row["f1_spam_mean"]))


def spam_scores(pipeline: Pipeline, messages: list[str]) -> np.ndarray:
    classes = list(pipeline.classes_)
    return pipeline.predict_proba(messages)[:, classes.index(SPAM_LABEL)]


def tune_threshold(
    y_validation: list[str],
    scores: np.ndarray,
    target_precision: float = TARGET_SPAM_PRECISION,
) -> dict[str, Any]:
    """Maximize validation recall subject to the stated spam precision target."""
    candidates: list[dict[str, float]] = []
    for threshold in np.unique(scores):
        predictions = np.where(scores >= threshold, SPAM_LABEL, HAM_LABEL)
        if not np.any(predictions == SPAM_LABEL):
            continue
        precision = float(precision_score(y_validation, predictions, pos_label=SPAM_LABEL, zero_division=0))
        if precision >= target_precision:
            candidates.append(
                {
                    "threshold": float(threshold),
                    "precision": precision,
                    "recall": float(recall_score(y_validation, predictions, pos_label=SPAM_LABEL, zero_division=0)),
                    "f1": float(f1_score(y_validation, predictions, pos_label=SPAM_LABEL, zero_division=0)),
                }
            )
    if candidates:
        best = max(candidates, key=lambda row: (row["recall"], row["precision"], row["threshold"]))
        met = True
    else:
        best = {
            "threshold": float(np.nextafter(float(np.max(scores)), np.inf)),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        met = False
    return {
        **best,
        "target_precision": target_precision,
        "constraint_met": met,
        "selection_rule": "maximum validation spam recall among thresholds with spam precision at least target_precision",
    }


def _save_plots(y_test: list[str], scores: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_test, predictions, labels=[HAM_LABEL, SPAM_LABEL])
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=[HAM_LABEL, SPAM_LABEL])
    ax.set_yticks([0, 1], labels=[HAM_LABEL, SPAM_LABEL])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Held-out test confusion matrix")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    y_binary = np.array([label == SPAM_LABEL for label in y_test], dtype=int)
    precision, recall, _ = precision_recall_curve(y_binary, scores)
    average_precision = float(average_precision_score(y_binary, scores))
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.step(recall, precision, where="post", color="#6c3fb5", label=f"Average precision = {average_precision:.3f}")
    ax.axhline(float(np.mean(y_binary)), color="#666666", linestyle="--", label="Spam prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Held-out test precision-recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "precision_recall_curve.png", dpi=160)
    plt.close(fig)
    return matrix.astype(int).tolist()


def _top_features(pipeline: Pipeline, limit: int = 15) -> dict[str, list[dict[str, Any]]]:
    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]
    names = vectorizer.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        spam_weight = classifier.coef_[0]
    else:
        spam_index = list(classifier.classes_).index(SPAM_LABEL)
        ham_index = list(classifier.classes_).index(HAM_LABEL)
        spam_weight = classifier.feature_log_prob_[spam_index] - classifier.feature_log_prob_[ham_index]
    spam_indices = np.argsort(spam_weight)[-limit:][::-1]
    ham_indices = np.argsort(spam_weight)[:limit]
    return {
        "spam": [{"feature": str(names[index]), "weight": float(spam_weight[index])} for index in spam_indices],
        "ham": [{"feature": str(names[index]), "weight": float(-spam_weight[index])} for index in ham_indices],
    }


def _message_length_stats(frame: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for label, group in frame.groupby("label", sort=True):
        lengths = group["message"].str.len()
        stats[str(label)] = {
            "messages": int(len(group)),
            "characters_mean": float(lengths.mean()),
            "characters_median": float(lengths.median()),
            "characters_min": int(lengths.min()),
            "characters_max": int(lengths.max()),
            "tokens_mean_after_preprocessing": float(
                np.mean([len(tokenize_text(normalize_text(message))) for message in group["message"]])
            ),
        }
    return stats


def _likely_reason(message: str, *, false_positive: bool) -> str:
    lower = message.lower()
    cues = ("free", "win", "won", "claim", "urgent", "offer", "prize", "call", "cash", "congrat")
    if false_positive:
        if any(term in lower for term in cues):
            return "Promotional or urgency wording overlaps vocabulary learned from spam examples."
        if "http" in lower or "www." in lower or "@" in lower:
            return "A link or contact detail is a common spam cue, though legitimate notices can contain one."
        return "The short or transactional wording resembles patterns in the training corpus; review context manually."
    if any(term in lower for term in ("verify", "account", "update", "service", "delivery", "click", "bank")):
        return "The message uses service/account wording without enough learned promotional cues to cross the tuned threshold."
    return "The wording or formatting differs from the older training examples and did not cross the tuned threshold."


def _evaluate_demo_messages(pipeline: Pipeline, threshold: float, demos: list[dict[str, str]]) -> list[dict[str, Any]]:
    texts = [validate_message(row["message"]) for row in demos]
    scores = spam_scores(pipeline, texts)
    results: list[dict[str, Any]] = []
    for demo, score in zip(demos, scores, strict=True):
        results.append(
            {
                "scenario": demo["scenario"],
                "expected_label_for_demo_only": demo["expected_label"],
                "predicted_label": SPAM_LABEL if score >= threshold else HAM_LABEL,
                "spam_probability": float(score),
                "threshold": float(threshold),
                "message_redacted": redact_for_log(demo["message"]),
            }
        )
    return results


def _evaluate_adversarial_demos(pipeline: Pipeline, threshold: float) -> list[dict[str, Any]]:
    texts = [validate_message(text) for _, text in ADVERSARIAL_DEMO_MESSAGES]
    scores = spam_scores(pipeline, texts)
    return [
        {
            "variant": variant,
            "predicted_label": SPAM_LABEL if score >= threshold else HAM_LABEL,
            "spam_probability": float(score),
            "threshold": float(threshold),
            "message_redacted": redact_for_log(text),
        }
        for (variant, text), score in zip(ADVERSARIAL_DEMO_MESSAGES, scores, strict=True)
    ]


def _write_analysis(
    metrics: dict[str, Any],
    frame: pd.DataFrame,
    y_test: list[str],
    predictions: np.ndarray,
    scores: np.ndarray,
    demo_rows: list[dict[str, Any]],
    adversarial_rows: list[dict[str, Any]],
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    false_positives: list[str] = []
    false_negatives: list[str] = []
    test_messages = metrics["_test_messages"]
    threshold = metrics["threshold"]["value"]
    for message, actual, predicted, score in zip(test_messages, y_test, predictions, scores, strict=True):
        item = (
            f"- Actual `{actual}`, predicted `{predicted}`, spam probability `{score:.6f}`: "
            f"{redact_for_log(message)}. Likely reason: "
            f"{_likely_reason(message, false_positive=(actual == HAM_LABEL and predicted == SPAM_LABEL))}"
        )
        if actual == HAM_LABEL and predicted == SPAM_LABEL:
            false_positives.append(item)
        elif actual == SPAM_LABEL and predicted == HAM_LABEL:
            false_negatives.append(item)

    top = metrics["top_features"]
    text = [
        "# Test-set error analysis",
        "",
        "This report was generated from the held-out test split after model selection and validation threshold tuning. The test labels were used only for this final evaluation and the error list below.",
        "",
        f"- Selected model: `{metrics['selected_model']['name']}`",
        f"- Validation threshold: `{threshold:.12g}` (precision target `{metrics['threshold']['target_precision']:.2f}`; constraint met: `{metrics['threshold']['constraint_met']}`)",
        f"- Test spam precision: `{metrics['test']['spam']['precision']:.6f}`; recall: `{metrics['test']['spam']['recall']:.6f}`; F1: `{metrics['test']['spam']['f1-score']:.6f}`",
        f"- Confusion matrix (actual rows ham/spam, predicted columns ham/spam): `{metrics['test']['confusion_matrix']}`",
        "",
        "## False positives (all held-out test errors)",
        "",
        *(false_positives or ["None on the held-out test split."]),
        "",
        "## False negatives (all held-out test errors)",
        "",
        *(false_negatives or ["None on the held-out test split."]),
        "",
        "## Indicative features",
        "",
        "These are learned model weights, not causal explanations.",
        "",
        "| More indicative of spam | Weight | More indicative of ham | Weight |",
        "|---|---:|---|---:|",
    ]
    for spam_feature, ham_feature in zip(top["spam"], top["ham"], strict=True):
        text.append(
            f"| `{spam_feature['feature']}` | {spam_feature['weight']:.5f} | "
            f"`{ham_feature['feature']}` | {ham_feature['weight']:.5f} |"
        )
    text.extend(["", "## Message length statistics after deduplication", "", "| Class | Messages | Mean chars | Median chars | Min | Max | Mean preprocessed tokens |", "|---|---:|---:|---:|---:|---:|---:|"])
    for label, row in metrics["message_length_stats"].items():
        text.append(
            f"| {label} | {row['messages']} | {row['characters_mean']:.2f} | "
            f"{row['characters_median']:.1f} | {row['characters_min']} | {row['characters_max']} | "
            f"{row['tokens_mean_after_preprocessing']:.2f} |"
        )
    text.extend(["", "## Modern SMS examples (DEMO ONLY)", "", "These hand-written examples are illustrative only. They are never included in training, validation, test metrics, or model selection.", "", "| Scenario | Expected demo label | Model prediction | Spam probability | Redacted message |", "|---|---|---|---:|---|"])
    for row in demo_rows:
        text.append(
            f"| {row['scenario']} | {row['expected_label_for_demo_only']} | {row['predicted_label']} | "
            f"{row['spam_probability']:.4f} | {row['message_redacted']} |"
        )
    text.extend(["", "## Obfuscation checks (DEMO ONLY)", "", "These adversarial-style examples are separate from the test set and are not evaluation metrics.", "", "| Variant | Prediction | Spam probability | Redacted example |", "|---|---|---:|---|"])
    for row in adversarial_rows:
        text.append(
            f"| {row['variant']} | {row['predicted_label']} | {row['spam_probability']:.4f} | {row['message_redacted']} |"
        )
    text.extend(
        [
            "",
            "Interpretation: zero-width characters are stripped before feature extraction, but punctuation, digit substitutions, spacing, and non-Latin homoglyphs can still change the token sequence. The observed predictions above are the only claims made about these examples.",
            "",
            "## Limitations",
            "",
            "The UCI corpus is old and drawn from English-language SMS sources, including UK and Singapore contexts. Its vocabulary, carriers, scam patterns, and class balance do not represent current messaging traffic. In particular, modern OTP/KYC/UPI and delivery scams in India can be out of domain. A high validation precision target does not guarantee the same precision in production. The model should be treated as a review aid, and messages near the decision boundary should not be automatically blocked without a human or policy layer.",
            "",
        ]
    )
    (DOCS_DIR / "ANALYSIS.md").write_text("\n".join(text), encoding="utf-8")


def _write_research(metrics: dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    data = metrics["dataset"]
    class_counts = data["raw_class_counts"]
    dedup_counts = data["class_counts_after_deduplication"]
    content = f"""# Research notes

## Dataset provenance and observed balance

This project uses the official UCI Machine Learning Repository **SMS Spam Collection**, cited as Almeida and Hidalgo (2011), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84). The file was downloaded from the [UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection), which lists a CC BY 4.0 license. The downloaded raw file contains {data['raw_rows']:,} labeled English SMS messages: {class_counts.get('ham', 0):,} ham and {class_counts.get('spam', 0):,} spam. Exact duplicate labeled rows were removed before splitting ({data['exact_duplicate_rows_removed']:,} rows); the deduplicated dataset contains {data['rows_after_deduplication']:,} rows ({dedup_counts.get('ham', 0):,} ham and {dedup_counts.get('spam', 0):,} spam). No conflicting labels for an identical message were observed in this copy.

## Method choices

- All vectorizers are inside scikit-learn pipelines so vocabulary and IDF statistics are learned only from the fold's training messages. Scikit-learn's leakage guidance recommends fitting transformations on training subsets and using pipelines for cross-validation ([common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)).
- Model selection uses stratified five-fold cross-validation on the 70% training split. The validation split is reserved for choosing the decision threshold; the held-out test split is evaluated only after those choices.
- Spam precision is prioritized because false positives can incorrectly flag legitimate messages. The validation threshold maximizes spam recall while requiring spam precision of at least {metrics['threshold']['target_precision']:.2f}. Precision-recall curves summarize this threshold tradeoff ([scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)).
- Preprocessing is offline: Unicode NFKC normalization, regex tokenization, placeholders for URLs/contact details/currency/numbers, and scikit-learn's built-in English stop-word set.

## Known limitations and corpus issues

- The corpus is English-only and dates from the early 2010s. UCI describes it as a collection of SMS messages with ham/spam labels; it should not be assumed to represent modern traffic.
- Exact duplicate labeled rows are removed before splitting. The data loader also discards any identical message text carrying conflicting labels, to prevent text overlap across splits; none were found in the downloaded copy.
- The corpus reflects older UK and Singapore message sources. It may not transfer to current Indian OTP, KYC, UPI, delivery, or regional-language scams. This domain-shift concern is a limitation inferred from the corpus age and its published source context, not a measured benchmark result.
- The original class balance is about {class_counts.get('spam', 0) / max(data['raw_rows'], 1):.1%} spam. Deployment traffic may have a different prevalence, which changes precision even when ranking quality is unchanged.
- User-written modern spam examples in `docs/ANALYSIS.md` are **DEMO ONLY** and are excluded from all fitted models and reported metrics.

## References

1. Almeida, T. & Hidalgo, J. (2011). [SMS Spam Collection, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
2. Scikit-learn developers. [Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
3. Scikit-learn developers. [Classification metrics and precision-recall evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
4. Scikit-learn developers. [Model persistence security considerations](https://scikit-learn.org/stable/model_persistence.html).
5. Joblib maintainers. [`joblib.load` warning](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html).
"""
    (DOCS_DIR / "RESEARCH.md").write_text(content, encoding="utf-8")


def _write_security_doc() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    content = """# Security notes

## Model artifacts

The trained pipeline, separately exported vectorizer, and JSON metadata are saved under `artifacts/`. Training rejects symlinked artifact directories and destinations before writing, writes each bundle file through a same-directory temporary file, and replaces the fixed destination atomically. `sha256-manifest.json` stores SHA-256 digests and byte sizes. Prediction verifies the complete fixed artifact bundle before loading the pipeline, refuses mismatches, and reads the verified model bytes from memory to avoid a path swap between verification and deserialization. The loader uses project-defined paths only; it does not accept a path from command-line input.

SHA-256 detects a changed artifact only while the manifest remains trusted. It is not a digital signature: an attacker able to replace both the model and its manifest can replace both values. Keep the model and manifest in a trusted, access-controlled location. Python `joblib` uses pickle-based deserialization and can execute code when loading a malicious file. Never use this loader for downloaded or otherwise untrusted model bundles. Scikit-learn recommends safer inspection-oriented formats such as `skops.io` when exchanging artifacts across trust boundaries ([model persistence documentation](https://scikit-learn.org/stable/model_persistence.html)); this project retains joblib because the assignment requests it.

## Message input and logging

- Inference requires a string, removes control and zero-width format characters, applies NFKC normalization, and rejects text longer than 1,000 characters before feature regexes run.
- Regex patterns use bounded/simple character classes and do not evaluate message contents. There is no `eval`, `exec`, shell, subprocess, or network call in inference.
- Logs redact URLs, email addresses, and phone numbers by default. Raw input is logged only when `--log-raw` is explicitly provided. The `logs/` directory is gitignored.
- Inference runs locally and does not call external services.

## Security checks

Tests cover checksum mismatch refusal, type/length/empty-input handling, control and zero-width characters, default log redaction including invisible separators and bare-domain links, and long adversarial strings. Static-analysis and dependency-audit results are recorded in `security-checks.json` and summarized in the project README. The local editable project package is skipped by `pip-audit` because it is not published to an index; its source is reviewed and tested in this repository.
"""
    (DOCS_DIR / "SECURITY.md").write_text(content, encoding="utf-8")


def _write_readme(metrics: dict[str, Any]) -> None:
    data = metrics["dataset"]
    report = metrics["test"]["classification_report"]
    lines = [
        "# Spam Message Classifier",
        "",
        "A classic machine-learning SMS spam classifier using an offline preprocessing pipeline, Count/TF-IDF features, Multinomial Naive Bayes, and Logistic Regression.",
        "",
        "## Dataset",
        "",
        f"Uses the official [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection). The raw copy has {data['raw_rows']:,} messages; {data['exact_duplicate_rows_removed']:,} exact duplicate rows were removed before the stratified split. The remaining {data['rows_after_deduplication']:,} messages are split 70%/15%/15% into train/validation/test with seed 42.",
        "",
        "## Setup and run",
        "",
        "```powershell",
        "python -m pip install -r requirements.txt",
        "python -m spam_classifier.train",
        'python -m spam_classifier.predict --text "Congratulations! You won a free prize, click now"',
        "python -m spam_classifier.predict --interactive",
        "pytest -q",
        "```",
        "",
        "## Held-out test metrics",
        "",
        f"Selected model: `{metrics['selected_model']['name']}`. Threshold `{metrics['threshold']['value']:.12g}` was selected on validation to maximize spam recall subject to spam precision ≥ {metrics['threshold']['target_precision']:.2f}; validation constraint met: **{metrics['threshold']['constraint_met']}**. Test data was evaluated once after model and threshold selection.",
        "",
        f"PR-AUC (average precision): **{metrics['test']['pr_auc']:.6f}** · ROC-AUC: **{metrics['test']['roc_auc']:.6f}**",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in (HAM_LABEL, SPAM_LABEL):
        row = report[label]
        lines.append(
            f"| {label} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1-score']:.6f} | {int(row['support'])} |"
        )
    lines.extend(
        [
            "",
            f"Confusion matrix in actual ham/spam rows and predicted ham/spam columns: `{metrics['test']['confusion_matrix']}`.",
            "",
            "![Held-out test confusion matrix](artifacts/confusion_matrix.png)",
            "",
            "![Held-out test precision-recall curve](artifacts/precision_recall_curve.png)",
            "",
            "## Cross-validation model comparison",
            "",
            "All means and standard deviations below come from stratified five-fold CV on the training split only. Models are ordered by mean spam precision, with spam F1 used to break ties.",
            "",
            "| Experiment | Spam precision mean ± SD | Spam recall mean ± SD | Spam F1 mean ± SD |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in metrics["model_comparison"]:
        lines.append(
            f"| `{row['name']}` | {row['precision_spam_mean']:.4f} ± {row['precision_spam_std']:.4f} | "
            f"{row['recall_spam_mean']:.4f} ± {row['recall_spam_std']:.4f} | "
            f"{row['f1_spam_mean']:.4f} ± {row['f1_spam_std']:.4f} |"
        )
    ablation = metrics["stop_word_ablation"]
    lines.extend(
        [
            "",
            "## Stop-word ablation",
            "",
            f"Controlled CV ablation using the selected model/vectorizer/ngram configuration: stop words removed → spam precision {ablation['removed']['precision_spam_mean']:.4f} ± {ablation['removed']['precision_spam_std']:.4f}, F1 {ablation['removed']['f1_spam_mean']:.4f} ± {ablation['removed']['f1_spam_std']:.4f}; stop words kept → spam precision {ablation['kept']['precision_spam_mean']:.4f} ± {ablation['kept']['precision_spam_std']:.4f}, F1 {ablation['kept']['f1_spam_mean']:.4f} ± {ablation['kept']['f1_spam_std']:.4f}.",
            "",
            "## Sample prediction log",
            "",
            "The following lines are taken from `logs/sample_predictions.log`. Messages in the log are redacted unless `--log-raw` is explicitly used.",
            "",
            "```text",
            *metrics["sample_log_preview"],
            "```",
            "",
            "## Deliverables",
            "",
            "- `src/spam_classifier/train.py`: training, training-only CV comparison, validation threshold tuning, final held-out evaluation, plots, and reports.",
            "- `notebooks/spam_classifier_walkthrough.ipynb`: concise reproducible workflow; `notebooks/spam_classifier_complete_project.ipynb`: full end-to-end, explained project notebook.",
            "- `artifacts/pipeline.joblib`: fitted preprocessing and model pipeline; `artifacts/vectorizer.joblib` is the fitted vectorizer export.",
            "- `artifacts/sha256-manifest.json` and `artifacts/model_metadata.json`: integrity manifest and model threshold metadata.",
            "- `artifacts/metrics.json`, `artifacts/confusion_matrix.png`, and `artifacts/precision_recall_curve.png`.",
            "- `docs/RESEARCH.md`, `docs/SECURITY.md`, and `docs/ANALYSIS.md`.",
            "",
            "## Security and limitations",
            "",
            "The prediction loader checks the fixed local artifact bundle against SHA-256 before joblib deserialization. Joblib is pickle-based: only load artifacts produced by this project in a trusted environment. Logs redact phone numbers, emails, and URLs by default, and inference has no network calls. Security details and test results are in [docs/SECURITY.md](docs/SECURITY.md).",
            "",
            "The corpus is old, English-only, and does not represent modern Indian OTP/KYC/UPI/delivery scams. Demo examples are labeled **DEMO ONLY** and excluded from all metrics. Review [docs/RESEARCH.md](docs/RESEARCH.md) and [docs/ANALYSIS.md](docs/ANALYSIS.md) before relying on predictions.",
            "",
            "## Security tool status",
            "",
            "| Tool | Command | Result |",
            "| --- | --- | --- |",
            f"| Bandit | `{metrics['security_checks']['bandit'].get('command') or 'not run'}` | {metrics['security_checks']['bandit'].get('status', 'not recorded')}: {metrics['security_checks']['bandit'].get('details', '')} |",
            f"| pip-audit | `{metrics['security_checks']['pip_audit'].get('command') or 'not run'}` | {metrics['security_checks']['pip_audit'].get('status', 'not recorded')}: {metrics['security_checks']['pip_audit'].get('details', '')} |",
            f"| pytest | `{metrics['security_checks']['pytest'].get('command') or 'not run'}` | {metrics['security_checks']['pytest'].get('status', 'not recorded')}: {metrics['security_checks']['pytest'].get('details', '')} |",
            "",
        ]
    )
    (PROJECT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_notebook() -> None:
    notebook_dir = PROJECT_ROOT / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Spam Message Classifier Walkthrough\n",
                "\n",
                "This notebook calls the same reproducible workflow as the command-line trainer. It uses the local UCI file, keeps vectorization inside training pipelines, tunes the spam threshold on validation, and evaluates the held-out test split at the end. Hand-written modern examples are DEMO ONLY.\n",
                "\n",
                "Run from the project root after `python -m pip install -r requirements.txt`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from spam_classifier.train import main\n",
                "\n",
                "main()\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "\n",
                "metrics = json.loads(Path('artifacts/metrics.json').read_text(encoding='utf-8'))\n",
                "pd.DataFrame(metrics['model_comparison']).sort_values(\n",
                "    ['precision_spam_mean', 'f1_spam_mean'], ascending=False\n",
                ")[[\n",
                "    'name', 'precision_spam_mean', 'precision_spam_std',\n",
                "    'recall_spam_mean', 'f1_spam_mean', 'f1_spam_std'\n",
                "]]\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from spam_classifier.predict import predict_message\n",
                "\n",
                "predict_message('Congratulations! You won a free prize, click now')\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (notebook_dir / "spam_classifier_walkthrough.ipynb").write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _load_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _load_security_checks() -> dict[str, Any]:
    """Read the latest locally recorded static/dependency audit results."""
    status_path = DOCS_DIR / "security-checks.json"
    if not status_path.is_file():
        return {
            "bandit": {"status": "not run", "command": None, "details": None},
            "pip_audit": {"status": "not run", "command": None, "details": None},
            "pytest": {"status": "not recorded", "command": None, "details": None},
        }
    try:
        checks = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "bandit": {"status": "status file unreadable", "command": None, "details": None},
            "pip_audit": {"status": "status file unreadable", "command": None, "details": None},
            "pytest": {"status": "status file unreadable", "command": None, "details": None},
        }
    if not isinstance(checks, dict):
        raise ValueError("Security check status must be a JSON object.")
    return {
        name: checks.get(name, {"status": "not recorded", "command": None, "details": None})
        for name in ("bandit", "pip_audit", "pytest")
    }


def run_training() -> dict[str, Any]:
    """Run the complete non-leaking experiment and write required deliverables."""
    frame, data_metadata = load_dataset(DATA_PATH)
    train_frame, validation_frame, test_frame = stratified_split(frame)
    x_train = train_frame["message"].tolist()
    y_train = train_frame["label"].tolist()
    x_validation = validation_frame["message"].tolist()
    y_validation = validation_frame["label"].tolist()
    x_test = test_frame["message"].tolist()
    y_test = test_frame["label"].tolist()

    model_comparison = [cross_validate_config(config, x_train, y_train) for config in experiment_grid()]
    model_comparison.sort(
        key=lambda row: (row["precision_spam_mean"], row["f1_spam_mean"]), reverse=True
    )
    selected = choose_model(model_comparison)
    pipeline = build_pipeline(selected)
    pipeline.fit(x_train, y_train)

    validation_scores = spam_scores(pipeline, x_validation)
    threshold = tune_threshold(y_validation, validation_scores)

    # Final test use: one set of scores, labels, and metrics after model/threshold selection.
    test_scores = spam_scores(pipeline, x_test)
    test_predictions = np.where(test_scores >= threshold["threshold"], SPAM_LABEL, HAM_LABEL)
    test_report = classification_report(
        y_test,
        test_predictions,
        labels=[HAM_LABEL, SPAM_LABEL],
        output_dict=True,
        zero_division=0,
    )
    matrix = _save_plots(y_test, test_scores, test_predictions)
    y_test_binary = np.array([label == SPAM_LABEL for label in y_test], dtype=int)
    test_metrics = {
        "classification_report": test_report,
        "ham": test_report[HAM_LABEL],
        "spam": test_report[SPAM_LABEL],
        "accuracy": float(test_report["accuracy"]),
        "macro_avg": test_report["macro avg"],
        "weighted_avg": test_report["weighted avg"],
        "pr_auc": float(average_precision_score(y_test_binary, test_scores)),
        "roc_auc": float(roc_auc_score(y_test_binary, test_scores)),
        "confusion_matrix": matrix,
        "confusion_matrix_labels": {"rows_actual": [HAM_LABEL, SPAM_LABEL], "columns_predicted": [HAM_LABEL, SPAM_LABEL]},
    }

    counterpart = next(
        row
        for row in model_comparison
        if row["vectorizer"] == selected["vectorizer"]
        and row["ngram_range"] == selected["ngram_range"]
        and row["model"] == selected["model"]
        and row["class_weight"] == selected["class_weight"]
        and row["remove_stop_words"] != selected["remove_stop_words"]
    )
    selected_for_stop_removed = selected if selected["remove_stop_words"] else counterpart
    selected_for_stop_kept = counterpart if selected["remove_stop_words"] else selected
    ablation = {"removed": selected_for_stop_removed, "kept": selected_for_stop_kept}

    metadata = {
        "threshold": threshold["threshold"],
        "target_spam_precision": TARGET_SPAM_PRECISION,
        "selected_model": selected,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
    }
    ensure_safe_artifact_targets()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    atomic_joblib_dump(pipeline, PIPELINE_PATH)
    atomic_joblib_dump(pipeline.named_steps["vectorizer"], VECTORIZER_PATH)
    atomic_write_artifact_text(METADATA_PATH, json.dumps(metadata, indent=2) + "\n")
    write_checksum_manifest()

    demo_predictions = _evaluate_demo_messages(pipeline, threshold["threshold"], MODERN_DEMO_MESSAGES)
    adversarial_predictions = _evaluate_adversarial_demos(pipeline, threshold["threshold"])

    # Keep the requested sample log useful while applying the default privacy redaction.
    from spam_classifier.predict import _append_log, predict_message

    for row in MODERN_DEMO_MESSAGES[:4]:
        _append_log(row["message"], predict_message(row["message"]), log_raw=False)

    log_preview: list[str] = []
    log_path = LOG_DIR / "sample_predictions.log"
    for line in log_path.read_text(encoding="utf-8").splitlines()[-4:]:
        try:
            item = json.loads(line)
            log_preview.append(
                f"{item['timestamp_utc']} | {item['message']} => {item['label']} "
                f"(spam_probability={float(item['spam_probability']):.4f}, threshold={float(item['threshold']):.6f})"
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    metrics: dict[str, Any] = {
        "dataset": {
            **data_metadata,
            "sha256": __import__("hashlib").sha256(DATA_PATH.read_bytes()).hexdigest(),
            "split_sizes": {
                "train": len(train_frame),
                "validation": len(validation_frame),
                "test": len(test_frame),
            },
            "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "random_state": RANDOM_STATE,
        },
        "model_selection": {
            "selected_by": "highest mean CV spam precision; ties broken by mean CV spam F1",
            "cv": {"folds": 5, "stratified": True, "training_split_only": True},
        },
        "selected_model": selected,
        "model_comparison": model_comparison,
        "stop_word_ablation": ablation,
        "threshold": {
            "value": threshold["threshold"],
            "target_precision": threshold["target_precision"],
            "validation_precision": threshold["precision"],
            "validation_recall": threshold["recall"],
            "validation_f1": threshold["f1"],
            "constraint_met": threshold["constraint_met"],
            "selection_rule": threshold["selection_rule"],
        },
        "test": test_metrics,
        "top_features": _top_features(pipeline),
        "message_length_stats": _message_length_stats(frame),
        "versions": _load_versions(),
        "artifact_paths": {
            "pipeline": str(PIPELINE_PATH.relative_to(PROJECT_ROOT)),
            "vectorizer": str(VECTORIZER_PATH.relative_to(PROJECT_ROOT)),
            "checksum_manifest": "artifacts/sha256-manifest.json",
            "metrics": "artifacts/metrics.json",
        },
        "security_checks": {
            "artifact_checksum_verification": "implemented and covered by pytest",
            **_load_security_checks(),
        },
        "_test_messages": x_test,
        "sample_log_preview": log_preview,
    }
    _write_analysis(metrics, frame, y_test, test_predictions, test_scores, demo_predictions, adversarial_predictions)
    _write_research(metrics)
    _write_security_doc()
    _write_notebook()

    # Private messages are used only to write the local error report, not metrics JSON.
    metrics.pop("_test_messages")
    metrics_path = ARTIFACT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(metrics)
    return metrics


def main() -> int:
    metrics = run_training()
    spam = metrics["test"]["spam"]
    print(f"Selected model: {metrics['selected_model']['name']}")
    print(f"Validation threshold: {metrics['threshold']['value']:.12g}")
    print(
        "Held-out test spam precision={:.4f}, recall={:.4f}, F1={:.4f}; PR-AUC={:.4f}, ROC-AUC={:.4f}".format(
            spam["precision"], spam["recall"], spam["f1-score"], metrics["test"]["pr_auc"], metrics["test"]["roc_auc"]
        )
    )
    print(f"Artifacts and reports written under {ARTIFACT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
