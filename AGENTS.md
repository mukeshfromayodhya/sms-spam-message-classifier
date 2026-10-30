# Project guidance

## Setup and commands

- Use Python 3.12 or newer.
- Install dependencies with `python -m pip install -r requirements.txt`.
- Train and regenerate reports with `python -m spam_classifier.train`.
- Predict with `python -m spam_classifier.predict --text "message text"` or `python -m spam_classifier.predict --interactive`.
- Run the requested checks with `pytest -q`.

## Durable rules

- Never fabricate dataset rows, evaluation metrics, predictions, or citations. Demo messages are explicitly labeled and never included in metrics.
- Fit vectorizers only inside a scikit-learn `Pipeline`, using training data only. Keep cross-validation on the training split, threshold tuning on validation, and test evaluation at the end.
- Never load a joblib artifact before verifying its SHA-256 against the local manifest. The loader must use fixed project paths and must not accept artifact paths from users.
- Treat all user message text as data. Never call `eval`, `exec`, a shell, or a subprocess with message content.
- Keep this project to classic machine learning; do not add deep-learning models.
- Do not log raw user messages unless the user explicitly supplies `--log-raw`. Redact URLs, email addresses, and phone numbers in default logs.
