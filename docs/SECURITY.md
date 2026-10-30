# Security notes

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
