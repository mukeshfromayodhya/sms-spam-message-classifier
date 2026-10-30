"""Checksum-verified loading for locally trained joblib artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import joblib

from spam_classifier.config import ARTIFACT_DIR

PIPELINE_PATH = ARTIFACT_DIR / "pipeline.joblib"
VECTORIZER_PATH = ARTIFACT_DIR / "vectorizer.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
MANIFEST_PATH = ARTIFACT_DIR / "sha256-manifest.json"
_TRUSTED_FILES = {
    PIPELINE_PATH.name: PIPELINE_PATH,
    VECTORIZER_PATH.name: VECTORIZER_PATH,
    METADATA_PATH.name: METADATA_PATH,
}


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact is absent, unexpected, or fails its checksum."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_safe_artifact_targets() -> None:
    """Reject pre-existing symlinks and non-files before training writes."""
    if ARTIFACT_DIR.is_symlink():
        raise ArtifactIntegrityError("Artifact directory must not be a symlink.")
    if ARTIFACT_DIR.exists() and not ARTIFACT_DIR.is_dir():
        raise ArtifactIntegrityError("Artifact path must be a directory.")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if ARTIFACT_DIR.is_symlink() or not ARTIFACT_DIR.is_dir():
        raise ArtifactIntegrityError("Artifact directory is not a trusted directory.")

    for path in (*_TRUSTED_FILES.values(), MANIFEST_PATH):
        if path.is_symlink():
            raise ArtifactIntegrityError(f"Artifact destination must not be a symlink: {path}")
        if path.exists() and not path.is_file():
            raise ArtifactIntegrityError(f"Artifact destination must be a regular file: {path}")


def _atomic_artifact_write(path: Path, writer: Callable[[Any], None]) -> None:
    """Write a fixed bundle member through a same-directory temporary file."""
    if path not in (*_TRUSTED_FILES.values(), MANIFEST_PATH):
        raise ArtifactIntegrityError("Refusing to write outside the fixed artifact bundle.")
    ensure_safe_artifact_targets()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=ARTIFACT_DIR, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Recheck the leaf and atomically replace it. os.replace replaces a
        # destination link itself instead of opening the link target.
        if path.is_symlink():
            raise ArtifactIntegrityError(f"Artifact destination became a symlink: {path}")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_joblib_dump(value: Any, path: Path) -> None:
    """Serialize a fixed model artifact to a safe temporary file, then replace."""
    _atomic_artifact_write(path, lambda handle: joblib.dump(value, handle, compress=3))


def atomic_write_artifact_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 metadata within the fixed artifact bundle."""
    encoded = text.encode("utf-8")
    _atomic_artifact_write(path, lambda handle: handle.write(encoded))


def write_checksum_manifest() -> dict[str, Any]:
    """Create a local SHA-256 manifest for the fixed artifact bundle."""
    ensure_safe_artifact_targets()
    files: dict[str, dict[str, Any]] = {}
    for name, path in _TRUSTED_FILES.items():
        if path.is_symlink() or not path.is_file():
            raise ArtifactIntegrityError(f"Expected a regular artifact file: {path}")
        content = path.read_bytes()
        files[name] = {"sha256": _sha256(content), "size_bytes": len(content)}
    manifest = {"schema_version": 1, "files": files}
    atomic_write_artifact_text(MANIFEST_PATH, json.dumps(manifest, indent=2) + "\n")
    return manifest


def verified_artifact_bytes() -> dict[str, bytes]:
    """Read fixed artifacts into memory and verify them before deserialization.

    No caller-controlled paths are accepted. Loading from verified bytes avoids
    a path swap between checksum verification and joblib deserialization.
    """
    if MANIFEST_PATH.is_symlink() or not MANIFEST_PATH.is_file():
        raise ArtifactIntegrityError("Artifact checksum manifest is missing or not a regular file.")
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest_files = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ArtifactIntegrityError("Artifact checksum manifest is unreadable.") from exc

    if set(manifest_files) != set(_TRUSTED_FILES):
        raise ArtifactIntegrityError("Artifact manifest has an unexpected file list.")

    verified: dict[str, bytes] = {}
    for name, path in _TRUSTED_FILES.items():
        if path.is_symlink() or not path.is_file():
            raise ArtifactIntegrityError(f"Expected a regular artifact file: {path}")
        content = path.read_bytes()
        expected = manifest_files.get(name, {})
        if len(content) != expected.get("size_bytes") or _sha256(content) != expected.get("sha256"):
            raise ArtifactIntegrityError(f"Checksum verification failed for {name}.")
        verified[name] = content
    return verified


def load_trusted_pipeline() -> Any:
    """Verify the full local bundle, then load only the bundled pipeline."""
    content = verified_artifact_bytes()[PIPELINE_PATH.name]
    try:
        return joblib.load(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001 - convert deserialization failure to a safe error
        raise ArtifactIntegrityError("Verified pipeline could not be deserialized.") from exc


def load_trusted_metadata() -> dict[str, Any]:
    """Verify the full local bundle and safely read its JSON metadata."""
    content = verified_artifact_bytes()[METADATA_PATH.name]
    try:
        metadata = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("Verified model metadata is invalid JSON.") from exc
    if not isinstance(metadata, dict) or "threshold" not in metadata:
        raise ArtifactIntegrityError("Verified model metadata is incomplete.")
    return metadata
