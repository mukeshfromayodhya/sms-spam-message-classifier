from __future__ import annotations

import json
from pathlib import Path

import pytest

from spam_classifier import security


def test_tampered_artifact_fails_checksum_before_joblib_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = tmp_path / "pipeline.joblib"
    vectorizer = tmp_path / "vectorizer.joblib"
    metadata = tmp_path / "model_metadata.json"
    manifest = tmp_path / "sha256-manifest.json"
    pipeline.write_bytes(b"trusted test bytes")
    vectorizer.write_bytes(b"vectorizer bytes")
    metadata.write_text(json.dumps({"threshold": 0.9}), encoding="utf-8")

    monkeypatch.setattr(security, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(security, "PIPELINE_PATH", pipeline)
    monkeypatch.setattr(security, "VECTORIZER_PATH", vectorizer)
    monkeypatch.setattr(security, "METADATA_PATH", metadata)
    monkeypatch.setattr(security, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(
        security,
        "_TRUSTED_FILES",
        {
            "pipeline.joblib": pipeline,
            "vectorizer.joblib": vectorizer,
            "model_metadata.json": metadata,
        },
    )
    security.write_checksum_manifest()
    pipeline.write_bytes(b"tampered test bytes")

    with pytest.raises(security.ArtifactIntegrityError, match="Checksum verification failed"):
        security.load_trusted_pipeline()


@pytest.mark.parametrize("linked_name", ["pipeline.joblib", "sha256-manifest.json"])
def test_training_preflight_rejects_artifact_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked_name: str
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    pipeline = artifact_dir / "pipeline.joblib"
    vectorizer = artifact_dir / "vectorizer.joblib"
    metadata = artifact_dir / "model_metadata.json"
    manifest = artifact_dir / "sha256-manifest.json"
    linked_path = artifact_dir / linked_name
    sentinel = tmp_path / "outside.txt"
    sentinel.write_text("must stay unchanged", encoding="utf-8")
    try:
        linked_path.symlink_to(sentinel)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")

    monkeypatch.setattr(security, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(security, "PIPELINE_PATH", pipeline)
    monkeypatch.setattr(security, "VECTORIZER_PATH", vectorizer)
    monkeypatch.setattr(security, "METADATA_PATH", metadata)
    monkeypatch.setattr(security, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(
        security,
        "_TRUSTED_FILES",
        {
            "pipeline.joblib": pipeline,
            "vectorizer.joblib": vectorizer,
            "model_metadata.json": metadata,
        },
    )

    with pytest.raises(security.ArtifactIntegrityError, match="must not be a symlink"):
        security.ensure_safe_artifact_targets()
    assert sentinel.read_text(encoding="utf-8") == "must stay unchanged"
