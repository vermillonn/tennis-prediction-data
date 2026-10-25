"""Raw ingestion: provenance, idempotency, integrity and actionable errors."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conftest import FAKE_COMMIT, match_row, write_config, write_csv
from tennis_data.config import load_config
from tennis_data.ingestion.fetch import (
    MissingSourceFileError,
    RawIntegrityError,
    SourceCommitError,
    ingest,
    sha256_file,
    verify_raw,
)
from tennis_data.ingestion.manifest import manifest_path
from tennis_data.normalization.normalize import SourceSchemaError
from tennis_data.normalization.schema import SOURCE_MATCH_COLUMNS
from tennis_data.pipeline import validate


@pytest.fixture
def cfg(tmp_path, archive):
    archive.set_season(2020, [match_row(1, 100, 200)])
    archive.set_season(2021, [match_row(1, 300, 400, tourney_id="2021-0001", tourney_date="20210104")])
    return load_config(write_config(tmp_path, archive.root, 2020, 2021))


def _snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def test_manifest_records_provenance(cfg, archive):
    ingest(cfg)
    m = json.loads(manifest_path(cfg).read_text())
    assert m["source"]["commit"] == FAKE_COMMIT
    assert m["source"]["commit_verified"] is False  # fixture is not a git clone; recorded honestly
    entry = m["files"]["atp/atp_matches_2020.csv"]
    assert entry["sha256"] == sha256_file(archive.root / "atp/atp_matches_2020.csv")
    assert entry["source_url"].startswith("file://") and entry["source_url"].endswith("atp/atp_matches_2020.csv")
    assert entry["retrieved_at"].endswith("+00:00")
    assert entry["family"] == "tour" and entry["season"] == 2020
    assert "LICENSE" in m["files"]


def test_raw_files_are_byte_identical_copies(cfg, archive):
    ingest(cfg)
    rel = "atp/atp_matches_2021.csv"
    assert (cfg.raw_source_dir / rel).read_bytes() == (archive.root / rel).read_bytes()


def test_ingestion_is_idempotent(cfg):
    first = ingest(cfg)
    before = _snapshot(cfg.paths.raw_dir)
    second = ingest(cfg)
    assert len(first.fetched) == first.total and second.fetched == []
    assert len(second.verified) == first.total
    assert _snapshot(cfg.paths.raw_dir) == before  # no new files, manifest unchanged
    assert verify_raw(cfg) == []


def test_interrupted_ingest_resumes_without_duplication(cfg):
    ingest(cfg)
    (cfg.raw_source_dir / "atp/atp_matches_2021.csv").unlink()
    result = ingest(cfg)
    assert result.fetched == ["atp/atp_matches_2021.csv"]
    assert len(json.loads(manifest_path(cfg).read_text())["files"]) == result.total


def test_modified_raw_file_is_detected(cfg):
    ingest(cfg)
    raw = cfg.raw_source_dir / "atp/atp_matches_2020.csv"
    raw.write_text(raw.read_text().replace("Testopen", "Tampered"))
    assert verify_raw(cfg) == ["checksum mismatch: atp/atp_matches_2020.csv"]
    with pytest.raises(RawIntegrityError, match="must never be edited"):
        ingest(cfg)
    with pytest.raises(RawIntegrityError):
        validate(cfg)


def test_missing_year_gives_actionable_error(tmp_path, archive):
    archive.set_season(2020, [match_row(1, 100, 200)])
    cfg = load_config(write_config(tmp_path, archive.root, 2020, 2022))
    with pytest.raises(MissingSourceFileError, match=r"atp_matches_2021\.csv[\s\S]*start_year/end_year"):
        ingest(cfg)


def test_missing_column_gives_actionable_error(tmp_path, archive):
    write_csv(archive.root / "atp/atp_matches_2020.csv", [match_row(1, 100, 200)],
              columns=[c for c in SOURCE_MATCH_COLUMNS if c != "w_svpt"])  # fmt: skip
    cfg = load_config(write_config(tmp_path, archive.root))
    ingest(cfg)
    with pytest.raises(SourceSchemaError, match=r"missing expected column\(s\): \['w_svpt'\]"):
        validate(cfg)


def test_local_mode_requires_pinned_commit_when_verifying(tmp_path, archive):
    archive.set_season(2020, [match_row(1, 100, 200)])
    cfg = load_config(write_config(tmp_path, archive.root, verify_local_commit=True))
    with pytest.raises(SourceCommitError):
        ingest(cfg)


def test_download_mode_uses_commit_pinned_urls(tmp_path, archive):
    archive.set_season(2020, [match_row(1, 100, 200)])
    cfg = load_config(write_config(tmp_path, archive.root, mode="download", base_url="https://example.invalid"))
    seen = []

    def fake_fetch(url: str, dest: Path, timeout: int) -> None:
        seen.append(url)
        rel = url.split(f"/{FAKE_COMMIT}/", 1)[1]
        shutil.copyfile(archive.root / rel, dest)

    ingest(cfg, fetcher=fake_fetch)
    assert f"https://example.invalid/example/fixture-archive/{FAKE_COMMIT}/atp/atp_matches_2020.csv" in seen
    assert ingest(cfg, fetcher=fake_fetch).fetched == []  # second run fetches nothing
