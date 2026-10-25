"""The raw-snapshot manifest: source identity plus one checksummed entry per file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tennis_data.config import Config

MANIFEST_VERSION = 1


@dataclass
class Manifest:
    source: dict
    files: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def new(cls, config: Config, commit_verified: bool) -> Manifest:
        src = config.source
        return cls(
            source={
                "name": src.name,
                "repo": src.repo,
                "commit": src.commit,
                "commit_verified": commit_verified,
                "mode": src.mode,
                "repo_url": f"https://github.com/{src.repo}",
                "license": src.license,
                "attribution": src.attribution,
            }
        )

    def check_matches(self, config: Config) -> None:
        if self.source.get("commit") != config.source.commit or self.source.get("repo") != config.source.repo:
            raise ValueError(
                f"Manifest belongs to {self.source.get('repo')}@{self.source.get('commit')}, "
                f"config pins {config.source.repo}@{config.source.commit}."
            )

    def to_json(self) -> str:
        payload = {"manifest_version": MANIFEST_VERSION, "source": self.source, "files": self.files}
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def manifest_path(config: Config) -> Path:
    # Stored beside, not inside, the snapshot so the snapshot holds only source bytes.
    return config.raw_source_dir.with_name(f"{config.source.commit}.manifest.json")


def load_manifest(path: Path) -> Manifest | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return Manifest(source=data["source"], files=data.get("files", {}))


def save_manifest(path: Path, manifest: Manifest) -> None:
    text = manifest.to_json()
    if path.exists() and path.read_text() == text:
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)
