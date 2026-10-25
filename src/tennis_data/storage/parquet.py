"""Deterministic, atomic Parquet output.

Outputs are rebuilt from scratch on every run (never appended to), rows are sorted
by a total order, and no wall-clock values are written, so the same raw snapshot
and code always produce byte-identical files.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import polars as pl

# Total order: tournament_date, then event, then round, then source position.
SORT_KEYS = ("tournament_date", "tournament_id", "round_order", "match_num", "source_file", "source_row_number")


def sort_canonical(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(list(SORT_KEYS), nulls_last=True, maintain_order=True)


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", compression_level=3, statistics=True, row_group_size=50_000)


def write_partitioned(df: pl.DataFrame, out_dir: Path, partition_col: str = "season") -> list[Path]:
    """Replace ``out_dir`` with hive-style partitions ``<col>=<value>/part-0.parquet``.

    The partition column is also kept inside each file, so every file carries the full
    canonical schema and can be read on its own.
    """
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    written = []
    for value in sorted(df[partition_col].unique().to_list()):
        part = df.filter(pl.col(partition_col) == value)
        rel = Path(f"{partition_col}={value}") / "part-0.parquet"
        _write(part, tmp / rel)
        written.append(out_dir / rel)
    _swap(tmp, out_dir)
    return written


def write_single(df: pl.DataFrame, path: Path) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    _write(df, tmp)
    tmp.replace(path)
    return path


def _swap(tmp: Path, target: Path) -> None:
    old = target.with_name(target.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    tmp.rename(target)
    shutil.rmtree(old, ignore_errors=True)


def tree_checksums(root: Path) -> dict[str, str]:
    """sha256 of every file under root, keyed by relative POSIX path (sorted)."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
