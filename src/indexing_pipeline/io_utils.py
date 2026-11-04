"""I/O primitives: atomic writes, hashing, JSONL, JSON-safe conversion.

No imports from this package. Pure stdlib + numpy.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    """ISO-8601 UTC timestamp with trailing Z, e.g. 2026-11-04T12:30:00Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_default(value: Any) -> Any:
    """Serialize Arrow, numpy, and date scalars for json.dumps(default=...)."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "as_py"):
        return value.as_py()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def json_safe(value: Any) -> Any:
    """Recursively convert Arrow/numpy types to plain JSON-safe Python.

    Used when building `metadata` JSONB payloads from Parquet row dicts.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if hasattr(value, "as_py"):
        return json_safe(value.as_py())
    return str(value)


def sha256_bytes(payload: bytes) -> str:
    """Return 'sha256:<hex>' — the format used everywhere in this pipeline."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file, returned as 'sha256:<hex>'."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def sha256_text(text: Any) -> str:
    """SHA-256 of a string's UTF-8 encoding, as 'sha256:<hex>'.

    Must match the format produced by chunker._hash_text so that hashes
    from different modules can coexist in the same database.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    return sha256_bytes(text.encode("utf-8"))


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write JSONL via temp-file + rename. Crash-safe: partial writes never
    leave a half-written file at `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=json_default,
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, value: dict) -> None:
    """Write a single JSON object atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield dicts from a JSONL file. Silent no-op if file is absent."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{line_no}")
            yield row


def write_jsonl_from_dict(path: Path, rows: dict[int, dict]) -> None:
    """Write a dict keyed by case_id as JSONL sorted by case_id — deterministic output."""
    atomic_write_jsonl(path, (rows[case_id] for case_id in sorted(rows)))
