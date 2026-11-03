#!/usr/bin/env python3
"""
adaptive-graphrag: download_data.py

Download one scale of the adaptive-graphrag dataset repository.

Usage:
    python3 download_data.py /workspace/tmp/data/raw
    python3 download_data.py --dir /workspace/tmp/data/raw --scale medium
    DATASET_SIZE=medium python3 download_data.py /workspace/tmp/data/raw
    python3 download_data.py /workspace/tmp/data/raw --force
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import importlib.util


def _require(pkgs: list[str]) -> None:
    missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
    if missing:
        sys.stderr.write(
            f"Missing Python packages: {', '.join(missing)}\n"
            "Run: pip install huggingface_hub\n"
        )
        sys.exit(1)


_require(["huggingface_hub"])

from huggingface_hub import snapshot_download  # noqa: E402


VALID_SIZES = {"min", "medium", "max"}
REPO_ID = "adaptive-graphrag-datasets"
DEFAULT_USER = "Athithya117"


def log(msg: str) -> None:
    print(msg, flush=True)


def log_err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--dir", dest="dir_opt")
    ap.add_argument("--scale", choices=sorted(VALID_SIZES), default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    target = args.dir_opt or args.path
    if not target:
        log_err("[error] target directory required")
        return 2

    size = (args.scale or os.environ.get("DATASET_SIZE", "min")).strip().lower()
    if size not in VALID_SIZES:
        log_err(f"[error] scale must be one of {sorted(VALID_SIZES)}")
        return 2

    user = os.environ.get("HF_USER_NAME", DEFAULT_USER).strip()
    repo_id = f"{user}/{REPO_ID}"
    tag = f"v1.0-{size}"

    target_dir = Path(target).expanduser().resolve()
    if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
        log(f"[skip] {target_dir} not empty (use --force)")
        return 0
    target_dir.mkdir(parents=True, exist_ok=True)

    log(f"Repo:   {repo_id}")
    log(f"Tag:    {tag}")
    log(f"Target: {target_dir}")

    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=tag,
            local_dir=str(target_dir),
            ignore_patterns=[
                ".cache/**",
                "**/.cache/**",
                "**/__pycache__/**",
                "intermediate/**",
            ],
        )
    except Exception as e:
        log_err(f"[error] download failed: {e}")
        log_err(f"  - Verify tag exists: https://huggingface.co/datasets/{repo_id}/tree/{tag}")
        return 1

    log(f"[ok] downloaded to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
