#!/usr/bin/env python3
"""
adaptive-graphrag: upload_datasets.py

Upload a local snapshot to Hugging Face with one repo + git tags.

Repo layout:
    Athithya117/adaptive-graphrag-datasets
        main          = latest uploaded scale
        tag v1.0-min
        tag v1.0-medium
        tag v1.0-max

Env:
    HF_USER_NAME    required, e.g. Athithya117
    HF_TOKEN        required, write scope
    DATASET_SIZE    min | medium | max (default: min)

Usage:
    export HF_USER_NAME=Athithya117
    export DATASET_SIZE=min
    python3 upload_datasets.py

    # After each --scale build:
    DATASET_SIZE=min    python3 upload_datasets.py
    DATASET_SIZE=medium python3 upload_datasets.py
    DATASET_SIZE=max    python3 upload_datasets.py

    python3 upload_datasets.py --dry-run
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

from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402


VALID_SIZES = {"min", "medium", "max"}
REPO_ID = "adaptive-graphrag-datasets"   # no suffix — one repo
DEFAULT_DIR = "/workspace/tmp/data/raw"


def log(msg: str) -> None:
    print(msg, flush=True)


def log_err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sanity_check(data_dir: Path) -> None:
    expected = [
        "README.md", "LICENSE", "DATA_SOURCES.md", "MANIFEST.json",
        "corpus/opinions.parquet",
        "graph/citations_filtered.parquet",
        "metadata/clusters_filtered.parquet",
        "eval_isolated/hojicha/corpus.parquet",
        "eval_isolated/hojicha/questions.parquet",
        "eval_isolated/synthesized_graph_qa/queries.parquet",
    ]
    missing = [p for p in expected if not (data_dir / p).exists()]
    if missing:
        log_err("[error] snapshot is incomplete — missing:")
        for p in missing:
            log_err(f"  - {p}")
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Upload adaptive-graphrag snapshot with git tags.",
    )
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--size", default=None, help="override DATASET_SIZE")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    user = os.environ.get("HF_USER_NAME", "").strip()
    if not user:
        log_err("[error] HF_USER_NAME not set")
        log_err("  export HF_USER_NAME=Athithya117")
        return 2

    size = (args.size or os.environ.get("DATASET_SIZE", "min")).strip().lower()
    if size not in VALID_SIZES:
        log_err(f"[error] size must be one of {sorted(VALID_SIZES)}")
        return 2

    data_dir = Path(args.dir).expanduser().resolve()
    if not data_dir.exists():
        log_err(f"[error] data dir does not exist: {data_dir}")
        return 2

    repo_id = f"{user}/{REPO_ID}"
    tag = f"v1.0-{size}"
    commit_msg = f"{size}: scale snapshot"

    log(f"User:   {user}")
    log(f"Size:   {size}")
    log(f"Repo:   {repo_id}")
    log(f"Tag:    {tag}")
    log(f"Source: {data_dir}")
    log(f"Target branch: main")

    sanity_check(data_dir)

    if args.dry_run:
        log("")
        log(f"[dry-run] would create/update {repo_id} at main and tag {tag}")
        return 0

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        log_err("[error] HF_TOKEN not set")
        return 2

    api = HfApi(token=token)

    # 1. Create repo if needed (idempotent)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset",
                        exist_ok=True, token=token)
        log(f"[ok] repo ready: https://huggingface.co/datasets/{repo_id}")
    except HfHubHTTPError as e:
        log_err(f"[error] create_repo failed: {e}")
        return 1

    # 2. Upload folder to main — overwrites files, keeps repo tree
    log(f"[upload] {data_dir} -> {repo_id}:main")
    try:
        commit = api.upload_folder(
            folder_path=str(data_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_msg,
            token=token,
            ignore_patterns=[
                "intermediate/**",          # intermediate parquet, not published
                "**/__pycache__/**",
                ".cache/**",
                "**/.cache/**",
            ],
        )
        log(f"[ok] uploaded: commit {commit.oid[:12]}")
    except HfHubHTTPError as e:
        log_err(f"[error] upload failed: {e}")
        return 1

    # 3. Create the tag at this commit (idempotent — overwrite if exists)
    log(f"[tag] {tag} -> {commit.oid[:12]}")
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=tag,
            revision=commit.oid,
            repo_type="dataset",
            token=token,
        )
        log(f"[ok] tag created: {tag}")
    except HfHubHTTPError as e:
        # A tag that already exists raises; update its ref instead
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            log(f"[tag] {tag} already exists — moving to {commit.oid[:12]}")
            try:
                api.create_tag(
                    repo_id=repo_id,
                    tag=tag,
                    revision=commit.oid,
                    repo_type="dataset",
                    token=token,
                    exist_ok=True,
                )
                log(f"[ok] tag updated: {tag}")
            except Exception as e2:
                log_err(f"[error] tag update failed: {e2}")
                return 1
        else:
            log_err(f"[error] tag creation failed: {e}")
            return 1

    log("")
    log(f"Uploaded to: https://huggingface.co/datasets/{repo_id}")
    log(f"Tag:         {tag}")
    log("")
    log("Pull this scale anywhere with:")
    log(f'  snapshot_download("{repo_id}", repo_type="dataset",')
    log(f'                    revision="{tag}", local_dir="data/raw")')
    log("")
    log("Next:")
    for s in sorted(VALID_SIZES):
        marker = " ← done" if s == size else ""
        log(f"  DATASET_SIZE={s:6s} python3 upload_datasets.py{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())