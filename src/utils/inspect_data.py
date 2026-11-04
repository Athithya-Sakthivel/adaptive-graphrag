#!/usr/bin/env python3
"""
inspect_data.py — inspect data/raw/ before chunking at scale.

Every measurement uses the actual embedding tokenizer. Answers:

  1. Are files present and schema-correct?
  2. What is the real token ratio? (fallback sizing only; pipeline uses tokenize=)
  3. Does _strip_html preserve paragraph breaks?  ← NEW, explains p50=1
  4. What percentage of opinions have section hits?
  5. Is the atomic filter applied inside _sentence_spans?  ← THE 3-BUG CHECK
  6. What does chunk_opinion actually emit?  ← NEW, validates output
  7. Do all graph edges resolve to real corpus opinions?  ← FULL corpus, not sample
  8. Do all corpus clusters exist in metadata?
  9. Is eval isolated from production?
 10. What chunker config does this data justify?

Usage:
    python src/utils/inspect_data.py /workspace/tmp/data/raw
    python src/utils/inspect_data.py /workspace/tmp/data/raw --sample 500
    python src/utils/inspect_data.py /workspace/tmp/data/raw --chunker-sample 50
    python src/utils/inspect_data.py /workspace/tmp/data/raw --inspect-one 1088107
    python src/utils/inspect_data.py /workspace/tmp/data/raw --json > report.json

Exit codes:
    0 — all critical checks pass, ready to chunk
    1 — non-critical warnings only
    2 — one or more critical findings
    3 — setup error (missing data or tokenizer required)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Make indexing_pipeline importable when running from src/utils/
_REPO_SRC = Path(__file__).resolve().parent.parent
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import pandas as pd
import pyarrow.parquet as pq

# Silence HF tokenizer warnings ("sequence length > max"); we count tokens,
# we do not run them through a model.
warnings.filterwarnings("ignore", message=".*sequence length.*")
warnings.filterwarnings("ignore", message=".*max_length.*")

# ── Chunker integration ────────────────────────────────────────────────
_CHUNKER_IMPORT_ERROR: str = ""
try:
    from indexing_pipeline import chunker as chunker_mod

    CHUNKER_AVAILABLE = True
except Exception as exc:
    chunker_mod = None
    CHUNKER_AVAILABLE = False
    _CHUNKER_IMPORT_ERROR = str(exc)

if CHUNKER_AVAILABLE and chunker_mod is not None:
    PARENT_TARGET_TOK = chunker_mod.PARENT_TARGET_TOK
    PARENT_MAX_TOK = chunker_mod.PARENT_MAX_TOK
    PARENT_MIN_TOK = chunker_mod.PARENT_MIN_TOK
    CHILD_TARGET_TOK = chunker_mod.CHILD_TARGET_TOK
    CHILD_MAX_TOK = chunker_mod.CHILD_MAX_TOK
    CHILD_MIN_TOK = chunker_mod.CHILD_MIN_TOK
    CHUNKER_CHARS_PER_TOKEN = chunker_mod.CHARS_PER_TOKEN
    CHUNKER_VERSION = chunker_mod.CHUNKER_VERSION
    COMPILED_SECTION_PATTERNS = chunker_mod.COMPILED_SECTION_PATTERNS
    ATOMIC_SPAN_PATTERNS = chunker_mod.ATOMIC_SPAN_PATTERNS
    _is_atomic_boundary = chunker_mod._is_atomic_boundary
    _sentence_spans = chunker_mod._sentence_spans
    _paragraph_spans = chunker_mod._paragraph_spans
    _detect_sections = chunker_mod._detect_sections
    _strip_html = chunker_mod._strip_html
    _normalize = chunker_mod._normalize
else:
    PARENT_TARGET_TOK, PARENT_MAX_TOK, PARENT_MIN_TOK = 1000, 1400, 500
    CHILD_TARGET_TOK, CHILD_MAX_TOK, CHILD_MIN_TOK = 320, 400, 160
    CHUNKER_CHARS_PER_TOKEN = 3.0
    CHUNKER_VERSION = f"fallback (chunker import failed: {_CHUNKER_IMPORT_ERROR})"
    COMPILED_SECTION_PATTERNS = ()
    ATOMIC_SPAN_PATTERNS = ()
    _is_atomic_boundary = lambda text, pos: True
    _sentence_spans = lambda text, s, e: [(s, e)]
    _paragraph_spans = lambda text, s, e: [(s, e)]
    _detect_sections = lambda text: [("body", 50, 0, len(text))]
    _strip_html = lambda text, kind: text
    _normalize = lambda text: text

# Embedder limits (must match embedder.py)
EMBEDDER_MAX_TOKENS = 464  # OpenVINO STATIC_SEQUENCE
EMBEDDER_TARGET_SEQ = 464  # same

EXPECTED_FILES = [
    "corpus/opinions.parquet",
    "graph/citations_filtered.parquet",
    "metadata/clusters_filtered.parquet",
    "eval_isolated/hojicha/corpus.parquet",
    "eval_isolated/hojicha/questions.parquet",
    "eval_isolated/synthesized_graph_qa/queries.parquet",
]

EXPECTED_COLUMNS: dict[str, set[str]] = {
    "corpus/opinions.parquet": {
        "cl_opinion_id",
        "cluster_id",
        "opinion_type",
        "text_kind",
        "text",
        "title",
        "citation",
        "court",
        "date",
        "jurisdiction",
        "doc_type",
    },
    "graph/citations_filtered.parquet": {
        "citing_opinion_id",
        "cited_opinion_id",
        "depth",
    },
    "metadata/clusters_filtered.parquet": {
        "id",
        "case_name",
        "case_name_short",
        "date_filed",
        "precedential_status",
        "citation_count",
        "nature_of_suit",
        "judges",
        "docket_id",
    },
}


# ── Reporter ───────────────────────────────────────────────────────────
class Reporter:
    def __init__(self, json_mode: bool = False) -> None:
        self.json_mode = json_mode
        self.sections: list[dict] = []
        self.current: dict = {"section": "", "checks": [], "findings": []}
        self.critical: list[str] = []
        self.warnings: list[str] = []

    def section(self, name: str) -> None:
        self.current = {"section": name, "checks": [], "findings": []}
        if not self.json_mode:
            print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")

    def check(self, name: str, value, ok: bool | None = None, note: str = "") -> None:
        self.current["checks"].append({"name": name, "value": value, "ok": ok, "note": note})
        if not self.json_mode:
            mark = "PASS" if ok is True else "FAIL" if ok is False else "INFO"
            print(f"  [{mark:4s}] {name}: {value}" + (f"  ({note})" if note else ""))

    def finding(self, msg: str, critical: bool = False) -> None:
        self.current["findings"].append(msg)
        if critical:
            self.critical.append(msg)
            if not self.json_mode:
                print(f"  [CRIT] {msg}")
        else:
            self.warnings.append(msg)
            if not self.json_mode:
                print(f"  ->   {msg}")

    def end(self) -> None:
        self.sections.append(self.current)

    def dump(self) -> dict:
        return {
            "sections": self.sections,
            "critical_count": len(self.critical),
            "critical": self.critical,
            "warning_count": len(self.warnings),
            "warnings": self.warnings,
        }


R = Reporter()


# ── Utilities ──────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(p * (len(s) - 1))))
    return float(s[idx])


def _make_token_counter(tok) -> Any:
    """Bind tokenizer to a counter. Uses add_special_tokens=True so counts
    match the pipeline's embedder-side validation."""

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tok.encode(text, add_special_tokens=True, truncation=False))

    return count


def load_tokenizer(explicit: Path | None) -> Any:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            _REPO_SRC / "indexing_pipeline" / "embedders" / "dinghy-law-0.6b-v1-openvino-fp16",
            _REPO_SRC / "indexing_pipeline" / "models" / "dinghy-law-0.6b-v1",
            Path("/kaggle/working/models/dinghy-law-0.6b-v1"),
        ]
    )
    from transformers import AutoTokenizer

    for c in candidates:
        if c and (c / "tokenizer.json").is_file():
            try:
                tok = AutoTokenizer.from_pretrained(str(c), local_files_only=True)
                # Lift model_max_length so encode(..., truncation=False) never
                # emits "sequence length > max" warnings on long opinions.
                try:
                    tok.model_max_length = 1_000_000
                except Exception:
                    pass
                if not R.json_mode:
                    print(f"  tokenizer: {c}")
                return tok
            except Exception as exc:
                if not R.json_mode:
                    print(f"  tokenizer at {c}: load failed ({exc})", file=sys.stderr)
    return None


def load_sample_rows(root: Path, sample_size: int) -> tuple[list[dict], list[str]]:
    """Load N sampled opinions, plus their (stripped+normalized) text."""
    cp = root / "corpus/opinions.parquet"
    if not cp.is_file():
        print(f"error: corpus missing: {cp}", file=sys.stderr)
        sys.exit(3)

    df = pd.read_parquet(
        cp,
        columns=[
            "cl_opinion_id",
            "cluster_id",
            "text",
            "text_kind",
            "opinion_type",
            "title",
            "citation",
            "court",
            "date",
        ],
    )
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    rows: list[dict] = []
    texts: list[str] = []
    for row in df.to_dict(orient="records"):
        raw = row.get("text")
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            stripped = _strip_html(raw, str(row.get("text_kind") or "plain_text"))
            normalized = _normalize(stripped)
        except Exception:
            continue
        if not normalized.strip():
            continue
        rows.append(row)
        texts.append(normalized)

    return rows, texts


def load_full_corpus_ids(root: Path) -> set[int]:
    cp = root / "corpus/opinions.parquet"
    if not cp.is_file():
        return set()
    df = pd.read_parquet(cp, columns=["cl_opinion_id"])
    return {int(x) for x in df["cl_opinion_id"].dropna().tolist()}


# ── Phase 1: files + schema ────────────────────────────────────────────
def inspect_files_schema(root: Path) -> None:
    R.section("1. Files, integrity, schema")

    manifest: dict = {}
    mp = root / "MANIFEST.json"
    if mp.is_file():
        try:
            manifest = json.loads(mp.read_text())
            files_obj = manifest.get("files", manifest)
            R.check(
                "MANIFEST entries",
                len(files_obj) if isinstance(files_obj, dict) else "unexpected shape",
            )
        except json.JSONDecodeError as exc:
            R.check("MANIFEST.json", f"invalid JSON: {exc}", ok=False)

    for rel in EXPECTED_FILES:
        p = root / rel
        if not p.is_file():
            R.check(rel, "MISSING", ok=False)
            R.finding(f"missing required file: {rel}", critical=True)
            continue
        size_mb = p.stat().st_size / (1024 * 1024)
        sha = sha256_file(p)

        expected_sha = None
        if isinstance(manifest, dict):
            files_obj = manifest.get("files", manifest)
            entry = files_obj.get(rel) if isinstance(files_obj, dict) else None
            if isinstance(entry, dict):
                expected_sha = entry.get("sha256")
            elif isinstance(entry, str):
                expected_sha = entry

        ok = None
        note = f"{size_mb:.1f} MB"
        if expected_sha:
            ok = sha == expected_sha
            note += ", sha256 " + ("match" if ok else "MISMATCH")
            if not ok:
                R.finding(f"sha256 mismatch: {rel}", critical=True)
        R.check(rel, sha[:12] + "...", ok=ok, note=note)

    for rel, expected in EXPECTED_COLUMNS.items():
        p = root / rel
        if not p.is_file():
            continue
        schema = pq.read_schema(p)
        actual = set(schema.names)
        missing = expected - actual
        if missing:
            R.check(f"{rel} missing cols", sorted(missing), ok=False)
            R.finding(f"{rel} missing required columns: {sorted(missing)}", critical=True)
        else:
            R.check(f"{rel} columns", f"{len(actual)} present", ok=True)


# ── Phase 2: token distribution ────────────────────────────────────────
def inspect_tokens(texts: list[str], count_fn) -> dict:
    R.section("2. Token distribution (real tokenizer)")

    tokens = [count_fn(t) for t in texts]
    chars = [len(t) for t in texts]
    ratios = [c / n if n > 0 else 0.0 for c, n in zip(chars, tokens, strict=True)]

    stats = {
        "sample_size": len(tokens),
        "tokens_p50": int(pct(tokens, 0.50)),
        "tokens_p95": int(pct(tokens, 0.95)),
        "tokens_p99": int(pct(tokens, 0.99)),
        "tokens_max": int(max(tokens)) if tokens else 0,
        "ratio_min": round(min(ratios), 3) if ratios else 0.0,
        "ratio_p05": round(pct(ratios, 0.05), 3),
        "ratio_p50": round(pct(ratios, 0.50), 3),
        "ratio_p95": round(pct(ratios, 0.95), 3),
    }

    R.check("sample size", stats["sample_size"])
    R.check("tokens/opinion p50", stats["tokens_p50"])
    R.check("tokens/opinion p95", stats["tokens_p95"])
    R.check("tokens/opinion max", stats["tokens_max"])
    R.check(
        "chars/token min",
        stats["ratio_min"],
        note="worst case; only matters if tokenize= is not passed",
    )
    R.check("chars/token p05", stats["ratio_p05"])
    R.check("chars/token p50", stats["ratio_p50"])

    # Informational: pipeline passes tokenize=, so this is fallback-only.
    if CHUNKER_CHARS_PER_TOKEN > stats["ratio_min"]:
        R.finding(
            f"chunker.CHARS_PER_TOKEN={CHUNKER_CHARS_PER_TOKEN} > measured min "
            f"{stats['ratio_min']}. Fallback path would underestimate tokens. "
            f"Informational only if pipeline passes tokenize= (it does via "
            f"chunk_worker). Set to <= {stats['ratio_p05']:.2f} for safety.",
            critical=False,
        )
    else:
        R.check(
            "CHARS_PER_TOKEN conservative",
            f"{CHUNKER_CHARS_PER_TOKEN} <= min {stats['ratio_min']}",
            ok=True,
        )

    return stats


# ── Phase 3: HTML extraction health (explains paragraphs p50 = 1) ──────
def inspect_html_health(rows: list[dict], max_per_kind: int = 3) -> None:
    R.section("3. HTML extraction health")

    if not rows:
        R.check("rows", "empty", ok=False)
        return

    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        kind = str(row.get("text_kind") or "unknown")
        by_kind.setdefault(kind, []).append(row)

    R.check("text_kind varieties", {k: len(v) for k, v in by_kind.items()})

    for kind, kind_rows in by_kind.items():
        # Per-opinion trace
        for i, row in enumerate(kind_rows[:max_per_kind]):
            raw = row.get("text") or ""
            try:
                stripped = _strip_html(raw, kind)
                normalized = _normalize(stripped)
            except Exception as exc:
                R.check(f"{kind}[{i}] pipeline failed", str(exc), ok=False)
                continue

            raw_p = raw.lower().count("<p")
            raw_br = raw.lower().count("<br")
            raw_div = raw.lower().count("<div")
            raw_dd = raw.count("\n\n")
            stripped_dd = stripped.count("\n\n")
            norm_dd = normalized.count("\n\n")
            paras = _paragraph_spans(normalized, 0, len(normalized))

            R.check(
                f"{kind}[{i}] chars={len(raw)}",
                f"raw tags: <p>={raw_p} <br>={raw_br} <div>={raw_div}",
            )
            R.check(
                f"{kind}[{i}] newlines",
                f"raw_\\n\\n={raw_dd} → stripped_\\n\\n={stripped_dd} → norm_\\n\\n={norm_dd}",
            )
            R.check(f"{kind}[{i}] paragraph spans", len(paras))

        # Aggregate stats per kind
        para_counts: list[int] = []
        for row in kind_rows:
            try:
                raw = row.get("text") or ""
                stripped = _strip_html(raw, kind)
                normalized = _normalize(stripped)
                paras = _paragraph_spans(normalized, 0, len(normalized))
                para_counts.append(len(paras))
            except Exception:
                para_counts.append(0)

        if para_counts:
            n_one = sum(1 for c in para_counts if c == 1)
            one_pct = 100 * n_one / len(para_counts)
            R.check(
                f"{kind}: paragraphs/opinion p50/p95",
                f"{int(pct(para_counts, 0.50))}/{int(pct(para_counts, 0.95))}",
            )
            R.check(
                f"{kind}: opinions with exactly 1 paragraph",
                f"{n_one}/{len(para_counts)} ({one_pct:.1f}%)",
                ok=(one_pct < 20.0),
                note="if high, HTML stripping or paragraph detection is broken"
                if one_pct >= 20
                else "",
            )
            if one_pct >= 20.0:
                R.finding(
                    f"{kind}: {one_pct:.0f}% of opinions have only 1 paragraph. "
                    f"Diagnose: does stripped text contain '\\n\\n'? If yes, "
                    f"_paragraph_spans is broken. If no, _strip_html is not "
                    f"emitting block breaks (check BLOCK_TAGS or raw structure).",
                    critical=False,
                )


# ── Phase 4: section pattern hit rate ──────────────────────────────────
def inspect_sections(texts: list[str]) -> dict:
    R.section("4. Section pattern hit rate")

    per_pattern: Counter = Counter()
    opinions_with_any = 0
    opinions_with_zero = 0
    section_counts: list[int] = []

    for text in texts:
        matched: set[str] = set()
        for name, _pri, regex in COMPILED_SECTION_PATTERNS:
            if regex.search(text):
                per_pattern[name] += 1
                matched.add(name)
        if matched:
            opinions_with_any += 1
        else:
            opinions_with_zero += 1
        try:
            section_counts.append(len(_detect_sections(text)))
        except Exception:
            section_counts.append(0)

    n = len(texts)
    fallback = opinions_with_zero / n if n else 1.0
    stats = {
        "opinions": n,
        "with_any_hit": opinions_with_any,
        "with_zero_hits": opinions_with_zero,
        "fallback_rate": round(fallback, 3),
        "per_pattern_hits": dict(per_pattern),
        "sections_p50": int(pct(section_counts, 0.50)),
        "sections_p95": int(pct(section_counts, 0.95)),
    }

    R.check(
        "opinions with ≥1 section hit",
        f"{opinions_with_any}/{n} ({100 * (1 - fallback):.1f}%)",
    )
    R.check("fallback rate (no section found)", f"{100 * fallback:.1f}%")
    R.check("sections/opinion p50", stats["sections_p50"])
    R.check("per-pattern hits", stats["per_pattern_hits"])

    if fallback > 0.80:
        R.finding(
            f"{100 * fallback:.0f}% fallback — section detection is decorative. "
            f"Consider paragraph-only chunking or add corpus-specific patterns.",
            critical=False,
        )
    elif fallback > 0.50:
        R.finding(
            f"{100 * fallback:.0f}% fallback — section detection catches <50%. "
            f"Benchmark parent-per-section vs paragraph-only before keeping.",
            critical=False,
        )
    else:
        R.check(
            "section detection viable",
            f"fallback {100 * fallback:.1f}% <= 50%",
            ok=True,
        )

    return stats


# ── Phase 5: boundary protection ───────────────────────────────────────
def _build_atomic_intervals(text: str) -> list[tuple[int, int]]:
    """One pass over all patterns. Returns sorted, merged intervals."""
    intervals: list[tuple[int, int]] = []
    for pat in ATOMIC_SPAN_PATTERNS:
        for m in pat.finditer(text):
            intervals.append((m.start(), m.end()))
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:  # overlapping or touching
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _pos_inside_intervals(intervals: list[tuple[int, int]], pos: int) -> bool:
    """Binary search: is pos strictly inside any interval?"""
    if not intervals:
        return False
    import bisect

    # Find the last interval whose start <= pos
    idx = bisect.bisect_right(intervals, (pos, float("inf"))) - 1
    if idx < 0:
        return False
    s, e = intervals[idx]
    return s < pos < e


def inspect_boundary_protection(texts: list[str], sample_size: int) -> dict:
    R.section("5. Boundary protection (the 3-bug check)")

    naive_re = re.compile(r'(?<=[.!?])[\x22\x27\u201d\u2019)]*\s+(?=[A-Z0-9("])')

    naive_total = naive_inside = 0
    chunker_total = chunker_inside = 0
    chunker_errors = 0

    # Sample fewer opinions for this expensive check — 100 is statistically
    # sufficient and 5x faster than 500.
    n_boundary = min(100, sample_size)

    for text in texts[:n_boundary]:
        if not text.strip():
            continue

        # Precompute once: O(patterns × text_length) instead of O(boundaries × patterns × text_length)
        intervals = _build_atomic_intervals(text)

        # Naive regex: count unprotected boundaries
        for m in naive_re.finditer(text):
            naive_total += 1
            if _pos_inside_intervals(intervals, m.start()):
                naive_inside += 1

        # Chunker's _sentence_spans: count unprotected boundaries
        try:
            spans = _sentence_spans(text, 0, len(text))
        except Exception:
            chunker_errors += 1
            continue
        for _s, e in spans:
            chunker_total += 1
            if _pos_inside_intervals(intervals, e):
                chunker_inside += 1

    naive_rate = naive_inside / naive_total if naive_total else 0.0
    chunker_rate = chunker_inside / chunker_total if chunker_total else 0.0

    stats = {
        "opinions_checked": n_boundary,
        "naive_boundaries": naive_total,
        "naive_inside": naive_inside,
        "naive_risk_rate": round(naive_rate, 4),
        "chunker_boundaries": chunker_total,
        "chunker_inside": chunker_inside,
        "chunker_risk_rate": round(chunker_rate, 4),
        "chunker_errors": chunker_errors,
    }

    R.check("opinions checked", n_boundary)
    R.check("naive boundaries", naive_total)
    R.check("naive inside atomic", naive_inside)
    R.check("naive risk rate", f"{100 * naive_rate:.2f}%")
    R.check("chunker boundaries", chunker_total)
    R.check("chunker inside atomic", chunker_inside)
    R.check("chunker risk rate", f"{100 * chunker_rate:.2f}%")

    if chunker_errors:
        R.finding(f"_sentence_spans raised {chunker_errors} times", critical=True)

    if chunker_inside > 0:
        R.finding(
            f"_sentence_spans emits {chunker_inside}/{chunker_total} boundaries "
            f"inside atomic spans. The _is_atomic_boundary filter is NOT applied. "
            f"This is the exact cause of Ala./v./Ms. splits.",
            critical=True,
        )
    elif naive_inside > 0:
        R.check(
            "chunker filters all naive atomic failures",
            f"naive={naive_inside} filtered → chunker=0",
            ok=True,
        )
    else:
        R.check("no atomic boundary risk", "0/0", ok=None)

    return stats


# ── Phase 6: paragraph structure ───────────────────────────────────────
def inspect_paragraphs(texts: list[str], count_fn) -> dict:
    R.section("6. Paragraph / sentence structure")

    para_counts: list[int] = []
    sent_counts: list[int] = []
    para_tokens: list[int] = []
    sent_tokens: list[int] = []
    oversized_sents = 0

    for text in texts:
        paras = _paragraph_spans(text, 0, len(text))
        para_counts.append(len(paras))
        for p_start, p_end in paras:
            pt = text[p_start:p_end]
            if not pt.strip():
                continue
            para_tokens.append(count_fn(pt))
            sents = _sentence_spans(text, p_start, p_end)
            sent_counts.append(len(sents))
            for s_start, s_end in sents:
                st = text[s_start:s_end]
                if not st.strip():
                    continue
                n = count_fn(st)
                sent_tokens.append(n)
                if n > CHILD_MAX_TOK:
                    oversized_sents += 1

    stats = {
        "paragraphs_p50": int(pct(para_counts, 0.50)),
        "paragraphs_p95": int(pct(para_counts, 0.95)),
        "paragraph_tokens_p50": int(pct(para_tokens, 0.50)),
        "paragraph_tokens_p95": int(pct(para_tokens, 0.95)),
        "paragraph_tokens_max": int(max(para_tokens)) if para_tokens else 0,
        "sentences_p50": int(pct(sent_counts, 0.50)),
        "sentences_p95": int(pct(sent_counts, 0.95)),
        "sentence_tokens_p95": int(pct(sent_tokens, 0.95)),
        "sentence_tokens_max": int(max(sent_tokens)) if sent_tokens else 0,
        "oversized_sentences": oversized_sents,
        "oversized_rate": round(oversized_sents / len(sent_tokens), 4) if sent_tokens else 0.0,
    }

    R.check("paragraphs/opinion p50", stats["paragraphs_p50"])
    R.check("paragraphs/opinion p95", stats["paragraphs_p95"])
    R.check("paragraph tokens p50", stats["paragraph_tokens_p50"])
    R.check("paragraph tokens p95", stats["paragraph_tokens_p95"])
    R.check("paragraph tokens max", stats["paragraph_tokens_max"])
    R.check("sentences/paragraph p50", stats["sentences_p50"])
    R.check("sentence tokens p95", stats["sentence_tokens_p95"])
    R.check("sentence tokens max", stats["sentence_tokens_max"])
    R.check(
        f"sentences > CHILD_MAX_TOK={CHILD_MAX_TOK}",
        f"{oversized_sents} ({100 * stats['oversized_rate']:.1f}%)",
    )

    if stats["paragraph_tokens_p50"] > CHILD_MAX_TOK:
        R.finding(
            f"p50 paragraph = {stats['paragraph_tokens_p50']} tokens > CHILD_MAX_TOK. "
            f"Every child needs _split_oversized — slow path.",
            critical=False,
        )

    if stats["oversized_rate"] > 0.10:
        R.finding(
            f"{100 * stats['oversized_rate']:.0f}% of sentences exceed CHILD_MAX_TOK — "
            f"_split_oversized fires often.",
            critical=False,
        )

    return stats


# ── Phase 7: chunker dry-run ───────────────────────────────────────────
def inspect_chunker_dry_run(
    rows: list[dict],
    tokenize,
    count_fn,
    sample_size: int,
) -> dict:
    R.section("7. Chunker dry-run (actual output validation)")

    if not CHUNKER_AVAILABLE or chunker_mod is None:
        R.check("chunker importable", "NO", ok=False)
        R.finding(
            f"cannot dry-run chunker: {_CHUNKER_IMPORT_ERROR}",
            critical=True,
        )
        return {}

    if tokenize is None:
        R.finding("no tokenizer — dry-run will use fallback heuristic", critical=False)

    children_per_op: list[int] = []
    parents_per_op: list[int] = []
    child_tokens: list[int] = []
    child_bad: list[tuple[str, int]] = []
    orphans = 0
    failures: list[tuple[Any, str]] = []

    t0 = time.monotonic()
    for row in rows[:sample_size]:
        try:
            chunks = chunker_mod.chunk_opinion(row, tokenize=tokenize)
        except Exception as exc:
            failures.append((row.get("cl_opinion_id"), f"{type(exc).__name__}: {exc}"))
            continue

        parents = [c for c in chunks if c.is_parent]
        children = [c for c in chunks if not c.is_parent]
        parents_per_op.append(len(parents))
        children_per_op.append(len(children))

        parent_ids = {c.chunk_id for c in parents}
        for c in children:
            if c.parent_id not in parent_ids:
                orphans += 1
            n = count_fn(c.text)  # full text = breadcrumb + body
            child_tokens.append(n)
            if n > EMBEDDER_MAX_TOKENS:
                child_bad.append((c.chunk_id, n))

    elapsed = time.monotonic() - t0

    stats = {
        "sample_size": min(sample_size, len(rows)),
        "successes": min(sample_size, len(rows)) - len(failures),
        "failures": len(failures),
        "orphans": orphans,
        "children_over_limit": len(child_bad),
        "parents_p50": int(pct(parents_per_op, 0.50)),
        "children_p50": int(pct(children_per_op, 0.50)),
        "children_p95": int(pct(children_per_op, 0.95)),
        "child_tokens_p50": int(pct(child_tokens, 0.50)),
        "child_tokens_p95": int(pct(child_tokens, 0.95)),
        "child_tokens_max": int(max(child_tokens)) if child_tokens else 0,
        "wall_seconds": round(elapsed, 1),
    }

    R.check("opinions dry-run", stats["sample_size"])
    R.check("successful", stats["successes"])
    R.check("failed", stats["failures"], ok=(stats["failures"] == 0))
    R.check("orphan children", stats["orphans"], ok=(stats["orphans"] == 0))
    R.check("parents/opinion p50", stats["parents_p50"])
    R.check("children/opinion p50", stats["children_p50"])
    R.check("children/opinion p95", stats["children_p95"])
    R.check("child full-text tokens p50", stats["child_tokens_p50"])
    R.check("child full-text tokens p95", stats["child_tokens_p95"])
    R.check("child full-text tokens max", stats["child_tokens_max"])
    R.check(
        f"children over EMBEDDER_MAX_TOKENS={EMBEDDER_MAX_TOKENS}",
        stats["children_over_limit"],
        ok=(stats["children_over_limit"] == 0),
    )
    R.check("wall time (s)", stats["wall_seconds"])

    if stats["failures"]:
        for case_id, msg in failures[:5]:
            R.finding(f"chunk_opinion failed: case={case_id} {msg}", critical=True)

    if stats["orphans"] > 0:
        R.finding(
            f"{stats['orphans']} children reference non-existent parents",
            critical=True,
        )

    if stats["children_over_limit"] > 0:
        for cid, n in child_bad[:5]:
            R.finding(
                f"child {cid} = {n} tokens > EMBEDDER_MAX_TOKENS={EMBEDDER_MAX_TOKENS}",
                critical=True,
            )
        R.finding(
            f"{stats['children_over_limit']} children exceed embedding limit; "
            f"embed would fail (torch embedder refuses oversized input)",
            critical=True,
        )

    return stats


# ── Phase 8: graph integrity (FULL corpus) ─────────────────────────────
def inspect_graph(root: Path) -> dict:
    R.section("8. Citation graph integrity (against full corpus)")

    p = root / "graph/citations_filtered.parquet"
    if not p.is_file():
        R.check("graph", "missing", ok=False)
        return {}

    g = pd.read_parquet(p)
    corpus_ids = load_full_corpus_ids(root)
    R.check("graph edges", len(g))
    R.check("corpus opinions", len(corpus_ids))

    src_missing = int((~g["citing_opinion_id"].isin(corpus_ids)).sum()) if corpus_ids else -1
    tgt_missing = int((~g["cited_opinion_id"].isin(corpus_ids)).sum()) if corpus_ids else -1

    R.check("edges citing_id not in corpus", src_missing, ok=(src_missing == 0))
    R.check("edges cited_id not in corpus", tgt_missing, ok=(tgt_missing == 0))

    if src_missing > 0 or tgt_missing > 0:
        R.finding(
            f"{src_missing + tgt_missing} dangling edges — graph traversal will "
            f"silently return nothing. Filter stage did not run correctly.",
            critical=True,
        )

    self_loops = int((g["citing_opinion_id"] == g["cited_opinion_id"]).sum())
    dup_edges = int(g.duplicated(subset=["citing_opinion_id", "cited_opinion_id"]).sum())
    R.check("self-loops", self_loops, ok=(self_loops == 0))
    R.check("duplicate (citing, cited)", dup_edges)

    out_deg = g.groupby("citing_opinion_id").size()
    in_deg = g.groupby("cited_opinion_id").size()
    R.check(
        "out-degree p50/p95/max",
        [int(out_deg.median()), int(out_deg.quantile(0.95)), int(out_deg.max())],
    )
    R.check(
        "in-degree p50/p95/max",
        [int(in_deg.median()), int(in_deg.quantile(0.95)), int(in_deg.max())],
    )

    return {"edges": len(g), "src_dangling": src_missing, "tgt_dangling": tgt_missing}


# ── Phase 9: metadata coverage (FULL corpus) ───────────────────────────
def inspect_metadata(root: Path) -> dict:
    R.section("9. Cluster metadata coverage (against full corpus)")

    mp = root / "metadata/clusters_filtered.parquet"
    cp = root / "corpus/opinions.parquet"
    if not mp.is_file() or not cp.is_file():
        R.check("metadata or corpus", "missing", ok=False)
        return {}

    m = pd.read_parquet(mp)
    c = pd.read_parquet(cp, columns=["cluster_id"])
    total = len(c)
    with_cluster = int(c["cluster_id"].notna().sum())
    null_cluster = total - with_cluster
    in_meta = int(c["cluster_id"].isin(m["id"]).sum())

    R.check("cluster rows", len(m))
    R.check("corpus opinions", total)
    R.check("with cluster_id", f"{with_cluster}/{total}")
    R.check(
        "null cluster_id",
        f"{null_cluster}/{total} ({100 * null_cluster / total:.2f}%)" if total else "n/a",
    )
    R.check(
        "clusters in metadata",
        f"{in_meta}/{total}",
        ok=(in_meta / total > 0.95) if total else None,
    )

    if with_cluster > 0 and in_meta < with_cluster * 0.95:
        R.finding(
            f"{with_cluster - in_meta} non-null cluster IDs missing from metadata. "
            f"Metadata filtering will exclude those opinions silently.",
            critical=False,
        )

    if "precedential_status" in m.columns:
        R.check(
            "precedential_status top-5",
            m["precedential_status"].value_counts(dropna=False).head(5).to_dict(),
        )

    return {"clusters": len(m), "corpus": total, "in_meta": in_meta}


# ── Phase 10: eval isolation ───────────────────────────────────────────
def inspect_eval(root: Path) -> None:
    R.section("10. Eval isolation")

    for rel in [
        "eval_isolated/hojicha/corpus.parquet",
        "eval_isolated/hojicha/questions.parquet",
        "eval_isolated/synthesized_graph_qa/queries.parquet",
    ]:
        p = root / rel
        if not p.is_file():
            R.check(rel, "MISSING", ok=False)
            R.finding(f"missing eval file: {rel}", critical=True)
            continue
        n = pq.ParquetFile(p).metadata.num_rows
        R.check(rel, f"{n} rows", ok=True)

    hq = root / "eval_isolated/hojicha/corpus.parquet"
    cp = root / "corpus/opinions.parquet"
    if hq.is_file() and cp.is_file():
        h = pd.read_parquet(hq, columns=["title"])
        c = pd.read_parquet(cp, columns=["title"])
        overlap = int(h["title"].isin(c["title"]).sum())
        R.check(
            "hojicha titles in production",
            overlap,
            ok=(overlap == 0),
            note="eval data must not appear in production",
        )


# ── Phase 11: text quality ─────────────────────────────────────────────
def inspect_quality(texts: list[str]) -> dict:
    R.section("11. Text quality (duplication, low-info, HTML residue)")

    para_hashes: Counter = Counter()
    total_paras = 0
    short_paras = 0
    nonalpha_paras = 0
    html_residue = 0
    unicode_chars = total_chars = 0
    html_tag_re = re.compile(r"<[a-zA-Z/][^>]*>")

    for text in texts:
        unicode_chars += sum(1 for ch in text if ord(ch) > 127)
        total_chars += len(text)
        for p_start, p_end in _paragraph_spans(text, 0, len(text)):
            pt = text[p_start:p_end].strip()
            if not pt:
                continue
            total_paras += 1
            para_hashes[hashlib.sha256(pt.encode()).hexdigest()] += 1
            if len(pt) < 50:
                short_paras += 1
            nonalpha = sum(1 for ch in pt if not ch.isalnum() and not ch.isspace())
            if nonalpha / len(pt) > 0.5:
                nonalpha_paras += 1
            if html_tag_re.search(pt):
                html_residue += 1

    dup_extra = sum(c - 1 for c in para_hashes.values() if c > 1)
    stats = {
        "total_paragraphs": total_paras,
        "duplicate_rate": round(dup_extra / total_paras, 4) if total_paras else 0.0,
        "short_paragraph_rate": round(short_paras / total_paras, 4) if total_paras else 0.0,
        "nonalpha_rate": round(nonalpha_paras / total_paras, 4) if total_paras else 0.0,
        "html_residue_rate": round(html_residue / total_paras, 4) if total_paras else 0.0,
        "unicode_density": round(unicode_chars / total_chars, 4) if total_chars else 0.0,
    }

    R.check("total paragraphs", total_paras)
    R.check("duplicate rate", f"{100 * stats['duplicate_rate']:.2f}%")
    R.check("short paragraphs (<50 chars)", f"{100 * stats['short_paragraph_rate']:.1f}%")
    R.check("nonalpha-dominant paragraphs", f"{100 * stats['nonalpha_rate']:.2f}%")
    R.check("HTML residue rate", f"{100 * stats['html_residue_rate']:.2f}%")
    R.check("unicode char density", f"{100 * stats['unicode_density']:.2f}%")

    if stats["html_residue_rate"] > 0.01:
        R.finding(
            f"{100 * stats['html_residue_rate']:.1f}% of paragraphs contain HTML tags. "
            f"_strip_html is failing or text_kind is wrong.",
            critical=True,
        )
    if stats["duplicate_rate"] > 0.05:
        R.finding(
            f"{100 * stats['duplicate_rate']:.1f}% paragraph duplication. "
            f"Add MinHash dedup at ingestion.",
            critical=False,
        )
    if stats["short_paragraph_rate"] > 0.10:
        R.finding(
            f"{100 * stats['short_paragraph_rate']:.1f}% of paragraphs are <50 chars. "
            f"Consider a min-content filter.",
            critical=False,
        )

    return stats


# ── Phase 12: config derivation ────────────────────────────────────────
def derive_recommendations(
    token_stats: dict | None,
    section_stats: dict | None,
    boundary_stats: dict | None,
    structure_stats: dict | None,
    quality_stats: dict | None,
    dry_run_stats: dict | None,
) -> None:
    R.section("12. Config recommendations (derived from measurements)")

    if not token_stats:
        R.finding("no tokenizer — cannot derive sizing recommendations", critical=False)
        return

    # 1. CHARS_PER_TOKEN — informational only (pipeline uses tokenize=)
    if CHUNKER_CHARS_PER_TOKEN <= token_stats["ratio_p05"]:
        R.check(
            "CHARS_PER_TOKEN safe",
            f"{CHUNKER_CHARS_PER_TOKEN} <= p05 {token_stats['ratio_p05']}",
            ok=True,
        )
    else:
        R.finding(
            f"CHARS_PER_TOKEN={CHUNKER_CHARS_PER_TOKEN} > p05 {token_stats['ratio_p05']}. "
            f"Fallback path unsafe on >5% of docs. Pipeline uses tokenize= so "
            f"non-fatal, but reduce to {token_stats['ratio_p05']:.2f} for safety.",
            critical=False,
        )

    # 2. Child sizing vs embedding limit
    breadcrumb_est = 40
    if CHILD_MAX_TOK + breadcrumb_est <= EMBEDDER_MAX_TOKENS:
        R.check(
            "child + breadcrumb fits embedding limit",
            f"{CHILD_MAX_TOK}+{breadcrumb_est} <= {EMBEDDER_MAX_TOKENS}",
            ok=True,
        )
    else:
        R.finding(
            f"CHILD_MAX_TOK={CHILD_MAX_TOK}+{breadcrumb_est} > "
            f"EMBEDDER_MAX_TOKENS={EMBEDDER_MAX_TOKENS}. Reduce CHILD_MAX_TOK "
            f"to <= {EMBEDDER_MAX_TOKENS - breadcrumb_est}.",
            critical=True,
        )

    # 3. Section detection viability
    if section_stats:
        if section_stats["fallback_rate"] > 0.80:
            R.finding(
                f"section fallback {100 * section_stats['fallback_rate']:.0f}% — disable "
                f"section detection or add corpus-specific patterns.",
                critical=False,
            )
        elif section_stats["fallback_rate"] < 0.20:
            R.check(
                "section detection reliable",
                f"fallback {100 * section_stats['fallback_rate']:.1f}%",
                ok=True,
            )

    # 4. Boundary protection
    if boundary_stats:
        if boundary_stats["chunker_inside"] > 0:
            R.finding(
                f"chunker still produces {boundary_stats['chunker_inside']} unprotected "
                f"boundaries — the fix is NOT applied. See section 5.",
                critical=True,
            )
        elif boundary_stats["naive_inside"] > 0:
            R.check(
                "boundary filter effective",
                f"{boundary_stats['naive_inside']} naive failures all filtered",
                ok=True,
            )

    # 5. Chunker dry-run output
    if dry_run_stats and dry_run_stats.get("children_over_limit", 0) > 0:
        R.finding(
            f"dry-run: {dry_run_stats['children_over_limit']} children exceed embedding "
            f"limit. Do NOT scale until fixed.",
            critical=True,
        )

    # 6. Quality warnings
    if quality_stats:
        if quality_stats["html_residue_rate"] > 0.01:
            R.finding("fix _strip_html before scaling", critical=True)
        if quality_stats["duplicate_rate"] > 0.05:
            R.finding("add MinHash dedup", critical=False)

    # 7. Structure
    if structure_stats and structure_stats["paragraphs_p50"] <= 1:
        R.finding(
            f"p50 paragraphs/opinion = {structure_stats['paragraphs_p50']}. "
            f"Diagnose via section 3 (HTML health). Chunking still works but "
            f"parent-per-section logic is ineffective.",
            critical=False,
        )


# ── Special mode: single opinion ───────────────────────────────────────
def inspect_one(root: Path, case_id: int, tokenize, count_fn) -> int:
    R.section(f"Single opinion: {case_id}")

    cp = root / "corpus/opinions.parquet"
    df = pd.read_parquet(cp)
    matched = df[df["cl_opinion_id"] == case_id]
    if len(matched) == 0:
        R.check("found", "NO", ok=False)
        return 1
    rec = matched.iloc[0].to_dict()

    raw = rec.get("text") or ""
    kind = str(rec.get("text_kind") or "plain_text")
    R.check("title", rec.get("title"))
    R.check("court", rec.get("court"))
    R.check("date", rec.get("date"))
    R.check("text_kind", kind)
    R.check("opinion_type", rec.get("opinion_type"))
    R.check("raw chars", len(raw))

    try:
        stripped = _strip_html(raw, kind)
        normalized = _normalize(stripped)
    except Exception as exc:
        R.check("strip/normalize failed", str(exc), ok=False)
        return 1

    R.check("stripped chars", len(stripped))
    R.check("normalized chars", len(normalized))
    R.check("normalized \\n\\n count", normalized.count("\n\n"))

    paras = _paragraph_spans(normalized, 0, len(normalized))
    R.check("paragraph spans", len(paras))
    for i, (s, e) in enumerate(paras[:5]):
        if count_fn:
            R.check(f"  para {i} tokens", count_fn(normalized[s:e]))

    if CHUNKER_AVAILABLE and chunker_mod is not None:
        try:
            chunks = chunker_mod.chunk_opinion(rec, tokenize=tokenize)
        except Exception as exc:
            R.check("chunk_opinion failed", str(exc), ok=False)
            return 1
        parents = [c for c in chunks if c.is_parent]
        children = [c for c in chunks if not c.is_parent]
        R.check("chunks emitted", len(chunks))
        R.check("parents", len(parents))
        R.check("children", len(children))
        for c in children[:10]:
            n = count_fn(c.text) if count_fn else 0
            R.check(
                f"  {c.chunk_id}",
                f"{n} tokens, section={c.section}, body_chars={len(c.text_original)}",
                ok=(n <= EMBEDDER_MAX_TOKENS) if n else None,
            )

    return 0


# ── Main ───────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", nargs="?", default="/workspace/tmp/data/raw")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sample", type=int, default=500, help="opinions for statistics (default 500)")
    ap.add_argument(
        "--chunker-sample",
        type=int,
        default=30,
        help="opinions for chunker dry-run (default 30)",
    )
    ap.add_argument("--tokenizer-path", type=Path, default=None)
    ap.add_argument(
        "--require-tokenizer",
        action="store_true",
        help="exit 3 if no tokenizer available",
    )
    ap.add_argument(
        "--inspect-one",
        type=int,
        default=None,
        help="deep-dive a single case_id and exit",
    )
    args = ap.parse_args()

    R.json_mode = bool(args.json)
    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 3

    if not R.json_mode:
        print(f"inspecting: {root}")
        print(f"chunker:    {CHUNKER_VERSION}")
        if CHUNKER_AVAILABLE:
            print(
                f"            parent {PARENT_TARGET_TOK}/{PARENT_MAX_TOK}/{PARENT_MIN_TOK}, "
                f"child {CHILD_TARGET_TOK}/{CHILD_MAX_TOK}/{CHILD_MIN_TOK}, "
                f"chars/token {CHUNKER_CHARS_PER_TOKEN}"
            )

    tok = load_tokenizer(args.tokenizer_path)
    if tok is None:
        if args.require_tokenizer:
            print("error: no tokenizer loaded (--require-tokenizer)", file=sys.stderr)
            return 3
        print(
            "WARNING: no tokenizer — token measurements will use fallback",
            file=sys.stderr,
        )
        count_fn = lambda s: len(s) // 4
    else:
        count_fn = _make_token_counter(tok)

    # Single-opinion deep dive
    if args.inspect_one is not None:
        if tok is None:
            print("error: --inspect-one requires a tokenizer", file=sys.stderr)
            return 3
        return inspect_one(root, args.inspect_one, tok, count_fn)

    # Sample for statistics
    rows, texts = load_sample_rows(root, args.sample)
    if not texts:
        print("error: no valid text rows in sample", file=sys.stderr)
        return 3
    if not R.json_mode:
        print(
            f"sample:     {len(texts)} opinions "
            f"(chunker dry-run: {min(args.chunker_sample, len(rows))})"
        )

    # Phase execution
    inspect_files_schema(root)
    R.end()

    token_stats = inspect_tokens(texts, count_fn)
    R.end()

    inspect_html_health(rows)
    R.end()

    section_stats = inspect_sections(texts)
    R.end()

    boundary_stats = inspect_boundary_protection(texts, sample_size=len(texts))
    R.end()

    structure_stats = inspect_paragraphs(texts, count_fn)
    R.end()

    dry_run_stats = (
        inspect_chunker_dry_run(
            rows, tok, count_fn, sample_size=min(args.chunker_sample, len(rows))
        )
        if CHUNKER_AVAILABLE
        else {}
    )
    R.end()

    inspect_graph(root)
    R.end()
    inspect_metadata(root)
    R.end()
    inspect_eval(root)
    R.end()
    quality_stats = inspect_quality(texts)
    R.end()

    derive_recommendations(
        token_stats,
        section_stats,
        boundary_stats,
        structure_stats,
        quality_stats,
        dry_run_stats,
    )
    R.end()

    # Output
    if R.json_mode:
        print(json.dumps(R.dump(), indent=2, default=str))
    else:
        print(f"\n{'=' * 72}")
        if R.critical:
            print(f"CRITICAL: {len(R.critical)} finding(s)")
            for c in R.critical:
                print(f"  - {c}")
            print("\nDo NOT chunk at scale until resolved.")
        elif R.warnings:
            print(f"OK: no critical findings, {len(R.warnings)} warning(s)")
            for w in R.warnings:
                print(f"  - {w}")
        else:
            print("CLEAN: no critical findings, no warnings. Ready to chunk.")
        print("=" * 72)

    return 2 if R.critical else (1 if R.warnings else 0)


if __name__ == "__main__":
    sys.exit(main())
