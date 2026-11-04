#!/usr/bin/env python3
"""inspect_chunks.py — pre-load quality checks on chunked/ output.

Reads chunked/{case_id}.jsonl without touching the database or embedder.
Mirrors gates 1-8 from db.py (gate_9 needs embed, so it lives elsewhere).

Answers:
  1. Are all case files present and well-formed?
  2. What is the actual token distribution (real tokenizer)?
  3. Do all children fit EMBEDDER_MAX_TOKENS=464?
  4. Does every chunk carry a valid breadcrumb?
  5. Do all children reference a valid parent in the same case?
  6. Is the chunker version consistent across all files?
  7. Do children's section/section_path match their parents'?
  8. Does every child body end on terminal punctuation?
  9. Are there any split-citation fragments at child body tails?
 10. Are there any intra-case duplicate child bodies?
 11. What does the dedup report say (if dedup has run)?
 12. Show sample chunks for eyeballing.
 13. (optional) Re-chunk N cases and verify stored ⊆ re-chunked.

Usage:
    python3 src/utils/inspect_chunks.py --chunked-dir /workspace/tmp/data/chunked
    python3 src/utils/inspect_chunks.py --chunked-dir /workspace/tmp/data/chunked \\
        --tokenizer-path /workspace/src/indexing_pipeline/embedders/dinghy-law-0.6b-v1-openvino-fp16
    python3 src/utils/inspect_chunks.py --chunked-dir ... --json > report.json
    python3 src/utils/inspect_chunks.py --chunked-dir ... --determinism-sample 20

Exit codes:
    0 — clean, ready to embed
    1 — warnings only
    2 — critical findings (do NOT embed)
    3 — setup error (missing dir, no tokenizer)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

# Must match chunker.EMBEDDER_MAX_TOKENS (default 464 for OpenVINO).
# Kept hardcoded here so this tool runs standalone even if the package
# cannot be imported.
EMBEDDER_MAX_TOKENS = 464

# Gate thresholds (must match db.py)
SIZE_P50_MAX = 400
SIZE_P95_MAX = 450
SIZE_CHILD_MAX = 450
SIZE_PARENT_MAX = 1600
BAD_ENDING_RATE_MAX = 0.02

# Breadcrumb regex — matches `_make_breadcrumb` output:
#   [title, citation, year · court · opinion_type · section]
# That is 4 ` · `-separated groups (3 separators). Must match db.py gate_2.
BREADCRUMB_RE = re.compile(r"^\[[^\]\n]+ · [^\]\n]+ · [^\]\n]+ · [^\]\n]+\]")

# Boundary integrity regex — matches db.py gate_3
END_PUNCT_RE = re.compile(r'[.!?…:](?:["”’)\]]+)?$')

# Citation-split patterns — must match db.py gate_4 (7 specific clauses).
# The previous catch-all `(?:[A-Z][a-z]+\.)+$` was removed: it matched
# any sentence ending in a Capitalized word (i.e. most English sentences)
# and produced false positives on ~3% of chunks.
CITATION_SPLIT_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"[0-9]+[ \t]+U[.]S[.][ \t]*$",  # 410 U.S.
        r"[0-9]+[ \t]+(?:So|P|A|N[.]E|N[.]W|S[.]E|S[.]W)[.][ \t]*[0-9]*[a-z]*[ \t]*$",  # 565 So.2d
        r"[0-9]+[ \t]+U[.]S[.]C[.][ \t]*$",  # 42 U.S.C.
        r"(?:[A-Z][a-z]+[.][ \t]*)+Code[ \t]*$",  # Ala. Code
        r"\bv[.][ \t]*$",  # Brown v.
        r"\b(?:Mr|Mrs|Ms|Dr|Hon|Prof|Rev|Gen|Col|Sgt|Capt|Lt|St)[.][ \t]*$",  # Mr.
        r"[§][ \t]*$",  # bare §
    )
)


# ── Reporter ────────────────────────────────────────────────────────────


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

    def check(self, name: str, value: Any, ok: bool | None = None, note: str = "") -> None:
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


# ── Helpers ─────────────────────────────────────────────────────────────


def pct(values: Sequence[int | float], p: float) -> float:
    """Return the p-th percentile of `values`. Accepts int or float input."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(p * (len(s) - 1))))
    return float(s[idx])


def load_tokenizer(explicit: Path | None) -> Any:
    from transformers import AutoTokenizer

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("/workspace/src/indexing_pipeline/embedders/dinghy-law-0.6b-v1-openvino-fp16"),
            Path("/workspace/src/indexing_pipeline/models/dinghy-law-0.6b-v1"),
        ]
    )
    for c in candidates:
        if c and (c / "tokenizer.json").is_file():
            try:
                tok = AutoTokenizer.from_pretrained(str(c), local_files_only=True)
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


def iter_cases(chunked_dir: Path) -> Iterator[tuple[int, list[dict]]]:
    """Yield (case_id, rows) for each case file; skip _-prefixed metadata."""
    for p in sorted(chunked_dir.glob("*.jsonl")):
        if p.name.startswith("_") or not p.stem.isdigit():
            continue
        case_id = int(p.stem)
        rows = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        yield case_id, rows


# ── Checks ──────────────────────────────────────────────────────────────


def check_inventory(chunked_dir: Path) -> dict:
    R.section("1. File inventory")
    files = [p for p in chunked_dir.glob("*.jsonl") if not p.name.startswith("_")]
    meta = [p for p in chunked_dir.iterdir() if p.name.startswith("_")]
    total_bytes = sum(p.stat().st_size for p in files)
    R.check("case files", len(files))
    R.check("metadata files", len(meta), note=", ".join(sorted(p.name for p in meta)))
    R.check("total size", f"{total_bytes / (1024 * 1024):.1f} MB")
    if len(files) == 0:
        R.finding("no case files", critical=True)
    return {"files": len(files), "bytes": total_bytes}


def check_inventory_chunks(chunked_dir: Path) -> dict:
    R.section("2. Chunk inventory (parents vs children)")
    parents = children = 0
    per_case_children: list[int] = []
    opinion_types: Counter = Counter()
    sections: Counter = Counter()
    for _case_id, rows in iter_cases(chunked_dir):
        case_children = 0
        for r in rows:
            opinion_types[r.get("opinion_type", "")] += 1
            sections[r.get("section", "")] += 1
            if r.get("is_parent"):
                parents += 1
            else:
                children += 1
                case_children += 1
        per_case_children.append(case_children)

    total = parents + children
    R.check("parents", parents)
    R.check("children", children)
    R.check("total chunks", total)
    if children == 0:
        R.finding("no children emitted — chunker produced only parents", critical=True)
    R.check(
        "children/case p50/p95",
        f"{int(pct(per_case_children, 0.5))}/{int(pct(per_case_children, 0.95))}",
    )
    R.check("opinion types", dict(opinion_types))
    R.check("sections", dict(sections.most_common(8)))
    return {"parents": parents, "children": children, "cases": len(per_case_children)}


def check_versions(chunked_dir: Path) -> dict:
    R.section("3. Chunker version consistency")
    versions: Counter = Counter()
    for _case_id, rows in iter_cases(chunked_dir):
        for r in rows:
            versions[r.get("chunker_version", "<missing>")] += 1
    R.check("versions", dict(versions))
    if len(versions) > 1:
        R.finding(
            f"mixed chunker_versions: {dict(versions)}. Re-run `cli chunk --force` to unify.",
            critical=True,
        )
    elif "<missing>" in versions:
        R.finding("chunks missing chunker_version", critical=True)
    return {"versions": dict(versions)}


def check_token_distribution(chunked_dir: Path, tokenize: Any) -> dict:
    R.section("4. Token distribution (real tokenizer)")
    tokens: list[int] = []
    over: list[tuple[str, int]] = []
    for _case_id, rows in iter_cases(chunked_dir):
        for r in rows:
            if r.get("is_parent"):
                continue
            n = tokenize(r["text"])
            tokens.append(n)
            if n > EMBEDDER_MAX_TOKENS:
                over.append((r["chunk_id"], n))

    if not tokens:
        R.check("children", 0)
        return {"tokens": [], "oversize": []}

    p50 = int(pct(tokens, 0.5))
    p95 = int(pct(tokens, 0.95))
    mx = max(tokens)
    R.check("children", len(tokens))
    R.check("tokens p50", p50)
    R.check("tokens p95", p95)
    R.check("tokens max", mx)
    R.check(
        f"oversize (>{EMBEDDER_MAX_TOKENS})",
        len(over),
        ok=(len(over) == 0),
    )
    if over:
        for cid, n in over[:5]:
            R.finding(f"child {cid} = {n} tokens > {EMBEDDER_MAX_TOKENS}", critical=True)
        R.finding(
            f"{len(over)} children exceed embedding limit — chunker regression or bug in v1.6",
            critical=True,
        )
    if p50 > SIZE_P50_MAX:
        R.finding(f"p50 {p50} > {SIZE_P50_MAX} — child target too high", critical=False)
    return {"tokens": tokens, "oversize": over}


def check_breadcrumb(chunked_dir: Path) -> dict:
    R.section("5. Breadcrumb presence")
    total = missing = 0
    examples: list[str] = []
    for _case_id, rows in iter_cases(chunked_dir):
        for r in rows:
            total += 1
            if not BREADCRUMB_RE.match(r.get("text", "")):
                missing += 1
                if len(examples) < 3:
                    examples.append(r.get("chunk_id", "?"))
    rate = missing / total if total else 0.0
    R.check("total chunks", total)
    R.check("missing breadcrumb", missing, ok=(missing == 0), note=f"sample: {examples}")
    R.check("breadcrumb rate", f"{100 * (1 - rate):.1f}%", ok=(missing == 0))
    if missing:
        R.finding(
            f"{missing}/{total} chunks lack the '[... · ... · ... · ...]' breadcrumb. "
            f"Chunker v1.6 broken or older output mixed in.",
            critical=True,
        )
    return {"total": total, "missing": missing}


def check_parent_child(chunked_dir: Path) -> dict:
    R.section("6. Parent-child integrity")
    total_children = 0
    orphan_children = 0
    for _case_id, rows in iter_cases(chunked_dir):
        parent_ids = {r["chunk_id"] for r in rows if r.get("is_parent")}
        for r in rows:
            if r.get("is_parent"):
                continue
            total_children += 1
            pid = r.get("parent_id")
            if pid not in parent_ids:
                orphan_children += 1

    R.check("children", total_children)
    R.check("orphan children", orphan_children, ok=(orphan_children == 0))
    if orphan_children:
        R.finding(
            f"{orphan_children}/{total_children} children reference a parent_id "
            f"not present in the same case file. Chunker bug.",
            critical=True,
        )

    # chunk_id uniqueness across the whole corpus. chunk_id embeds case_id,
    # so a collision between different cases means a bug in _make_chunk_id.
    seen_ids: dict[str, int] = {}
    for _case_id, rows in iter_cases(chunked_dir):
        for r in rows:
            seen_ids[r["chunk_id"]] = seen_ids.get(r["chunk_id"], 0) + 1
    collisions = {cid: n for cid, n in seen_ids.items() if n > 1}
    if collisions:
        R.finding(
            f"{len(collisions)} chunk_id collisions across cases (sample: {list(collisions)[:3]})",
            critical=True,
        )
    return {"orphans": orphan_children, "total_children": total_children}


def check_sections_consistent(chunked_dir: Path) -> dict:
    R.section("7. Section / section_path consistency")
    total = bad_section = bad_path = 0
    for _case_id, rows in iter_cases(chunked_dir):
        parents = {r["chunk_id"]: r for r in rows if r.get("is_parent")}
        for r in rows:
            if r.get("is_parent"):
                continue
            total += 1
            parent = parents.get(r.get("parent_id"))
            if not parent:
                continue
            if r.get("section") != parent.get("section"):
                bad_section += 1
            if r.get("section_path") != parent.get("section_path"):
                bad_path += 1
    R.check("children", total)
    R.check("section mismatch", bad_section)
    R.check("section_path mismatch", bad_path)
    rate = (bad_section + bad_path) / (2 * total) if total else 0.0
    R.check("consistency", f"{100 * (1 - rate):.1f}%", ok=(rate == 0))
    if rate > 0:
        R.finding(
            f"{bad_section} section + {bad_path} section_path mismatches "
            f"between children and parents",
            critical=False,
        )
    return {"mismatches": bad_section + bad_path, "total": total}


def check_boundary_integrity(chunked_dir: Path) -> dict:
    R.section("8. Boundary integrity (terminal punctuation)")
    total = bad = 0
    examples: list[str] = []
    for _case_id, rows in iter_cases(chunked_dir):
        for r in rows:
            if r.get("is_parent"):
                continue
            total += 1
            body = r.get("text_original", "").rstrip()
            if not END_PUNCT_RE.search(body):
                bad += 1
                if len(examples) < 5:
                    examples.append(r["chunk_id"])
    rate = bad / total if total else 0.0
    R.check("children", total)
    R.check("bad endings", bad, note=f"sample: {examples}")
    R.check("bad ending rate", f"{100 * rate:.2f}%", ok=(rate <= BAD_ENDING_RATE_MAX))
    if rate > BAD_ENDING_RATE_MAX:
        R.finding(
            f"{100 * rate:.1f}% of children end mid-sentence — boundary integrity gate failed",
            critical=False,
        )
    return {"bad": bad, "total": total}


def check_citation_integrity(chunked_dir: Path) -> dict:
    R.section("9. Citation integrity (split-fragment detection)")
    total = hits = 0
    examples: list[tuple[str, str]] = []
    for _case_id, rows in iter_cases(chunked_dir):
        for r in rows:
            if r.get("is_parent"):
                continue
            total += 1
            body = r.get("text_original", "").rstrip()
            for pat in CITATION_SPLIT_RES:
                if pat.search(body):
                    hits += 1
                    if len(examples) < 5:
                        examples.append((r["chunk_id"], body[-40:].replace("\n", " ")))
                    break
    R.check("children", total)
    R.check("split-citation tails", hits, ok=(hits == 0), note=f"sample: {examples[:3]}")
    if hits:
        R.finding(
            f"{hits}/{total} children end on a citation fragment. "
            f"Chunker v1.6 atomic-span filter is failing on these patterns.",
            critical=True,
        )
    return {"hits": hits, "total": total}


def check_duplicates(chunked_dir: Path) -> dict:
    R.section("10. Intra-case duplicate detection")
    total = dups = 0
    examples: list[tuple[int, str]] = []
    for case_id, rows in iter_cases(chunked_dir):
        seen: Counter = Counter()
        for r in rows:
            if r.get("is_parent"):
                continue
            total += 1
            seen[r.get("text_hash", "")] += 1
        for h, n in seen.items():
            if n > 1:
                dups += n - 1
                if len(examples) < 5:
                    examples.append((case_id, h[:20]))
    R.check("children", total)
    R.check("intra-case duplicates", dups, ok=(dups == 0), note=f"sample: {examples[:3]}")
    if dups:
        R.finding(
            f"{dups} intra-case duplicate child bodies — chunker regression. "
            f"Cross-case dedup runs later via dedup.py; intra-case is a bug.",
            critical=True,
        )
    return {"dups": dups, "total": total}


def check_dedup_report(chunked_dir: Path) -> dict:
    R.section("11. Dedup report")
    path = chunked_dir / "_dedup_report.json"
    if not path.is_file():
        R.check("_dedup_report.json", "absent", note="dedup has not run yet")
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        R.check("_dedup_report.json", f"unparseable: {exc}", ok=False)
        return {}
    totals = data.get("totals", {})
    R.check("duplicate_hashes", data.get("duplicate_hashes", "?"))
    R.check("chunks_dropped", totals.get("chunks_dropped", "?"))
    R.check("cases_rewritten", totals.get("cases_rewritten", "?"))
    R.check("duplicate_rate", f"{100 * data.get('duplicate_rate', 0):.2f}%")
    if data.get("intra_case_duplicates"):
        R.finding(
            f"dedup found {len(data['intra_case_duplicates'])} intra-case duplicate groups — "
            f"chunker regression. Check section 10 above.",
            critical=True,
        )
    return data


def show_samples(chunked_dir: Path, tokenize: Any, count: int = 3) -> None:
    R.section("12. Sample chunks (eyeball check)")
    shown = 0
    for case_id, rows in iter_cases(chunked_dir):
        if shown >= count:
            break
        for r in rows:
            if r.get("is_parent"):
                continue
            preview = r["text"][:240].replace("\n", " ⏎ ")
            n = tokenize(r["text"])
            R.check(
                f"case {case_id} | {r['chunk_id']}",
                f"{n} tok, section={r['section']}",
                note=preview[:180] + ("..." if len(preview) > 180 else ""),
            )
            shown += 1
            if shown >= count:
                break


def check_determinism(chunked_dir: Path, tokenize: Any, sample: int) -> dict:
    """Re-chunk the first N cases and verify stored ⊆ re-chunked.

    After dedup, stored chunks are a SUBSET of re-chunked chunks (cross-
    case duplicates were removed). So the check is: every stored chunk
    appears in the re-chunked set. Any stored chunk NOT re-producible is
    a mismatch (chunker regression or source text drift without version
    bump).
    """
    R.section(f"13. Determinism (re-chunk {sample} cases)")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from indexing_pipeline import chunker
        from indexing_pipeline.filters import is_noise_chunk
    except Exception as exc:
        R.check("chunker importable", f"no: {exc}", ok=False)
        return {}

    import pyarrow.parquet as pq

    candidates: list[int] = []
    for p in sorted(chunked_dir.glob("*.jsonl")):
        if p.name.startswith("_") or not p.stem.isdigit():
            continue
        candidates.append(int(p.stem))
        if len(candidates) >= sample:
            break
    if not candidates:
        R.check("candidates", 0)
        return {}

    data_dir = chunked_dir.parent
    corpus_path = data_dir / "raw" / "corpus" / "opinions.parquet"
    if not corpus_path.is_file():
        R.check("corpus path", str(corpus_path), ok=False, note="cannot determinism-check")
        return {}

    tokenize_fn = lambda t: tokenize(t)  # noqa: E731
    checked = mismatches = 0
    for batch in pq.ParquetFile(corpus_path).iter_batches(batch_size=200):
        for row in batch.to_pylist():
            if int(row["cl_opinion_id"]) not in candidates:
                continue
            case_id = int(row["cl_opinion_id"])
            try:
                recomputed = chunker.chunk_opinion(
                    row, tokenize=tokenize_fn, is_noise=is_noise_chunk
                )
            except Exception as exc:
                R.finding(f"case {case_id}: re-chunk failed: {exc}", critical=True)
                mismatches += 1
                continue
            recomputed_ids = {c.chunk_id for c in recomputed}
            stored = [
                json.loads(line)
                for line in (chunked_dir / f"{case_id}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            stored_ids = {r["chunk_id"] for r in stored}
            missing = stored_ids - recomputed_ids
            if missing:
                R.finding(
                    f"case {case_id}: {len(missing)} stored chunk_ids not reproducible",
                    critical=True,
                )
                mismatches += 1
            else:
                checked += 1

    R.check("checked", checked)
    R.check("mismatches", mismatches, ok=(mismatches == 0))
    return {"checked": checked, "mismatches": mismatches}


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--chunked-dir", type=Path, default=Path("/workspace/tmp/data/chunked"))
    ap.add_argument("--tokenizer-path", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--determinism-sample",
        type=int,
        default=0,
        help="re-chunk N cases and compare (default 0 = skip; needs corpus access)",
    )
    args = ap.parse_args()

    R.json_mode = bool(args.json)
    if not args.chunked_dir.is_dir():
        print(f"error: {args.chunked_dir} does not exist", file=sys.stderr)
        return 3

    if not R.json_mode:
        print(f"inspecting: {args.chunked_dir}")

    tok = load_tokenizer(args.tokenizer_path)
    if tok is None:
        print("error: no tokenizer available", file=sys.stderr)
        return 3

    tokenize = lambda t: len(tok.encode(t, add_special_tokens=True, truncation=False))  # noqa: E731

    check_inventory(args.chunked_dir)
    R.end()
    check_inventory_chunks(args.chunked_dir)
    R.end()
    check_versions(args.chunked_dir)
    R.end()
    check_token_distribution(args.chunked_dir, tokenize)
    R.end()
    check_breadcrumb(args.chunked_dir)
    R.end()
    check_parent_child(args.chunked_dir)
    R.end()
    check_sections_consistent(args.chunked_dir)
    R.end()
    check_boundary_integrity(args.chunked_dir)
    R.end()
    check_citation_integrity(args.chunked_dir)
    R.end()
    check_duplicates(args.chunked_dir)
    R.end()
    check_dedup_report(args.chunked_dir)
    R.end()
    show_samples(args.chunked_dir, tokenize)
    R.end()

    if args.determinism_sample > 0:
        check_determinism(args.chunked_dir, tok, args.determinism_sample)
        R.end()

    if R.json_mode:
        print(json.dumps(R.dump(), indent=2, default=str))
    else:
        print(f"\n{'=' * 72}")
        if R.critical:
            print(f"CRITICAL: {len(R.critical)} finding(s)")
            for c in R.critical:
                print(f"  - {c}")
            print("\nDo NOT run `embed` until resolved.")
        elif R.warnings:
            print(f"OK: no critical findings, {len(R.warnings)} warning(s)")
            for w in R.warnings:
                print(f"  - {w}")
        else:
            print("CLEAN: no critical findings, no warnings. Ready to embed.")
        print("=" * 72)

    return 2 if R.critical else (1 if R.warnings else 0)


if __name__ == "__main__":
    sys.exit(main())
