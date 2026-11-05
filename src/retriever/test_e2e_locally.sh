#!/usr/bin/env bash
#
# End-to-end retriever acceptance test — hardened.
#
# Runs all 7 acceptance tests in a SINGLE Python process so the
# QueryEmbedder is loaded once and reused across HOLDING/VALIDITY/
# APPLICATION/DOCTRINE. LOOKUP and EXPLICIT OPINION ID never touch it.
#
# Every test now asserts 10+ invariants:
#   - class matches expected
#   - required arms present in arms_used
#   - every arm in arms_used has an arm_diagnostics entry with hits>0
#     (except graph_1hop / graph_2hop, which may legitimately have 0)
#   - case diversity: distinct_cases == hit_count in top-K (case_dedup worked)
#   - latency under per-test bound
#   - no arm carries an error in its diagnostic
#   - per-test extras (opinion-ID resolution, trigram resolution, APPLICATION
#     semantic direction, doctrine-graph-skip, etc.)
#
# Usage (from anywhere):
#   bash src/retriever/test_e2e_locally.sh             # 7 tests, ~30s
#   bash src/retriever/test_e2e_locally.sh --fast      # 3 tests, ~2s, no embedder
#   bash src/retriever/test_e2e_locally.sh --top-k 5
#   bash src/retriever/test_e2e_locally.sh --verbose   # full tracebacks on fail
#
# Exit code 0 iff every executed test passes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

[[ -d "${SRC_DIR}/.venv" ]] || {
    echo "error: .venv not found in ${SRC_DIR}" >&2
    exit 2
}

# shellcheck source=/dev/null
source "${SRC_DIR}/.venv/bin/activate"

: "${DATABASE_URL:=postgresql://legalrag:e6a1c2e5bf8d@127.0.0.1:5432/legalrag}"
export DATABASE_URL

MODEL_DIR="${SRC_DIR}/indexing_pipeline/embedders/dinghy-law-0.6b-v1-openvino-fp16"
[[ -d "${MODEL_DIR}" ]] || {
    echo "error: model directory not found: ${MODEL_DIR}" >&2
    exit 2
}
export RETRIEVER_MODEL_DIR="${MODEL_DIR}"

cd "${SRC_DIR}"

TMP_PY="$(mktemp "${TMPDIR:-/tmp}/e2e_hardened.XXXXXX.py")"
cleanup() { rm -f "${TMP_PY}"; }
trap cleanup EXIT INT TERM

cat > "${TMP_PY}" <<'PYEOF'
"""Hardened end-to-end acceptance test for the retriever."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psycopg

from retriever import Filters, QueryClass, exact, route
from retriever.classify import extract_case_reference
from retriever.config import RetrieverConfig
from retriever.fusion import Hit


# ── Embedder wrapper ──────────────────────────────────────────────────


class LazyEmbedder:
    """Load QueryEmbedder on first use; track cold-start and per-query latency."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._instance: Any = None
        self.cold_start_s: float | None = None
        self.query_count = 0
        self.query_total_s = 0.0

    def _get(self) -> Any:
        if self._instance is not None:
            return self._instance
        from retriever.embedder import QueryEmbedder

        print(f"  [loading QueryEmbedder from {self.model_dir.name}...]", flush=True)
        started = time.monotonic()
        self._instance = QueryEmbedder(model_dir=self.model_dir)
        self.cold_start_s = time.monotonic() - started
        print(f"  [cold start: {self.cold_start_s:.2f}s — reused for the rest of the run]", flush=True)
        return self._instance

    def embed_query(self, text: str) -> Any:
        t0 = time.monotonic()
        try:
            return self._get().embed_query(text)
        finally:
            self.query_total_s += time.monotonic() - t0
            self.query_count += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)

    def close(self) -> None:
        if self._instance is None:
            return
        instance = self._instance
        self._instance = None
        try:
            instance.close()
        except Exception:
            pass


# ── Data-test target discovery ────────────────────────────────────────


def find_target_case(conn) -> tuple[int, str]:
    """Find a case with citers whose name round-trips cleanly.

    Constraints:
        * no comma / semicolon / colon in the name (classifier's
          case-reference pattern truncates at those characters)
        * extract_case_reference(f"which cases cite {name}") == name
        * exact.resolve_case_ids(name) contains this case_id
          (exact match, not trigram — target must be a clean resolution)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.case_id, c.case_name,
                   COUNT(DISTINCT e.source_id) AS citers
            FROM cases c
            JOIN case_edges e ON e.target_id = c.case_id
            WHERE c.case_name IS NOT NULL AND c.case_name <> ''
            GROUP BY c.case_id, c.case_name
            ORDER BY citers DESC, c.case_id ASC
            LIMIT 50
            """
        )
        candidates = [(int(r[0]), str(r[1])) for r in cur.fetchall()]

    for case_id, case_name in candidates:
        if any(ch in case_name for ch in ",;:"):
            continue
        extracted = extract_case_reference(f"which cases cite {case_name}")
        if not extracted:
            continue
        if extracted.strip().lower() != case_name.strip().lower():
            continue
        if case_id not in exact.resolve_case_ids(conn, extracted, limit=10):
            continue
        return case_id, case_name

    raise SystemExit(
        "no clean target case found: corpus needs a case with citers "
        "and a name without ,;: that round-trips through "
        "extract_case_reference + exact.resolve_case_ids"
    )


def find_trigram_case(conn) -> tuple[int, str, str]:
    """Find a case with citers whose name has a comma/semicolon/colon.

    Returns (case_id, full_name, truncated_name). The truncated name
    (text before the first separator) must NOT resolve via exact match
    but MUST resolve via trigram fallback.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.case_id, c.case_name,
                   COUNT(DISTINCT e.source_id) AS citers
            FROM cases c
            JOIN case_edges e ON e.target_id = c.case_id
            WHERE c.case_name IS NOT NULL AND c.case_name <> ''
              AND (c.case_name LIKE '%,%' OR c.case_name LIKE '%;%' OR c.case_name LIKE '%:%')
            GROUP BY c.case_id, c.case_name
            ORDER BY citers DESC, c.case_id ASC
            LIMIT 20
            """
        )
        candidates = [(int(r[0]), str(r[1])) for r in cur.fetchall()]

    for case_id, case_name in candidates:
        truncated = (
            case_name.replace(";", ",").replace(":", ",").split(",")[0].strip()
        )
        if not truncated or truncated.strip().lower() == case_name.strip().lower():
            continue
        extracted = extract_case_reference(f"which cases cite {truncated}")
        if not extracted:
            continue
        resolved = exact.resolve_case_ids(conn, extracted, limit=10)
        if case_id not in resolved:
            continue
        return case_id, case_name, truncated

    raise SystemExit(
        "no trigram test case found: corpus needs a case with citers "
        "whose name contains ,/;/: and whose truncated form resolves "
        "via pg_trgm"
    )


def citers_of(conn, case_id: int) -> set[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_id FROM case_edges WHERE target_id = %s",
            (case_id,),
        )
        return {int(r[0]) for r in cur.fetchall()}


# ── Test definitions and runner ──────────────────────────────────────


# Aliases between fusion-arm names (in arms_used) and diagnostic names.
# case_resolution contributes hits via resolved_case_parents; the graph
# arms also log a graph_parent_expansion step.
_ARM_TO_DIAG: dict[str, tuple[str, ...]] = {
    "case_resolution": ("resolved_case_parents",),
    "graph_1hop": ("graph_1hop", "graph_parent_expansion"),
    "graph_2hop": ("graph_2hop", "graph_parent_expansion"),
}


@dataclass
class TestCase:
    name: str
    query: str
    expected_class: QueryClass
    required_arms: frozenset[str] = frozenset()
    forbidden_arms: frozenset[str] = frozenset()
    max_latency_ms: float = 5000.0
    needs_embedder: bool = False
    target_case_id: int | None = None
    extra: Callable[[Any, psycopg.Connection, list[str]], None] | None = None


@dataclass
class TestResult:
    name: str
    passed: bool = True
    skipped: bool = False
    skip_reason: str = ""
    actual_class: QueryClass | None = None
    actual_arms: list[str] = field(default_factory=list)
    hit_count: int = 0
    distinct_cases: int = 0
    latency_ms: float = 0.0
    top_hit: Hit | None = None
    arm_diagnostics: list = field(default_factory=list)
    fail_reasons: list[str] = field(default_factory=list)


def _check_arm_contributions(result: Any, failures: list[str]) -> None:
    """Verify every arm in arms_used has a corresponding diag with hits>0.

    graph_1hop and graph_2hop are exempt (target may legitimately have
    zero citers). Errors in any diagnostic always fail.
    """
    for arm in result.arms_used:
        diag_names = _ARM_TO_DIAG.get(arm, (arm,))
        entries = [d for d in result.arm_diagnostics if d.name in diag_names]
        if not entries:
            failures.append(f"arm {arm!r} has no diagnostic entry")
            continue
        for d in entries:
            if d.error:
                failures.append(f"arm {d.name!r} raised: {d.error}")
        if arm not in ("graph_1hop", "graph_2hop"):
            total_hits = sum(d.hits for d in entries)
            if total_hits == 0:
                failures.append(f"arm {arm!r} contributed 0 hits")


def run_test(
    conn: psycopg.Connection,
    embedder: LazyEmbedder | None,
    tc: TestCase,
    top_k: int,
    verbose: bool,
    config: RetrieverConfig,
) -> TestResult:
    print()
    print(f"─── TEST: {tc.name} " + "─" * max(0, 55 - len(tc.name)))
    print(f"    query: {tc.query!r}")

    tr = TestResult(name=tc.name)

    if tc.needs_embedder and embedder is None:
        tr.skipped = True
        tr.skip_reason = "embedder unavailable in fast mode"
        print(f"    [SKIP] {tr.skip_reason}")
        return tr

    started = time.monotonic()
    response = None
    exception_text = None
    tb = None
    try:
        response = route(
            conn,
            embedder,
            tc.query,
            filters=Filters(),
            top_k=top_k,
            config=config,
        )
    except Exception as exc:
        exception_text = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
    tr.latency_ms = (time.monotonic() - started) * 1000

    if exception_text:
        tr.passed = False
        tr.fail_reasons.append(f"exception: {exception_text}")
        if verbose and tb:
            print(tb)
        print(f"    [FAIL] {tr.latency_ms:.0f}ms  → {exception_text}")
        return tr

    assert response is not None
    tr.actual_class = response.query_class
    tr.actual_arms = list(response.arms_used)
    tr.hit_count = len(response.hits)
    tr.distinct_cases = len({h.case_id for h in response.hits})
    tr.top_hit = response.hits[0] if response.hits else None
    tr.arm_diagnostics = list(response.arm_diagnostics)

    f = tr.fail_reasons

    # 1. class
    if response.query_class != tc.expected_class:
        f.append(
            f"class: expected {tc.expected_class.value}, "
            f"got {response.query_class.value}"
        )

    # 2. required arms
    missing = tc.required_arms - set(response.arms_used)
    if missing:
        f.append(f"missing required arms: {sorted(missing)}")

    # 3. forbidden arms
    present = tc.forbidden_arms & set(response.arms_used)
    if present:
        f.append(f"forbidden arms present: {sorted(present)}")

    # 4. hit count
    if tr.hit_count == 0:
        f.append("no hits returned")

    # 5. arm contribution + errors
    _check_arm_contributions(response, f)

    # 6. case diversity (dedup worked)
    if tr.hit_count >= 2 and tr.distinct_cases != tr.hit_count:
        f.append(
            f"case diversity: {tr.distinct_cases} distinct / "
            f"{tr.hit_count} hits (case_dedup should have collapsed)"
        )

    # 7. latency
    if tr.latency_ms > tc.max_latency_ms:
        f.append(
            f"latency: {tr.latency_ms:.0f}ms > "
            f"{tc.max_latency_ms:.0f}ms bound"
        )

    # 8. extra (per-test)
    if tc.extra is not None:
        tc.extra(response, conn, f)

    tr.passed = not f

    mark = "PASS" if tr.passed else "FAIL"
    print(
        f"    [{mark}] class={response.query_class.value} "
        f"arms={','.join(response.arms_used) or '—'} "
        f"hits={tr.hit_count} distinct={tr.distinct_cases} "
        f"{tr.latency_ms:.0f}ms"
    )
    for d in response.arm_diagnostics:
        suffix = f"  ERROR: {d.error}" if d.error else ""
        print(
            f"      arm={d.name:<24} attempted={d.attempted!s:<5} "
            f"hits={d.hits:<4} {d.latency_ms:>7.1f}ms{suffix}"
        )
    if tr.top_hit:
        flag = "P" if tr.top_hit.is_parent else "C"
        preview = (tr.top_hit.meta.get("preview") or "")[:70].replace("\n", " ")
        print(
            f"    top-1: [{flag}] {tr.top_hit.chunk_id} "
            f"case_id={tr.top_hit.case_id}"
        )
        if preview:
            print(f"           {preview}")
    for reason in tr.fail_reasons:
        print(f"    → {reason}")
    return tr


# ── Extra assertion factories ────────────────────────────────────────


def _extra_lookup_resolution(expected_case_id: int) -> Callable:
    """LOOKUP must attempt case_name_resolution with hits>0."""
    def check(response: Any, conn: psycopg.Connection, failures: list[str]) -> None:
        res_diag = next(
            (d for d in response.arm_diagnostics if d.name == "case_name_resolution"),
            None,
        )
        if res_diag is None:
            failures.append("case_name_resolution diagnostic missing")
        elif res_diag.hits == 0:
            failures.append(
                f"case_name_resolution returned 0 hits for target {expected_case_id}"
            )
        # graph_1hop diagnostic must show the target's citers were expanded
        graph_diag = next(
            (d for d in response.arm_diagnostics if d.name == "graph_1hop"),
            None,
        )
        if graph_diag is None:
            failures.append("graph_1hop diagnostic missing")
        elif graph_diag.hits == 0:
            failures.append(
                f"graph_1hop returned 0 citers for target {expected_case_id}"
            )
    return check


def _extra_opinion_id_resolution(expected_case_id: int) -> Callable:
    """EXPLICIT OPINION ID path must go through opinion_id_resolution."""
    def check(response: Any, conn: psycopg.Connection, failures: list[str]) -> None:
        res_diag = next(
            (d for d in response.arm_diagnostics if d.name == "opinion_id_resolution"),
            None,
        )
        if res_diag is None:
            failures.append("opinion_id_resolution diagnostic missing")
        elif res_diag.hits == 0:
            failures.append(
                f"opinion_id_resolution returned 0 hits for id {expected_case_id}"
            )
        graph_diag = next(
            (d for d in response.arm_diagnostics if d.name == "graph_1hop"),
            None,
        )
        if graph_diag is None or graph_diag.hits == 0:
            failures.append(
                f"graph_1hop returned 0 citers for opinion {expected_case_id}"
            )
    return check


def _extra_trigram_resolution(expected_case_id: int) -> Callable:
    """Trigram fallback must have resolved the truncated name."""
    def check(response: Any, conn: psycopg.Connection, failures: list[str]) -> None:
        res_diag = next(
            (d for d in response.arm_diagnostics if d.name == "case_name_resolution"),
            None,
        )
        if res_diag is None:
            failures.append(
                "case_name_resolution diagnostic missing (trigram path may not have run)"
            )
        elif res_diag.hits == 0:
            failures.append(
                "case_name_resolution returned 0 hits — trigram fallback failed"
            )
        graph_diag = next(
            (d for d in response.arm_diagnostics if d.name == "graph_1hop"),
            None,
        )
        if graph_diag is None or graph_diag.hits == 0:
            failures.append(
                f"graph_1hop returned 0 citers for trigram-resolved {expected_case_id}"
            )
    return check


def _extra_application_semantic(target_case_id: int) -> Callable:
    """APPLICATION must return citers of target, not the target itself."""
    def check(response: Any, conn: psycopg.Connection, failures: list[str]) -> None:
        # Audit fix: target must not appear in results
        returned_ids = {h.case_id for h in response.hits}
        if target_case_id in returned_ids:
            failures.append(
                f"APPLICATION returned target case {target_case_id} — "
                f"audit fix (filter target from pass-1) not applied"
            )
        # At least one hit must be a real citer of target
        citers = citers_of(conn, target_case_id)
        if not (citers & returned_ids):
            failures.append(
                f"APPLICATION returned no citers of {target_case_id} "
                f"(citers={len(citers)}, returned={sorted(returned_ids)})"
            )
        # graph_1hop must have contributed hits
        graph_diag = next(
            (d for d in response.arm_diagnostics if d.name == "graph_1hop"),
            None,
        )
        if graph_diag is None or graph_diag.hits == 0:
            failures.append(
                f"graph_1hop returned 0 citers of {target_case_id}"
            )
    return check


def _extra_doctrine_skips_graph() -> Callable:
    """DOCTRINE APPLICATION must skip graph entirely (no seed)."""
    def check(response: Any, conn: psycopg.Connection, failures: list[str]) -> None:
        for forbidden in ("graph_1hop", "graph_2hop"):
            if forbidden in response.arms_used:
                failures.append(
                    f"doctrine APPLICATION attempted {forbidden}; "
                    f"no seed should mean no graph"
                )
        if any("graph" in d.name for d in response.arm_diagnostics):
            failures.append(
                "doctrine APPLICATION has graph diagnostic entries; "
                "graph arm should never have been attempted"
            )
    return check


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    if args.help:
        print("Usage: test_e2e_locally.sh [--fast] [--top-k N] [--verbose]")
        return 0

    if args.top_k < 1:
        print("error: --top-k must be positive", file=sys.stderr)
        return 2

    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        print("error: DATABASE_URL not set", file=sys.stderr)
        return 2

    model_dir_str = os.environ.get("RETRIEVER_MODEL_DIR", "").strip()
    if not model_dir_str:
        print("error: RETRIEVER_MODEL_DIR not set", file=sys.stderr)
        return 2
    model_dir = Path(model_dir_str)

    config = RetrieverConfig.from_env()

    print("=" * 72)
    print("Retriever end-to-end acceptance test — HARDENED")
    print("=" * 72)
    print(f"  database:   {dsn.rsplit('@', 1)[-1]}")
    print(f"  model_dir:  {model_dir.name}")
    print(f"  top_k:      {args.top_k}")
    print(f"  fast mode:  {args.fast}")
    if not args.fast:
        print(
            "  note:       first HOLDING query absorbs QueryEmbedder "
            "cold-start (~3s); later tests reuse it"
        )

    try:
        conn = psycopg.connect(dsn, connect_timeout=10)
    except Exception as exc:
        print(f"error: cannot connect to database: {exc}", file=sys.stderr)
        return 2

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            row = cur.fetchone()
            chunks = int(row[0]) if row else 0
        if chunks == 0:
            print("error: chunks table is empty", file=sys.stderr)
            return 2
        print(f"  chunks:     {chunks}")

        # Discover test targets
        try:
            target_id, target_name = find_target_case(conn)
        except SystemExit as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"  target:     {target_name!r} (case_id={target_id})")

        trigram_id: int | None = None
        trigram_full: str = ""
        trigram_truncated: str = ""
        try:
            trigram_id, trigram_full, trigram_truncated = find_trigram_case(conn)
            print(
                f"  trigram:    {trigram_full!r} "
                f"(case_id={trigram_id}, truncated→{trigram_truncated!r})"
            )
        except SystemExit as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        embedder: LazyEmbedder | None = None if args.fast else LazyEmbedder(model_dir)
        results: list[TestResult] = []

        test_cases: list[tuple[TestCase, bool]] = [
            # (test_case, is_fast_mode_ok)
            (
                TestCase(
                    name="LOOKUP (case name → incoming graph)",
                    query=f"which cases cite {target_name}",
                    expected_class=QueryClass.LOOKUP,
                    required_arms=frozenset({"graph_1hop", "case_resolution"}),
                    max_latency_ms=500.0,
                    needs_embedder=False,
                    extra=_extra_lookup_resolution(target_id),
                ),
                True,
            ),
            (
                TestCase(
                    name="EXPLICIT OPINION ID",
                    query=f"Find cases citing opinion {target_id}",
                    expected_class=QueryClass.LOOKUP,
                    required_arms=frozenset({"graph_1hop"}),
                    max_latency_ms=500.0,
                    needs_embedder=False,
                    extra=_extra_opinion_id_resolution(target_id),
                ),
                True,
            ),
            (
                TestCase(
                    name="TRIGRAM FALLBACK (comma-truncated name)",
                    query=f"which cases cite {trigram_truncated}",
                    expected_class=QueryClass.LOOKUP,
                    required_arms=frozenset({"graph_1hop"}),
                    max_latency_ms=1000.0,
                    needs_embedder=False,
                    extra=_extra_trigram_resolution(trigram_id),
                ),
                True,
            ),
            (
                TestCase(
                    name="HOLDING (dense + sparse)",
                    query="what did the court hold on qualified immunity",
                    expected_class=QueryClass.HOLDING,
                    required_arms=frozenset({"dense", "sparse"}),
                    max_latency_ms=5000.0,
                    needs_embedder=True,
                ),
                False,
            ),
            (
                TestCase(
                    name="VALIDITY (graph → dense over citers)",
                    query=f"has {target_name} been overruled",
                    expected_class=QueryClass.VALIDITY,
                    required_arms=frozenset({"graph_1hop", "dense_citers"}),
                    max_latency_ms=5000.0,
                    needs_embedder=True,
                ),
                False,
            ),
            (
                TestCase(
                    name="APPLICATION (incoming graph from target)",
                    query=f"how has {target_name} been applied",
                    expected_class=QueryClass.APPLICATION,
                    required_arms=frozenset({"graph_1hop"}),
                    max_latency_ms=5000.0,
                    needs_embedder=True,
                    target_case_id=target_id,
                    extra=_extra_application_semantic(target_id),
                ),
                False,
            ),
            (
                TestCase(
                    name="DOCTRINE APPLICATION (no case ref → dense+sparse)",
                    query="how has qualified immunity been applied",
                    expected_class=QueryClass.APPLICATION,
                    required_arms=frozenset({"dense", "sparse"}),
                    forbidden_arms=frozenset({"graph_1hop", "graph_2hop"}),
                    max_latency_ms=5000.0,
                    needs_embedder=True,
                    extra=_extra_doctrine_skips_graph(),
                ),
                False,
            ),
        ]

        try:
            for tc, fast_ok in test_cases:
                if args.fast and not fast_ok:
                    r = TestResult(name=tc.name, skipped=True,
                                   skip_reason="fast mode")
                    results.append(r)
                    print()
                    print(f"─── TEST: {tc.name} " + "─" * max(0, 55 - len(tc.name)))
                    print(f"    [SKIP] fast mode")
                    continue
                # APPLICATION and DOCTRINE use smaller top-k to keep
                # the citer prominence check robust under pass-2 weights.
                k = args.top_k
                if tc.expected_class == QueryClass.APPLICATION:
                    k = min(args.top_k, 3)
                elif tc.expected_class == QueryClass.HOLDING:
                    k = min(args.top_k, 3)
                elif tc.expected_class == QueryClass.VALIDITY:
                    k = min(args.top_k, 3)
                results.append(
                    run_test(conn, embedder, tc, k, args.verbose, config)
                )
        finally:
            if embedder is not None:
                embedder.close()

        # ── Summary ─────────────────────────────────────────────────
        print()
        print("=" * 72)
        print("SUMMARY")
        print("=" * 72)
        passed = 0
        failed = 0
        skipped = 0
        for r in results:
            if r.skipped:
                skipped += 1
                print(f"  [SKIP] {r.name}  ({r.skip_reason})")
                continue
            mark = "PASS" if r.passed else "FAIL"
            if r.passed:
                passed += 1
            else:
                failed += 1
            cls = r.actual_class.value if r.actual_class else "—"
            print(
                f"  [{mark}] {r.name}"
                f"  class={cls} hits={r.hit_count} "
                f"distinct={r.distinct_cases} {r.latency_ms:.0f}ms"
            )
            for reason in r.fail_reasons:
                print(f"         → {reason}")

        if embedder is not None and embedder.cold_start_s is not None:
            avg_ms = (
                (embedder.query_total_s / embedder.query_count) * 1000
                if embedder.query_count > 0
                else 0.0
            )
            print()
            print(f"  QueryEmbedder cold start:  {embedder.cold_start_s:.2f}s")
            print(f"  embed_query calls:         {embedder.query_count}")
            print(f"  avg embed_query:           {avg_ms:.0f}ms")
            cold_ok = embedder.cold_start_s < 5.0
            avg_ok = avg_ms < 1500.0
            print(f"  cold-start budget <5s:     {'PASS' if cold_ok else 'FAIL'}")
            print(f"  avg-embed budget <1500ms:  {'PASS' if avg_ok else 'FAIL'}")
            if not cold_ok:
                failed += 1
            if not avg_ok:
                failed += 1

        total = passed + failed
        print()
        print(f"  {passed}/{total} passed ({skipped} skipped)")
        return 0 if failed == 0 else 1

    finally:
        conn.close()


if __name__ == "__main__":
    # Guarded so a forked child (e.g. OpenVINO worker) doesn't re-run
    # the suite on module import.
    sys.exit(main())
PYEOF

# PYTHONPATH must be set because the temp file lives in /tmp; without
# it `import retriever` fails even though the bash script already cd'd
# to SRC_DIR.
PYTHONPATH="${SRC_DIR}" python3 "${TMP_PY}" "$@"
