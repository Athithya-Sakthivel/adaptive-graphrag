"""Immutable retriever configuration.

Every tuning knob previously hardcoded across router.py, dense.py,
graph.py, exact.py, and fusion.py lives here. Each field can be
overridden from the environment (``RETRIEVER_*`` env vars) or via
``RetrieverConfig.with_overrides()`` for eval-time sweeps without
touching source files.

The config is a leaf: it imports nothing from the rest of retriever,
so importing it never pulls in a DB connection, embedder, or router.

Design notes
------------

Defaults match the current on-disk behavior of router.py except where
marked (hardening) or (audit-fix). Existing tests that call
``route(conn, embedder, query, filters=..., top_k=...)`` continue to
work without modification — route() uses a module-level
``default_config()`` singleton when no config is passed.

Every field is validated at construction time. There is no separate
"validated()" call to remember — ``RetrieverConfig(...)`` raises on an
invalid value, and ``replace()`` re-validates, so it is impossible to
build an invalid config.

Field groups
------------

* Per-arm limits: how many rows each arm returns before fusion.
* RRF k constants: per-arm, per-query-class rank smoothing.
* Graph retrieval: per-class direction, cap_per_hop, depth.
* Graph ranking: which signal orders the graph arm.
* HNSW: pgvector iterative scan and scan-budget controls.
* Resolution: pg_trgm fallback for case-name matching.
* Case diversity: per-class max chunks per case (fusion post-filter).
* Feature flags: 2-hop ablation gate.

Typed helper returns
--------------------

``application_pass2_kwargs()`` returns ``ApplicationPass2Kwargs`` — a
``TypedDict`` rather than ``dict[str, float | int]``. This lets
callers spread its result with ``**`` into ``rrf_pass2`` while
preserving pyright's per-key type checking: a plain ``dict[str,
float | int]`` return would force pyright to assume every value could
land in every parameter, including ``config: RetrieverConfig | None``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Literal, TypedDict

# ── Constrained literal types ──────────────────────────────────────────

GraphDirection = Literal["incoming", "outgoing", "both"]
GraphRankBy = Literal["in_graph_degree", "citation_count", "case_id"]
IterativeScanMode = Literal["off", "relaxed_order", "strict_order"]
QueryClassName = Literal["HOLDING", "LOOKUP", "VALIDITY", "APPLICATION"]


# ── Typed return for pass-2 kwargs ─────────────────────────────────────


class ApplicationPass2Kwargs(TypedDict):
    """Keyword arguments for ``fusion.rrf_pass2`` on APPLICATION queries.

    Shape is ``weight_seeds: float, weight_graph: float, k_seeds: int,
    k_graph: int``. A ``TypedDict`` preserves this per-key shape when
    the caller spreads the result with ``**`` — a plain ``dict[str,
    float | int]`` would lose the key-to-type mapping.
    """

    weight_seeds: float
    weight_graph: float
    k_seeds: int
    k_graph: int


# ── Environment helpers ────────────────────────────────────────────────

_ENV_PREFIX = "RETRIEVER_"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX + name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX + name} must be a float, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_ENV_PREFIX + name} must be a boolean (1/0/true/false/yes/no/on/off), got {raw!r}"
    )


def _env_choice(name: str, default: str, choices: frozenset[str]) -> str:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip()
    if not raw:
        return default
    if raw not in choices:
        raise ValueError(f"{_ENV_PREFIX + name} must be one of {sorted(choices)}, got {raw!r}")
    return raw


# ── RetrieverConfig ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    """Immutable retriever configuration.

    Field defaults match the current on-disk behavior except where
    marked. Validation runs in ``__post_init__`` on every construction
    (including ``replace()``), so an invalid config cannot be built.
    """

    # ── Per-arm result limits ───────────────────────────────────────
    #
    # How many candidates each arm returns before RRF fusion. Higher
    # values give more diversity reserve for case_dedup but cost more
    # memory and SQL work.

    dense_limit: int = 50
    sparse_limit: int = 50
    graph_limit: int = 50  # LOOKUP/VALIDITY graph arm; APPLICATION
    # uses application_graph_limit below

    # Fusion requests rrf_output_multiplier * top_k from
    # rrf_pass1/rrf_pass2, then truncates to top_k. 3 gives case_dedup
    # a 30-candidate pool for a top_k=10 query.
    rrf_output_multiplier: int = 3

    # Distinct case IDs fed into graph traversal.
    lookup_seed_limit: int = 5
    application_seed_limit: int = 20

    # VALIDITY dense-over-citers arm result count.
    validity_citers_limit: int = 30

    # ── RRF k constants (per-arm, per-class) ───────────────────────
    #
    # Formula: RRF(d) = sum_arms weight_arm / (k_arm + rank_arm(d)).
    # Lower k concentrates the arm's top ranks.
    #
    # These are the "literature default" for the current corpus. The
    # T2-RAGBench result (k=10 beat k=60) and Qdrant's shipped default
    # (k=2) show the optimum is corpus-specific. Sweep via env vars
    # when eval says otherwise.

    dense_k: int = 60
    sparse_k_holding: int = 60  # HOLDING: both arms approximate
    sparse_k_lookup: int = 15  # LOOKUP: sparse is exact for citations
    sparse_k_validity: int = 20  # VALIDITY: less exact, mid-weight
    sparse_k_application: int = 20  # APPLICATION: pass-1 seeds
    exact_k: int = 15  # Exact matches are deterministic
    graph_k: int = 10  # Direct CITES edges concentrate at top
    dense_citers_k: int = 60  # VALIDITY dense-over-citers

    # Application pass-2 fusion (seeds + graph candidates)
    seeds_pass2_k: int = 60
    graph_pass2_k: int = 10
    weight_seeds_pass2: float = 0.85
    weight_graph_pass2: float = 0.15

    # ── Graph retrieval (per-class) ────────────────────────────────

    lookup_graph_direction: GraphDirection = "incoming"
    lookup_graph_cap_per_hop: int = 50

    validity_graph_direction: GraphDirection = "incoming"
    validity_graph_cap_per_hop: int = 50

    # (audit-fix) "outgoing" -> "incoming".
    # "how has X been applied" semantically wants cases citing X, not
    # cases X cites. The audit flagged outgoing direction as P0.
    application_graph_direction: GraphDirection = "incoming"
    application_graph_cap_per_hop: int = 30
    # APPLICATION returns up to 20 seeds × 30-hop-1 cap = 600 before
    # dedup; cap the final list at 200.
    application_graph_limit: int = 200

    # ── Graph ranking ──────────────────────────────────────────────
    #
    # in_graph_degree: rank by COUNT(case_edges WHERE source_id = X).
    #   Cheap, in-graph, reflects how often the case is cited WITHIN
    #   the sampled corpus. Use this for the graph arm.
    # citation_count: rank by CourtListener's global count.
    #   Not in-graph — a case with 5000 global citations but 0
    #   in-graph citers ranks above a real hub. Avoid.
    # case_id: numeric ordering. Deterministic, semantically arbitrary.
    #   Included for ablation only.
    graph_rank_by: GraphRankBy = "in_graph_degree"

    # ── HNSW (dense arm) ───────────────────────────────────────────
    #
    # (hardening) "strict_order" -> "relaxed_order".
    # Benchmark: strict_order loses at every selectivity, up to 100k
    # rescored rows / 742ms at 8 clients. relaxed_order gives better
    # recall. Requires materialized-CTE re-sort in dense.py.
    iterative_scan_mode: IterativeScanMode = "relaxed_order"

    ef_search: int = 100
    # 2× pgvector default (20,000). At 1% selectivity, the default
    # ceiling caps recall around 0.76; 40,000 buys ~0.92 without
    # partial indexes. Only effective when iterative_scan_mode != off.
    max_scan_tuples: int = 40_000
    scan_mem_multiplier: float = 1.0

    # ── Case-name resolution (exact arm) ───────────────────────────
    #
    # Exact match first. On no rows, fall back to pg_trgm similarity
    # via the % operator (GIN index). 0.60 is the practical floor:
    # 0.30 (pg_trgm default) matches "Smith v. Jones" against
    # "Smith v. Johnson"; 0.85 (my first proposal) rejects legitimate
    # comma-truncated names like "Andrews v. Merritt Oil".
    trigram_similarity_threshold: float = 0.60
    trigram_limit: int = 5

    # ── Case diversity (fusion post-filter) ────────────────────────
    #
    # Without this, a popular case can occupy all top-K slots,
    # collapsing R@10 even when retrieval is correct. Per-class
    # max_per_case lets HOLDING (1 = best chunk per case) and
    # VALIDITY (2 = best + one more for context) coexist.
    case_diversity_enabled: bool = True
    case_max_holding: int = 1
    case_max_lookup: int = 2
    case_max_validity: int = 2
    case_max_application: int = 2

    # ── Feature flags ──────────────────────────────────────────────
    #
    # 2-hop is disabled until the ablation shows ≥5 points R@10 gain
    # over 1-hop on multi-hop queries. Do not enable preemptively.
    application_2hop_enabled: bool = False

    # ── Derived properties ─────────────────────────────────────────

    @property
    def application_graph_depth(self) -> Literal[1, 2]:
        """2-hop when the ablation gate has been cleared, else 1-hop."""
        return 2 if self.application_2hop_enabled else 1

    # ── Validation ────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # ---- positive integers ----
        positive_ints = (
            "dense_limit",
            "sparse_limit",
            "graph_limit",
            "application_graph_limit",
            "rrf_output_multiplier",
            "lookup_seed_limit",
            "application_seed_limit",
            "validity_citers_limit",
            "trigram_limit",
            "dense_k",
            "sparse_k_holding",
            "sparse_k_lookup",
            "sparse_k_validity",
            "sparse_k_application",
            "exact_k",
            "graph_k",
            "dense_citers_k",
            "seeds_pass2_k",
            "graph_pass2_k",
            "lookup_graph_cap_per_hop",
            "validity_graph_cap_per_hop",
            "application_graph_cap_per_hop",
            "ef_search",
            "max_scan_tuples",
            "case_max_holding",
            "case_max_lookup",
            "case_max_validity",
            "case_max_application",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

        # ---- HNSW specific bounds ----
        if self.ef_search > 1000:
            raise ValueError(f"ef_search must be <= 1000 (pgvector limit), got {self.ef_search}")
        if self.scan_mem_multiplier <= 0:
            raise ValueError(f"scan_mem_multiplier must be > 0, got {self.scan_mem_multiplier}")
        if self.iterative_scan_mode not in {
            "off",
            "relaxed_order",
            "strict_order",
        }:
            raise ValueError(
                f"iterative_scan_mode must be off/relaxed_order/"
                f"strict_order, got {self.iterative_scan_mode!r}"
            )

        # ---- RRF weights ----
        for name in ("weight_seeds_pass2", "weight_graph_pass2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.weight_seeds_pass2 + self.weight_graph_pass2 <= 0:
            raise ValueError(
                "at least one pass-2 fusion weight must be positive "
                f"(got seeds={self.weight_seeds_pass2}, "
                f"graph={self.weight_graph_pass2})"
            )

        # ---- trigram threshold ----
        if not 0.0 < self.trigram_similarity_threshold <= 1.0:
            raise ValueError(
                f"trigram_similarity_threshold must be in (0, 1], "
                f"got {self.trigram_similarity_threshold}"
            )

        # ---- graph direction ----
        for name in (
            "lookup_graph_direction",
            "validity_graph_direction",
            "application_graph_direction",
        ):
            value = getattr(self, name)
            if value not in {"incoming", "outgoing", "both"}:
                raise ValueError(f"{name} must be incoming/outgoing/both, got {value!r}")

        # ---- graph rank ----
        if self.graph_rank_by not in {
            "in_graph_degree",
            "citation_count",
            "case_id",
        }:
            raise ValueError(
                f"graph_rank_by must be in_graph_degree/citation_count/"
                f"case_id, got {self.graph_rank_by!r}"
            )

        # ---- bools ----
        for name in ("case_diversity_enabled", "application_2hop_enabled"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool, got {value!r}")

    # ── Construction from environment ─────────────────────────────

    @classmethod
    def from_env(cls) -> RetrieverConfig:
        """Build a config from RETRIEVER_* env vars, defaulting to dataclass fields.

        Unset env vars fall through to the field default. Set env vars
        override. Invalid values raise ValueError with the env var name.
        """
        defaults = cls()
        return defaults.with_overrides(
            dense_limit=_env_int("DENSE_LIMIT", defaults.dense_limit),
            sparse_limit=_env_int("SPARSE_LIMIT", defaults.sparse_limit),
            graph_limit=_env_int("GRAPH_LIMIT", defaults.graph_limit),
            application_graph_limit=_env_int(
                "APPLICATION_GRAPH_LIMIT", defaults.application_graph_limit
            ),
            rrf_output_multiplier=_env_int("RRF_OUTPUT_MULTIPLIER", defaults.rrf_output_multiplier),
            lookup_seed_limit=_env_int("LOOKUP_SEED_LIMIT", defaults.lookup_seed_limit),
            application_seed_limit=_env_int(
                "APPLICATION_SEED_LIMIT", defaults.application_seed_limit
            ),
            validity_citers_limit=_env_int("VALIDITY_CITERS_LIMIT", defaults.validity_citers_limit),
            dense_k=_env_int("DENSE_K", defaults.dense_k),
            sparse_k_holding=_env_int("SPARSE_K_HOLDING", defaults.sparse_k_holding),
            sparse_k_lookup=_env_int("SPARSE_K_LOOKUP", defaults.sparse_k_lookup),
            sparse_k_validity=_env_int("SPARSE_K_VALIDITY", defaults.sparse_k_validity),
            sparse_k_application=_env_int("SPARSE_K_APPLICATION", defaults.sparse_k_application),
            exact_k=_env_int("EXACT_K", defaults.exact_k),
            graph_k=_env_int("GRAPH_K", defaults.graph_k),
            dense_citers_k=_env_int("DENSE_CITERS_K", defaults.dense_citers_k),
            seeds_pass2_k=_env_int("SEEDS_PASS2_K", defaults.seeds_pass2_k),
            graph_pass2_k=_env_int("GRAPH_PASS2_K", defaults.graph_pass2_k),
            weight_seeds_pass2=_env_float("WEIGHT_SEEDS_PASS2", defaults.weight_seeds_pass2),
            weight_graph_pass2=_env_float("WEIGHT_GRAPH_PASS2", defaults.weight_graph_pass2),
            lookup_graph_direction=_env_choice(
                "LOOKUP_GRAPH_DIRECTION",
                defaults.lookup_graph_direction,
                frozenset({"incoming", "outgoing", "both"}),
            ),
            lookup_graph_cap_per_hop=_env_int(
                "LOOKUP_GRAPH_CAP_PER_HOP",
                defaults.lookup_graph_cap_per_hop,
            ),
            validity_graph_direction=_env_choice(
                "VALIDITY_GRAPH_DIRECTION",
                defaults.validity_graph_direction,
                frozenset({"incoming", "outgoing", "both"}),
            ),
            validity_graph_cap_per_hop=_env_int(
                "VALIDITY_GRAPH_CAP_PER_HOP",
                defaults.validity_graph_cap_per_hop,
            ),
            application_graph_direction=_env_choice(
                "APPLICATION_GRAPH_DIRECTION",
                defaults.application_graph_direction,
                frozenset({"incoming", "outgoing", "both"}),
            ),
            application_graph_cap_per_hop=_env_int(
                "APPLICATION_GRAPH_CAP_PER_HOP",
                defaults.application_graph_cap_per_hop,
            ),
            graph_rank_by=_env_choice(
                "GRAPH_RANK_BY",
                defaults.graph_rank_by,
                frozenset({"in_graph_degree", "citation_count", "case_id"}),
            ),
            iterative_scan_mode=_env_choice(
                "ITERATIVE_SCAN_MODE",
                defaults.iterative_scan_mode,
                frozenset({"off", "relaxed_order", "strict_order"}),
            ),
            ef_search=_env_int("EF_SEARCH", defaults.ef_search),
            max_scan_tuples=_env_int("MAX_SCAN_TUPLES", defaults.max_scan_tuples),
            scan_mem_multiplier=_env_float("SCAN_MEM_MULTIPLIER", defaults.scan_mem_multiplier),
            trigram_similarity_threshold=_env_float(
                "TRIGRAM_SIMILARITY_THRESHOLD",
                defaults.trigram_similarity_threshold,
            ),
            trigram_limit=_env_int("TRIGRAM_LIMIT", defaults.trigram_limit),
            case_diversity_enabled=_env_bool(
                "CASE_DIVERSITY_ENABLED", defaults.case_diversity_enabled
            ),
            case_max_holding=_env_int("CASE_MAX_HOLDING", defaults.case_max_holding),
            case_max_lookup=_env_int("CASE_MAX_LOOKUP", defaults.case_max_lookup),
            case_max_validity=_env_int("CASE_MAX_VALIDITY", defaults.case_max_validity),
            case_max_application=_env_int("CASE_MAX_APPLICATION", defaults.case_max_application),
            application_2hop_enabled=_env_bool(
                "APPLICATION_2HOP_ENABLED", defaults.application_2hop_enabled
            ),
        )

    # ── Overrides for CLI and eval ────────────────────────────────

    def with_overrides(self, **kwargs: object) -> RetrieverConfig:
        """Return a new config with the given fields replaced.

        Only keys that correspond to actual fields are accepted. None
        values are ignored, so argparse defaults of None work without
        per-call special-casing.

        Raises ValueError on unknown keys and on any field that would
        fail validation after replacement.
        """
        allowed = {f.name for f in fields(self)}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        applied = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **applied)  # type: ignore[arg-type]

    # ── Helper accessors consumed by router.py ────────────────────

    def holding_rrf_k(self) -> dict[str, int]:
        """k dict for rrf_pass1 on HOLDING queries."""
        return {
            "dense": self.dense_k,
            "sparse": self.sparse_k_holding,
            "exact": self.exact_k,
        }

    def lookup_rrf_k(self) -> dict[str, int]:
        """k dict for rrf_pass1 on LOOKUP queries."""
        return {
            "sparse": self.sparse_k_lookup,
            "exact": self.exact_k,
            "graph_1hop": self.graph_k,
        }

    def validity_rrf_k(self) -> dict[str, int]:
        """k dict for rrf_pass1 on VALIDITY queries."""
        return {
            "sparse": self.sparse_k_validity,
            "exact": self.exact_k,
            "graph_1hop": self.graph_k,
            "dense_citers": self.dense_citers_k,
        }

    def application_pass1_rrf_k(self) -> dict[str, int]:
        """k dict for rrf_pass1 (seed selection) on APPLICATION queries."""
        return {
            "dense": self.dense_k,
            "sparse": self.sparse_k_application,
        }

    def application_pass2_kwargs(self) -> ApplicationPass2Kwargs:
        """kwargs for rrf_pass2 (seeds + graph fusion) on APPLICATION queries.

        Return type is a ``TypedDict`` rather than ``dict[str,
        float | int]`` so that callers can spread this result with
        ``**`` into ``rrf_pass2`` and have pyright check each keyword
        parameter independently. A plain dict would lose the
        key-to-type mapping and force pyright to assume every value
        could flow into every parameter.
        """
        return {
            "weight_seeds": self.weight_seeds_pass2,
            "weight_graph": self.weight_graph_pass2,
            "k_seeds": self.seeds_pass2_k,
            "k_graph": self.graph_pass2_k,
        }

    def case_max_per_class(self, class_name: str) -> int:
        """max chunks per case for a given query class.

        ``class_name`` is 'HOLDING', 'LOOKUP', 'VALIDITY', or
        'APPLICATION' (case-insensitive). Raises ValueError on an
        unknown class.
        """
        attr = f"case_max_{class_name.lower()}"
        value = getattr(self, attr, None)
        # bool is a subclass of int; explicitly reject it.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"unknown query class: {class_name!r}; "
                f"expected one of HOLDING, LOOKUP, VALIDITY, APPLICATION"
            )
        return value


# ── Module-level default singleton ────────────────────────────────────
#
# Built lazily so importing the config module never touches env vars at
# import time. ``default_config()`` is what route() uses when the
# caller passes config=None.

_DEFAULT: RetrieverConfig | None = None


def default_config() -> RetrieverConfig:
    """Return the process-wide default config, built from env on first call."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RetrieverConfig.from_env()
    return _DEFAULT


def reset_default_config() -> None:
    """Test helper — clear the singleton so from_env() runs again."""
    global _DEFAULT
    _DEFAULT = None


__all__ = [
    "ApplicationPass2Kwargs",
    "GraphDirection",
    "GraphRankBy",
    "IterativeScanMode",
    "QueryClassName",
    "RetrieverConfig",
    "default_config",
    "reset_default_config",
]
