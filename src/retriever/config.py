"""Immutable retriever configuration with environment variable support.

Every tuning knob lives here. Fields can be overridden from environment
(RETRIEVER_* prefix) or via with_overrides() for eval-time sweeps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Literal

GraphDirection = Literal["incoming", "outgoing", "both"]
GraphRankBy = Literal["in_graph_degree", "citation_count", "case_id"]
IterativeScanMode = Literal["off", "relaxed_order", "strict_order"]
QueryClassName = Literal["HOLDING", "LOOKUP", "VALIDITY"]

_ENV_PREFIX = "RETRIEVER_"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX + name} must be int, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX + name} must be float, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_ENV_PREFIX + name} must be bool, got {raw!r}")


def _env_choice(name: str, default: str, choices: frozenset[str]) -> str:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip()
    if not raw:
        return default
    if raw not in choices:
        raise ValueError(f"{_ENV_PREFIX + name} must be one of {sorted(choices)}, got {raw!r}")
    return raw


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    """Immutable retriever configuration with validation."""

    dense_limit: int = 50
    sparse_limit: int = 50
    graph_limit: int = 50
    rrf_output_multiplier: int = 3
    lookup_seed_limit: int = 5
    validity_citers_limit: int = 30

    dense_k: int = 60
    sparse_k_holding: int = 60
    sparse_k_lookup: int = 15
    sparse_k_validity: int = 20
    exact_k: int = 15
    graph_k: int = 10
    dense_citers_k: int = 60

    lookup_graph_direction: GraphDirection = "incoming"
    lookup_graph_cap_per_hop: int = 50
    validity_graph_direction: GraphDirection = "incoming"
    validity_graph_cap_per_hop: int = 50
    graph_rank_by: GraphRankBy = "in_graph_degree"

    iterative_scan_mode: IterativeScanMode = "relaxed_order"
    ef_search: int = 100
    max_scan_tuples: int = 40_000
    scan_mem_multiplier: float = 1.0

    trigram_similarity_threshold: float = 0.60
    trigram_limit: int = 5

    case_diversity_enabled: bool = True
    case_max_holding: int = 1
    case_max_lookup: int = 2
    case_max_validity: int = 2

    def __post_init__(self) -> None:
        positive_ints = (
            "dense_limit",
            "sparse_limit",
            "graph_limit",
            "rrf_output_multiplier",
            "lookup_seed_limit",
            "validity_citers_limit",
            "dense_k",
            "sparse_k_holding",
            "sparse_k_lookup",
            "sparse_k_validity",
            "exact_k",
            "graph_k",
            "dense_citers_k",
            "lookup_graph_cap_per_hop",
            "validity_graph_cap_per_hop",
            "ef_search",
            "max_scan_tuples",
            "trigram_limit",
            "case_max_holding",
            "case_max_lookup",
            "case_max_validity",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive int, got {value!r}")

        if self.ef_search > 1000:
            raise ValueError(f"ef_search must be <= 1000, got {self.ef_search}")
        if self.scan_mem_multiplier <= 0:
            raise ValueError("scan_mem_multiplier must be > 0")
        if self.iterative_scan_mode not in {"off", "relaxed_order", "strict_order"}:
            raise ValueError(f"invalid iterative_scan_mode: {self.iterative_scan_mode!r}")
        if not 0.0 < self.trigram_similarity_threshold <= 1.0:
            raise ValueError("trigram_similarity_threshold must be in (0, 1]")

        for name in ("lookup_graph_direction", "validity_graph_direction"):
            if getattr(self, name) not in {"incoming", "outgoing", "both"}:
                raise ValueError(f"invalid {name}: {getattr(self, name)!r}")

        if self.graph_rank_by not in {"in_graph_degree", "citation_count", "case_id"}:
            raise ValueError(f"invalid graph_rank_by: {self.graph_rank_by!r}")

    @classmethod
    def from_env(cls) -> RetrieverConfig:
        defaults = cls()
        return defaults.with_overrides(
            dense_limit=_env_int("DENSE_LIMIT", defaults.dense_limit),
            sparse_limit=_env_int("SPARSE_LIMIT", defaults.sparse_limit),
            graph_limit=_env_int("GRAPH_LIMIT", defaults.graph_limit),
            rrf_output_multiplier=_env_int("RRF_OUTPUT_MULTIPLIER", defaults.rrf_output_multiplier),
            lookup_seed_limit=_env_int("LOOKUP_SEED_LIMIT", defaults.lookup_seed_limit),
            validity_citers_limit=_env_int("VALIDITY_CITERS_LIMIT", defaults.validity_citers_limit),
            dense_k=_env_int("DENSE_K", defaults.dense_k),
            sparse_k_holding=_env_int("SPARSE_K_HOLDING", defaults.sparse_k_holding),
            sparse_k_lookup=_env_int("SPARSE_K_LOOKUP", defaults.sparse_k_lookup),
            sparse_k_validity=_env_int("SPARSE_K_VALIDITY", defaults.sparse_k_validity),
            exact_k=_env_int("EXACT_K", defaults.exact_k),
            graph_k=_env_int("GRAPH_K", defaults.graph_k),
            dense_citers_k=_env_int("DENSE_CITERS_K", defaults.dense_citers_k),
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
        )

    def with_overrides(self, **kwargs: object) -> RetrieverConfig:
        allowed = {f.name for f in fields(self)}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        applied = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **applied)

    def holding_rrf_k(self) -> dict[str, int]:
        return {"dense": self.dense_k, "sparse": self.sparse_k_holding, "exact": self.exact_k}

    def lookup_rrf_k(self) -> dict[str, int]:
        return {"sparse": self.sparse_k_lookup, "exact": self.exact_k, "graph_1hop": self.graph_k}

    def validity_rrf_k(self) -> dict[str, int]:
        return {
            "sparse": self.sparse_k_validity,
            "exact": self.exact_k,
            "graph_1hop": self.graph_k,
            "dense_citers": self.dense_citers_k,
        }

    def case_max_per_class(self, class_name: str) -> int:
        attr = f"case_max_{class_name.lower()}"
        value = getattr(self, attr, None)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"unknown query class: {class_name!r}; expected HOLDING, LOOKUP, or VALIDITY"
            )
        return value


_DEFAULT: RetrieverConfig | None = None


def default_config() -> RetrieverConfig:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RetrieverConfig.from_env()
    return _DEFAULT


def reset_default_config() -> None:
    global _DEFAULT
    _DEFAULT = None


__all__ = [
    "GraphDirection",
    "GraphRankBy",
    "IterativeScanMode",
    "QueryClassName",
    "RetrieverConfig",
    "default_config",
    "reset_default_config",
]
