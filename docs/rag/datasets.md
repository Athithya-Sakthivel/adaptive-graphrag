# Adaptive-GraphRAG — Dataset Reference

Scope: Defines every dataset in the adaptive-graphrag retrieval system — role, schema, provenance, join topology, scale presets, and the retrieval techniques each dataset supports.
Audience: Engineers building ingestion, indexing, retrieval, or evaluation against these datasets.

---

## 1. Purpose and Scope

The adaptive-graphrag system answers legal research queries over United States case law with four retrieval arms: dense vector, sparse lexical, citation graph, and deterministic metadata filtering. Adaptive routing selects the cheapest arm(s) per query at runtime.

This document specifies the datasets that back those arms. It is the single authoritative reference for:

- What data exists in the system and why
- The exact schema of each dataset
- How the datasets join to one another
- What scale each dataset operates at
- What each dataset proves in the evaluation harness
- What is deliberately not shipped

It does not describe ingestion mechanics, chunking strategy, embedding models, the retrieval router, or the evaluation methodology. Those are separate documents under docs/rag/.

---

## 2. System Context

The retrieval thesis is that legal queries are heterogeneous, and no single retrieval arm is optimal for all of them. Three classes dominate:

Query class: Holding
Example: "What did the court hold on qualified immunity?"
Optimal arm: Dense + sparse

Query class: Citation lookup
Example: "Find cases citing Roe v. Wade"
Optimal arm: Sparse + graph

Query class: Precedent chain
Example: "Trace how the undue-burden standard evolved"
Optimal arm: Graph (2-hop)

Query class: Metadata-filtered
Example: "Published Ninth Circuit opinions after 2022 on X"
Optimal arm: Deterministic filter + dense

To support these classes, the system maintains four independent stores rather than one monolithic index:

1. A full-text corpus for semantic and lexical retrieval
2. A citation graph for precedent traversal
3. A cluster metadata table for deterministic filtering
4. Two evaluation sets isolated from production

Each is sourced, filtered, and versioned independently. This separation is intentional — it allows the graph to grow, the corpus to be re-embedded, and the metadata to be re-filtered without invalidating the others.

---

## 3. The Four Data Roles

Role: Corpus
Dataset: docketx/us-caselaw (streamed sample)
Ships to HF: Yes
Purpose: Full-text opinions + per-opinion metadata

Role: Graph
Dataset: CourtListener citation-map (filtered)
Ships to HF: Yes
Purpose: CITES edges between opinions

Role: Metadata
Dataset: CourtListener opinion-clusters (filtered)
Ships to HF: Yes
Purpose: Cluster-level fields for deterministic filtering

Role: Evaluation
Dataset: feyninc/hojicha + synthesized QA
Ships to HF: Yes
Purpose: Chunking ablation + citation-lookup ground truth

Every dataset below lives under data/raw/ in the published snapshot. The directory tree is identical at every scale; only row counts change.

    data/raw/
    ├── corpus/opinions.parquet
    ├── graph/citations_filtered.parquet
    ├── metadata/clusters_filtered.parquet
    └── eval_isolated/
        ├── hojicha/{corpus,questions}.parquet
        └── synthesized_graph_qa/queries.parquet

---

## 4. Dataset 1 — Corpus (corpus/opinions.parquet)

### 4.1 Role

The production retrieval corpus. Every chunk embedded into the dense index, every term indexed into the sparse index, and every document returned by a retrieval arm originates here.

This is the only dataset indexed for retrieval. All others are either lookup tables (graph, metadata) or evaluation artifacts.

### 4.2 Source

HuggingFace: docketx/us-caselaw — a mirror of the CourtListener full-text corpus, retrieved from CourtListener bulk snapshot 2026-06-30. Public domain.

### 4.3 Schema

Column: id
Type: string
Semantics: docketx internal record identifier
Nullable: No

Column: cl_opinion_id
Type: int64
Semantics: CourtListener opinion ID — primary join key
Nullable: No

Column: cluster_id
Type: int64
Semantics: CourtListener cluster ID — joins to metadata
Nullable: Yes

Column: opinion_type
Type: string
Semantics: majority / dissent / concurrence / plurality / per_curiam / unknown
Nullable: Yes

Column: text_kind
Type: string
Semantics: original / combined / generated — provenance of the text
Nullable: Yes

Column: title
Type: string
Semantics: Case display name (e.g., Roe v. Wade)
Nullable: Yes

Column: text
Type: string
Semantics: Full opinion body — the field that gets chunked
Nullable: No

Column: citation
Type: string
Semantics: Bluebook citation string (e.g., 410 U.S. 113)
Nullable: Yes

Column: court
Type: string
Semantics: Court slug (e.g., ca9, scotus, nysupct)
Nullable: Yes

Column: date
Type: string
Semantics: Decision date, ISO 8601 (YYYY-MM-DD)
Nullable: Yes

Column: jurisdiction
Type: string
Semantics: Jurisdiction code (e.g., us, us-ca9)
Nullable: Yes

Column: doc_type
Type: string
Semantics: opinion / order / other
Nullable: Yes

Column: license
Type: string
Semantics: Upstream license identifier
Nullable: Yes

Column: source
Type: string
Semantics: Source system (e.g., courtlistener)
Nullable: Yes

Column: retrieved_at
Type: string
Semantics: ISO timestamp of docketx retrieval
Nullable: Yes

### 4.4 Field Notes

- cl_opinion_id is the primary key. Every join in the system originates from this column. It is unique within the corpus (deduplicated at ingestion).
- cluster_id is the secondary key. Required for metadata joins. Approximately 0.3% of opinions lack a cluster ID (typical of very old or partially ingested records); these are retained in the corpus but excluded from cluster-level filtering.
- text is the chunking target. No other column feeds the dense or sparse indexes. Chunking operates exclusively on text with title and citation retained as metadata.
- text_kind distinguishes multiple texts per opinion. Some opinions have both original and generated variants; the pipeline currently uses all rows and deduplicates at ingestion by (cl_opinion_id, text_kind) preference order: original > combined > generated.
- court and jurisdiction provide court metadata directly. This is why the dockets table is not downloaded — the corpus already carries what the metadata table would redundantly supply.
- date is the opinion decision date, not the cluster date. Cluster dates may differ by hours to days for cluster-level metadata; for temporal filters on opinion text, corpus.date is authoritative.

### 4.5 Cardinality

Scale min: 5,000 corpus rows, ~25K estimated chunks
Scale medium: 100,000 corpus rows, ~500K estimated chunks
Scale max: 400,000 corpus rows, ~2M estimated chunks

Full opinion bodies range from short orders (~500 tokens) to landmark decisions (~50,000 tokens). The chunker's hierarchical strategy produces variable chunk counts per opinion; the table above uses a ~5-chunks-per-document average.

### 4.6 What This Dataset Proves

The corpus enables:

- Dense retrieval of semantically similar passages
- Sparse retrieval of exact citations, case names, and statutes
- Hierarchical chunking (child/parent structure)
- Temporal and jurisdictional filtering via date and jurisdiction

---

## 5. Dataset 2 — Citation Graph (graph/citations_filtered.parquet)

### 5.1 Role

The precedent citation network. Provides the edges that enable graph traversal — the retrieval arm that vector and lexical search cannot replace.

### 5.2 Source

CourtListener bulk snapshot 2026-06-30, file citation-map-2026-06-30.csv.bz2 (526 MB compressed, 77,460,014 raw edges). Filtered to edges where both endpoints exist in the sampled corpus. Public Domain Mark 1.0.

### 5.3 Schema

Column: citing_opinion_id
Type: int64
Semantics: CourtListener ID of the citing opinion (the source of the citation)

Column: cited_opinion_id
Type: int64
Semantics: CourtListener ID of the cited opinion (the target)

Column: depth
Type: int32
Semantics: CourtListener citation depth heuristic (1 = direct, higher = inferred)

### 5.4 Field Notes

- citing_opinion_id and cited_opinion_id are foreign keys into corpus.opinions.cl_opinion_id. Both are guaranteed to exist in the corpus — the filter stage drops any edge whose endpoints are not in the sample.
- depth is not a confidence score. CourtListener uses depth to indicate citation extraction method, not reliability. depth = 1 means the citation was extracted from a directly quotable reference; depth > 1 means the extractor inferred it from context. All depths are retained; downstream ranking may weight by depth.
- Direction matters. A -> B means A cites B. Reversed traversal (finding cases that cite A) requires querying cited_opinion_id = A. Both directions are supported by the schema without additional indexes.

### 5.5 Cardinality

Scale min: 5,000 corpus rows, 1,951 edges, 0.39 avg out-degree
Scale medium: 100,000 corpus rows, 426,462 edges, 4.26 avg out-degree
Scale max: 400,000 corpus rows, ~1.7M edges (est.), ~4.25 avg out-degree

min scale produces a graph too sparse for multi-hop traversal. At 1,951 edges across 5,000 opinions, only a handful of 2-hop paths exist. This scale is a pipeline smoke test only.

medium scale is the smallest demo-able graph. 426,462 edges across 100,000 opinions yields ~4.3 citations per opinion — enough that precedent chains have multiple valid intermediate nodes, and adaptive routing between graph and vector arms produces measurable differences.

### 5.6 What This Dataset Proves

The citation graph enables:

- 2-hop precedent traversal — "cases cited by cases that cite X"
- In-degree ranking — "the most-cited opinions in this corpus"
- Adaptive routing wins — graph-first retrieval for citation-lookup queries where vector search is approximate and graph is exact

The graph is also the source of the synthesized evaluation set (see section 7.2). Every "Find cases citing X" query has a deterministic ground truth from citations_filtered.parquet.

---

## 6. Dataset 3 — Cluster Metadata (metadata/clusters_filtered.parquet)

### 6.1 Role

Cluster-level metadata for deterministic filtering. Enables metadata-prefiltered retrieval — narrowing the candidate set before ANN search rather than after.

### 6.2 Source

CourtListener bulk snapshot 2026-06-30, file opinion-clusters-2026-06-30.csv.bz2 (2.4 GB compressed, ~72M physical CSV rows). Cleaned (see section 6.5) and filtered to clusters referenced by the sampled corpus. Public Domain Mark 1.0.

### 6.3 Schema

Column: id
Type: int64
Semantics: CourtListener cluster ID — joins to corpus.cluster_id
Indexed for filtering: Yes

Column: case_name
Type: string
Semantics: Full case name, typically Plaintiff v. Defendant
Indexed for filtering: No

Column: case_name_short
Type: string
Semantics: Abbreviated case name (often just the defendant)
Indexed for filtering: No

Column: date_filed
Type: string
Semantics: Cluster decision date, ISO 8601
Indexed for filtering: Yes

Column: precedential_status
Type: string
Semantics: Published / Unpublished / Errata / Separate / In-chambers / Relating-to
Indexed for filtering: Yes

Column: citation_count
Type: int64
Semantics: Number of opinions citing this cluster
Indexed for filtering: No

Column: nature_of_suit
Type: string
Semantics: Subject-matter classification (e.g., Civil Rights, Habeas Corpus)
Indexed for filtering: Yes

Column: judges
Type: string
Semantics: Bench composition, free-form
Indexed for filtering: No

Column: docket_id
Type: int64
Semantics: FK into CourtListener dockets table (not shipped)
Indexed for filtering: No

### 6.4 Field Notes

- court_id is deliberately absent. CourtListener's schema places court_id on the dockets table, two joins removed from opinion-clusters:

      opinions.cluster_id  ->  clusters.id
      clusters.docket_id   ->  dockets.id
      dockets.court_id     ->  courts.id

  Downloading dockets (larger than clusters) would add ~2 hours of pipeline for zero retrieval benefit, because corpus.court and corpus.jurisdiction already carry court metadata. The join is documented but not materialized.

- precedential_status is a correctness filter, not a preference. Unpublished opinions are not citable as precedent in most US jurisdictions. Any query asking "what binds?" must filter precedential_status = 'Published'; the metadata table is the only reliable source for this.

- case_name may contain HTML fragments for a subset of CourtListener exports. The upstream CSV embeds <parties>, <docketnumber>, <court> tags in unquoted multi-line fields. The cleaning stage validates int(row[0]) — the leading cluster ID — which is unaffected by downstream field corruption. Verified: 0 of 4,780 target clusters at --scale min had truncated case_name.

- citation_count is a corpus-independent influence signal. It reflects citations in the entire CourtListener corpus, not just the sampled subset. Useful for influence ranking, but should not be confused with in-degree within citations_filtered.parquet.

### 6.5 Cleaning Pipeline

The upstream CSV has two documented irregularity classes:

1. Multi-line fields. case_name, judges, and nature_of_suit may contain literal newlines that csv.reader interprets as row boundaries. Approximately 86% of physical rows are continuation fragments of multi-line fields, not real cluster records.

2. Field-shifted rows. When a multi-line field breaks a record, downstream columns shift right. A shifted row may put a case name in the id column.

The cleaning stage applies a single invariant: id is field 0 and must parse as a positive integer. A multi-line field at any position >= 1 cannot corrupt row[0]. Rows failing this check are dropped as fragments.

Result: 10.2M clean cluster records from 71.4M physical CSV lines (14.3% retention rate). The 4,780 clusters matching --scale min and 92,157 clusters matching --scale medium were all recovered from the retained records — no truncation of case_name.

### 6.6 Cardinality

Scale min: 4,780 cluster records, 100% case_name complete
Scale medium: 92,157 cluster records, 100% case_name complete
Scale max: ~370,000 cluster records, 100% case_name complete (expected)

### 6.7 What This Dataset Proves

The metadata table enables:

- Pre-filter before ANN — filter by precedential_status, date_filed, nature_of_suit before HNSW traversal, not after
- Bindingness filtering — Published only for precedent queries
- Temporal filtering — date_filed for "cases after 2020" queries
- Subject-matter filtering — nature_of_suit for domain-scoped retrieval

---

## 7. Dataset 4 — Evaluation Sets (eval_isolated/)

Evaluation sets are never indexed in the production retrieval store. They exist solely to measure whether the system retrieves, ranks, and grounds correctly. Contaminating production with evaluation data invalidates every metric.

Two complementary evaluation sets are shipped.

### 7.1 Chunking Ablation — eval_isolated/hojicha/

Source: feyninc/hojicha (HuggingFace), derived from CUAD legal contracts. CC BY 4.0.

Purpose: Measure whether the project's hierarchical chunker outperforms a fixed-size baseline. This is the only dataset in the system designed to isolate chunking quality from retrieval-model quality.

corpus.parquet — 194 rows

Column: title
Type: string
Semantics: Contract identifier

Column: text
Type: string
Semantics: Full contract text

questions.parquet — 1,568 rows

Column: question
Type: string
Semantics: Clause-identification question

Column: document_title
Type: string
Semantics: Links the question to a corpus row

Column: chunk-must-contain
Type: string
Semantics: Ground truth: passage that MUST appear in the retrieved chunk

Column: question_type
Type: string
Semantics: Clause category (e.g., Governing Law)

Why it is domain-adjacent, not domain-matched. Contracts are legal text but not case law. The ablation measures a chunker property (does the passage survive chunking?) that is independent of document type. Using contracts avoids contamination with the case-law corpus while still exercising legal-text structure (nested clauses, cross-references, numeric provisions).

What it proves: Delta Recall@5 between fixed-size and hierarchical chunkers. Not a legal RAG quality claim; a chunker quality claim.

### 7.2 Synthesized Citation-Lookup — eval_isolated/synthesized_graph_qa/

Source: Derived from graph/citations_filtered.parquet — not downloaded. CC0 1.0 (derived).

Purpose: Provide deterministic ground truth for citation-lookup and precedent-traversal queries. Unlike a hand-labeled evaluation set, this set's answers are derivable from the graph itself, making it reproducible and auditable.

queries.parquet — 200 rows

Column: query
Type: string
Semantics: Natural-language query (Find cases citing opinion {id})

Column: eval_type
Type: string
Semantics: citation_lookup

Column: cited_opinion_id
Type: int64
Semantics: The target opinion

Column: ground_truth_citing_ids
Type: list of int64
Semantics: Every citing_opinion_id for this target in the graph

Column: ground_truth_count
Type: int64
Semantics: Length of the ground truth list

Column: derivation
Type: string
Semantics: courtlistener_citation_map_set_membership — provenance of the ground truth

Construction rule. For each of the top-200 most-cited opinions in the filtered graph, emit one query whose ground truth is the complete set of citing opinions. The query is a citation-lookup task: given a named case, return the full set of cases that cite it.

Why this is a valid benchmark. The ground truth is not an opinion — it is a set membership test in citations_filtered.parquet. Any reviewer can re-derive the same ground truth from the same file. This is stronger than 100 hand-labeled queries because it eliminates labeler subjectivity.

What it proves: Recall@K and NDCG@K for the graph retrieval arm. Also, hit rate of the adaptive router when it dispatches citation-lookup queries to the graph arm rather than dense retrieval.

### 7.3 What These Two Sets Jointly Prove

Metric: Chunking quality (Recall@5)
Evaluated against: hojicha

Metric: Graph retrieval quality (Recall@10, NDCG@10)
Evaluated against: synthesized_graph_qa

Metric: Citation precision
Evaluated against: synthesized_graph_qa

Metric: Router accuracy (graph vs. dense dispatch)
Evaluated against: Both

Metric: End-to-end latency
Evaluated against: Both (per-route p50/p95)

The two sets are intentionally complementary: hojicha isolates chunker behaviour from retrieval behaviour; synthesized_graph_qa isolates graph traversal correctness from dense semantic quality.

---

## 8. Join Topology

All datasets connect through two keys:

    corpus.opinions.cl_opinion_id   <->  graph.citations.citing_opinion_id
    corpus.opinions.cl_opinion_id   <->  graph.citations.cited_opinion_id

    corpus.opinions.cluster_id      <->  metadata.clusters.id
    metadata.clusters.docket_id     ->   dockets.id         (not shipped)
    dockets.court_id                ->   courts.id          (court metadata on corpus)

Referential integrity guarantees:

Constraint: Every citations.citing_opinion_id is in corpus.cl_opinion_id
Guaranteed by: Citation filter stage

Constraint: Every citations.cited_opinion_id is in corpus.cl_opinion_id
Guaranteed by: Citation filter stage

Constraint: Every clusters.id is in corpus.cluster_id
Guaranteed by: Cluster filter stage

Constraint: corpus.cluster_id may be NULL
Guaranteed by: Upstream; ~0.3% of opinions

These guarantees hold at every scale. The filter stages are the mechanism: they run set() membership tests against the sampled corpus, dropping any upstream record whose key is not present. There is no partial-coverage scenario where a graph edge points to a non-existent corpus row.

---

## 9. Scale Presets

The three scales are not just row-count variations. They serve distinct engineering purposes, and each is the correct choice for a specific phase of the project.

Scale min:
Corpus: 5,000
Graph: 1,951
Clusters: 4,780
Purpose: Pipeline smoke test. Verify end-to-end execution in ~3 minutes. Graph is too sparse for retrieval evaluation.

Scale medium:
Corpus: 100,000
Graph: 426,462
Clusters: 92,157
Purpose: Demo scale. Smallest scale where 2-hop traversal has meaningful branching. Default for README examples and portfolio demos.

Scale max:
Corpus: 400,000
Graph: ~1.7M
Clusters: ~370,000
Purpose: Portfolio claim. Supports "~2M vectors" without inflating chunk-per-document counts. Build time ~45 min.

Do not evaluate at min. The 1,951 edges produce a graph where 2-hop traversal returns empty results for most queries. Any graph-vs-dense comparison at min scale is measuring noise.

Do not ship a portfolio at min. The README claim should be tied to medium or max. min exists for CI.

---

## 10. Technique to Dataset Mapping

Each retrieval technique in the system uses specific datasets. This table documents which dataset supports which technique, so changes to one dataset can be traced to the techniques they affect.

Technique: Dense retrieval
Depends on: Corpus (text), embeddings
Mechanism: ANN over chunk embeddings

Technique: Sparse retrieval
Depends on: Corpus (text, citation, title)
Mechanism: BM25 / tsvector over chunks

Technique: Hierarchical chunking
Depends on: Corpus (text)
Mechanism: Child/parent structure over opinion bodies

Technique: Chunking ablation
Depends on: hojicha (isolated)
Mechanism: Compare chunkers by passage-retention Recall@5

Technique: Graph traversal (1-hop)
Depends on: citations_filtered
Mechanism: Direct neighbors of a seed opinion

Technique: Graph traversal (2-hop)
Depends on: citations_filtered
Mechanism: Neighbors of neighbors, cycle-safe

Technique: Metadata filtering
Depends on: clusters_filtered
Mechanism: Pre-ANN filter by precedential_status, date_filed, nature_of_suit

Technique: Adaptive routing
Depends on: All four
Mechanism: Classifier dispatches query to cheapest arm

Technique: Citation-lookup eval
Depends on: synthesized_graph_qa
Mechanism: Set-membership ground truth

Technique: Citation precision
Depends on: synthesized_graph_qa + corpus
Mechanism: Do returned cases have the claimed citation string

Implication for changes: if clusters_filtered is dropped, techniques 7 and 8 lose their metadata prefilter — the router cannot honour precedential_status = Published queries. If citations_filtered is dropped, techniques 5, 6, 8, 9 all fail. If hojicha is dropped, technique 4 loses its ablation. The dataset table in section 3 is not documentation — it is the dependency graph for the retrieval system.

---

## 11. Provenance, Licensing, Attribution

Dataset: Corpus
Upstream: docketx/us-caselaw
License: public-domain-edict-of-government
Redistribution: Unrestricted (US government works)

Dataset: Graph
Upstream: CourtListener citation-map-2026-06-30
License: Public Domain Mark 1.0
Redistribution: Unrestricted

Dataset: Metadata
Upstream: CourtListener opinion-clusters-2026-06-30
License: Public Domain Mark 1.0
Redistribution: Unrestricted

Dataset: Chunking eval
Upstream: feyninc/hojicha
License: CC BY 4.0
Redistribution: Attribution required

Dataset: Synthesized eval
Upstream: Derived from graph
License: CC0 1.0
Redistribution: Unrestricted (derived work)

Attribution requirements:

- hojicha is under CC BY 4.0. Any redistribution of eval_isolated/hojicha/ must include the attribution notice in DATA_SOURCES.md.
- CourtListener bulk data is Public Domain Mark 1.0. No attribution required, but upstream URL should be cited for reproducibility.
- docketx/us-caselaw derives from CourtListener and inherits the public-domain status of US court opinions.

Snapshots are immutable. Each published scale corresponds to a git tag (v1.0-min, v1.0-medium, v1.0-max). Consumers pin by tag, not by branch. Upstream changes to docketx or CourtListener require a new tag, never an overwrite.

---

## 12. Non-Shipped Artifacts

The published snapshot contains only the six retrieval artifacts plus four metadata files. The following are deliberately excluded:

Artifact: intermediate/citation_map.parquet
Size: ~15 GB
Reason for exclusion: Full 77M-row citation map before corpus filter. Rebuildable from upstream in ~4 minutes with lbzip2.

Artifact: intermediate/clusters_clean.parquet
Size: ~3.5 GB
Reason for exclusion: Cleaned cluster metadata before corpus filter. Rebuildable in ~8 minutes.

Artifact: graph/citation-map-*.csv.bz2
Size: 526 MB
Reason for exclusion: Upstream compressed source. Rebuildable.

Artifact: graph/citation-map-*.csv
Size: 2.7 GB
Reason for exclusion: Decompressed intermediate. Deleted immediately after Parquet conversion.

Artifact: metadata/opinion-clusters-*.csv.bz2
Size: 2.4 GB
Reason for exclusion: Upstream compressed source. Rebuildable.

Artifact: metadata/opinion-clusters-*.csv
Size: 12 GB
Reason for exclusion: Decompressed intermediate. Deleted immediately after clean-Parquet conversion.

Total published at --scale medium: ~1.1 GB across 10 files.
Total intermediates: ~36 GB.

The 33x ratio is why the split exists — consumers download 1.1 GB; producers pay 36 GB once.

Intermediates are excluded from MANIFEST.json and from upload_folder() via ignore_patterns. They live under intermediate/, which is also excluded from the manifest walk.

---

## 13. Known Limitations

Graph typing. The citation graph carries only CITES edges. There are no typed relationships (OVERRULES, DISTINGUISHES, CONFLICTS_WITH). This reflects the state of public legal datasets — typed precedent edges are proprietary (Shepard's, KeyCite) and not available in open data. Consequences:

- Precedent validity queries ("is X still good law?") cannot be answered by graph traversal alone. They require secondary retrieval over corpus text.
- Adaptive routing for validity queries falls back to dense retrieval, not graph traversal.

Cluster metadata coverage. Approximately 0.3% of corpus opinions lack a cluster_id (typically very old records). These opinions remain in the corpus (indexed for retrieval) but are excluded from metadata-filtered queries.

case_name HTML markup. For clusters whose upstream CSV row was fragmented by multi-line fields, case_name may contain residual HTML markup. This affects display strings only — join keys, filtering, and citation matching are unaffected. Verified at --scale min (all 4,780 clusters, 0 truncated names).

Opinion date granularity. corpus.date reflects the opinion decision date. For temporal filtering at day granularity (e.g., "decisions on 2020-01-15"), this is authoritative. For week or year granularity, sub-day differences between opinion and cluster dates are immaterial.

citation_count is corpus-independent. The value comes from CourtListener's full-corpus count, not the sampled corpus. It is a valid influence signal but should not be compared to in-degree of citations_filtered.parquet — the latter reflects only the sample.

Synthesized eval scale. 200 queries is a valid deterministic benchmark but small relative to the corpus. Recall@K on 200 queries has wider confidence intervals than on 2,000. Users seeking statistical rigor should extend the eval set to at least 1,000 queries (trivial to do — the synthesis method is a single value_counts() call).

---

## 14. Consumption Contract

Any code consuming this dataset should rely on the following guarantees.

Guaranteed:

1. All join keys preserve referential integrity (section 8)
2. corpus.text is non-null and non-empty for every row
3. citations_filtered endpoints are always in corpus.cl_opinion_id
4. clusters_filtered.id values are always in corpus.cluster_id
5. MANIFEST.json records SHA-256 and row count for every shipped file
6. Files are Parquet with fixed schema (column names, types stable across scales)

Not guaranteed:

1. Row count stability across scales (by design — scales are different)
2. Field completeness for optional columns (citation, opinion_type, judges, etc.)
3. case_name HTML-freedom
4. Cross-scale key overlap (min and max share some rows but not all)
5. Upstream stability — new snapshots require new git tags

Do not:

1. Cache path assumptions — the tree is stable, but the code should read from data/raw/ regardless of where it was cloned
2. Assume min is a subset of medium is a subset of max — the sampling is stratified by opinion ID, not cumulative
3. Merge eval_isolated/ into corpus/ for any reason — evaluation contamination is unrecoverable

---

## 15. Versioning

Each published snapshot is a git tag on Athithya117/adaptive-graphrag-datasets:

    main          = latest uploaded scale (currently medium)
    v1.0-min      = 5,000 corpus, 1,951 edges
    v1.0-medium   = 100,000 corpus, 426,462 edges
    v1.0-max      = 400,000 corpus, ~1.7M edges

Breaking changes require a new tag. A breaking change is any of:

- Column rename or type change in a shipped Parquet file
- Removal of a join key
- Change to the filtering semantics (e.g., dropping edges below depth threshold)
- Change to the synthesis rule for synthesized_graph_qa

Non-breaking changes (adding a column, adding rows without removing) may ship under the existing tag only if they are additive and the MANIFEST.json update is included. In practice, prefer a new tag — the cost of a new tag is near-zero, and the reproducibility benefit is high.

Upstream drift. CourtListener publishes quarterly bulk snapshots. docketx/us-caselaw is rebuilt periodically. Any refresh of an upstream source requires a new tag: v1.1-min, v1.1-medium, v1.1-max. Do not re-upload under v1.0-*.

---

## 16. Summary Table

Dataset: corpus/opinions.parquet
Rows (medium): 100,000
Bytes (medium): ~700 MB
Joins to: (primary)
Retrieval role: Dense, sparse, chunking
Eval role: —

Dataset: graph/citations_filtered.parquet
Rows (medium): 426,462
Bytes (medium): ~15 MB
Joins to: cl_opinion_id
Retrieval role: Graph traversal
Eval role: Citation-lookup truth

Dataset: metadata/clusters_filtered.parquet
Rows (medium): 92,157
Bytes (medium): ~20 MB
Joins to: cluster_id
Retrieval role: Metadata filter
Eval role: Bindingness filter

Dataset: eval_isolated/hojicha/corpus.parquet
Rows (medium): 194
Bytes (medium): ~10 MB
Joins to: —
Retrieval role: —
Eval role: Chunking ablation

Dataset: eval_isolated/hojicha/questions.parquet
Rows (medium): 1,568
Bytes (medium): ~1 MB
Joins to: —
Retrieval role: —
Eval role: Chunking ablation

Dataset: eval_isolated/synthesized_graph_qa/queries.parquet
Rows (medium): 200
Bytes (medium): <1 MB
Joins to: —
Retrieval role: —
Eval role: Graph retrieval eval

Total published: ~750 MB at --scale medium.

---
