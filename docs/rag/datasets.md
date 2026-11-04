# Adaptive-GraphRAG — Dataset Reference

**Scope:** Dataset roles, schemas, provenance, join keys, scale presets, and dataset-to-technique dependencies for the Adaptive-GraphRAG retrieval system.

**Audience:** Engineers working on ingestion, indexing, retrieval, and evaluation.

**Status:** Snapshot-specific reference. Row counts and file sizes below describe the stated `2026-06-30` upstream snapshot and the `min`, `medium`, and `max` build presets. Values marked *estimated* are planning estimates, not release guarantees. For an actual release, `MANIFEST.json` is authoritative for file names, row counts, hashes, and sizes.

---

## 1. Purpose and scope

Adaptive-GraphRAG retrieves information from United States case law using four retrieval capabilities:

1. **Dense retrieval** over embeddings of opinion chunks.
2. **Sparse retrieval** over chunk text and selected lexical metadata.
3. **Citation-graph retrieval** over directed `CITES` edges between opinions.
4. **Metadata filtering** over opinion-cluster metadata and opinion-level corpus fields.

An adaptive router may select one or more capabilities for a query. The examples in this document describe intended query-to-retrieval mappings; they are not a claim that one route is optimal for every query in a class.

This document defines the datasets consumed by those capabilities and the isolated evaluation data shipped with them. It does not specify the chunking algorithm, embedding model, index implementation, router logic, or evaluation protocol. Those belong in their respective documents under `docs/rag/`.

### 1.1 Query examples and intended retrieval inputs

| Query class | Example | Relevant data |
|---|---|---|
| Holding or issue question | “What did the court hold on qualified immunity?” | Opinion text; chunk embeddings; lexical index |
| Citation lookup | “Find cases citing Roe v. Wade.” | Citation graph; corpus metadata for display and resolution |
| Precedent-chain exploration | “Trace how the undue-burden standard evolved.” | Citation graph plus opinion text for substantive analysis |
| Metadata-constrained search | “Published Ninth Circuit opinions after 2022 on X.” | Cluster metadata for `precedential_status` and `date_filed`; corpus fields for `court`, `jurisdiction`, and opinion text |

The datasets have separate responsibilities:

- The **corpus** contains opinion text and opinion-level metadata.
- The **citation graph** contains directed opinion-to-opinion citation edges.
- The **cluster metadata** file contains one record per selected CourtListener cluster.
- The **evaluation data** contains a chunking-ablation set and a graph-derived citation-lookup set. Neither evaluation set is production corpus data.

### 1.2 Published directory layout

The dataset snapshot uses the following layout at each scale. Row counts vary by scale; the path layout is intended to remain stable.

```text
data/raw/
├── corpus/
│   └── opinions.parquet
├── graph/
│   └── citations_filtered.parquet
├── metadata/
│   └── clusters_filtered.parquet
└── eval_isolated/
    ├── hojicha/
    │   ├── corpus.parquet
    │   └── questions.parquet
    └── synthesized_graph_qa/
        └── queries.parquet
```

The six Parquet files above are the dataset artifacts. Repository/release metadata files are separate and should be counted from the release manifest rather than included in a dataset-file count.

---

## 2. Dataset 1 — Opinion corpus

**Path:** `corpus/opinions.parquet`
**Upstream:** Hugging Face dataset `docketx/us-caselaw`, described as a mirror of CourtListener full-text opinions.
**Upstream snapshot:** CourtListener bulk snapshot dated `2026-06-30`.
**Role:** Source records for opinion text, opinion-level metadata, chunk creation, and opinion display.

The corpus is the source of the text that is chunked and embedded. The graph and metadata datasets are joined back to corpus opinions; they are not substitutes for the opinion text.

### 2.1 Schema

The following schema is the schema declared by the source dataset documentation. Confirm the released Parquet schema against `MANIFEST.json` when publishing a new snapshot.

| Column | Type | Nullable | Meaning |
|---|---|---:|---|
| `id` | string | No | `docketx` internal record identifier. |
| `cl_opinion_id` | int64 | No | CourtListener opinion identifier. Used to join opinions to citation-graph endpoints. |
| `cluster_id` | int64 | Yes | CourtListener cluster identifier. Used to join opinions to `clusters_filtered.id`. |
| `opinion_type` | string | Yes | Opinion type, such as `majority`, `dissent`, `concurrence`, `plurality`, `per_curiam`, or `unknown`. |
| `text_kind` | string | Yes | Text provenance/category, such as `original`, `combined`, or `generated`. |
| `title` | string | Yes | Case display name. |
| `text` | string | No | Opinion body supplied to the chunking pipeline. |
| `citation` | string | Yes | Citation string, for example `410 U.S. 113`. |
| `court` | string | Yes | Court slug, for example `ca9`, `scotus`, or `nysupct`. |
| `date` | string | Yes | Opinion decision date in `YYYY-MM-DD` form. |
| `jurisdiction` | string | Yes | Jurisdiction code, for example `us` or `us-ca9`. |
| `doc_type` | string | Yes | Document category, such as `opinion`, `order`, or `other`. |
| `license` | string | Yes | Upstream license identifier. |
| `source` | string | Yes | Source-system label, for example `courtlistener`. |
| `retrieved_at` | string | Yes | Retrieval timestamp in ISO format. |

### 2.2 Key semantics and indexing inputs

There are two distinct join keys in the corpus:

- `cl_opinion_id` is the **opinion-level key**. It connects corpus opinions to both endpoint columns in the citation graph.
- `cluster_id` is the **cluster-level key**. It connects corpus opinions to `clusters_filtered.id`. It may be null.

These keys are not interchangeable. Not every relationship in the dataset is keyed by `cl_opinion_id`.

`text` is the body supplied to the chunker. `title` and `citation` are retained as opinion metadata and may be used by the sparse-retrieval representation or for result display. The retrieval index should therefore be described as consuming chunk text plus selected metadata, not as consuming `text` alone for every retrieval mode.

Opinion-level `court`, `jurisdiction`, and `date` are available without joining the CourtListener `dockets` or `courts` tables. `date` is the opinion date; `clusters_filtered.date_filed` is the cluster-level date. Use the date field whose semantics match the query being implemented, and document any fallback policy.

The source documentation describes a preference order for competing text variants: `original` before `combined` before `generated`. The final corpus must have a clearly defined row-selection policy. In particular, the opinion-level graph join assumes that `cl_opinion_id` identifies one canonical corpus opinion row. If multiple rows per `cl_opinion_id` are retained, joins from graph results to corpus records become one-to-many and that behavior must be explicit. The published artifact should be checked for this invariant rather than relying on the phrase “primary key” alone.

### 2.3 Scale and chunk-count estimates

| Preset | Corpus rows | Estimated chunks | Interpretation |
|---|---:|---:|---|
| `min` | 5,000 | ~25,000 | Pipeline smoke-test scale. |
| `medium` | 100,000 | ~500,000 | Demo scale. |
| `max` | 400,000 | ~2,000,000 | Large demonstration scale. |

Chunk counts are estimates, not corpus row counts and not guaranteed embedding counts. The original estimate assumes roughly five chunks per opinion. Actual counts depend on the chunker, the source opinion lengths, and the final build configuration. Use the chunk-build output or index manifest for an exact vector count; do not present these estimates as measured vector totals.

Opinion bodies vary substantially in length. The source description gives examples ranging from short orders of roughly 500 tokens to long decisions of roughly 50,000 tokens. Chunk sizing, parent/child structure, and embedding limits are defined in the chunking and embedding documentation, not by this dataset reference.

### 2.4 Uses

The corpus supports:

- Dense retrieval after opinion text has been chunked and embedded.
- Sparse retrieval over chunk text and selected fields such as case title and citation.
- Hierarchical chunk construction from opinion bodies.
- Opinion-level filtering or display using fields such as `court`, `jurisdiction`, and `date`.

The corpus alone does not provide citation relationships or cluster-level `precedential_status` and `nature_of_suit` values.

---

## 3. Dataset 2 — Citation graph

**Path:** `graph/citations_filtered.parquet`
**Upstream:** CourtListener citation-map bulk file `citation-map-2026-06-30.csv.bz2`.
**Upstream snapshot:** `2026-06-30`.
**Role:** Directed citation edges between opinions in the selected corpus.

The source file is reported as 526 MB compressed and 77,460,014 raw edges before filtering. The published graph retains edges for which both endpoint opinion IDs are present in the selected corpus.

### 3.1 Schema

| Column | Type | Meaning |
|---|---|---|
| `citing_opinion_id` | int64 | CourtListener ID of the opinion making the citation (edge source). |
| `cited_opinion_id` | int64 | CourtListener ID of the opinion being cited (edge target). |
| `depth` | int32 | CourtListener citation-depth/extraction field, as described by the source. |

### 3.2 Direction and `depth`

An edge `A -> B` means **opinion A cites opinion B**.

- To find opinions cited by A, select rows where `citing_opinion_id = A`.
- To find opinions that cite A, select rows where `cited_opinion_id = A`.

The source documentation describes `depth = 1` as a direct citation and values greater than 1 as inferred from context. It also explicitly cautions that `depth` is not a confidence score. Preserve the field as upstream extraction metadata; do not interpret a larger value as a stronger or weaker citation without a separate, documented ranking rule. All depth values are stated to be retained in the filtered graph.

### 3.3 Filtering and referential integrity

Both `citing_opinion_id` and `cited_opinion_id` refer to `corpus.opinions.cl_opinion_id`. The graph filter is described as retaining an edge only when **both** endpoint IDs are present in the selected corpus. Under that filtering rule, every published edge endpoint should resolve to a corpus opinion.

This guarantee is about graph endpoints. It does not imply that every corpus opinion has at least one incoming or outgoing edge.

### 3.4 Scale

| Preset | Corpus rows | Filtered edges | Edges per corpus opinion |
|---|---:|---:|---:|
| `min` | 5,000 | 1,951 | 0.39 |
| `medium` | 100,000 | 426,462 | 4.26 |
| `max` | 400,000 | ~1,700,000 (estimated) | ~4.25 (estimated) |

“Edges per corpus opinion” is `filtered edge count / corpus opinion count`. It is a corpus-normalized edge density. It should not be presented as the average degree among only those nodes that appear in the graph unless the graph's vertex denominator is defined that way.

The `min` graph is intended for pipeline validation, not for drawing retrieval-quality conclusions. Its smaller edge count means fewer available citation relationships. Whether a particular 1-hop or 2-hop query has results must be measured against the graph; edge count alone does not establish the number of reachable paths or the quality of a graph-versus-dense comparison.

### 3.5 Uses and limits

The graph supports:

- Direct citation lookup in either direction.
- In-degree and out-degree calculations, when the denominator is defined explicitly.
- One-hop and multi-hop graph traversal.
- Graph-derived ground truth for the direct citation-lookup evaluation set described in §6.2.

The graph contains `CITES` relationships only. It does not encode typed legal relationships such as `OVERRULES`, `DISTINGUISHES`, or `CONFLICTS_WITH`. Citation traversal alone therefore cannot establish whether a case remains good law.

---

## 4. Dataset 3 — Cluster metadata

**Path:** `metadata/clusters_filtered.parquet`
**Upstream:** CourtListener opinion-clusters bulk file `opinion-clusters-2026-06-30.csv.bz2`.
**Upstream snapshot:** `2026-06-30`.
**Role:** Cluster-level metadata used for deterministic filtering and case display.

The upstream file is reported as 2.4 GB compressed and approximately 72 million physical CSV rows. The cleaned file is filtered to cluster IDs referenced by the selected corpus.

### 4.1 Schema

| Column | Type | Meaning | Listed as filter field? |
|---|---|---|---:|
| `id` | int64 | CourtListener cluster ID; joins to `corpus.opinions.cluster_id`. | Yes |
| `case_name` | string | Full case name. | No |
| `case_name_short` | string | Abbreviated case name. | No |
| `date_filed` | string | Cluster-level date in ISO date form. | Yes |
| `precedential_status` | string | Status such as `Published`, `Unpublished`, `Errata`, `Separate`, `In-chambers`, or `Relating-to`. | Yes |
| `citation_count` | int64 | CourtListener citation count for the cluster. | No |
| `nature_of_suit` | string | Subject-matter classification, such as Civil Rights or Habeas Corpus. | Yes |
| `judges` | string | Free-form bench-composition value. | No |
| `docket_id` | int64 | CourtListener docket ID; the dockets table is not shipped. | No |

“Listed as filter field” reproduces the intended use stated for this dataset; the actual database/index configuration should be checked separately.

### 4.2 Join and field semantics

`clusters_filtered.id` joins to `corpus.opinions.cluster_id`. Multiple opinions may belong to the same cluster, so cluster-row counts and corpus-opinion counts are different quantities.

`date_filed` is a cluster-level date. It is not the same field as `corpus.opinions.date`, which is described as the opinion decision date. Query logic should choose the appropriate field and should not silently substitute one for the other.

`court_id` is not present in this file. In the upstream CourtListener schema, the path to court metadata is described as:

```text
corpus.opinions.cluster_id -> clusters.id
clusters.docket_id         -> dockets.id
 dockets.court_id           -> courts.id
```

The `dockets` and `courts` tables are not shipped in this snapshot. The corpus already contains `court` and `jurisdiction` fields, so court filtering should use those corpus fields unless a separate, documented reason requires the external normalized join.

`citation_count` is described as a CourtListener-wide count, not a count restricted to the sampled graph. It is therefore not interchangeable with the in-degree calculated from `citations_filtered.parquet`.

The source documentation treats `precedential_status` as the field for filtering to published opinions. It should not be treated as a universal legal conclusion about whether an opinion is binding in every jurisdiction; jurisdiction-specific precedent rules are outside the scope of this dataset schema.

### 4.3 Cleaning notes

The upstream CSV is described as containing multi-line fields and field-shifted physical rows. The documented cleaning rule accepts a row when field 0 (the cluster ID) parses as a positive integer and drops rows that fail that check.

The source reports the following snapshot-specific counts:

- Approximately 71.4 million physical CSV lines processed.
- Approximately 10.2 million retained clean cluster records.
- Approximately 14.3% retention from physical lines to retained records.

These figures describe the reported cleaning run; they are not a general guarantee that the same parser will recover every future CourtListener snapshot. A new source snapshot should be checked for row counts, duplicate IDs, key coverage, and field completeness.

The original build notes report no truncated `case_name` values among the 4,780 target clusters checked at `min` scale. Do not generalize that specific check to every future snapshot or scale without rerunning the validation.

### 4.4 Scale

| Preset | Cluster rows | Meaning |
|---|---:|---|
| `min` | 4,780 | Distinct selected cluster records reported for the 5,000-opinion corpus. |
| `medium` | 92,157 | Distinct selected cluster records reported for the 100,000-opinion corpus. |
| `max` | ~370,000 (estimated) | Estimated selected cluster records for the 400,000-opinion corpus. |

These counts are cluster records, not opinions. The difference between corpus rows and cluster rows cannot be used to calculate the percentage of null `cluster_id` values: several opinions can share one cluster. The earlier source note reports approximately 0.3% of opinions with a null `cluster_id`, but that percentage must be confirmed with a direct null-count query against `opinions.parquet`; it is not derivable from the cluster-row counts above.

### 4.5 Uses and limits

The metadata file supports filtering by:

- `precedential_status`.
- `date_filed`.
- `nature_of_suit`.

It also supplies display and analysis fields such as `case_name`, `case_name_short`, `judges`, and `citation_count`. Court and jurisdiction filtering should use the corresponding opinion-level corpus fields unless the unshipped `dockets` and `courts` tables are added.

---

## 5. Evaluation data

Evaluation files live under `eval_isolated/` and are not part of the production opinion corpus. They should not be merged into production indexing data.

Two evaluation datasets are described below. They test different properties and should not be treated as interchangeable benchmarks.

### 5.1 Chunking ablation — Hojicha

**Path:** `eval_isolated/hojicha/`
**Upstream:** Hugging Face dataset `feyninc/hojicha`, described as derived from CUAD legal contracts.
**Declared license:** CC BY 4.0.
**Purpose:** Compare chunking strategies using clause-identification questions and a required-passage field.

`corpus.parquet` contains 194 rows:

| Column | Type | Meaning |
|---|---|---|
| `title` | string | Contract identifier. |
| `text` | string | Full contract text. |

`questions.parquet` contains 1,568 rows:

| Column | Type | Meaning |
|---|---|---|
| `question` | string | Clause-identification question. |
| `document_title` | string | Links the question to a contract row. |
| `chunk-must-contain` | string | Required ground-truth passage that must be present in a retrieved chunk. |
| `question_type` | string | Clause category, for example Governing Law. |

This is a **domain-adjacent** chunking benchmark: the documents are contracts, not case-law opinions. Its stated purpose is to test whether chunking preserves a required passage. A reported metric is Recall@5, comparing fixed-size and hierarchical chunkers. Results should be described as chunking results on this dataset, not as evidence of end-to-end case-law RAG quality.

### 5.2 Synthesized direct citation-lookup set

**Path:** `eval_isolated/synthesized_graph_qa/queries.parquet`
**Source:** Derived from `graph/citations_filtered.parquet`; not downloaded as an independent upstream dataset.
**Declared license:** CC0 1.0 for the derived evaluation artifact.
**Purpose:** Evaluate direct citation lookup using ground truth computed from the filtered citation graph.

The file contains 200 rows. Its documented schema is:

| Column | Type | Meaning |
|---|---|---|
| `query` | string | Natural-language query of the form “Find cases citing opinion {id}”. |
| `eval_type` | string | Task label; documented value is `citation_lookup`. |
| `cited_opinion_id` | int64 | Opinion being cited (the lookup target). |
| `ground_truth_citing_ids` | list<int64> | All `citing_opinion_id` values whose `cited_opinion_id` equals the target in the filtered graph. |
| `ground_truth_count` | int64 | Number of IDs in `ground_truth_citing_ids`. |
| `derivation` | string | Ground-truth derivation label; documented value is `courtlistener_citation_map_set_membership`. |

The construction rule is to select the 200 most-cited target opinions in the filtered graph and emit one query per target. For a target `X`, the ground truth is the set of opinions `Y` for which the graph contains the edge `Y -> X`.

This is a deterministic **one-hop incoming-citation lookup** benchmark. The documented `eval_type` and ground-truth columns do not define a 2-hop precedent-chain benchmark, holding-question benchmark, or semantic-relevance benchmark. Such evaluations require separate query sets and ground-truth definitions.

The graph-derived ground truth is reproducible from the same filtered graph and construction rule. It is not independent human annotation: the evaluation measures retrieval against the graph's own recorded edges.

### 5.3 What the supplied evaluation data supports

| Evaluation question | Dataset | Supported interpretation |
|---|---|---|
| Does a chunk contain the required passage? | Hojicha | Chunking passage-retention performance, reported as Recall@5. |
| Can the system retrieve opinions that cite a specified opinion? | Synthesized citation lookup | Direct graph lookup, reported with metrics such as Recall@K and NDCG@K. |

The datasets do **not**, by themselves, define a complete router-accuracy benchmark, an end-to-end latency benchmark, a holding-quality benchmark, or a 2-hop precedent-chain benchmark. Those measurements may be part of the wider evaluation harness, but their query sets, labels, metrics, and protocol must be documented in the evaluation methodology rather than inferred from these two datasets.

---

## 6. Join topology and integrity rules

There are two separate join paths from the corpus:

```text
Opinion-level joins
-------------------
corpus.opinions.cl_opinion_id
    <-> graph.citations_filtered.citing_opinion_id
    <-> graph.citations_filtered.cited_opinion_id

Cluster-level join
------------------
corpus.opinions.cluster_id
    <-> metadata.clusters_filtered.id

Optional upstream CourtListener path (not shipped)
--------------------------------------------------
metadata.clusters_filtered.docket_id -> dockets.id
 dockets.court_id                     -> courts.id
```

### 6.1 Required checks for each release

| Check | Expected condition | How it is established |
|---|---|---|
| Graph source endpoint | Every `citing_opinion_id` resolves to a corpus `cl_opinion_id`. | Graph filter retains only edges whose source endpoint is in the sampled corpus. |
| Graph target endpoint | Every `cited_opinion_id` resolves to a corpus `cl_opinion_id`. | Graph filter retains only edges whose target endpoint is in the sampled corpus. |
| Cluster key direction | Every shipped `clusters_filtered.id` is referenced by a non-null corpus `cluster_id`. | Cluster filter selects clusters referenced by the sampled corpus. |
| Cluster coverage | Every non-null corpus `cluster_id` has a metadata row, if complete metadata coverage is part of the release contract. | Verify with an anti-join; subset filtering alone does not prove the reverse direction. |
| Corpus opinion-key cardinality | `cl_opinion_id` is unique if graph-to-corpus joins are intended to be one-to-one. | Verify after text-variant resolution. |
| Null cluster IDs | Nulls are allowed in `corpus.cluster_id`. | Measure directly from the corpus; do not infer from cluster-row counts. |

The graph filter description establishes both endpoint checks. The cluster filter description establishes that metadata rows are selected from cluster IDs referenced by the corpus; complete reverse coverage should still be asserted with an anti-join if consumers depend on it. Similarly, the one-row-per-opinion assumption should be checked after applying the text-variant selection policy.

A release validation should record at least:

```sql
-- Graph endpoint coverage
SELECT COUNT(*)
FROM graph.citations_filtered g
LEFT JOIN corpus.opinions c
  ON c.cl_opinion_id = g.citing_opinion_id
WHERE c.cl_opinion_id IS NULL;

SELECT COUNT(*)
FROM graph.citations_filtered g
LEFT JOIN corpus.opinions c
  ON c.cl_opinion_id = g.cited_opinion_id
WHERE c.cl_opinion_id IS NULL;

-- Corpus opinions without a cluster ID
SELECT COUNT(*)
FROM corpus.opinions
WHERE cluster_id IS NULL;

-- Non-null corpus cluster IDs without metadata
SELECT COUNT(DISTINCT c.cluster_id)
FROM corpus.opinions c
LEFT JOIN metadata.clusters_filtered m
  ON m.id = c.cluster_id
WHERE c.cluster_id IS NOT NULL
  AND m.id IS NULL;

-- Duplicate opinion IDs (must be zero for a one-row-per-opinion contract)
SELECT cl_opinion_id, COUNT(*) AS row_count
FROM corpus.opinions
GROUP BY cl_opinion_id
HAVING COUNT(*) > 1;
```

Adapt table names and SQL syntax to the actual storage engine. The queries describe the checks; they are not evidence that a particular release has passed them. Store their results with the build manifest or validation report.

---

## 7. Scale presets

The presets represent different corpus sample sizes. Graph and metadata counts depend on filtering those upstream datasets to the selected corpus.

| Preset | Corpus opinions | Citation edges | Cluster rows | Intended use |
|---|---:|---:|---:|---|
| `min` | 5,000 | 1,951 | 4,780 | CI and end-to-end pipeline smoke tests. Not the preferred scale for retrieval-quality claims. |
| `medium` | 100,000 | 426,462 | 92,157 | Default demo and README scale. |
| `max` | 400,000 | ~1.7M (estimated) | ~370,000 (estimated) | Larger demonstration and scale testing. |

The scale values are snapshot-specific. In particular:

- Corpus counts are opinion rows.
- Graph counts are directed citation edges after both endpoints are filtered to the corpus.
- Metadata counts are cluster records, not opinion rows.
- Chunk counts and vector counts are separate derived quantities and are only estimates until measured from a completed chunk/index build.

Do not assume that `min` is a strict subset of `medium`, or that `medium` is a strict subset of `max`. The source documentation describes the sampling as stratified by opinion ID rather than cumulative.

Use `min` to validate that the pipeline runs. Use `medium` or `max` for demonstrations and retrieval experiments, and report the chosen preset with every result. Do not infer 2-hop path availability or benchmark validity from row counts alone.

---

## 8. Dataset-to-technique dependencies

| Technique | Required dataset or derived artifact | Dependency |
|---|---|---|
| Dense retrieval | Corpus plus chunk and embedding outputs | Opinion `text` is chunked; derived chunks are embedded and stored in the ANN index. |
| Sparse retrieval | Corpus plus chunk/index outputs | Chunk text and selected lexical metadata, including title and citation where configured. |
| Hierarchical chunking | Corpus | Opinion `text` is the source body. Chunk structure and token limits are defined in the chunking specification. |
| Citation graph traversal (1-hop) | `citations_filtered.parquet` | Direct incoming or outgoing neighbors of an opinion ID. |
| Citation graph traversal (2-hop) | `citations_filtered.parquet` | Neighbors-of-neighbors traversal; cycle handling and edge-direction semantics belong to the graph implementation. |
| Metadata filtering | `clusters_filtered.parquet` plus corpus fields | Cluster fields such as `precedential_status`, `date_filed`, and `nature_of_suit`; opinion-level court and jurisdiction fields remain in the corpus. |
| Chunking ablation | Hojicha evaluation files | Required-passage retention comparison between chunking strategies. |
| Direct citation-lookup evaluation | `synthesized_graph_qa/queries.parquet` plus the same filtered graph | Set-membership ground truth for incoming citation edges. |

The router's dispatch policy is an application-layer dependency over the retrieval capabilities. These datasets provide the underlying records, but the route classifier, cost model, and route-quality labels are not defined here.

---

## 9. Provenance, license declarations, and versioning

The following provenance and license labels are those stated in the source dataset documentation. Verify the upstream license files and the repository's `DATA_SOURCES.md` before redistributing a refreshed snapshot.

| Dataset | Upstream/source | Declared license or status |
|---|---|---|
| Opinion corpus | `docketx/us-caselaw`, derived from CourtListener | `public-domain-edict-of-government` / public-domain US court opinions, as declared by the source documentation. |
| Citation graph | CourtListener citation-map snapshot `2026-06-30` | Public Domain Mark 1.0. |
| Cluster metadata | CourtListener opinion-clusters snapshot `2026-06-30` | Public Domain Mark 1.0. |
| Hojicha evaluation | `feyninc/hojicha` | CC BY 4.0; attribution required. |
| Synthesized citation evaluation | Derived from the filtered citation graph | CC0 1.0, as declared for the derived artifact. |

Attribution requirements recorded for this project:

- Redistributing Hojicha files requires the CC BY 4.0 attribution notice in `DATA_SOURCES.md`.
- CourtListener source URLs and snapshot identifiers should be retained for reproducibility even where the declared status does not require attribution.
- The corpus provenance should identify `docketx/us-caselaw` and its CourtListener source snapshot.

The source documentation describes immutable release tags:

```text
v1.0-min
v1.0-medium
v1.0-max
```

It also describes `main` as the latest uploaded scale. A consumer requiring reproducibility should pin an immutable tag and inspect that tag's manifest, rather than relying on a mutable branch name.

A new upstream CourtListener or `docketx` snapshot should receive a new dataset version/tag. Do not overwrite an existing version with changed source data or changed filtering semantics.

### 9.1 Breaking changes

Treat the following as breaking changes:

- Renaming or changing the type of a shipped column.
- Removing or changing the meaning of a join key.
- Changing graph filtering semantics, including any change to retained `depth` values.
- Changing the text-variant selection policy in a way that changes corpus identity or row cardinality.
- Changing the construction rule or ground-truth semantics for `synthesized_graph_qa`.

Adding a nullable or otherwise backward-compatible column may be additive, but still requires an updated manifest and schema validation. Prefer a new version tag when a change could affect reproducibility.

---

## 10. Excluded build intermediates

The following intermediates are described as build-time artifacts and are not part of the published dataset tree:

| Artifact | Reported size | Reason for exclusion |
|---|---:|---|
| `intermediate/citation_map.parquet` | ~15 GB | Full citation map before filtering to the selected corpus. Rebuildable from the upstream citation-map file. |
| `intermediate/clusters_clean.parquet` | ~3.5 GB | Cleaned cluster data before filtering to corpus cluster IDs. |
| `graph/citation-map-*.csv.bz2` | 526 MB | Compressed upstream source file. |
| `graph/citation-map-*.csv` | 2.7 GB | Decompressed CSV intermediate used for Parquet conversion. |
| `metadata/opinion-clusters-*.csv.bz2` | 2.4 GB | Compressed upstream source file. |
| `metadata/opinion-clusters-*.csv` | 12 GB | Decompressed CSV intermediate used for cleaning and Parquet conversion. |

The reported intermediate sizes sum to approximately 36 GB. The source documentation states that intermediates are excluded from `MANIFEST.json` and the dataset upload through ignore rules. The build/release process should verify the ignore configuration so that source CSVs and intermediate Parquet files are not accidentally published.

---

## 11. Known limitations

### 11.1 Citation relation types

The graph contains `CITES` edges only. It does not encode `OVERRULES`, `DISTINGUISHES`, or `CONFLICTS_WITH`. Queries about whether a case remains good law require analysis beyond this graph, such as retrieving and examining relevant opinion text or using a separate citator source.

### 11.2 Cluster metadata coverage

Opinions with a null `cluster_id` cannot join to cluster-level metadata. The source notes report an approximate 0.3% null rate, but the exact rate must be computed from the released corpus. Do not derive it by subtracting the cluster-row count from the opinion-row count, because multiple opinions can share a cluster.

### 11.3 Case-name formatting

The source build notes identify possible HTML fragments in `case_name` from irregular upstream CSV rows. Treat case names as display strings that may require sanitization. Join keys and numeric filter fields should be validated independently. The reported no-truncation check at `min` scale is specific to that checked sample.

### 11.4 Opinion date versus cluster date

`corpus.date` and `clusters_filtered.date_filed` have different documented semantics. Temporal filters must state which date is authoritative for the use case. Do not silently mix the two fields.

### 11.5 Corpus-wide versus sample-specific citation counts

`clusters_filtered.citation_count` is described as an upstream corpus-wide count. Graph degree computed from `citations_filtered.parquet` is sample-specific. They are different measures and should not be compared as though they used the same population.

### 11.6 Evaluation scope and sample size

The synthesized set contains 200 direct citation-lookup queries derived from the filtered graph. It is reproducible, but it reflects the graph's own edge coverage and query-construction rule. Results should be reported with the query count and dataset version. It does not substitute for a separately labeled semantic, holding, or 2-hop precedent-chain evaluation.

Hojicha contains 1,568 clause-identification questions over 194 contracts. Its domain is contracts rather than case law, so its results should remain scoped to chunking passage-retention behavior.

---

## 12. Consumer contract

### 12.1 Release invariants to validate

A release should not be described as conforming until its manifest and validation report establish the following:

1. All six dataset Parquet files exist at the documented paths.
2. Parquet schemas match the versioned schema contract.
3. Every graph source and target opinion ID resolves to the corpus.
4. Every shipped metadata cluster ID is referenced by the corpus; verify reverse coverage as well if consumers require metadata for every non-null corpus `cluster_id`.
5. The intended cardinality of `cl_opinion_id` is verified after applying the text-variant policy.
6. Null counts for `cluster_id` and optional fields are measured and recorded.
7. `MANIFEST.json` records the hash and row count for each shipped Parquet file; record byte size as well if the release process supports it.
8. Evaluation files remain isolated from production corpus and retrieval indexing.

### 12.2 Consumer assumptions that are not guaranteed

Consumers must not assume:

- Exact row counts across scales.
- Non-null values in optional metadata fields such as `citation`, `opinion_type`, `judges`, or `nature_of_suit`.
- HTML-free case names.
- That the `min`, `medium`, and `max` samples are cumulative subsets.
- That upstream source data or schemas remain unchanged between snapshots.
- That estimated chunk counts equal the number of chunks or vectors in a particular build.
- That every opinion has a citation edge or that every graph traversal has a result.

### 12.3 Consumer rules

- Resolve dataset paths relative to the configured dataset root; do not assume a fixed checkout location.
- Use `cl_opinion_id` for opinion-level graph joins and `cluster_id` for cluster-metadata joins.
- Treat null `cluster_id` as a valid corpus state; define how metadata-filtered queries handle those opinions.
- Keep `eval_isolated/` separate from production indexing.
- Pin a version tag and retain its manifest when reproducing an experiment.

---

## 13. Medium-scale file-size reference

The source documentation reports the following approximate sizes for the six dataset files at `medium` scale:

| File | Rows | Approximate size |
|---|---:|---:|
| `corpus/opinions.parquet` | 100,000 | ~700 MB |
| `graph/citations_filtered.parquet` | 426,462 | ~15 MB |
| `metadata/clusters_filtered.parquet` | 92,157 | ~20 MB |
| `eval_isolated/hojicha/corpus.parquet` | 194 | ~10 MB |
| `eval_isolated/hojicha/questions.parquet` | 1,568 | ~1 MB |
| `eval_isolated/synthesized_graph_qa/queries.parquet` | 200 | <1 MB |

These rounded per-file figures sum to approximately 746 MB, or roughly 750 MB. The earlier source text also states that the complete published snapshot is approximately 1.1 GB across 10 files. That larger total is not explained by the six dataset-file sizes above. Do not publish both numbers as equivalent totals: use the actual release manifest to report either (a) the sum of the six dataset Parquet files or (b) the total size of all files in the release, and list what is included in the latter.

---

## 14. Summary

| Dataset | Medium rows | Join key / relationship | Production or evaluation role |
|---|---:|---|---|
| `corpus/opinions.parquet` | 100,000 | `cl_opinion_id` for graph; `cluster_id` for metadata | Opinion text and opinion-level metadata; source for chunking and retrieval indexes. |
| `graph/citations_filtered.parquet` | 426,462 | Both endpoint fields join to `cl_opinion_id` | Directed citation graph and direct citation-lookup ground truth. |
| `metadata/clusters_filtered.parquet` | 92,157 | `id` joins to `cluster_id` | Cluster-level filtering and display metadata. |
| `eval_isolated/hojicha/corpus.parquet` | 194 | `title` links to evaluation questions | Contract text for chunking ablation. |
| `eval_isolated/hojicha/questions.parquet` | 1,568 | `document_title` links to contract `title` | Clause-identification questions and required-passage ground truth. |
| `eval_isolated/synthesized_graph_qa/queries.parquet` | 200 | `cited_opinion_id` identifies graph target | Direct incoming-citation lookup evaluation. |

**Operational rule:** Treat the corpus, graph, and metadata as distinct datasets with distinct keys. Treat evaluation data as isolated test material. Treat all row counts and file sizes as versioned snapshot values, and use the manifest and validation report—not estimates in prose—as the final release record.
