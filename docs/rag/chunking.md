# Chunking Strategy — adaptive-graphrag

**Status:** Finalized
**Scope:** US case-law opinions (`corpus/opinions.parquet` → `chunks` table)
**Chunker version:** `chunk-v1`
**Depends on:** [datasets.md](./datasets.md), [rag_design.md](./rag_design.md)
**Last revised:** 2026-11-03

---

## 1. Purpose and scope

This document specifies how a court opinion is transformed into retrievable
units. It is authoritative for:

- the segmentation algorithm,
- the parent/child size contract,
- the overlap policy,
- the metadata and breadcrumb contract,
- the validation gates that must pass before a corpus is published,
- the ablation that decides whether the strategy is kept.

It does **not** cover embedding model selection, HNSW parameters, RRF, or graph
traversal. Those are separate documents.

**Normative language.** `MUST` = a gate that blocks publication. `SHOULD` = a
default that may be overridden only with a recorded ablation result.

---

## 2. Why chunking is the highest-leverage decision in this system

Chunking is not a preprocessing detail. It determines what the retriever can
possibly return, and no downstream component — RRF, graph traversal, or the
generator — can recover evidence that was severed at ingestion.

The strongest evidence is StrucChunk, evaluated on five corpora across three
legal traditions and two languages:

| Metric | Recursive splitting | StrucChunk (structure-aware) |
|---|---|---|
| Boundary precision (CrPC) | 7.7% | **100%** |
| Documented cross-references preserved | 0% | **97.6%** |
| Char-Recall@8 (LegalBench-RAG, n=6,858) | 23.59% (best baseline) | **48.18% (+104%)** |
| Recall@5 (CrPC, n=501) | 0.557 | **0.804 (+44.4%)** |

Critically, the paper attributes **88% (CrPC) and 73% (CPC) of the total
retrieval gain to chunking**, not to the embedding model or the fusion method
. Section-boundary alignment is reported as the dominant factor
across languages, with cross-reference augmentation adding a further +2.3% on
cross-reference queries .

Two supporting findings constrain the design:

- **LegalBench-RAG** found Recursive Text Character Splitter (RTCS) without
  overlap and without a reranker to be the strongest configuration on legal text
  [reference:2].
- **Paragraph Group Chunking** — grouping whole paragraphs rather than cutting
  at fixed token counts — was the strongest strategy specifically in the legal
  domain, across 36 segmentation methods and six domains [reference:3].

**Consequence for this project:** we do not use fixed-size token windows. We use
structure-aligned, paragraph-respecting, parent-child chunking with explicit
metadata inheritance.

---

## 3. What the corpus actually looks like

The corpus is US case-law opinions. The fields that matter for chunking are:

| Column | Role in chunking |
|---|---|
| `cl_opinion_id` | Deterministic ID component; join key to graph |
| `cluster_id` | Joins to `metadata/clusters_filtered.parquet` |
| `opinion_type` | `majority`, `dissent`, `concurrence`, `plurality`, `per_curiam` — **dissents are not binding authority** |
| `text_kind` | `original`, `combined`, `generated` — dedupe preference: original > combined > generated |
| `text` | The chunking target |
| `citation`, `title`, `court`, `date` | Breadcrumb and metadata inheritance |

**Structural reality.** Opinions are not flat prose. They contain:

- a caption/syllabus region,
- numbered or headed argument sections (`I.`, `II.`, `A.`, `B.`, `Facts`,
  `Discussion`, `Analysis`, `Conclusion`),
- paragraphs that carry the actual reasoning,
- embedded block quotations, statutory citations, and case citations.

**A chunk boundary that falls in the middle of a citation, a block quote, or a
statutory reference is a defect, not an approximation.** The StrucChunk
cross-reference result (97.6% preserved vs. 0%) is the direct evidence that
token-window splitting destroys these structures .

**A note on recall vs. precision.** StrucChunk's character-level
*precision*@1 (4.51%) is lower than recursive splitting (6.64%), because its
chunks are full legal provisions (300–700 chars) rather than the minimal
ground-truth span (50–200 chars). The authors argue this trade-off is correct
for RAG: a complete provision with its qualifying context is preferable to a
minimal fragment that lacks the conditions needed for interpretation
[reference:5]. We adopt that position and therefore optimize
**Char-Recall and Recall@K, not character precision.**

---

## 4. Design principles

1. **Structure before size.** Split on the document's own boundaries first. Use
   size only as a fallback when a structural unit is too large.
2. **Two levels, not one.** Children are for retrieval; parents are for
   generation. A single-level chunk cannot serve both.
3. **Every child is self-contained.** A child chunk read in isolation must
   identify its case, court, year, opinion type, and section. Metadata stored
   only in a side column is invisible to the embedding model.
4. **Atomic spans are never split.** Citations, statutory references, dollar
   amounts, dates, and block quotations are indivisible tokens.
5. **Chunking is versioned.** A change to any parameter in §6 requires a new
   `chunker_version` and a full re-embed, because the vector space changes.

---

## 5. The pipeline

```
opinion.text
   │
   ▼
[0] Normalize                → strip headers/footers, collapse whitespace,
                               preserve paragraph breaks, normalize quotes
   │
   ▼
[1] Detect sections          → caption, syllabus, I., II., A., B., Facts,
                               Discussion, Conclusion, footnote block
   │
   ▼
[2] Build parents            → one parent per detected section
                               target 1,200 tok · cap 1,600 · min 600
   │
   ▼
[3] Build children           → split each parent on paragraph boundaries,
                               then sentence boundaries if still too large
                               target 350 tok · cap 450 · min 180
   │
   ▼
[4] Protect atomic spans     → if a split would cut a citation/quote/statute,
                               move the boundary to the nearest safe point
   │
   ▼
[5] Attach breadcrumb        → prepend hierarchy header to every child
   │
   ▼
[6] Emit two text fields     → text (breadcrumb + body) for BM25
                               text_original (body only) for dense ablation
   │
   ▼
[7] Validate                 → §8 gates
   │
   ▼
[8] Write to chunks
```

### 5.1 Stage 1 — Section detection

Use an ordered separator hierarchy, not a fixed token count. This is the
production pattern used by the Vaquill Indian legal pipeline [reference:6]:

```python
TEXT_SEPARATORS = [
    "\n\n\n",   # major section break
    "\n\n",     # paragraph break
    "\n",       # line break
    ". ",       # sentence end
    " ",        # word break
    "",         # character level (last resort)
]
```

On top of the separator pass, apply a **section classifier** over line starts.
Recognized section markers, in priority order, with a retrieval boost weight:

| Section type | Detection signal | Priority |
|---|---|---|
| `conclusion` | `CONCLUSION`, `We therefore`, `Accordingly` | 95 |
| `judgment` / `disposition` | `It is so ordered`, `Reversed`, `Affirmed` | 90 |
| `holding` | `We hold`, `We conclude` | 88 |
| `analysis` / `discussion` | `I.`, `II.`, `III.`, `Discussion`, `Analysis` | 85 |
| `facts` / `background` | `Facts`, `Background`, `I. Background` | 70 |
| `syllabus` / `caption` | leading block, `<parties>`, docket line | 40 (stored, not promoted) |
| `footnote` | `*` / `1` at line start, dense citation | 20 (never promoted) |

Section priorities mirror the 15-category scheme used in production Indian legal
RAG pipelines [reference:7]. Priority is stored on the chunk as
`section_priority` and used only as a **tie-breaker in RRF**, never as a
substitute for relevance.

**Section types are advisory.** If no marker matches, the whole opinion body is
one parent, split by paragraph. We do not invent structure that is not present.

### 5.2 Stage 2 — Parents

A parent is a **complete structural section**.

| Parameter | Value |
|---|---|
| Target | 1,200 tokens |
| Hard cap | 1,600 tokens |
| Minimum | 600 tokens (smaller sections merge with the adjacent section) |
| Overlap between parents | **0 tokens** |
| One parent per | `(cl_opinion_id, opinion_type, section_index)` |

Parents are stored with `parent_id IS NULL` and are **not** embedded. They exist
only for expansion at generation time.

The 1,000–2,000-token range for parents is the documented range for legal and
knowledge-dense documents [reference:8]. We pick 1,200 as the target because it
leaves room for two to three parents inside a 1,500–2,000-token generation
budget while remaining large enough to hold a complete argumentative unit.

### 5.3 Stage 3 — Children

A child is the **retrieval unit**. It is a paragraph, or a group of paragraphs,
inside a parent.

| Parameter | Value |
|---|---|
| Target | 350 tokens |
| Hard cap | 450 tokens |
| Minimum | 180 tokens (smaller paragraphs merge with the adjacent paragraph in the same parent) |
| Overlap | **0 tokens by default**; 32 tokens only when a paragraph must be mechanically split |
| One child per | `(parent_id, paragraph_index)` |

**Why 350 and not 512.** LegalBench-RAG ground-truth spans are short clauses
(typically 50–200 characters), while complete legal provisions run 300–700
characters [reference:9]. A 350-token child covers one to two complete provisions —
enough for the sentence to be interpretable, small enough that the embedding is
not diluted by unrelated reasoning.

**Why not smaller.** Paragraph-group chunking is the strongest strategy in the
legal domain precisely because it preserves coherent argumentative units
[reference:10]. A single sentence is a fragment, not an argument. We do not
index fragments.

**Parent-child sizing is validated in the literature.** 512-token parent /
128-token child is described as the minimum for knowledge-dense legal documents
[reference:11]; 1,024-token parent / 256-token child benchmarks at ~64% Recall@10
on legal text versus ~52% for fixed-size 512 . Our 1,200/350 is the
same shape at a size that matches the observed length of a legal provision.

### 5.4 Stage 4 — Overlap policy (revised)

**Default overlap is 0 tokens.** This is a deliberate departure from the common
"10–15% overlap" advice, and it is evidence-backed.

A controlled study on Austrian VAT law found that **zero overlap outperformed
both tested overlap settings**:

| Overlap | Accuracy |
|---|---|
| **0 tokens** | **89.19%** |
| 50 tokens | 83.78% |
| 150 tokens | 79.73% |

[reference:13]

The mechanism is straightforward: overlap duplicates text across chunks, which
means the same passage competes with itself in fusion, and the parent-expansion
step already restores surrounding context that overlap was meant to approximate.

**Overlap is therefore a fallback, not a default:**

| Situation | Overlap |
|---|---|
| Boundary is a paragraph or section boundary | **0** |
| Boundary is a sentence inside an over-long paragraph | **0** |
| Boundary is a mechanical split of a single sentence > 450 tokens | **32 tokens** |

**Overlap is an ablation parameter, not a constant** (§9, experiment E). If the
ablation shows a gain, it is promoted to a default with the measured number
recorded here. Until then, 0.

### 5.5 Stage 5 — Atomic spans that must never be split

Before writing a boundary, run an atomic-span check. If the proposed cut falls
inside any of the following, move the boundary to the nearest safe point:

| Span | Pattern |
|---|---|
| Case citation | `\d+\s+[A-Z][A-Za-z.']*\s+\d+` (e.g. `410 U.S. 113`) |
| Statutory citation | `\d+\s+U\.S\.C\.\s+§+\s*\d+[\w().]*` |
| Section reference | `§+\s*\d+[\w().-]*` |
| Docket number | `No\.\s*\d{2}-\d+` |
| Currency / amount | `\$\s*\d[\d,]*(\.\d+)?` |
| Date | `\b(19|20)\d{2}\b`, `\b\d{1,2}/\d{1,2}/\d{2,4}\b` |
| Block quotation | text between a line ending `:` and the closing citation line |

This is the operational form of the cross-reference preservation result: a split
through `410 U.S. 113` produces two chunks that neither dense nor BM25 retrieval
can recover .

### 5.6 Stage 6 — Breadcrumb and metadata inheritance

Every child `MUST` begin with a breadcrumb that makes it semantically
self-contained. This is not cosmetic: embedding hierarchical metadata directly
into each chunk is what allows the vector to "carry legal identity besides
semantic content" [reference:15].

**Breadcrumb format:**

```
[{case_name_short}, {citation} ({year}) · {court} · {opinion_type} · § {section_path}]
```

**Example:**

```
[Roe v. Wade, 410 U.S. 113 (1973) · scotus · majority · § III.B]
The Court held that the right of privacy...
```

**Inherited fields (stored as columns, not only in text):**

| Field | Source | Purpose |
|---|---|---|
| `chunk_id` | deterministic (§5.8) | Idempotent upsert |
| `case_id` | `cl_opinion_id` | Join to graph and metadata |
| `parent_id` | parent chunk | Context expansion |
| `section` | detected section type | RRF tie-breaker |
| `section_path` | `III.B` | Breadcrumb, section-aware filtering |
| `opinion_type` | corpus | Down-weight or exclude dissents for binding queries |
| `court`, `date`, `citation` | corpus | Metadata filter without a join |
| `page_start`, `page_end` | parser | User-facing provenance |
| `char_start`, `char_end` | chunker | Highlighting and citation verification |
| `chunker_version` | constant | Re-embed trigger |
| `text_hash` | `sha256(text_original)` | Duplicate detection |

Two text fields are stored, following the production Indian legal schema
[reference:16]:

| Field | Content | Used for |
|---|---|---|
| `text` | breadcrumb + body | `tsvector` / BM25, citation display |
| `text_original` | body only | Dense embedding, character-offset highlighting |

**Dense embedding input is an ablation, not an assumption.** The default embeds
`text` (breadcrumb included), because the IEEE result shows metadata injection
improves legal-identity retrieval [reference:17]. `text_original` is the
comparison arm in experiment D (§9).

### 5.7 Stage 7 — Opinion-type handling

| `opinion_type` | Chunking behavior |
|---|---|
| `majority`, `plurality`, `per_curiam` | Normal treatment; breadcrumb tags the type |
| `concurrence` | Normal treatment; breadcrumb tags the type |
| `dissent` | Chunked normally, but `opinion_type = 'dissent'` and breadcrumb tags it. **Excluded from binding-authority queries by the retrieval filter, not by dropping the chunk** |

Dissents are valuable evidence for "how has this been argued," and useless (or
misleading) for "what is the holding." That is a retrieval-time filter decision,
so the chunk must exist and be labelled.

### 5.8 Stage 8 — Deterministic chunk IDs

```
chunk_id = f"{cl_opinion_id}:{opinion_type}:{section_index}:{child_index}"
```

The ID must be a pure function of `(document, section position, child position)`.
This makes re-ingestion idempotent: a re-run overwrites the same rows rather than
creating duplicates. It also makes the parent link stable:
`parent_id = f"{cl_opinion_id}:{opinion_type}:{section_index}"`.

### 5.9 Stage 9 — Versioning

```python
CHUNKER_VERSION = "chunk-v1"
```

Any change to a parameter in §6 changes `CHUNKER_VERSION`. A change to
`CHUNKER_VERSION` **requires** a full re-embed of the affected corpus. Chunks
with different `chunker_version` values must never coexist in the same HNSW
index.

---

## 6. Final parameters

| Parameter | Value | Rationale |
|---|---|---|
| Splitter | Recursive, structure-aware separator hierarchy | 100% boundary precision vs 7.7%  |
| Section detection | Marker + separator hybrid, 15 section types | Production legal pipeline pattern [reference:19] |
| Parent target / cap / min | 1,200 / 1,600 / 600 tokens | 1,000–2,000 documented range [reference:20] |
| Child target / cap / min | 350 / 450 / 180 tokens | Provision length 300–700 chars [reference:21] |
| Overlap (structure boundary) | **0 tokens** | Zero overlap won 89.19% vs 83.78% / 79.73% [reference:22] |
| Overlap (mechanical split) | 32 tokens | Boundary-loss fallback only |
| Breadcrumb | `[case, citation (year) · court · type · § path]` | Metadata injection [reference:23] |
| Embedding input | `text` (breadcrumb + body) | Ablation D may override |
| Lexical input | `text` | Breadcrumb tokens help exact case-name match |
| Atomic spans | Citation, statute, docket, amount, date, quote | Cross-reference preservation  |
| Dissents | Chunked, labelled, filtered at query time | Not a chunking decision |
| `chunker_version` | `chunk-v1` | Re-embed trigger |

---

## 7. Schema

```sql
CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,           -- {cl_opinion_id}:{type}:{s}:{c}
    case_id         BIGINT NOT NULL REFERENCES cases(case_id),
    parent_id       TEXT,                       -- NULL for parent chunks
    is_parent       BOOLEAN NOT NULL DEFAULT FALSE,

    -- text
    text            TEXT NOT NULL,              -- breadcrumb + body  (BM25)
    text_original   TEXT NOT NULL,              -- body only          (dense / display)
    text_hash       TEXT NOT NULL,              -- sha256(text_original)

    -- structure
    section         TEXT,                       -- conclusion | holding | analysis | facts | ...
    section_path    TEXT,                       -- e.g. 'III.B'
    section_priority INT,                       -- 0-100, RRF tie-breaker only
    opinion_type    TEXT,                       -- majority | dissent | concurrence | ...
    page_start      INT,
    page_end        INT,
    char_start      INT,
    char_end        INT,

    -- inherited metadata (denormalised for filter-without-join)
    court           TEXT,
    date            DATE,
    citation        TEXT,

    -- vectors and lexical
    embedding       VECTOR(1024),
    tsv             TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,

    chunker_version TEXT NOT NULL DEFAULT 'chunk-v1',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_chunks_tsv       ON chunks USING gin (tsv);
CREATE INDEX idx_chunks_case      ON chunks (case_id);
CREATE INDEX idx_chunks_parent    ON chunks (parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX idx_chunks_court_date ON chunks (court, date);
CREATE UNIQUE INDEX uq_chunks_text_hash
    ON chunks (case_id, text_hash) WHERE is_parent = FALSE;
```

`parent_id` is `TEXT`, not `BIGINT`, because the deterministic ID is the
contract. A `BIGSERIAL` parent key would break idempotency on re-ingestion.

---

## 8. Validation gates (block publication)

A corpus is publishable only if all eight gates pass. These run as
`scripts/validate_chunks.py` and are executed in CI.

| # | Gate | Query / check | Threshold |
|---|---|---|---|
| 1 | Parent coverage | `SELECT count(*) FROM chunks WHERE parent_id IS NULL` vs children referencing them | 100% of non-parent chunks have an existing parent |
| 2 | Breadcrumb presence | regex `^\[.+, .+\(\d{4}\) · .+ · .+ · § ` on `text` | 100% |
| 3 | Boundary integrity | count of chunks where `text_original` ends mid-sentence | ≤ 2% |
| 4 | Citation integrity | count of `\d+ U\.S\. \d+` split across a chunk boundary | **0** |
| 5 | Size distribution | p50 / p95 / max of token count for children | p50 ≤ 400, p95 ≤ 450, max ≤ 600 |
| 6 | Duplicate content | duplicate `(case_id, text_hash)` rows | **0** |
| 7 | Section coverage | detected sections wholly contained in exactly one parent | ≥ 98% |
| 8 | Determinism | re-run chunker on 100 random opinions, compare `chunk_id` sets | identical |

Gate 4 is the direct operationalisation of the cross-reference result: a single
split citation is a retrieval failure that no downstream component can repair
. Gate 8 is what allows re-ingestion to be safe.

---

## 9. Ablation — the decision procedure for keeping this strategy

Chunking parameters are not accepted on authority. Every published corpus is
accompanied by the result of this ablation on the same evaluation set (see
`eval_isolated/` in [datasets.md](./datasets.md)).

| Config | Strategy | Purpose |
|---|---|---|
| **A** | Fixed 512 tokens, 0 overlap | Floor baseline |
| **B** | Recursive character splitting, 512 tokens, 0 overlap | LegalBench-RAG best baseline [reference:26] |
| **C** | **Structure-aware + parent-child (this document)** | The proposal |
| **D** | C, but dense embeds `text_original` (no breadcrumb) | Tests metadata injection [reference:27] |
| **E** | C, but child overlap ∈ {0, 32, 64} | Tests the overlap default [reference:28] |
| **F** | C, but child size ∈ {256, 350, 512} | Tests the size target |

**Primary metric:** Char-Recall@8 and Recall@10 on the citation-lookup and
holding-query sets. **Secondary:** nDCG@10, p50/p95 ingestion time, index size.

**Kill criteria — if any of these hold, the strategy is reverted, not defended:**

| Condition | Action |
|---|---|
| C does not beat B by ≥10% Char-Recall@8 | Revert to B |
| C does not beat B by ≥3 points Recall@10 | Revert to B |
| D beats C by ≥5% on legal-identity queries | Drop breadcrumb from embedding input |
| E shows overlap > 0 beats 0 by ≥3 points | Promote that overlap value to default |
| F shows 256 or 512 beats 350 by ≥5% | Adopt that value |
| Gate 3 or Gate 4 fails at scale | Block publication; fix the splitter |

**Precision expectation.** C will have lower character-level precision@1 than B,
because its chunks are complete provisions rather than minimal spans
[reference:29]. This is the intended trade-off and is not a failure condition.

---

## 10. What this strategy explicitly rejects

| Rejected | Why |
|---|---|
| **Fixed-size token windows** | 7.7% boundary precision on legal text; severs argument structure  |
| **Pure semantic chunking** | Vectara 2026: 54% accuracy vs 69% for recursive 512. High retrieval recall (91.9%) but fragments averaging 43 tokens leave the LLM without enough context [reference:31] |
| **Single-sentence chunks** | Gold spans are 3–5 sentences; isolated sentences lose the holding |
| **Default 10–15% overlap** | Zero overlap scored 89.19% vs 83.78% / 79.73% in a controlled legal study [reference:32] |
| **Single-level chunking (no parent)** | Cannot serve retrieval precision and generation context simultaneously [reference:33] |
| **Metadata only in columns** | Invisible to the embedding; the vector cannot carry legal identity [reference:34] |
| **Splitting tables, citations, or block quotes** | Cross-references are destroyed; unrecoverable at retrieval time  |

**One honest caveat.** Structure-aware chunking's advantage shrinks as documents
become less hierarchically structured. StrucChunk reports its smallest absolute
gain on MAUD (M&A contracts, +12.15 pp) precisely because those contracts lack
explicit section numbering [reference:36]. Court opinions are strongly structured,
so the strategy is well matched to this corpus — but the ablation in §9 exists
to confirm that on **our** data, not on the paper's.

---

## 11. Operational notes

**Ingestion order.** Load all rows with `embedding = NULL` first, create the
HNSW index **after** bulk load, then backfill embeddings. Building HNSW on an
empty-to-growing table is slower and produces a worse graph.

**Re-embedding.** A change to `CHUNKER_VERSION` or to the embedding model creates
a **new index**, not an in-place update. Write to `chunks_v2`, evaluate, then
switch the read alias. In-place mutation destroys the rollback path.

**Idempotency.** Because `chunk_id` is deterministic, `ON CONFLICT (chunk_id) DO
UPDATE` is safe. Gate 8 verifies this on every CI run.

**Cost.** At `medium` scale (100K opinions), expect roughly 500K children at 350
tokens each. At `max` (400K opinions), roughly 2M children. The 40.3%
fewer-chunks result on GDPR  is the reason we merge small sections
into parents instead of emitting one chunk per subsection.

---

## 12. Summary

**Structure-aligned recursive splitting; 1,200-token parents and 350-token
children; zero overlap except when a single sentence exceeds the cap; every
child carries a case/citation/section breadcrumb in its text; citations, statutes
and block quotes are atomic; deterministic chunk IDs; eight validation gates;
six-arm ablation with explicit kill criteria.**

The evidence that this matters is that chunking accounts for 73–88% of the
retrieval gain in the strongest legal chunking study available , and
that the same study reports +104% Char-Recall@8 over the best recursive baseline
on LegalBench-RAG . The evidence that it must be measured rather than
assumed is that semantic chunking, despite the highest isolated retrieval recall,
produced the worst end-to-end accuracy [reference:40].

---

## References

1. Singh, S. D. & Tripathi, A. *StrucChunk: Structure-Aware Retrieval for Legal
   Document RAG.* arXiv:2610.26313, 19 Oct 2026.
   https://arxiv.org/abs/2610.26313
2. *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal
   Domain.* arXiv:2408.10343, 2024.
   https://arxiv.org/abs/2408.10343
3. *Mitigating Context Loss in Legal RAG: A Metadata-Aware Chunking Strategy for
   Hierarchical Indian Statutes.* IEEE, conference 24–25 Apr 2026, added to
   Xplore 26 May 2026. https://ieeexplore.ieee.org/document/11528014
4. Shaukat, M. A., Adnan, M., Kuhn, C. C. N. *A Systematic Investigation of
   Document Chunking Strategies and Embedding Sensitivity.* arXiv:2603.06976,
   Aug 2026. (Paragraph Group Chunking strongest in legal.)
5. *Using large language models for legal decision-making in Austrian
   value-added tax law: a comparative study.* 2026. (Zero-overlap result.)
6. Vectara / FloTorch chunking benchmark, 2026 — recursive 512 tokens 69% vs
   semantic 54%.
7. Vaquill AI, `open-india-law` — `UNIFIED_CHUNKING_SCHEMA.md` and
   `legal_chunker.py`. Production chunk schema, separator hierarchy, dual text
   fields. https://github.com/Vaquill-AI/open-india-law
