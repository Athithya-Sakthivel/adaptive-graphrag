# Chunking Strategy — adaptive-graphrag

**Status:** Finalized
**Scope:** US case-law opinions (`corpus/opinions.parquet` → `chunks` table)
**Chunker version:** `chunk-v1`

---

## 1. Purpose

Authoritative specification for:
- Segmentation algorithm
- Parent/child size contract
- Overlap policy
- Metadata and breadcrumb contract
- Validation gates
- Ablation procedure

---

## 2. Design Rationale

Chunking determines what the retriever can return. Structure-aware chunking accounts for 73–88% of retrieval gain in legal RAG, with +104% Char-Recall@8 over recursive baselines.

**Key findings:**
- Zero overlap outperforms 10–15% overlap (89.19% vs 83.78%)
- Paragraph-group chunking is strongest in legal domain
- Splitting citations destroys cross-references (0% preserved vs 97.6%)
- 1,200-token parents / 350-token children matches legal provision length (300–700 chars)

---

## 3. Pipeline

```
opinion.text
   │
   ▼
[0] Normalize                → strip headers/footers, collapse whitespace
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
[4] Protect atomic spans     → move boundary if cut falls inside citation/quote/statute
   │
   ▼
[5] Attach breadcrumb        → prepend hierarchy header to every child
   │
   ▼
[6] Emit two text fields     → text (breadcrumb + body) for BM25
                               text_original (body only) for dense ablation
   │
   ▼
[7] Validate                 → §7 gates
   │
   ▼
[8] Write to chunks
```

### 3.1 Section Detection

Ordered separator hierarchy:

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

Section markers with retrieval boost weight:

| Section Type | Detection Signal | Priority |
|---|---|---|
| `conclusion` | `CONCLUSION`, `We therefore`, `Accordingly` | 95 |
| `judgment` / `disposition` | `It is so ordered`, `Reversed`, `Affirmed` | 90 |
| `holding` | `We hold`, `We conclude` | 88 |
| `analysis` / `discussion` | `I.`, `II.`, `III.`, `Discussion`, `Analysis` | 85 |
| `facts` / `background` | `Facts`, `Background`, `I. Background` | 70 |
| `syllabus` / `caption` | leading block, `<parties>`, docket line | 40 |
| `footnote` | `*` / `1` at line start, dense citation | 20 |

Priority stored as `section_priority`, used as RRF tie-breaker only.

### 3.2 Parents

Complete structural sections.

| Parameter | Value |
|---|---|
| Target | 1,200 tokens |
| Hard cap | 1,600 tokens |
| Minimum | 600 tokens (merge with adjacent if smaller) |
| Overlap | **0 tokens** |
| One parent per | `(cl_opinion_id, opinion_type, section_index)` |

Stored with `parent_id IS NULL`, not embedded. Exist only for generation-time expansion.

### 3.3 Children

Retrieval units: paragraphs or paragraph groups inside a parent.

| Parameter | Value |
|---|---|
| Target | 350 tokens |
| Hard cap | 450 tokens |
| Minimum | 180 tokens (merge with adjacent if smaller) |
| Overlap | **0 tokens** (32 tokens only for mechanical sentence splits) |
| One child per | `(parent_id, paragraph_index)` |

### 3.4 Atomic Spans

Never split inside:

| Span | Pattern |
|---|---|
| Case citation | `\d+\s+[A-Z][A-Za-z.']*\s+\d+` |
| Statutory citation | `\d+\s+U\.S\.C\.\s+§+\s*\d+[\w().]*` |
| Section reference | `§+\s*\d+[\w().-]*` |
| Docket number | `No\.\s*\d{2}-\d+` |
| Currency / amount | `\$\s*\d[\d,]*(\.\d+)?` |
| Date | `\b(19\|20)\d{2}\b`, `\b\d{1,2}/\d{1,2}/\d{2,4}\b` |
| Block quotation | text between line ending `:` and closing citation |

Move boundary to nearest safe point if cut would fall inside.

### 3.5 Breadcrumb

Every child begins with:

```
[{case_name_short}, {citation} ({year}) · {court} · {opinion_type} · § {section_path}]
```

Example:
```
[Roe v. Wade, 410 U.S. 113 (1973) · scotus · majority · § III.B]
The Court held that the right of privacy...
```

### 3.6 Text Fields

| Field | Content | Used For |
|---|---|---|
| `text` | breadcrumb + body | `tsvector` / BM25, citation display |
| `text_original` | body only | Dense embedding, character-offset highlighting |

### 3.7 Opinion Types

| `opinion_type` | Behavior |
|---|---|
| `majority`, `plurality`, `per_curiam` | Normal treatment |
| `concurrence` | Normal treatment |
| `dissent` | Chunked and labelled; excluded from binding-authority queries at retrieval time |

### 3.8 Deterministic Chunk IDs

```
chunk_id = f"{cl_opinion_id}:{opinion_type}:{section_index}:{child_index}"
```

Pure function of `(document, section position, child position)`. Makes re-ingestion idempotent.

Parent link: `parent_id = f"{cl_opinion_id}:{opinion_type}:{section_index}"`

### 3.9 Versioning

```python
CHUNKER_VERSION = "chunk-v1"
```

Any parameter change requires new `chunker_version` and full re-embed. Different versions must never coexist in same HNSW index.

---

## 4. Final Parameters

| Parameter | Value |
|---|---|
| Splitter | Recursive, structure-aware separator hierarchy |
| Section detection | Marker + separator hybrid, 15 section types |
| Parent target / cap / min | 1,200 / 1,600 / 600 tokens |
| Child target / cap / min | 350 / 450 / 180 tokens |
| Overlap (structure boundary) | **0 tokens** |
| Overlap (mechanical split) | 32 tokens |
| Breadcrumb | `[case, citation (year) · court · type · § path]` |
| Embedding input | `text` (breadcrumb + body) |
| Lexical input | `text` |
| Atomic spans | Citation, statute, docket, amount, date, quote |
| Dissents | Chunked, labelled, filtered at query time |
| `chunker_version` | `chunk-v1` |

---

## 5. Schema

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
    embedding       halfvec(384),
    tsv             TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,

    chunker_version TEXT NOT NULL DEFAULT 'chunk-v1',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_chunks_tsv       ON chunks USING gin (tsv);
CREATE INDEX idx_chunks_case      ON chunks (case_id);
CREATE INDEX idx_chunks_parent    ON chunks (parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX idx_chunks_court_date ON chunks (court, date);
CREATE UNIQUE INDEX uq_chunks_text_hash
    ON chunks (case_id, text_hash) WHERE is_parent = FALSE;
```

`parent_id` is `TEXT` for idempotent re-ingestion.

---

## 6. Validation Gates

Corpus publishable only if all gates pass. Run via `scripts/validate_chunks.py` in CI.

| # | Gate | Check | Threshold |
|---|---|---|---|
| 1 | Parent coverage | All non-parent chunks have existing parent | 100% |
| 2 | Breadcrumb presence | regex `^\[.+, .+\(\d{4}\) · .+ · .+ · § ` on `text` | 100% |
| 3 | Boundary integrity | `text_original` ending mid-sentence | ≤ 2% |
| 4 | Citation integrity | `\d+ U\.S\. \d+` split across chunk boundary | **0** |
| 5 | Size distribution | p50 / p95 / max token count for children | p50 ≤ 400, p95 ≤ 450, max ≤ 600 |
| 6 | Duplicate content | duplicate `(case_id, text_hash)` rows | **0** |
| 7 | Section coverage | detected sections wholly contained in exactly one parent | ≥ 98% |
| 8 | Determinism | re-run chunker on 100 random opinions, compare `chunk_id` sets | identical |

Gate 4 is non-negotiable: a split citation is unrecoverable at retrieval time.

---

## 7. Ablation

Every published corpus accompanied by ablation results on `eval_isolated/` dataset.

| Config | Strategy | Purpose |
|---|---|---|
| **A** | Fixed 512 tokens, 0 overlap | Floor baseline |
| **B** | Recursive char splitting, 512 tokens, 0 overlap | LegalBench-RAG best baseline |
| **C** | **Structure-aware + parent-child (this document)** | The proposal |
| **D** | C, but dense embeds `text_original` (no breadcrumb) | Tests metadata injection |
| **E** | C, but child overlap ∈ {0, 32, 64} | Tests overlap default |
| **F** | C, but child size ∈ {256, 350, 512} | Tests size target |

**Primary metric:** Char-Recall@8 and Recall@10. **Secondary:** nDCG@10, p50/p95 ingestion time, index size.

**Kill criteria:**

| Condition | Action |
|---|---|
| C does not beat B by ≥10% Char-Recall@8 | Revert to B |
| C does not beat B by ≥3 points Recall@10 | Revert to B |
| D beats C by ≥5% on legal-identity queries | Drop breadcrumb from embedding input |
| E shows overlap > 0 beats 0 by ≥3 points | Promote that overlap value to default |
| F shows 256 or 512 beats 350 by ≥5% | Adopt that value |
| Gate 3 or Gate 4 fails at scale | Block publication; fix the splitter |

C will have lower character-level precision@1 than B (complete provisions vs minimal spans). This is the intended trade-off.

---

## 8. Rejected Strategies

| Strategy | Why Rejected |
|---|---|
| Fixed-size token windows | 7.7% boundary precision; severs argument structure |
| Pure semantic chunking | 54% accuracy vs 69% for recursive 512; fragments averaging 43 tokens |
| Single-sentence chunks | Gold spans are 3–5 sentences; isolated sentences lose the holding |
| Default 10–15% overlap | Zero overlap scored 89.19% vs 83.78% / 79.73% |
| Single-level chunking (no parent) | Cannot serve retrieval precision and generation context simultaneously |
| Metadata only in columns | Invisible to embedding; vector cannot carry legal identity |
| Splitting tables, citations, or block quotes | Cross-references destroyed; unrecoverable at retrieval time |

---

## 9. Operational Notes

**Ingestion order:** Load all rows with `embedding = NULL` first, create HNSW index after bulk load, then backfill embeddings.

**Re-embedding:** Change to `CHUNKER_VERSION` or embedding model creates new index. Write to `chunks_v2`, evaluate, then switch read alias.

**Idempotency:** `chunk_id` is deterministic, so `ON CONFLICT (chunk_id) DO UPDATE` is safe.

**Cost:** At `medium` scale (100K opinions), expect ~500K children at 350 tokens each. At `max` (400K opinions), ~2M children.

---

## 10. Summary

Structure-aligned recursive splitting; 1,200-token parents and 350-token children; zero overlap except when a single sentence exceeds the cap; every child carries a case/citation/section breadcrumb in its text; citations, statutes and block quotes are atomic; deterministic chunk IDs; eight validation gates; six-arm ablation with explicit kill criteria.
