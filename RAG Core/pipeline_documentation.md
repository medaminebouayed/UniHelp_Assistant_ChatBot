# IIT/ISB Graph RAG Pipeline — Full Technical Documentation

> Complete reference for the data pipeline built across this conversation.
> Covers architecture, every script, design decisions, and lessons learned.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Data Sources](#3-data-sources)
4. [Phase 1 — Data Cleaning](#4-phase-1--data-cleaning)
5. [Phase 2 — Chunking](#5-phase-2--chunking)
6. [Phase 3 — Data Augmentation](#6-phase-3--data-augmentation)
7. [Phase 4 — Synthetic Generation](#7-phase-4--synthetic-generation)
8. [Phase 5 — Graph Extraction (Neo4j)](#8-phase-5--graph-extraction-neo4j)
9. [Phase 6 — Embedding (ChromaDB)](#9-phase-6--embedding-chromadb)
10. [Phase 7 — BM25 Keyword Index](#10-phase-7--bm25-keyword-index)
11. [Retrieval Pipeline](#11-retrieval-pipeline)
12. [RAG Types Used](#12-rag-types-used)
13. [File Structure](#13-file-structure)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Running the Full Pipeline](#15-running-the-full-pipeline)
16. [Known Limitations](#16-known-limitations)

---

## 1. Project Overview

A **Hybrid Graph RAG system** built for IIT/ISB Tunisian university program data.

**Goal**: answer complex student queries across two schools:

- *"Quels Masters en Marketing Digital ont des partenariats avec des universités suisses ?"*
- *"Quel est le coût du programme 3iL Ingénieurs depuis la Tunisie ?"*
- *"Quels modules GC1 sont enseignés au semestre S1 en Génie Civil ?"*

**Stack**:

| Component | Technology |
|---|---|
| LLM (local) | Qwen2.5:7b via Ollama |
| Graph DB | Neo4j (bolt://localhost:7687) |
| Vector DB | ChromaDB |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 |
| Keyword index | rank-bm25 (BM25Okapi) |
| Orchestration | LangChain |
| Hardware | RTX 5070 8GB VRAM, i9-14900HX, 32GB RAM |

---

## 2. Architecture

```
Cleaned text files
        │
        ▼
    Chunking (SemanticDataChunker v4)
        │
        ├─────────────────────────┐
        ▼                         ▼
 Data Augmentation           Synthetic Generation
 (augment_metadata.py)      (synthetic_chunks.py)
        │                         │
        ▼                         ▼
 *_augmented.json         *_synthetic.json
        │
        ├──────────────────────────┐
        ▼                          ▼
  Graph Extraction            Embedding to PostgreSQL (PGVector)
  (kge_engine.py → Neo4j)    (embed_to_postgreSQL.py)
        │                          │
        ▼                          ▼
  :Chunk nodes + MENTIONS      Vector index
  :Institution, :Program...
        │
        ▼
    BM25 Index (build_bm25_index.py)
        │
        ▼
  Hybrid Retriever
  BM25 + Vector + Graph → RRF → Reranker → LLM
```

---

## 3. Data Sources

**IIT** (Institut des Ingénieurs de Tunis):
- `iit/scraped_data_cleaned/` — 1061 chunks from PDFs + website scraping
- Categories: `mobilite_internationale` (744), `wp_content` (176), `Website Content` (141)
- Language: 99% French

**ISB** (International School of Business):
- `isb/scraped_data_cleaned/` — 1074 chunks
- Categories: `international` (784), `Website Content` (288), `wp_content` (2)
- Language: 96% French

**Total real corpus**: 2135 chunks, avg 162 tokens, all 30–200 tokens.

---

## 4. Phase 1 — Data Cleaning

### SmartDeduplicator (`utils/preprocessing/`)

Removes redundancy at 3 levels:

```python
# Level 1: Repeated words (regex)
remove_repeated_words("le le chat est est là") → "le chat est là"

# Level 2: N-gram deduplication (adaptive threshold = mean + 2×std)
# Auto-detects repeated phrases without prior knowledge

# Level 3: Semantic deduplication (sentence-transformers + cosine)
# Keeps the most informative sentence per cluster
```

**Key insight**: `paraphrase-multilingual-MiniLM-L12-v2` handles French + English in one model.

---

## 5. Phase 2 — Chunking

### SemanticDataChunker v4 (`chunk_data.py`)

```python
chunker = SemanticDataChunker(chunk_size=200, chunk_overlap=30)
# chunk_size and chunk_overlap in TOKENS (not chars)
```

**Critical design decisions**:

1. **Token-based sizing** — uses `tiktoken (cl100k_base)` for accurate measurement. French text has ratio ~3.1 chars/token vs English ~4.5. Character-based sizing causes 33% overflow for French content.

2. **`from_tiktoken_encoder()`** — passes token budgets directly to LangChain, eliminating the chars/token conversion ratio entirely.

3. **Effective chunk size** — breadcrumb tokens are subtracted from budget:
   ```python
   breadcrumb_tokens = count_tokens(current_breadcrumb)   # ~11-15 tokens
   available_tokens  = max(50, chunk_size - breadcrumb_tokens)
   ```

4. **Section detection** — `get_sections()` detects academic section headers (`Objectifs`, `Admissions`, `Frais`, uppercase lines, colon-terminated lines).

5. **Breadcrumb prefix** — every chunk starts with `[SITE | TITLE] SECTION > ` for provenance tracking.

**Output format** (`iit_chunks.json`):
```json
{
  "page_content": "[IIT | Master Marketing Digital] Objectifs > ...",
  "metadata": {
    "site": "IIT", "title": "Master Marketing Digital",
    "category": "Website Content", "url": "...",
    "header": "Objectifs", "chunk_length": 584, "token_count": 198
  }
}
```

---

## 6. Phase 3 — Data Augmentation

### augment_metadata.py

Enriches each chunk with 4 LLM-generated fields using Qwen2.5:7b via Ollama.

**3 content-type variants** (auto-detected):

| Type | Detection | Focus |
|---|---|---|
| `study_plan` | module codes regex `[A-Z]{2}\d[-\s]\d{3}` | Module codes, semester labels, specializations |
| `mobility` | category contains `mobilite` or `international` | Partner universities, costs, deadlines |
| `general` | everything else | Programs, accreditations, key facts |

**Output fields per chunk**:

```json
{
  "metadata": {
    "questions"    : ["4 student queries in French"],
    "keywords"     : ["GC1 101", "Génie Civil", "S1", "Bac+3", ...],
    "section_title": "Plan d'études Génie Civil 1ère année",
    "summary"      : "1-2 sentence dense summary",
    "content_type" : "study_plan"
  },
  "enriched_content": "section_title + summary + questions + keywords + original_text",
  "bm25_text"       : "section_title + summary + keywords (NO breadcrumb)"
}
```

**Key design decisions**:

- `temperature=0` — deterministic output, safe to re-run failed chunks
- `format="json"` — Ollama's native JSON mode forces valid JSON at sampling level
- `extract_section_title()` — accepts 6 key name variants (`section_title`, `titre_section`, `titre`, `title`, `section`, `heading`) because Qwen sometimes renames keys
- Broken headers fixed: `"Plan D"` → `"Plan d'études Génie Civil 1ère année"` (LLM reads content and reconstructs)
- Checkpoint every 20 chunks, resume with `--start-from N`

**`enriched_content` vs `page_content`**:

- `enriched_content` → embedded into ChromaDB (3-4× more signal)
- `bm25_text` → indexed by BM25 (no breadcrumb, clean keyword signal)
- `page_content` → passed to LLM for answer generation (original, never modified)

---

## 7. Phase 4 — Synthetic Generation

### synthetic_chunks.py

Generates N synthetic chunks by rewriting existing augmented chunks in 5 styles.

**5 rewrite styles**:

| Style | Weight | Purpose | Questions extraction |
|---|---|---|---|
| `query` | 25% | Google-style search queries | Bullet lines (light filter) |
| `qa` | 40% | Q/A pairs | Flexible regex (strict filter) |
| `reformulate` | 20% | Same info, different phrasing | Inherited from source |
| `student` | 8% | First-person casual narrative | Inherited from source + bonus |
| `structured` | 7% | Title + bullets + paragraph | Inherited from source + bonus |

**Quality gates (applied in order)**:

1. **Length gate**: `len(generated) >= 60 chars`
2. **Token cap (REFINEMENT 1)**: `generated_clean` truncated to 720 chars (~180 tokens) at sentence boundary — prevents silent truncation by embedding model (all-MiniLM-L6-v2 max: 256 tokens)
3. **Similarity gate**: per-style dynamic windows with hard floor 0.48
4. **Deduplication**: sliding window of last 300 accepted embeddings (O(300) vs O(N))

**Per-style similarity windows**:
```python
SIM_THRESHOLDS = {
    "query"      : (0.48, 0.88),
    "qa"         : (0.58, 0.88),
    "reformulate": (0.65, 0.90),
    "student"    : (0.55, 0.85),
    "structured" : (0.60, 0.88),
}
```

**Source saturation fix** (key engineering decision):
The old code: `source = random.choices(all_sources)` → `if saturated: continue` caused 93% of loop iterations to be empty. The fix: build `free_indices = [i for i where usage[i] < MAX_PER_SOURCE]` and sample only from those. When all saturated, reset counters.

**Adaptive threshold relaxation**: if LLM acceptance < 25%, relax lower bounds by 0.03 every 50 calls (max 0.10 total relaxation).

**QA pair storage (IMP 1)**: structured `{"q": ..., "a": ...}` pairs stored in `metadata["qa_pairs"]` in addition to flat `metadata["questions"]`. Answers also included in `enriched_content` for better vector retrieval.

**Noise injection** (`--noise` flag): 15% of query-style chunks get realistic Tunisian search typos (char swap/drop/repeat at 8% word rate). Applied **after** question extraction to avoid corrupting bullet format.

**Scalability**: 2135 real sources × MAX_PER_SOURCE=8 = 17,080 max synthetic. 10k achievable overnight (~12-15 hours on RTX 5070).

**Production run command**:
```bash
py utils/augmentation/synthetic_chunks.py \
    --count 10000 \
    --merge \
    --noise \
    --checkpoint-every 100 \
    --output data/synthetic/full_10k.json
```

**Good result benchmark** (from test_200.json):
```
Questions filled  : 200/200 chunks  avg 3.7  ✅
QA pairs filled   : 76/200, avg 1.0 pairs    (aim for 2.0+)
Noised chunks     : 8 (4.0%)                 ✅ below 5%
Avg sim score     : 0.706                    ✅ healthy
Styles            : qa=38%, query=30%, reformulate=18%  ✅
Token count       : should be max ~180 after refinement 1
```

---

## 8. Phase 5 — Graph Extraction (Neo4j)

### kge_engine.py

Extracts entities and relationships from augmented chunks into Neo4j.

**Node schema** (matched to real IIT/ISB corpus):

| Node type | Examples |
|---|---|
| `Institution` | IIT, ISB, 3iL Ingénieurs, CESI, EPF, Kedge |
| `Program` | Génie Civil, Master Marketing Digital, MBA, MSc |
| `Module` | GC1 101, Analyse numérique, BIM |
| `Level` | Licence, Master, Bac+3, Bac+5, Cycle Ingénieur |
| `PartnerCountry` | France, Espagne, Canada |
| `City` | Paris, Lyon, Toulouse, Limoges |
| `Cost` | 670€/mois, 26000€/an |
| `Skill` | Python, SEO, CAO, BIM, DELF |
| `Accreditation` | CGE, AACSB, ISO 9001 |
| `Chunk` | Source chunk linked to entities |

**Relationship schema**:

```
OFFERS          → Institution → Program
HAS_LEVEL       → Program → Level
HAS_MODULE      → Program → Module
PARTNERS_WITH   → Institution → Institution
LOCATED_IN      → Institution → City
COSTS           → Program → Cost
REQUIRES_SKILL  → Program/Module → Skill
ACCREDITED_BY   → Institution/Program → Accreditation
MENTIONS        → Chunk → any entity (key Graph RAG link)
MENTIONED_IN    → any entity → Chunk (bidirectional)
```

**Why `MENTIONS` is critical**: Without it, the graph can answer "IIT partners with 3iL" but cannot retrieve the source text that contains the conditions. With it, every graph fact is backed by a retrievable source chunk.

**LLM prompt context**: uses `summary` + `keywords` from augmented metadata as context hints — better extraction than raw OCR text alone.

**Canonical name normalization**: `"Carte d'identité"` and `"Photocopie CIN"` → both `canonical_name: "CIN"` to prevent duplicate nodes.

**Checkpoint**: uses `_kge_done.json` sidecar file with stable chunk IDs for safe resume.

---

## 9. Phase 6 — Embedding (PostgreSQL / PGVector)

**Critical**: embed `enriched_content`, NOT `page_content`.

```python
# WRONG — embeds garbled OCR text
collection.add(documents=[chunk["page_content"]], ...)

# CORRECT — embeds clean, enriched signal
collection.add(documents=[chunk["enriched_content"]], ...)
```

`enriched_content` = `section_title + summary + questions + qa_answers + keywords + original_text`

This gives the embedding model ~4× more semantic signal than the raw chunk text.

---

## 10. Phase 7 — BM25 Keyword Index

### build_bm25_index.py

```python
# Index bm25_text, NOT page_content
# Reason: breadcrumb "[IIT | ...]" appears in every chunk
# and would pollute BM25 scores equally for everything

bm25_text = chunk.get("bm25_text", "")
# = section_title + summary + keywords (no breadcrumb)
```

**French-aware tokenizer**:
```python
tokens = re.findall(r"[a-zéàèùâêîôûçœæ0-9][a-zéàèùâêîôûçœæ0-9\-]*", text.lower())
# Keeps module codes: "GC1", "GM2-5", "BIM"
# Handles French accents: "étude", "spécialité"
```

---

## 11. Retrieval Pipeline

### Query time flow

```
User query
    │
    ├── Entity Linking → identify entities (IIT, Master IA, etc.)
    │
    ├── Graph Traversal (Neo4j Cypher)
    │   MATCH (u:University)-[:PARTNER_WITH]->(p:University)
    │   MATCH (u)-[:OFFERS]->(c:Course {domain: "Digital Marketing"})
    │   RETURN c.name, u.name, c.page_content
    │
    ├── BM25 Search (exact keyword matching)
    │   bm25.get_scores(query.split())
    │   → finds "IIT", "730€", "GC1 101" exactly
    │
    ├── Vector Search (ChromaDB cosine similarity)
    │   collection.query(query_embeddings=[q_vector], n_results=20)
    │   → finds semantically similar content
    │
    ├── RRF Fusion (Reciprocal Rank Fusion, k=60)
    │   score = 1/(rank_bm25 + k) + 1/(rank_vec + k)
    │
    ├── Reranker (BAAI/bge-reranker-v2-m3)
    │   top-20 → scored → top-5 most relevant
    │
    └── LLM Synthesis
        context = graph_facts + top-5 chunks
        answer = llm.invoke(context + question)
```

### Why 3 retrieval branches

| Branch | Catches | Misses |
|---|---|---|
| Graph RAG | Multi-hop structured queries | Vague questions |
| BM25 | Exact terms: codes, costs, names | Paraphrases, synonyms |
| Vector RAG | Semantic similarity | Exact numbers, proper nouns |

Together: ~90%+ recall vs ~55% for vector alone.

---

## 12. RAG Types Used

| Type | Status | Why |
|---|---|---|
| Vector RAG | ✅ Implemented | Semantic similarity base |
| Graph RAG | ✅ Implemented | Multi-hop structured queries |
| Hybrid (BM25+Vector) | ✅ Implemented | Exact term recall |
| Reranker | ✅ Recommended | Eliminate IIT/ISB confusion in top-5 |
| HyDE | 🔶 Optional | Short query improvement |
| Self-RAG | ❌ Skip | Adds latency, closed corpus doesn't need it |
| Agentic RAG | ❌ Skip | Over-engineering for this use case |
| Multimodal | ❌ Skip | Unless PDFs have important diagrams |

---

## 13. File Structure

```
PROJECT_ROOT/
├── data/
│   ├── augmented/
│   │   ├── iit_chunks_augmented.json    ← 1061 chunks + questions/keywords/summary
│   │   └── isb_chunks_augmented.json    ← 1074 chunks + augmentation
│   ├── raw/
│   │   ├── iit_chunks.json              ← original chunks (no augmentation)
│   │   └── isb_chunks.json
│   └── synthetic/
│       ├── test_200.json                ← test synthetic run
│       └── full_10k.json                ← production corpus
├── iit/scraped_data_cleaned/            ← source .txt files
├── isb/scraped_data_cleaned/
├── utils/
│   ├── augmentation/
│   │   ├── augment_metadata.py           ← Phase 3: augmentation
│   │   ├── synthetic_chunks.py          ← Phase 4: synthetic generation
│   │   └── clean_and_patch.py           ← post-processing cleaner
│   ├── graph/
│   │   └── kge_engine.py                ← Phase 5: Neo4j extraction
│   ├── preprocessing/
│   │   ├── chunk_data.py                ← Phase 2: chunking
│   │   └── data_cleaner.py              ← Phase 1: deduplication
│   └── evaluation/
├── .env                                 ← OLLAMA_URL, HF_TOKEN, NEO4J_PASSWORD
└── requirements.txt
```

---

## 14. Key Design Decisions

### Why Qwen2.5:7b locally instead of GPT-4o-mini

- Hardware: RTX 5070 8GB VRAM → 7B models run at 40-50 tok/s fully on GPU
- Cost: free vs ~$0.50 per 2135-chunk augmentation run
- Privacy: IIT/ISB data never leaves the machine
- Quality: Qwen2.5 was specifically trained for structured JSON output and multilingual French

### Why `enriched_content` ≠ `page_content`

The embedding model compresses a chunk into a single 384-dimensional vector. Raw chunk text from OCR/scraping contains noise, truncated headers, and lacks context. `enriched_content` provides 4× more signal by concatenating summary + questions + keywords + original — giving the embedding model a much more precise target.

### Why BM25 indexes `bm25_text` not `page_content`

The breadcrumb `[IIT | plan-detude-2025 2026-GC] Plan D >` appears in every single IIT study plan chunk. If indexed, BM25 would assign equal scores for breadcrumb tokens across all chunks, destroying precision. `bm25_text = summary + keywords` (no breadcrumb) gives clean keyword scores.

### Why token counting matters for French

```
"d'Intelligence Artificielle" → 4 tokens (French, accents fragment)
"Artificial Intelligence"     → 2 tokens (English, larger chunks learned)
```

`chunk_size=800 chars` sounds safe but contains ~267 tokens of French text, silently exceeding the 200-token target. `chunk_size=200 tokens` via `from_tiktoken_encoder()` is exact regardless of language.

### Why source saturation was the real bottleneck

```
Old loop: random.choices(all_182_sources) → if saturated → continue
→ 93% of iterations: empty skip (no LLM call)
→ reported "5% acceptance" was actually 68% of real LLM calls

Fix: sample only from free_sources, reset when all saturated
→ every loop iteration does real work
→ scales to 10k+ without MAX_ATTEMPTS hitting
```

---

## 15. Running the Full Pipeline

### Step 1: Environment

```bash
pip install langchain-text-splitters tiktoken sentence-transformers
pip install rank-bm25 neo4j requests tqdm python-dotenv scikit-learn
```

`.env` file:
```
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b
HF_TOKEN=hf_your_token_here
NEO4J_PASSWORD=graphrag123
```

### Step 2: Chunking

```bash
py utils/preprocessing/chunk_data.py
# Output: data/raw/iit_chunks.json, data/raw/isb_chunks.json
```

### Step 3: Augmentation

```bash
py utils/augmentation/augment_metadata.py --site iit
py utils/augmentation/augment_metadata.py --site isb
# Output: data/augmented/iit_chunks_augmented.json, isb_chunks_augmented.json
```

### Step 4: Synthetic generation

```bash
# Test first (50 chunks, ~4 min)
py utils/augmentation/synthetic_chunks.py --count 50 --output data/synthetic/test.json

# Production run (10k, overnight)
py utils/augmentation/synthetic_chunks.py \
    --count 10000 --merge --noise \
    --checkpoint-every 100 \
    --output data/synthetic/full_10k.json
```

### Step 5: Clean and patch

```bash
py utils/augmentation/clean_and_patch.py all \
    --synthetic data/synthetic/full_10k.json \
    --originals data/augmented/iit_chunks_augmented.json \
               data/augmented/isb_chunks_augmented.json \
    --output data/synthetic/corpus_final.json
```

### Step 6: Graph extraction

```bash
# Neo4j must be running (localhost:7687)
py utils/graph/kge_engine.py --site both --limit 100  # test first
py utils/graph/kge_engine.py --site both               # full run
py utils/graph/kge_engine.py --stats-only              # verify
```

### Step 7: Embed to ChromaDB

```bash
# Use enriched_content field
py utils/embed_to_postgreSQL.py --corpus data/synthetic/corpus_final.json
```

### Step 8: Build BM25 index

```bash
py utils/build_bm25_index.py  # reads bm25_text field, saves bm25_index.pkl
```

---

## 16. Known Limitations

**Token count in synthetic chunks**: `token_count` in metadata is estimated as `len(generated_clean) // 4` (character approximation). For precise counts, use tiktoken post-generation. The actual embedding is capped by the model at 256 tokens regardless.

**Only 100 chunks per source file tested**: augmentation scripts were validated on 100-chunk subsets. Full 2135-chunk runs may reveal edge cases in `get_sections()` for unusual document structures.

**QA pairs avg 1.0** (from test runs): The QA prompt sometimes generates only 1 pair per chunk instead of 3-4. The flexible regex extractor in the final version improves this but cannot fix cases where Qwen generates free-form text instead of structured Q/R pairs. Expected with full augmented corpus: avg 2.0-2.5 pairs.

**Mobility category underrepresented in synthetic**: `content_type=study_plan` dominates (54%) because study plan chunks have higher token counts and get higher sampling weights. If mobility retrieval is critical, use `--type mobility` for a dedicated synthetic run.

**Neo4j graph quality**: Graph extraction was run on raw chunks (before augmentation in the original pipeline). Re-running `kge_engine.py` on augmented chunks and using `clean_and_patch.py` to MERGE missing entities into the existing graph will improve quality without requiring a full rebuild.

---

*Documentation generated from the full pipeline development conversation. Last validated against test_200.json (200 chunks, 100% questions filled, 4% noise, avg sim 0.706).*
