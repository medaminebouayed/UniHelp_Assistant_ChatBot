@echo off
REM ============================================================================
REM  run_pipeline.bat — Full IIT/ISB Graph RAG Pipeline
REM ============================================================================
REM  What this does:
REM    1. Merges real + synthetic chunks into a unified corpus_final.json
REM    2. Builds a BM25 keyword index from bm25_text
REM    3. Embeds all chunks into PostgreSQL (PGVector)
REM    4. Extracts a Knowledge Graph into Neo4j
REM    5. Runs hybrid RAG evaluation (dense + BM25 + reranker)
REM    6. Runs A/B retrieval evaluation (V1 vs V2) with intrinsic quality check
REM
REM  Prerequisites:
REM    - PostgreSQL running with PGVector extension
REM    - Neo4j running on bolt://localhost:7687
REM    - Ollama running with qwen2.5:7b
REM    - pip install -r requirements.txt
REM
REM  Usage:
REM    run_pipeline.bat              (full pipeline)
REM    run_pipeline.bat --quick      (skip graph + eval, just build corpus)
REM ============================================================================

setlocal enabledelayedexpansion

REM ── Optional: skip graph + eval for quick iteration ─────────────────────────
set QUICK_MODE=0
if "%1"=="--quick" set QUICK_MODE=1

echo ============================================================================
echo  IIT/ISB Graph RAG Pipeline
echo  Started: %DATE% %TIME%
echo  Quick mode: %QUICK_MODE%
echo ============================================================================

REM ── STEP 1: Build unified corpus ────────────────────────────────────────────
REM  Merges real augmented chunks (iit + isb) with synthetic chunks (full_10kv2).
REM  Patches schema inconsistencies: adds qa_pairs, has_noise, is_synthetic, etc.
echo.
echo [1/6] Merging real + synthetic into corpus_final.json...
python utils/postprocessing/clean_and_patch.py all ^
    --synthetic data/synthetic/full_10kv2.json ^
    --originals data/augmented/iit_chunks_augmented.json ^
                data/augmented/isb_chunks_augmented.json ^
    --output data/synthetic/corpus_final.json
if %ERRORLEVEL% neq 0 echo [ERROR] Step 1 failed && exit /b %ERRORLEVEL%

REM ── STEP 2: Build BM25 index ────────────────────────────────────────────────
REM  Indexes bm25_text (NOT page_content) with French-aware tokenizer.
REM  Output: data/bm25_index.pkl
echo.
echo [2/6] Building BM25 keyword index...
python utils/build_bm25_index.py --corpus data/synthetic/corpus_final.json
if %ERRORLEVEL% neq 0 echo [ERROR] Step 2 failed && exit /b %ERRORLEVEL%

REM ── STEP 3: Embed to PostgreSQL (PGVector) ──────────────────────────────────
REM  Embeds enriched_content (NOT page_content) using all-MiniLM-L6-v2.
REM  Stores 384-dim vectors in the `chunks` table for cosine similarity search.
REM  --rebuild truncates the table before inserting.
echo.
echo [3/6] Embedding chunks into PostgreSQL...
python utils/embed_to_postgreSQL.py --corpus data/synthetic/corpus_final.json --rebuild
if %ERRORLEVEL% neq 0 echo [ERROR] Step 3 failed && exit /b %ERRORLEVEL%

REM ── STEP 4: Knowledge Graph Extraction (Neo4j) ──────────────────────────────
REM  Extracts entities (Institution, Program, Module, etc.) and relationships
REM  from each chunk using Qwen2.5:7b, then writes to Neo4j.
REM  Each Chunk node is linked to extracted entities via :MENTIONS edges.
if %QUICK_MODE%==0 (
    echo.
    echo [4/6] Extracting Knowledge Graph to Neo4j...
    python utils/graph/kge_engine.py --input data/synthetic/corpus_final.json
    if !ERRORLEVEL! neq 0 echo [ERROR] Step 4 failed && exit /b !ERRORLEVEL!
) else (
    echo.
    echo [4/6] Skipping Neo4j graph extraction (--quick mode)...
)

REM ── STEP 5: Hybrid RAG Evaluation ───────────────────────────────────────────
REM  Tests dense (E5), hybrid (dense + BM25 + RRF), and reranked retrieval.
REM  Uses 300 stratified queries from real chunks (study_plan/general/mobility).
if %QUICK_MODE%==0 (
    echo.
    echo [5/6] Running Hybrid RAG evaluation...
    python utils/evaluation/evaluate_hybrid_rag.py ^
        --corpus data/synthetic/corpus_final.json ^
        --real both --queries 300 --debug-bm25
    if !ERRORLEVEL! neq 0 echo [ERROR] Step 5 failed && exit /b !ERRORLEVEL!
) else (
    echo.
    echo [5/6] Skipping hybrid evaluation (--quick mode)...
)

REM ── STEP 6: A/B Retrieval Evaluation ────────────────────────────────────────
REM  Compares V1 (augmented only) vs V2 (merged corpus) dense retrieval.
REM  --intrinsic-quality prints near-dup rates, factual recall, filter sim stats.
if %QUICK_MODE%==0 (
    echo.
    echo [6/6] Running A/B retrieval evaluation...
    python utils/evaluation/evaluate_v1_v2_rag.py ^
        --v1 data/augmented/iit_chunks_augmented.json ^
        --v2 data/synthetic/corpus_final.json ^
        --real both --max-queries 300 --intrinsic-quality
    if !ERRORLEVEL! neq 0 echo [ERROR] Step 6 failed && exit /b !ERRORLEVEL!
) else (
    echo.
    echo [6/6] Skipping A/B evaluation (--quick mode)...
)

echo.
echo ============================================================================
echo  Pipeline complete!  %DATE% %TIME%
echo  Corpus: data/synthetic/corpus_final.json
echo  BM25:   data/bm25_index.pkl
echo  Neo4j:  run "python utils/graph/kge_engine.py --stats-only" to verify graph
echo  Eval:   check output above for metrics
echo ============================================================================
endlocal
