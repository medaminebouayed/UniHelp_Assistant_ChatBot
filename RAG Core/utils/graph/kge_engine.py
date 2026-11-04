"""
kge_engine.py — Knowledge Graph Extraction for IIT/ISB Graph RAG
=================================================================
Designed specifically for the real IIT/ISB Tunisia corpus:
  - 3 content types: study_plan, mobility, general
  - Uses augmented chunks (summary + keywords as context hints)
  - Creates :Chunk nodes linked to entities via :MENTIONS
  - Canonical name normalization to prevent duplicate nodes
  - Resume/checkpoint support for 2135 chunks
  - Schema matched to real entities found in the corpus

Node types (matched to actual corpus content):
  Institution     → IIT, ISB, 3iL Ingénieurs, CESI, EPF, Kedge...
  Program         → Génie Civil, Master Marketing, MBA, MSc...
  Module          → GC1 101, Analyse numérique, BIM...
  Level           → Licence, Master, MSc, Bac+3, Bac+5...
  PartnerCountry  → France, Espagne, Canada...
  City            → Paris, Lyon, Toulouse...
  Cost            → 670€/mois, 26000€...
  Skill           → Python, SEO, CAO, BIM...
  Accreditation   → CGE, AACSB, ISO...
  Chunk           → source chunk linked to entities

Relationships:
  OFFERS          → Institution → Program
  HAS_LEVEL       → Program → Level
  HAS_MODULE      → Program → Module
  PARTNERS_WITH   → Institution → Institution
  LOCATED_IN      → Institution → City
  COSTS           → Program → Cost
  REQUIRES_SKILL  → Program → Skill
  ACCREDITED_BY   → Institution/Program → Accreditation
  MENTIONS        → Chunk → any entity (the key link for Graph RAG)

Dependencies:
    pip install neo4j requests tqdm
"""

import os
import json
import re
import time
import warnings
import requests
from pathlib import Path
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures

# Load credentials from .env
load_dotenv()

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")   # From .env
KGE_PARALLEL_WORKERS = max(1, int(os.getenv("KGE_PARALLEL_WORKERS", "8")))


# ════════════════════════════════════════════════════════════════════════
# KGE PROMPT — tailored to real IIT/ISB corpus
# ════════════════════════════════════════════════════════════════════════

KGE_PROMPT = """
Tu es un expert en extraction de graphes de connaissances pour des données universitaires tunisiennes (IIT/ISB).

Extrais les entités et relations du texte ci-dessous.
Retourne UNIQUEMENT du JSON valide. Aucun texte avant ou après.

SCHEMA DES NOEUDS:
- Institution  : IIT, ISB, 3iL Ingénieurs, CESI, EPF, Kedge, MBS, ESC Clermont...
- Program      : Génie Civil, Master Marketing Digital, MBA, MSc Strategic Design...
- Module       : GC1 101, Analyse numérique, BIM, Résistance des matériaux...
- Level        : Licence, Master, MSc, Bac+3, Bac+4, Bac+5, Cycle Ingénieur...
- PartnerCountry: France, Espagne, Canada, Allemagne...
- City         : Paris, Lyon, Toulouse, Limoges, Clermont-Ferrand...
- Cost         : 670€/mois, 26000€/an, 14200€... (inclure l'unité)
- Skill        : Python, SEO, CAO, BIM, DELF, Excel, Robotique, IA...
- Accreditation: CGE, AACSB, ISO 9001, Conférence des Grandes Écoles...

SCHEMA DES RELATIONS:
- OFFERS          : Institution → Program
- HAS_LEVEL       : Program → Level
- HAS_MODULE      : Program → Module
- PARTNERS_WITH   : Institution → Institution
- LOCATED_IN      : Institution → City
- LOCATED_IN      : City → PartnerCountry
- COSTS           : Program → Cost
- REQUIRES_SKILL  : Program/Module → Skill
- ACCREDITED_BY   : Institution ou Program → Accreditation

REGLES:
1. canonical_name = nom normalisé sans accents inutiles, sans abréviations ambiguës
   Exemple: "Carte d'identité" et "CIN" → canonical_name: "CIN"
   Exemple: "3iL ingénieurs" et "3iL Ingénieurs" → canonical_name: "3iL Ingénieurs"
2. N'extraire QUE les entités clairement mentionnées dans le texte
3. Si rien n'est extractable, retourner: {{"nodes": [], "edges": []}}
4. Maximum 15 nodes et 20 edges par chunk

FORMAT JSON (strict):
{{
  "nodes": [
    {{"label": "Institution", "name": "Nom original", "canonical_name": "Nom normalisé"}}
  ],
  "edges": [
    {{"source": "canonical_name_source", "target": "canonical_name_target", "type": "REL_TYPE"}}
  ]
}}

CONTEXTE (résumé et mots-clés déjà extraits):
Résumé: {summary}
Mots-clés: {keywords}

TEXTE:
{text}
"""


# ════════════════════════════════════════════════════════════════════════
# NEO4J SETUP — constraints and indexes
# ════════════════════════════════════════════════════════════════════════

SETUP_QUERIES = [
    "CREATE CONSTRAINT uniq_institution  IF NOT EXISTS FOR (n:Institution)   REQUIRE n.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT uniq_program      IF NOT EXISTS FOR (n:Program)        REQUIRE n.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT uniq_module       IF NOT EXISTS FOR (n:Module)         REQUIRE n.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT uniq_level        IF NOT EXISTS FOR (n:Level)          REQUIRE n.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT uniq_country      IF NOT EXISTS FOR (n:PartnerCountry) REQUIRE n.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT uniq_city         IF NOT EXISTS FOR (n:City)           REQUIRE n.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT uniq_cost         IF NOT EXISTS FOR (n:Cost)           REQUIRE n.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT uniq_skill        IF NOT EXISTS FOR (n:Skill)          REQUIRE n.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT uniq_accreditation IF NOT EXISTS FOR (n:Accreditation) REQUIRE n.canonical_name IS UNIQUE",
    "CREATE CONSTRAINT uniq_chunk        IF NOT EXISTS FOR (n:Chunk)          REQUIRE n.chunk_id IS UNIQUE",
    "CREATE INDEX idx_institution_site   IF NOT EXISTS FOR (n:Institution) ON (n.site)",
    "CREATE INDEX idx_program_category   IF NOT EXISTS FOR (n:Program)     ON (n.category)",
    "CREATE INDEX idx_chunk_site         IF NOT EXISTS FOR (n:Chunk)        ON (n.site)",
]


# ════════════════════════════════════════════════════════════════════════
# GRAPH EXTRACTOR CLASS
# ════════════════════════════════════════════════════════════════════════

class KGEEngine:

    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            print("Connected to Neo4j.")
            self._run_setup()
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            raise

    def close(self):
        self.driver.close()

    def _run_setup(self):
        """Create constraints and indexes — safe to run multiple times."""
        with self.driver.session() as session:
            for q in SETUP_QUERIES:
                try:
                    session.run(q)
                except Exception:
                    pass  # already exists
        print("Schema constraints and indexes ready.")

    # ── LLM call ──────────────────────────────────────────────────────

    def _call_ollama(self, chunk: dict) -> dict:
        """Build context-aware prompt from augmented chunk and call Ollama."""
        text     = chunk.get("page_content", "")[:1200]  # cap at 1200 chars
        summary  = chunk.get("metadata", {}).get("summary", "")
        keywords = ", ".join(chunk.get("metadata", {}).get("keywords", [])[:15])

        prompt = KGE_PROMPT.format(
            summary  = summary  or "Non disponible",
            keywords = keywords or "Non disponible",
            text     = text,
        )

        for attempt in range(3):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model"  : OLLAMA_MODEL,
                        "prompt" : prompt,
                        "stream" : False,
                        "format" : "json",
                        "options": {"temperature": 0, "num_predict": 2048},
                        "keep_alive": "5m",
                    },
                    timeout=120,
                )
                resp_data = response.json()
                if not isinstance(resp_data, dict):
                    raise ValueError(f"Ollama returned non-dict: {type(resp_data).__name__}")

                raw = resp_data.get("response", "{}").strip()
                raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
                raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE)
                data = json.loads(raw.strip())

                if not isinstance(data, dict):
                    raise ValueError(f"LLM returned non-object JSON: {type(data).__name__}")

                if "nodes" not in data:
                    data["nodes"] = []
                if "edges" not in data:
                    data["edges"] = []
                return data

            except json.JSONDecodeError:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return {"nodes": [], "edges": []}
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [LLM error] {e}")
                    return {"nodes": [], "edges": []}

        return {"nodes": [], "edges": []}

    # ── Neo4j write ───────────────────────────────────────────────────

    def _ingest(self, tx, chunk: dict, graph_data: dict):
        """
        Write nodes, edges, and Chunk node with MENTIONS relationships.
        All using MERGE to be safe for re-runs.
        """
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes and not edges:
            return

        # Build canonical_name lookup for edge resolution
        canon_map = {n["canonical_name"]: n for n in nodes if n.get("canonical_name")}

        # 1. Create entity nodes
        for node in nodes:
            label  = re.sub(r'\W', '', node.get("label", "Entity"))
            cname  = (node.get("canonical_name") or node.get("name", "")).strip()
            name   = (node.get("name") or cname).strip()
            site   = chunk.get("metadata", {}).get("site", "")

            if not cname:
                continue

            tx.run(
                f"""
                MERGE (n:{label} {{canonical_name: $cname}})
                ON CREATE SET n.name = $name,
                              n.site = $site
                """,
                cname=cname, name=name, site=site,
            )

        # 2. Create relationships between entities
        for edge in edges:
            src = (edge.get("source") or "").strip()
            tgt = (edge.get("target") or "").strip()
            rel = re.sub(r'\W', '_', edge.get("type", "RELATED_TO")).upper()

            if not src or not tgt or not rel:
                continue

            src_node = canon_map.get(src)
            tgt_node = canon_map.get(tgt)
            if not src_node or not tgt_node:
                continue

            src_label = re.sub(r'\W', '', src_node.get("label", "Entity"))
            tgt_label = re.sub(r'\W', '', tgt_node.get("label", "Entity"))

            tx.run(
                f"""
                MATCH (a:{src_label} {{canonical_name: $src}})
                MATCH (b:{tgt_label} {{canonical_name: $tgt}})
                MERGE (a)-[:{rel}]->(b)
                """,
                src=src, tgt=tgt,
            )

        # 3. Create :Chunk node and link it to every extracted entity
        meta      = chunk.get("metadata", {})
        chunk_id  = meta.get("synthetic_id") or f"{meta.get('site','')}__{meta.get('title','')}__{meta.get('header','')}__{hash(chunk.get('page_content',''))}"
        summary   = meta.get("summary", "")
        site      = meta.get("site", "")
        category  = meta.get("category", "")
        url       = meta.get("url", "")
        sec_title = meta.get("section_title", meta.get("header", ""))
        is_synthetic = meta.get("is_synthetic", False)
        synthetic_style = meta.get("synthetic_style", "")

        tx.run(
            """
            MERGE (c:Chunk {chunk_id: $chunk_id})
            ON CREATE SET c.page_content    = $page_content,
                          c.enriched        = $enriched,
                          c.summary         = $summary,
                          c.site            = $site,
                          c.category        = $category,
                          c.url             = $url,
                          c.section_title   = $sec_title,
                          c.token_count     = $token_count,
                          c.is_synthetic    = $is_synthetic,
                          c.synthetic_style = $synthetic_style
            """,
            chunk_id     = chunk_id,
            page_content = chunk.get("page_content", "")[:500],  # cap for Neo4j
            enriched     = chunk.get("enriched_content", "")[:500],
            summary      = summary,
            site         = site,
            category     = category,
            url          = url,
            sec_title    = sec_title,
            token_count  = meta.get("token_count", 0),
            is_synthetic = is_synthetic,
            synthetic_style = synthetic_style,
        )

        # Link Chunk → each entity with :MENTIONS
        for node in nodes:
            cname = (node.get("canonical_name") or "").strip()
            label = re.sub(r'\W', '', node.get("label", "Entity"))
            if not cname:
                continue
            tx.run(
                f"""
                MATCH (c:Chunk {{chunk_id: $chunk_id}})
                MATCH (e:{label} {{canonical_name: $cname}})
                MERGE (c)-[:MENTIONS]->(e)
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                chunk_id=chunk_id, cname=cname,
            )

    # ── Main process loop ─────────────────────────────────────────────

    def is_chunk_processed(self, chunk_id: str) -> bool:
        """Check if chunk already exists in Neo4j to avoid re-extracting."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN 1 LIMIT 1", chunk_id=chunk_id)
                return result.single() is not None
        except Exception:
            return False

    def process_chunks(
        self,
        chunks: list,
        doc_key_fn=None,
        checkpoint_file: str = None,
        batch_callback=None,
    ):
        """
        Process a list of chunk dicts directly (no file I/O).
        Each chunk may be in the format:
          { "page_content": ..., "enriched_content": ..., "metadata": {...}, ... }
          OR
          { "page_content": ..., "metadata": {...}, ... }

        Args:
            chunks: List of chunk dicts
            doc_key_fn: Optional function(chunk) -> str to compute stable chunk_id.
                        Default: uses synthetic_id or site__title__header__hash
            checkpoint_file: Path to checkpoint JSON (list of done chunk_ids).
            batch_callback: Optional callback(batch_idx, total) for progress tracking.
        """
        total      = len(chunks)
        done_ids   = set()
        ckpt_path  = checkpoint_file
        if ckpt_path and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done_ids = set(json.load(f))
            print(f"Checkpoint loaded: {len(done_ids)} chunks recorded as locally processed.")

        stats = {"extracted": 0, "empty": 0, "error": 0, "skipped": 0}

        def _chunk_id(chunk):
            if doc_key_fn:
                return doc_key_fn(chunk)
            meta = chunk.get("metadata", {})
            return meta.get("synthetic_id") or f"{meta.get('site','')}__{meta.get('title','')}__{meta.get('header','')}__{hash(chunk.get('page_content',''))}"

        def _extract_worker(i, chunk):
            try:
                graph_data = self._call_ollama(chunk)
                return i, chunk, graph_data, None
            except Exception as e:
                return i, chunk, None, e

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=KGE_PARALLEL_WORKERS) as executor:
                futures = {}
                for i in range(total):
                    chunk = chunks[i]
                    chunk_id = _chunk_id(chunk)

                    if chunk_id in done_ids:
                        stats["skipped"] += 1
                        continue

                    if len(chunk.get("page_content", "")) < 30:
                        done_ids.add(chunk_id)
                        continue

                    if self.is_chunk_processed(chunk_id):
                        done_ids.add(chunk_id)
                        stats["skipped"] += 1
                        continue

                    future = executor.submit(_extract_worker, i, chunk)
                    futures[future] = (chunk_id, i)

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="KGE"):
                    chunk_id, i = futures[future]
                    i, chunk, graph_data, err = future.result()

                    if err:
                        tqdm.write(f"  [extract error chunk {i}] {err}")
                        stats["error"] += 1
                        continue

                    if graph_data["nodes"]:
                        stats["extracted"] += 1
                    else:
                        stats["empty"] += 1

                    try:
                        with self.driver.session() as session:
                            session.execute_write(self._ingest, chunk, graph_data)
                    except Exception as e:
                        tqdm.write(f"  [neo4j error chunk {i}] {e}")
                        stats["error"] += 1

                    done_ids.add(chunk_id)

                    if ckpt_path and len(done_ids) % 25 == 0:
                        with open(ckpt_path, "w", encoding="utf-8") as f:
                            json.dump(list(done_ids), f)

                    if batch_callback:
                        batch_callback(i, total)

        except KeyboardInterrupt:
            print(f"\n[interrupt] KGE Engine interrupted. Saving state... ({len(done_ids)} chunks recorded)")
            raise
        finally:
            if ckpt_path:
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(list(done_ids), f)

            print(f"\nDone. extracted={stats['extracted']} empty={stats['empty']} errors={stats['error']} skipped={stats['skipped']}")

    def process_file(
        self,
        file_path  : str,
        limit      : int  = None,
        start_from : int  = 0,
        checkpoint_file: str = None,
    ):
        if not os.path.exists(file_path):
            print(f"[warn] {file_path} not found.")
            return

        with open(file_path, encoding="utf-8") as f:
            chunks = json.load(f)

        to_process = chunks[:limit] if limit else chunks
        total      = len(to_process)

        # Load checkpoint
        done_ids   = set()
        ckpt_path  = checkpoint_file or file_path.replace(".json", "_kge_done.json")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done_ids = set(json.load(f))
            print(f"Checkpoint loaded: {len(done_ids)} chunks recorded as locally processed.")

        stats = {"extracted": 0, "empty": 0, "error": 0, "skipped": 0}

        print(f"\nKGE: {total - start_from} chunks from {file_path}...")

        def _extract_worker(i, chunk):
            try:
                graph_data = self._call_ollama(chunk)
                return i, chunk, graph_data, None
            except Exception as e:
                return i, chunk, None, e

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=KGE_PARALLEL_WORKERS) as executor:
                futures = {}
                for i in range(start_from, total):
                    chunk = to_process[i]

                    # Build stable ID
                    meta     = chunk.get("metadata", {})
                    chunk_id = meta.get("synthetic_id") or f"{meta.get('site','')}__{meta.get('title','')}__{meta.get('header','')}__{hash(chunk.get('page_content',''))}"

                    if chunk_id in done_ids:
                        stats["skipped"] += 1
                        continue

                    # Skip very short chunks
                    if len(chunk.get("page_content", "")) < 30:
                        done_ids.add(chunk_id)
                        continue

                    # Query Neo4j to avoid re-run
                    if self.is_chunk_processed(chunk_id):
                        done_ids.add(chunk_id)
                        stats["skipped"] += 1
                        continue

                    future = executor.submit(_extract_worker, i, chunk)
                    futures[future] = chunk_id

                # Await jobs to finish
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extraction"):
                    chunk_id = futures[future]
                    i, chunk, graph_data, err = future.result()

                    if err:
                        tqdm.write(f"  [extract error chunk {i}] {err}")
                        stats["error"] += 1
                        continue

                    # Debug: print first extraction arbitrarily
                    # if i == start_from:
                    #     tqdm.write(f"\n[DEBUG first extraction]\n{json.dumps(graph_data, ensure_ascii=False, indent=2)[:600]}\n")

                    if graph_data["nodes"]:
                        stats["extracted"] += 1
                    else:
                        stats["empty"] += 1

                    # Write to Neo4j sequentially to avoid deadlocks
                    try:
                        with self.driver.session() as session:
                            session.execute_write(self._ingest, chunk, graph_data)
                    except Exception as e:
                        tqdm.write(f"  [neo4j error chunk {i}] {e}")
                        stats["error"] += 1

                    done_ids.add(chunk_id)

                    # Checkpoint periodically
                    if len(done_ids) % 25 == 0:
                        with open(ckpt_path, "w", encoding="utf-8") as f:
                            json.dump(list(done_ids), f)

        except KeyboardInterrupt:
            print(f"\n[interrupt] KGE Engine interrupted. Saving state... ({len(done_ids)} chunks recorded)")
            raise
        finally:
            # Final checkpoint
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(list(done_ids), f)

            print(f"\nDone. extracted={stats['extracted']} empty={stats['empty']} errors={stats['error']} skipped={stats['skipped']}")

    # ── Graph stats ───────────────────────────────────────────────────

    def print_stats(self):
        """Print a summary of what's in the graph."""
        queries = [
            ("Institutions", "MATCH (n:Institution) RETURN count(n) AS c"),
            ("Programs",     "MATCH (n:Program)     RETURN count(n) AS c"),
            ("Modules",      "MATCH (n:Module)      RETURN count(n) AS c"),
            ("Chunks",       "MATCH (n:Chunk)       RETURN count(n) AS c"),
            ("MENTIONS rels","MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c"),
            ("OFFERS rels",  "MATCH ()-[r:OFFERS]->()   RETURN count(r) AS c"),
            ("PARTNERS_WITH","MATCH ()-[r:PARTNERS_WITH]->() RETURN count(r) AS c"),
        ]
        print("\n── Graph Stats ──────────────────")
        with self.driver.session() as session:
            for label, q in queries:
                try:
                    result = session.run(q).single()
                    print(f"  {label:<20}: {result['c']}")
                except Exception:
                    print(f"  {label:<20}: (error)")
        print("────────────────────────────────")


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site",       choices=["iit", "isb", "both"], default="both")
    parser.add_argument("--input",      type=str, default=None, help="Path to merged corpus json (e.g. data/synthetic/corpus_final.json)")
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    engine = KGEEngine()

    try:
        if args.stats_only:
            engine.print_stats()
        else:
            parsed_input = args.input
            if not parsed_input:
                print("Recommended: run with --input data/synthetic/corpus_final.json (merged real + synthetic).")
                print("Falling back to legacy per-site augmented files...\n")
                if args.site in ("iit", "both"):
                    engine.process_file(
                        "data/augmented/iit_chunks_augmented.json",
                        limit      = args.limit,
                        start_from = args.start_from,
                    )
                if args.site in ("isb", "both"):
                    engine.process_file(
                        "data/augmented/isb_chunks_augmented.json",
                        limit      = args.limit,
                        start_from = args.start_from if args.site == "isb" else 0,
                    )
            else:
                engine.process_file(
                    parsed_input,
                    limit      = args.limit,
                    start_from = args.start_from,
                )

            engine.print_stats()
            print("\nNext step: run utils/embed_to_postgreSQL.py with enriched_content field.")
    finally:
        engine.close()