"""
rag_api — Standalone RAG microservice for UniHelp.

Usage:
    python -m uvicorn rag_api.main:app --host 0.0.0.0 --port 8002

Endpoints:
    GET   /health
    POST  /process              — Process a document (extract → clean → chunk → augment → insert)
    POST  /index-unindexed      — Run full indexing pipeline (embed → KGE → BM25)
    POST  /query                — Hybrid search + reranker + Ollama generation
    DELETE /chunks              — Delete chunks by site + title
    POST  /generate-email       — Generate administrative email via Ollama
"""

import os
import sys
import json
import hashlib
import time
from functools import lru_cache
from pathlib import Path
import re
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import quote

# ── Query cache ────────────────────────────────────────────────────────
_query_cache: dict[str, tuple[float, str, list]] = {}
CACHE_TTL = 300  # 5 minutes

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env — try several locations (Docker vs local dev)
_env_loaded = load_dotenv(PROJECT_ROOT / "backend" / ".env")
if not _env_loaded:
    _env_loaded = load_dotenv(PROJECT_ROOT / ".env")
if not _env_loaded:
    _env_loaded = load_dotenv(PROJECT_ROOT / ".env.docker")

sys.path.insert(0, str(PROJECT_ROOT))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
RAG_API_PORT = int(os.getenv("RAG_API_PORT", "8002"))

# ── LLM Backend selection ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_USE_GROQ = bool(GROQ_API_KEY)

print(f"[RAG-API] LLM backend: {'Groq Cloud' if _USE_GROQ else 'Ollama local'} ({GROQ_MODEL if _USE_GROQ else OLLAMA_MODEL})")

# ── Pydantic models ──────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    file_path: str
    site: str = "IIT"
    title: str = ""
    category: str = "Website Content"
    url: str = ""

class ProcessResponse(BaseModel):
    status: str
    chunks: int = 0
    inserted: int = 0
    detail: str = ""

class IndexRequest(BaseModel):
    use_graph: bool = True
    gen_synthetic: bool = True
    synth_limit: int = 20

class IndexResponse(BaseModel):
    status: str
    detail: str = ""

class QueryRequest(BaseModel):
    question: str
    top_k: int = 20
    use_graph: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list = []

class DeleteChunksRequest(BaseModel):
    site: str
    title: str

class DeleteChunksResponse(BaseModel):
    deleted: int

class EmailRequest(BaseModel):
    type_email: str
    nom: str
    prenom: str
    universite: str
    details: str = ""

class EmailResponse(BaseModel):
    objet: str
    destinataire: str
    contenu: str

# ── FastAPI app ───────────────────────────────────────────────────────

app = FastAPI(
    title="UniHelp RAG API",
    description="Standalone RAG microservice for document processing, indexing, and querying",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8002", "http://127.0.0.1:8002", "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: eager model loading ─────────────────────────────────────

@app.on_event("startup")
def _load_models_at_startup():
    print("[RAG-API] Pre-loading models at startup...")
    from rag.models import get_dense_model, get_reranker
    print(f"[RAG-API] Loading dense model...")
    get_dense_model()
    print(f"[RAG-API] Loading reranker...")
    get_reranker()
    print(f"[RAG-API] All models loaded successfully.")


# ── Helpers ───────────────────────────────────────────────────────────

def _extract_text_from_json(fp: str) -> tuple[str, str, str]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    content = data.get("content", "") or ""
    title = data.get("title", "")
    category = data.get("category", "Website Content")
    return content, title, category


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-api"}


@app.post("/process", response_model=ProcessResponse)
def process_document(body: ProcessRequest):
    fp = body.file_path
    path = Path(fp)
    if not path.exists():
        raise HTTPException(400, f"File not found: {fp}")

    site = body.site
    title = body.title or path.stem.replace("_", " ").replace("-", " ")[:255]
    category = body.category
    url = body.url or fp  # use provided URL (download link) else fallback to file path

    try:
        ext = path.suffix.lower()
        if ext == ".json":
            raw_text, json_title, json_cat = _extract_text_from_json(fp)
            title = json_title or title
            category = json_cat or category
        else:
            from upload_doc import extract_text
            raw_text = extract_text(fp)

        if not raw_text.strip():
            return ProcessResponse(status="error", detail="No text extracted")

        from upload_doc import clean_text, chunk_text, augment_chunks, insert_chunks

        cleaned = clean_text(raw_text, site)
        chunks = chunk_text(cleaned, site, title, category, url)
        if not chunks:
            return ProcessResponse(status="error", detail="No chunks generated")

        augmented = augment_chunks(chunks)
        n = insert_chunks(augmented)

        return ProcessResponse(
            status="ok",
            chunks=len(chunks),
            inserted=n,
            detail=f"Inserted {n} chunks",
        )
    except Exception as e:
        return ProcessResponse(status="error", detail=str(e))


@app.post("/index-unindexed", response_model=IndexResponse)
def index_unindexed(body: IndexRequest):
    try:
        from rag.pipeline import index_unindexed as run_index

        run_index(
            use_graph=body.use_graph,
            gen_synthetic=body.gen_synthetic,
            synth_limit=body.synth_limit,
        )
        return IndexResponse(status="ok", detail="Indexing complete")
    except Exception as e:
        return IndexResponse(status="error", detail=str(e))


# ── Query Augmentation ────────────────────────────────────────────────

_GREETING_RESPONSES = {
    "bonjour": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "bonsoir": "Bonsoir ! Comment puis-je vous aider ?",
    "salut": "Salut ! De quoi avez-vous besoin ?",
    "hello": "Hello ! How can I help you ?",
    "bon matin": "Bonjour ! Comment puis-je vous aider ?",
    "re": "Rebonjour ! Comment puis-je vous aider ?",
    "hey": "Hey ! Comment puis-je vous aider ?",
    "yo": "Salut ! Comment puis-je vous aider ?",
    "coucou": "Coucou ! Comment puis-je vous aider ?",
    "au revoir": "Au revoir ! N'hesitez pas a revenir si vous avez d'autres questions sur vos etudes a l'IIT ou l'ISB.",
    "bye": "Bye ! A bientot pour d'autres questions sur vos etudes a l'IIT ou l'ISB.",
    "bonne journee": "Bonne journee ! Revenez quand vous voulez pour des questions sur vos etudes.",
    "bonne soiree": "Bonne soiree ! A bientot sur UniHelp.",
    "bonne nuit": "Bonne nuit ! A bientot pour d'autres questions sur vos etudes.",
    "a bientot": "A bientot ! N'hesitez pas a revenir si vous avez besoin d'aide.",
    "a plus": "A plus ! Revenez quand vous voulez.",
}

_GREETING_ONLY_WORDS = frozenset({
    "bonjour", "bonsoir", "salut", "hello", "coucou", "hey", "yo", "bonnuit", "bonmatin",
    "rebonjour", "reee", "bon", "ok", "super", "slt", "slm", "sava",
    "au revoir", "au revoir", "bye", "bonne journee", "bonne soiree",
    "bonne nuit", "a bientot", "a plus", "a tout a l heure",
})

_CHITCHAT_PHRASES = frozenset({
    "comment ca va", "comment vas tu", "comment allez vous", "ca va",
    "ca va bien", "je vais bien", "tres bien", "super et toi",
    "bien et toi", "ca roule", "tranquille", "quoi de neuf",
    "cv", "salut cv", "sava", "salut cava", "salut ca va",
    "quoi de beau", "merci", "merci beaucoup", "merci bien",
    "merci infiniment", "merci a vous", "merci a toi",
    "de rien", "pas de souci", "pas de probleme", "pas grave",
    "avec plaisir", "je vous en prie", "je t en prie",
    "oui", "non", "ok", "d accord", "d'accord", "super",
    "genial", "cool", "parfait", "excellent", "magnifique",
    "tres cool", "trop cool", "nickel", "impeccable",
    "parfaitement", "bien recu", "compris", "entendu",
    "dac", "ok merci", "merci ok", "super merci", "bon",
    "tres bien merci", "bien merci",
})

_GREETING_ONLY_WORDS = frozenset({
    "bonjour", "bonsoir", "salut", "hello", "coucou", "hey", "yo", "bonnuit", "bonmatin",
    "rebonjour", "reee", "bon", "ok", "super",
    "au revoir", "au revoir", "bye", "bonne journee", "bonne soiree",
    "bonne nuit", "a bientot", "a plus", "a tout a l heure",
})

_DISCIPLINE_KEYWORDS = {
    "informatique": frozenset({"meca", "indus", "civil", "procede", "mecanique",
                               "mecaniques", "industriel", "gc", "genie civil",
                               "genie mecanique", "genie procedes", "genie industriel"}),
    "genie logiciel": frozenset({"meca", "indus", "civil", "procede", "mecanique",
                                  "gc", "genie civil", "genie mecanique"}),
    "genie civil": frozenset({"informatique", "logiciel", "meca", "procede",
                               "indus", "mecanique"}),
    "genie mecanique": frozenset({"informatique", "logiciel", "civil", "procede",
                                   "indus", "gc"}),
    "genie procedes": frozenset({"informatique", "logiciel", "civil", "meca",
                                  "indus", "mecanique", "gc"}),
    "genie industriel": frozenset({"informatique", "logiciel", "civil", "meca",
                                    "procede", "gc"}),
    "architecture": frozenset({"informatique", "logiciel", "meca", "indus",
                                "civil", "procede", "genie"}),
}


def _filter_by_discipline(results: list, query: str) -> list:
    """Filter out sources whose title mentions a DIFFERENT discipline
    than what the user asked about."""
    q_lower = query.lower()
    mentioned_discipline = None
    for disc, _ in _DISCIPLINE_KEYWORDS.items():
        if disc in q_lower:
            mentioned_discipline = disc
            break
    if not mentioned_discipline:
        return results
    blocklist = _DISCIPLINE_KEYWORDS[mentioned_discipline]
    filtered = []
    for r in results:
        title = ((r.get("title") or "") + " " + (r.get("section_title") or "")).lower()
        has_blocked = any(b in title for b in blocklist)
        if not has_blocked:
            filtered.append(r)
    if filtered:
        return filtered
    return results


_IRRELEVANT_PATTERNS = [
    "blague", "humour", "rire", "histoire drole", "poem", "poeme",
    "chanson", "recette", "meteo", "temps", "sport", "football",
    "politique", "actualite", "cinema", "film", "musique",
    "qui est le president", "quel jour", "quelle heure",
    "danse", "cuisine", "voyage", "jeu", "jouer",
]

_ACRONYM_CLEANUP = re.compile(r'\b(IIT|ISB)\s*\([^)]{3,}\)')


async def _augment_query(question: str) -> str:
    """Minimal query cleaning. Handles only critical typos that hurt retrieval.
    BM25 expansion already handles accent variations at search time."""
    q = question.strip()
    if len(q) < 3:
        return q
    return q


def _is_irrelevant(question: str) -> bool:
    """Check if query is clearly irrelevant (joke, chitchat, etc)."""
    q = question.lower().strip()
    if len(q) < 3:
        return True
    for pat in _IRRELEVANT_PATTERNS:
        if pat in q:
            return True
    # Check if query is just random words with no academic intent
    academic_keywords = ["formation", "frais", "inscription", "cours", "matiere",
                         "programme", "diplome", "master", "licence", "doctorat",
                         "ecole", "universite", "iit", "isb", "stage", "mobilite",
                         "bourse", "admission", "concours", "note", "examen",
                         "professeur", "departement", "laboratoire", "recherche",
                         "accreditation", "partenaire", "international", "etudier",
                         "etude", "plan", "semestre", "module", "cursus", "curriculum",
                         "genie", "informatique", "logiciel", "mecanique", "civil",
                         "industriel", "procede", "prepa", "cycle", "ingenieur",
                         "inscrit", "candidature", "postuler", "rentree", "scolarite",
                         "campus", "bibliotheque", "visa", "logement", "transport",
                         "emploi", "carriere", "debouche", "metier", "competence",
                         "langue", "anglais", "francais", "certification",
                         "soutenance", "these", "memoire", "rapport", "projet",
                         "classe", "groupe", "horaire", "emploi du temps",
                         "reglement", "interieur", "discipline", "absence",
                         "attestation", "releve", "note", "bulletin",
                         "europeen", "erasmus", "echange", "etranger",
                         "alternance", "apprentissage", "professionnel",
                         "accompagnement", "orientation", "conseil",
                          "handicap", "bourse", "aide", "financement",
                          "logement", "cite", "restaurant", "crous",
                          "annee", "anne", "eme", "ieme", "premiere", "deuxieme",
                          "troisieme", "derniere", "duree", "dure"]
    words = re.findall(r'[a-zà-öù-ÿ]+', q)
    if len(words) >= 4:
        academic_count = sum(1 for w in words if w in academic_keywords)
        if academic_count == 0:
            return True
    return False


def _is_greeting(question: str) -> str | None:
    """Detect greeting/chitchat. Returns canned response or None."""
    q = question.lower().strip()
    # Single-word greetings
    if q in _GREETING_ONLY_WORDS:
        return _GREETING_RESPONSES.get(q, "Bonjour ! Comment puis-je vous aider ?")
    # Multi-word greetings that are only greeting words
    words = re.findall(r'[a-zà-öù-ÿ]+', q)
    if len(words) <= 3:
        # Check if all words are greeting only words
        if all(w in _GREETING_ONLY_WORDS for w in words):
            return _GREETING_RESPONSES.get(words[0], "Bonjour ! Comment puis-je vous aider ?")
        q_norm = q.replace("'", " ").replace("-", " ").replace("é", "e").replace("è", "e")
        q_norm = " ".join(q_norm.split())
        if q_norm in _CHITCHAT_PHRASES:
            return (
                "Je vais bien, merci ! Comment puis-je vous aider "
                "avec vos etudes, formations ou demarches administratives "
                "a l'IIT ou l'ISB ?"
            )
    return None


def _strip_acronym_expansion(text: str) -> str:
    """Remove parenthesized expansions of IIT/ISB from generated text."""
    return _ACRONYM_CLEANUP.sub(r'\1', text)


# ── Query Endpoint ────────────────────────────────────────────────────

async def _prepare_rag_context(body: QueryRequest) -> tuple[str | None, list, str, str]:
    q_raw = body.question.strip()
    
    greeting_response = _is_greeting(q_raw)
    if greeting_response:
        return greeting_response, [], "", q_raw

    if _is_irrelevant(q_raw):
        return ("Je suis un assistant universitaire pour IIT et ISB. "
                "Je ne peux repondre qu'aux questions liees aux etudes, "
                "formations, procedures administratives et vie universitaire. "
                "Veuillez reformuler votre question."), [], "", q_raw

    q = await _augment_query(q_raw)
    from rag.pipeline import query as rag_query
    results = rag_query(q, top_k=min(body.top_k, 30), use_graph=body.use_graph)

    if not results:
        return None, [], "", q

    from rag.graph_index import _extract_entities
    from rag.retriever import _accent_fold
    query_kw = _extract_entities(q)
    total = len(results)
    scored = []
    for i, r in enumerate(results):
        title = _accent_fold(r.get("title", "") or "").lower()
        section = _accent_fold(r.get("section_title", "") or "").lower()
        kw_matches = sum(1 for kw in query_kw if kw.lower() in title or kw.lower() in section)
        kw_score = kw_matches / max(len(query_kw), 1) if query_kw else 0
        pos_score = 1.0 - (i / max(total, 1))
        combined = 0.4 * kw_score + 0.6 * pos_score
        scored.append((combined, r))

    scored.sort(key=lambda x: -x[0])

    filtered = [r for s, r in scored if s > 0.2]
    if len(filtered) < 3:
        filtered = [r for _, r in scored[:10]]

    MAX_PER_DOC = 3
    doc_counts: dict = {}
    final = []
    for r in filtered:
        title = r.get("title", "") or "unknown"
        cnt = doc_counts.get(title, 0)
        if cnt >= MAX_PER_DOC:
            continue
        doc_counts[title] = cnt + 1
        final.append(r)

    _BAN5 = frozenset(["ation", "ement", "ment ", "tion ", "satio"])
    def _ng5(text):
        s = _accent_fold(text).lower()
        if len(s) < 5:
            return set()
        return {s[i:i+5] for i in range(len(s)-4)} - _BAN5
    kw5 = set()
    for k in query_kw:
        kw5 |= _ng5(k)
    def _overlap(r):
        t = _ng5(r.get("title", "") or "")
        s = _ng5(r.get("section_title", "") or "")
        return len(kw5 & (t | s))
    final = [r for r in final if _overlap(r) > 0]
    if not final and len(results) > 0:
        final = results[:body.top_k]

    q_lower = q.lower()
    sites_in_query = [s.lower() for s in ["IIT", "ISB"]
                      if re.search(r'\b' + re.escape(s.lower()) + r'\b', q_lower)]
    if sites_in_query:
        final = [r for r in final
                 if (r.get("site") or "").lower() in sites_in_query]
        if not final and len(results) > 0:
            final = [r for r in results[:body.top_k]
                     if (r.get("site") or "").lower() in sites_in_query]

    final = _filter_by_discipline(final, q)

    context_parts = []
    sources = []
    seen_titles = set()
    for idx, r in enumerate(final, 1):
        title = r.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        text = r.get("enriched_content") or r.get("page_content", "")
        ctype = r.get("content_type", "general")
        header = f"[S{len(sources)+1}] type={ctype} | {r.get('site','')} | {title}"
        section = r.get("section_title", "")
        if section:
            header += f" | {section}"
        context_parts.append(f"{header}\n{text[:800]}")
        download_url = f"/files/{quote(r.get('doc_key',''), safe='')}" if r.get("doc_key") else ""
        sources.append({
            "id": f"S{len(sources)+1}", "site": r.get("site", ""),
            "title": title, "section": section,
            "type": ctype, "url": download_url
        })
        if len(sources) >= 5:
            break

    context = "\n\n---\n\n".join(context_parts)
    return None, sources, context, q


@app.post("/query", response_model=QueryResponse)
async def query_rag(body: QueryRequest):
    try:
        direct_ans, sources, context, q = await _prepare_rag_context(body)
        if direct_ans is not None:
            return QueryResponse(answer=direct_ans, sources=sources)
        
        cache_key = hashlib.sha256(q.lower().encode()).hexdigest()
        now = time.time()
        if cache_key in _query_cache:
            ts, answer, cached_sources = _query_cache[cache_key]
            if now - ts < CACHE_TTL:
                return QueryResponse(answer=answer, sources=cached_sources)
                
        answer = await _ask_llm(q, context=context)
        answer = _strip_acronym_expansion(answer)
        _query_cache[cache_key] = (now, answer, sources)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return QueryResponse(answer=f"Erreur RAG: {str(e)}", sources=[])


from fastapi.responses import StreamingResponse

@app.post("/query/stream")
async def query_rag_stream(body: QueryRequest):
    async def event_generator():
        try:
            direct_ans, sources, context, q = await _prepare_rag_context(body)
            # Send sources first
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            if direct_ans is not None:
                yield f"data: {json.dumps({'type': 'chunk', 'text': direct_ans})}\n\n"
            else:
                # Check cache
                cache_key = hashlib.sha256(q.lower().encode()).hexdigest()
                now = time.time()
                if cache_key in _query_cache and (now - _query_cache[cache_key][0] < CACHE_TTL):
                    cached_answer = _query_cache[cache_key][1]
                    yield f"data: {json.dumps({'type': 'chunk', 'text': cached_answer})}\n\n"
                else:
                    # Stream LLM
                    full_answer = ""
                    async for text_chunk in _stream_llm(q, context=context):
                        clean_chunk = _strip_acronym_expansion(text_chunk)
                        full_answer += clean_chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'text': clean_chunk})}\n\n"
                    # Cache the full answer
                    _query_cache[cache_key] = (now, full_answer, sources)
                    
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/chunks", response_model=DeleteChunksResponse)
def delete_chunks(body: DeleteChunksRequest):
    try:
        from rag.store import delete_chunks_by_document
        n = delete_chunks_by_document(body.site, body.title)
        return DeleteChunksResponse(deleted=n)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/generate-email", response_model=EmailResponse)
async def generate_email(body: EmailRequest):
    full_name = f"{body.prenom} {body.nom}"
    prompt = f"""Geneere un email administratif formel en francais.
Type: {body.type_email}
Etudiant: {full_name}
Universite: {body.universite}
Details: {body.details}

Le contenu doit etre signe par l'etudiant.
Apres 'Sincereement', ecris le prenom et nom de l'etudiant ({full_name}) — NE mets PAS de crochets, PAS de placeholders, PAS de coordonnees fictives.

REGLE STRICTE : Le 'contenu' de l'email doit etre du TEXTE BRUT. 
INTERDICTION ABSOLUE d'utiliser du Markdown (pas de **gras**, pas de #, pas de tirets).
Retourne UNIQUEMENT un JSON avec les cles: objet, destinataire, contenu."""

    if _USE_GROQ:
        client = _get_groq_client()
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return EmailResponse(**data)
    else:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL, 
                    "prompt": prompt, 
                    "stream": False, 
                    "format": "json",
                    "keep_alive": "10m",
                    "num_predict": 400,
                    "num_ctx": 1024
                },
            )
            response.raise_for_status()
            data = json.loads(response.json()["response"])
            return EmailResponse(**data)


@app.get("/files/{doc_key:path}")
async def download_file(doc_key: str):
    """Download the original file for a chunk from PostgreSQL."""
    try:
        from rag.store import get_conn
        import psycopg2
        import psycopg2.extras
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT file_data, file_name, file_type
            FROM chunks
            WHERE doc_key = %s
              AND file_data IS NOT NULL
            LIMIT 1
        """, (doc_key,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            raise HTTPException(404, "Fichier non trouve")
        file_bytes = row["file_data"].tobytes() if hasattr(row["file_data"], "tobytes") else bytes(row["file_data"])
        file_name = row["file_name"] or "document"
        file_type = row["file_type"] or "application/octet-stream"
        return Response(content=file_bytes, media_type=file_type,
                        headers={"Content-Disposition": f'attachment; filename="{file_name}"'})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


def _build_prompt(question: str, context: str = "") -> str:
    anti_markdown_instruction = """REGLE FONDAMENTALE DE FORMATAGE : 
- INTERDICTION ABSOLUE D'UTILISER LE FORMATAGE MARKDOWN.
- N'utilise JAMAIS de double astérisques (ex: pas de **texte**) ni d'astérisque simple (ex: pas de *texte*) pour mettre en gras ou en italique.
- N'utilise JAMAIS de symbole dièse (#, ##, ###, etc.) pour les titres.
- Pour structurer ta réponse, utilise uniquement du TEXTE BRUT : des MAJUSCULES pour les titres, et des tirets (-) pour les listes.
- Ta réponse doit être un texte professionnel propre sans aucune balise ni caractère spécial markdown."""

    if context:
        return f"""Tu es UniHelp, assistant IIT/ISB.
{anti_markdown_instruction}

Tu ne reponds qu'avec les sources fournies. REGLES ABSOLUES (toute violation est interdite) :

1. TITRE d'abord : le TITRE de chaque source doit obligatoirement correspondre au sujet de la question. Si le titre ne correspond pas, IGNORE la source MEME si son texte semble pertinent.
2. PRIORITE aux sources SPECIALISEES : si une source a un titre SPECIFIQUE au sujet (ex: "Accreditations - IIT" pour une question sur accreditations), elle est PLUS FIABLE qu'une source generique (ex: "Presentation-Passeport", "Culture choc", "R&D").
3. CONTRADICTION : si deux sources se contredisent, garde UNIQUEMENT celle dont le TITRE est le plus specifique au sujet.
4. Si AUCUNE source valide ne repond a la question, dis-le clairement.
5. SIGLES INTERDIT : N'ECRIS JAMAIS IIT(Institut ...) ou ISB(...). JAMAIS de parentheses apres IIT ou ISB. Si tu vois "IIT" dans les sources, ecris juste "IIT". Meme chose pour "ISB".
6. DETAILS EXHAUSTIFS : Ta reponse doit etre EXTREMEMENT DETAILLEE. Ne resume pas, explique en profondeur. Structure obligatoirement ta reponse avec des listes a puces (bullet points) pour plus de clarte. Cite les montants exacts, les credits ECTS, les noms de modules, etc.
7. NE JAMAIS utiliser d'expressions comme "Selon le contexte", "D'apres les sources fournies", "Les documents indiquent que". Tu DOIS repondre directement, de maniere naturelle et fluide, comme si c'etait ta propre connaissance.
8. NE JAMAIS inventer des montants, chiffres ou donnees qui ne figurent pas dans les sources. Si un detail n'est pas explicitement mentionne, dis-le clairement.
9. RAPPEL FORMATAGE : N'utilise aucun markdown. Pas de **, pas de #. Uniquement du texte brut avec des majuscules pour les titres et tirets (-) pour les listes.

CONTEXTE (type, titre, section) :
{context}

QUESTION : {question}

{anti_markdown_instruction}

REPONSE:"""
    return f"""Tu es UniHelp, assistant universitaire pour IIT et ISB. Tu ne reponds qu'aux questions sur les etudes, formations, procedures administratives et vie universitaire a l'IIT et ISB Tunisie.
{anti_markdown_instruction}

QUESTION: {question}

{anti_markdown_instruction}

REPONSE:"""


# ── Groq backend ──────────────────────────────────────────────────────
_groq_client: any = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import AsyncGroq
        _groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    return _groq_client


async def _ask_groq(question: str, context: str = "") -> str:
    prompt = _build_prompt(question, context)
    client = _get_groq_client()
    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


# ── Ollama backend (local) ────────────────────────────────────────────
_ollama_client: httpx.AsyncClient | None = None


def _get_ollama_client() -> httpx.AsyncClient:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = httpx.AsyncClient(timeout=120.0)
    return _ollama_client


async def _ask_ollama(question: str, context: str = "") -> str:
    prompt = _build_prompt(question, context)
    client = _get_ollama_client()
    response = await client.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",  # FIX #4: keep model hot between messages
            "num_predict": 800,   # FIX #4: was 1500 — 800 is enough for detailed answers
            "num_ctx": 2048,      # FIX #4: was 4096 — prompt fits in 2K tokens easily
        },
    )
    response.raise_for_status()
    return response.json()["response"]


# ── Helpers ───────────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think\s*/>", "", text)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text


async def _ask_llm(question: str, context: str = "") -> str:
    if _USE_GROQ:
        return _strip_think(await _ask_groq(question, context))
    return _strip_think(await _ask_ollama(question, context))

async def _stream_groq(question: str, context: str = ""):
    prompt = _build_prompt(question, context)
    client = _get_groq_client()
    stream = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.3,
        stream=True
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

async def _stream_ollama(question: str, context: str = ""):
    prompt = _build_prompt(question, context)
    client = _get_ollama_client()
    async with client.stream(
        "POST",
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "10m",  # FIX #4: keep model hot between messages
            "num_predict": 800,   # FIX #4: was 1500
            "num_ctx": 2048,      # FIX #4: was 4096
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line:
                try:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield _strip_think(data["response"])
                except Exception:
                    pass

async def _stream_llm(question: str, context: str = ""):
    if _USE_GROQ:
        async for chunk in _stream_groq(question, context):
            yield chunk
    else:
        async for chunk in _stream_ollama(question, context):
            yield chunk
