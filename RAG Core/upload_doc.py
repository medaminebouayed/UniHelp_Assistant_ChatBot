"""
upload_doc.py - Upload a document (PDF/Word/TXT) -> full pipeline to searchable.

Usage:
    python upload_doc.py path/to/document.pdf
    python upload_doc.py path/to/document.docx --no-synth --no-kge
    python upload_doc.py path/to/file.txt --site IIT --title "Mon Document"

Steps:
    1. Extract text from PDF/Word/TXT
    2. Clean text (boilerplate removal, deduplication)
    3. Chunk into 200-token pieces with breadcrumbs
    4. Augment each chunk (LLM: questions, keywords, summary)
    5. Insert into PostgreSQL
    6. Generate synthetic variants (unless --no-synth)
    7. Embed into PGVector
    8. Extract entities to Neo4j via LLM KGE (unless --no-kge)
    9. Rebuild BM25 index
    10. Mark all as indexed - document is now searchable
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════════════════════════════
# STEP 0: Text Extraction (PDF / DOCX / TXT)
# ════════════════════════════════════════════════════════════════════════

def extract_text(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text() for page in doc])
            doc.close()
            return text
        except ImportError:
            print("[extract] PyMuPDF not found, trying pypdf...")
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])

    if ext in (".docx", ".doc"):
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            print("[extract] python-docx not found, trying docx2txt...")
            import docx2txt
            return docx2txt.process(file_path)

    raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .docx, or .txt")


# ════════════════════════════════════════════════════════════════════════
# STEP 1: Clean text
# ════════════════════════════════════════════════════════════════════════

def clean_text(raw: str, site: str = "IIT") -> str:
    """Light cleaning: remove boilerplate + metadata headers, normalize whitespace."""
    text = raw.replace('\xa0', ' ')

    # Strip metadata header lines FIRST (before whitespace normalization)
    header_keys = ["Source URL", "Source File", "Title", "Category", "Site", "Type"]
    header_pattern = r"^(?:" + "|".join(header_keys) + r"):\s*.*\n?"
    text = re.sub(header_pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    boilerplate = [
        r"Skip to content", r"Continuer la lecture", r"En savoir plus",
        r"Read More", r"Voir plus", r"Copyright.*", r"All rights reserved",
        r"Contactez-nous", r"Suivez-nous sur", r"Accès rapide",
        r"Espace étudiant", r"Vidéothéque", r"Visite virtuelle",
        r"Brochure", r"Prenez votre avenir en main",
        r"Vous devriez également aimer",
        r"Nom et prénom", r"Votre email", r"Numéro de téléphone",
    ]
    for pat in boilerplate:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ════════════════════════════════════════════════════════════════════════
# STEP 2: Chunking
# ════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, site: str, title: str, category: str, url: str) -> List[dict]:
    """Chunk cleaned text using SemanticDataChunker."""
    from utils.preprocessing.chunk_data import SemanticDataChunker, count_tokens

    chunker = SemanticDataChunker(chunk_size=200, chunk_overlap=30)

    meta = {
        "site": site,
        "title": title,
        "category": category,
        "url": url or "Unknown",
    }

    base_breadcrumb = f"[{meta['site']} | {meta['title']}]"
    chunks = []

    # Use a single "Content" section (avoid aggressive section detection on arbitrary text)
    section = "Content"
    current_breadcrumb = f"{base_breadcrumb} {section} > "
    splitter = chunker._make_splitter(current_breadcrumb)
    sub_chunks = splitter.split_text(text)

    for sc in sub_chunks:
        if len(re.sub(r'\W', '', sc)) < 25:
            continue
        rich_chunk = current_breadcrumb + sc
        token_count = count_tokens(rich_chunk)

        chunks.append({
            "page_content": rich_chunk,
            "metadata": {
                "site": meta["site"],
                "title": meta["title"],
                "category": meta["category"],
                "url": meta["url"],
                "header": section,
                "chunk_length": len(rich_chunk),
                "token_count": token_count,
            },
        })

    print(f"  -> {len(chunks)} chunks generated")
    return chunks


# ════════════════════════════════════════════════════════════════════════
# STEP 3: Augment each chunk (LLM)
# ════════════════════════════════════════════════════════════════════════

def augment_chunks(chunks: List[dict]) -> List[dict]:
    """Augment all chunks with LLM (questions, keywords, summary, section_title)."""
    from utils.augmentation.augment_metadata import augment_chunk as augment_one
    from tqdm import tqdm

    augmented = []
    for c in tqdm(chunks, desc="Augmenting"):
        if len(c["page_content"]) < 30:
            c["metadata"]["content_type"] = "noise"
            c["metadata"]["questions"] = []
            c["metadata"]["keywords"] = []
            c["metadata"]["section_title"] = ""
            c["metadata"]["summary"] = ""
            c["enriched_content"] = c["page_content"]
            c["bm25_text"] = ""
            augmented.append(c)
            continue
        try:
            result = augment_one(c)
            augmented.append(result)
        except Exception as e:
            print(f"  [augment error] {e}")
            # Insert minimal version anyway
            c["metadata"]["content_type"] = "general"
            c["metadata"]["questions"] = []
            c["metadata"]["keywords"] = []
            c["metadata"]["section_title"] = c["metadata"].get("header", "")
            c["metadata"]["summary"] = ""
            c["enriched_content"] = c["page_content"]
            c["bm25_text"] = c["metadata"].get("section_title", "")
            augmented.append(c)

    return augmented


# ════════════════════════════════════════════════════════════════════════
# STEP 4: Insert into PostgreSQL
# ════════════════════════════════════════════════════════════════════════

def insert_chunks(chunks: List[dict]) -> int:
    """Insert augmented chunks into PostgreSQL as unindexed rows."""
    from rag.store import ensure_schema, insert_chunk

    ensure_schema()
    inserted = 0

    for c in chunks:
        meta = c.get("metadata", {})
        site = meta.get("site", "").strip().lower()
        title = meta.get("title", "").strip().lower()
        section = (meta.get("section_title") or meta.get("header") or "").strip().lower()
        style = meta.get("synthetic_style", "")
        doc_key = f"{site}|{title}|{section}|{style}" if style else f"{site}|{title}|{section}"

        ok = insert_chunk(
            doc_key=doc_key,
            page_content=c.get("page_content", ""),
            enriched_content=c.get("enriched_content", c.get("page_content", "")),
            bm25_text=c.get("bm25_text", ""),
            metadata=meta,
        )
        if ok:
            inserted += 1

    return inserted


# ════════════════════════════════════════════════════════════════════════
# STEP 5-10: Full pipeline via rag.index
# ════════════════════════════════════════════════════════════════════════

def run_pipeline(use_graph: bool, gen_synthetic: bool, synth_limit: int):
    """Run the full incremental pipeline on all unindexed chunks."""
    from rag.pipeline import index_unindexed

    print("\n" + "=" * 60)
    print("RUNNING FULL PIPELINE (synth gen -> embed -> KGE -> BM25)")
    print("=" * 60)

    index_unindexed(
        use_graph=use_graph,
        gen_synthetic=gen_synthetic,
        synth_limit=synth_limit,
    )


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Upload a document -> full pipeline to searchable in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python upload_doc.py document.pdf
    python upload_doc.py dossier.docx --site ISB --title "Rapport Stage"
    python upload_doc.py notes.txt --skip-pipeline   # just insert, don't index
    python upload_doc.py document.pdf --no-synth      # skip synthetic gen
    python upload_doc.py document.pdf --no-kge        # skip Neo4j extraction
        """,
    )
    parser.add_argument("file", help="Path to the document (.pdf, .docx, .txt)")
    parser.add_argument("--site", default=None, help="Site: IIT or ISB (auto-detected from title if not set)")
    parser.add_argument("--title", default=None, help="Title override (default: filename)")
    parser.add_argument("--category", default="Website Content", help="Category (default: Website Content)")
    parser.add_argument("--url", default="", help="Source URL (optional)")
    parser.add_argument("--no-synth", action="store_true", help="Skip synthetic generation")
    parser.add_argument("--no-kge", action="store_true", help="Skip Neo4j knowledge graph extraction")
    parser.add_argument("--synth-limit", type=int, default=50, help="Max synthetic variants per doc")
    parser.add_argument("--skip-pipeline", action="store_true", help="Only insert into DB, don't run pipeline")

    args = parser.parse_args()

    file_path = args.file
    if not Path(file_path).exists():
        print(f"[error] File not found: {file_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"UPLOADING: {Path(file_path).name}")
    print(f"{'=' * 60}")

    # ── Step 0: Extract text ───────────────────────────────────────────
    print("\n[1/6] Extracting text...")
    try:
        raw_text = extract_text(file_path)
    except Exception as e:
        print(f"[error] Text extraction failed: {e}")
        sys.exit(1)

    if not raw_text.strip():
        print("[error] No text extracted from the document")
        sys.exit(1)
    print(f"  {len(raw_text)} characters extracted")

    # ── Metadata from filename and args ─────────────────────────────────
    stem = Path(file_path).stem
    title = args.title or stem.replace("_", " ").replace("-", " ")[:80]
    site = args.site or ("ISB" if "isb" in stem.lower() else "IIT")
    url = args.url or f"upload://{Path(file_path).name}"

    # ── Step 1: Clean ──────────────────────────────────────────────────
    print("\n[2/6] Cleaning text...")
    cleaned = clean_text(raw_text, site)
    print(f"  {len(cleaned)} characters after cleaning")

    # ── Step 2: Chunk ──────────────────────────────────────────────────
    print("\n[3/6] Chunking...")
    chunks = chunk_text(cleaned, site, title, args.category, url)
    if not chunks:
        print("[error] No chunks produced - text may be too short")
        sys.exit(1)

    # ── Step 3: Augment ─────────────────────────────────────────────────
    print(f"\n[4/6] Augmenting {len(chunks)} chunks with LLM (this may take a while)...")
    augmented = augment_chunks(chunks)

    # ── Step 4: Insert ──────────────────────────────────────────────────
    print("\n[5/6] Inserting into PostgreSQL...")
    n = insert_chunks(augmented)
    print(f"  {n} new chunks inserted")

    # ── Steps 5-10: Pipeline ────────────────────────────────────────────
    if args.skip_pipeline:
        print("\n[6/6] Skipped pipeline (--skip-pipeline).")
        print(f"\n  Run manually: python -m rag.index")
    else:
        print(f"\n[6/6] Running full pipeline...")
        run_pipeline(
            use_graph=not args.no_kge,
            gen_synthetic=not args.no_synth,
            synth_limit=args.synth_limit,
        )

    # ── Done ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"DONE - {Path(file_path).name} is now searchable!")
    print(f"{'=' * 60}")
    print(f"  File    : {file_path}")
    print(f"  Site    : {site}")
    print(f"  Title   : {title}")
    print(f"  Chunks  : {len(chunks)} (augmented)")
    print(f"  Inserted: {n}")
    print(f"  Synth   : {'yes' if not args.no_synth else 'no'}")
    print(f"  KGE     : {'yes' if not args.no_kge else 'no'}")
    print(f"\n  Try: python -m rag.query \"{title[:60]}\"")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
