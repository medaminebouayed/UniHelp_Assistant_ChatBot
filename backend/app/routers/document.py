import io
import os
import shutil
import tempfile
from pathlib import Path
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import httpx
from app.database import get_db
from app.models.document import Document
from app.schemas.document import DocumentCreate, DocumentOut
from app.utils.security import require_role

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8002")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000/api")

SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".txt", ".json"}
DOC_SCAN_DIRS = [
    PROJECT_ROOT / "iit" / "scraped_data_categorized",
    PROJECT_ROOT / "isb" / "scraped_data_categorized",
]

router = APIRouter(prefix="/documents", tags=["Documents"])


def _path_to_category(file_path: Path) -> str:
    parts = file_path.relative_to(PROJECT_ROOT).parts
    if "scraped_data_categorized" in parts:
        idx = parts.index("scraped_data_categorized") + 1
        if idx < len(parts):
            return parts[idx]
    return "autre"


def _path_to_site(file_path: Path) -> str:
    rel = str(file_path.relative_to(PROJECT_ROOT))
    if rel.startswith("iit"):
        return "IIT"
    if rel.startswith("isb"):
        return "ISB"
    return ""


def _path_to_source(file_path: Path) -> str:
    rel = str(file_path.relative_to(PROJECT_ROOT))
    if "scraped_data_categorized" in rel:
        return "scraped"
    if "uploads" in rel:
        return "uploaded"
    return "synced"


def _call_rag_process(file_path: str, site: str, title: str, category: str, url: str = "") -> dict:
    """Call RAG API /process and return the response JSON."""
    body = {
        "file_path": file_path,
        "site": site,
        "title": title,
        "category": category,
    }
    if url:
        body["url"] = url

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{RAG_API_URL}/process", json=body)
        resp.raise_for_status()
        return resp.json()


async def _call_rag_process_async(file_path: str, site: str, title: str, category: str, url: str = "") -> dict:
    body = {
        "file_path": file_path,
        "site": site,
        "title": title,
        "category": category,
    }
    if url:
        body["url"] = url

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{RAG_API_URL}/process", json=body)
        resp.raise_for_status()
        return resp.json()


def _call_rag_index():
    """Call RAG API /index-unindexed."""
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{RAG_API_URL}/index-unindexed",
            json={"use_graph": True, "gen_synthetic": True, "synth_limit": 20},
        )
        resp.raise_for_status()
        return resp.json()


async def _call_rag_index_async():
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{RAG_API_URL}/index-unindexed",
            json={"use_graph": True, "gen_synthetic": True, "synth_limit": 20},
        )
        resp.raise_for_status()
        return resp.json()


def _call_rag_delete_chunks(site: str, title: str) -> int:
    """Delete RAG chunks for a document. Returns number deleted."""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.delete(
                f"{RAG_API_URL}/chunks",
                json={"site": site, "title": title},
            )
            if resp.status_code == 200:
                return resp.json().get("deleted", 0)
    except Exception as e:
        print(f"[rag] Warning: could not delete chunks: {e}")
    return 0


def _process_file_with_rag(doc_id: int, temp_path: Path, site: str, title: str, category: str) -> dict:
    """Write temp file → call RAG /process + /index-unindexed → clean up temp file."""
    dl_url = f"{PUBLIC_API_URL}/documents/{doc_id}/download"
    try:
        proc_data = _call_rag_process(str(temp_path), site, title, category, url=dl_url)
        if proc_data.get("status") != "ok":
            return {
                "status": "error",
                "detail": proc_data.get("detail", "Echec pipeline"),
                "chunks": 0,
                "inserted": 0,
            }
        index_data = _call_rag_index()
        return {
            "status": "ok",
            "detail": "Indexation reussie",
            "chunks": proc_data.get("chunks", 0),
            "inserted": proc_data.get("inserted", 0),
        }
    except httpx.RequestError as e:
        return {"status": "error", "detail": f"RAG API inaccessible: {e}", "chunks": 0, "inserted": 0}
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


async def _process_file_with_rag_async(doc_id: int, temp_path: Path, site: str, title: str, category: str) -> dict:
    dl_url = f"{PUBLIC_API_URL}/documents/{doc_id}/download"
    try:
        proc_data = await _call_rag_process_async(str(temp_path), site, title, category, url=dl_url)
        if proc_data.get("status") != "ok":
            return {
                "status": "error",
                "detail": proc_data.get("detail", "Echec pipeline"),
                "chunks": 0,
                "inserted": 0,
            }
        index_data = await _call_rag_index_async()
        return {
            "status": "ok",
            "detail": "Indexation reussie",
            "chunks": proc_data.get("chunks", 0),
            "inserted": proc_data.get("inserted", 0),
        }
    except httpx.RequestError as e:
        return {"status": "error", "detail": f"RAG API inaccessible: {e}", "chunks": 0, "inserted": 0}
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# ── CREATE (json body) ────────────────────────────────────────────────

@router.post("", response_model=DocumentOut, status_code=201)
def create(body: DocumentCreate, db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    data = body.model_dump()
    if data.get("administrateur_id") is None:
        data["administrateur_id"] = int(_["sub"])
    doc = Document(**data)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


# ── UPLOAD (multipart) ────────────────────────────────────────────────

@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    site: str = Form("IIT"),
    title: Optional[str] = Form(None),
    category: str = Form("Website Content"),
    run_pipeline: bool = Form(True),
    db: Session = Depends(get_db),
    _=Depends(require_role("admin")),
):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail="Format non supporte. Utilisez PDF, DOCX, TXT ou JSON.")

    raw_bytes = await file.read()
    file_size_kb = round(len(raw_bytes) / 1024, 2)
    doc_title = title or Path(file.filename).stem

    doc = Document(
        titre=doc_title,
        site=site,
        source="uploaded",
        categorie=category,
        fichier=None,
        contenu=raw_bytes,
        nom_fichier=file.filename,
        taille=file_size_kb,
        dateAjout=date.today(),
        administrateur_id=int(_["sub"]),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    if not run_pipeline:
        return {
            "message": "Fichier uploade sans indexation",
            "document": {"id": doc.idDocument, "titre": doc.titre, "nom_fichier": doc.nom_fichier},
        }

    # Write temp file for RAG processing
    temp_path = UPLOAD_DIR / f"__rag_{doc.idDocument}_{file.filename}"
    try:
        temp_path.write_bytes(raw_bytes)
        result = await _process_file_with_rag_async(doc.idDocument, temp_path, site, doc_title, category)
        if result["status"] == "ok":
            return {
                "message": "Document uploade et indexe avec succes",
                "document": {"id": doc.idDocument, "titre": doc.titre, "nom_fichier": doc.nom_fichier},
                "pipeline": {"chunks": result["chunks"], "inserted": result["inserted"]},
            }
        return {
            "message": f"Fichier uploade mais echec pipeline: {result['detail']}",
            "document": {"id": doc.idDocument, "titre": doc.titre, "nom_fichier": doc.nom_fichier},
        }
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return {
            "message": f"Fichier uploade mais erreur: {e}",
            "document": {"id": doc.idDocument, "titre": doc.titre, "nom_fichier": doc.nom_fichier},
        }


# ── RE-PROCESS (RAG pipeline on existing doc) ─────────────────────────

@router.post("/process/{doc_id}")
def process_document(doc_id: int, db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    doc = db.query(Document).filter(Document.idDocument == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouve")

    if doc.contenu:
        # Bytea doc — write temp file
        fname = doc.nom_fichier or f"doc_{doc_id}"
        temp_path = UPLOAD_DIR / f"__rag_{doc_id}_{fname}"
        try:
            temp_path.write_bytes(bytes(doc.contenu))
            result = _process_file_with_rag(doc_id, temp_path, doc.site or "IIT", doc.titre or "", doc.categorie or "")
            if result["status"] != "ok":
                raise HTTPException(status_code=500, detail=result["detail"])
            return {"message": "Document re-indexe avec succes", "chunks": result["chunks"], "inserted": result["inserted"]}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
    elif doc.fichier and Path(doc.fichier).exists():
        # Legacy file-path doc
        dl_url = f"{PUBLIC_API_URL}/documents/{doc_id}/download" if doc.source == "uploaded" else ""
        try:
            proc_data = _call_rag_process(doc.fichier, doc.site or "IIT", doc.titre or "", doc.categorie or "", url=dl_url)
            if proc_data.get("status") != "ok":
                raise HTTPException(status_code=500, detail=proc_data.get("detail", "Echec pipeline"))
            _call_rag_index()
            return {"message": "Document re-indexe avec succes", "chunks": proc_data.get("chunks", 0), "inserted": proc_data.get("inserted", 0)}
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"RAG API inaccessible: {e}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Aucun fichier disponible (ni contenu bytea, ni chemin fichier valide)")


# ── UPDATE FILE (PUT) ─────────────────────────────────────────────────

@router.put("/{doc_id}/file", status_code=200)
async def update_document_file(
    doc_id: int,
    file: Optional[UploadFile] = File(None),
    site: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    _=Depends(require_role("admin")),
):
    doc = db.query(Document).filter(Document.idDocument == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouve")

    # Update metadata fields
    if site is not None:
        doc.site = site
    if title is not None:
        doc.titre = title
    if category is not None:
        doc.categorie = category
    db.commit()

    if not file:
        return {
            "message": "Document mis a jour (metadonnees)",
            "document": {"id": doc.idDocument, "titre": doc.titre, "nom_fichier": doc.nom_fichier},
        }

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail="Format non supporte. Utilisez PDF, DOCX, TXT ou JSON.")

    old_site = doc.site or ""
    old_title = doc.titre or ""

    raw_bytes = await file.read()
    doc.contenu = raw_bytes
    doc.nom_fichier = file.filename
    doc.taille = round(len(raw_bytes) / 1024, 2)
    doc.fichier = None  # mark as bytea doc
    db.commit()
    db.refresh(doc)

    # Delete old RAG chunks
    if old_site and old_title:
        deleted = _call_rag_delete_chunks(old_site, old_title)
        print(f"[update] Deleted {deleted} old chunks for '{old_site}/{old_title}'")

    # Re-process with new file
    temp_path = UPLOAD_DIR / f"__rag_{doc_id}_{file.filename}"
    try:
        temp_path.write_bytes(raw_bytes)
        result = await _process_file_with_rag_async(doc_id, temp_path, doc.site or "IIT", doc.titre or "", doc.categorie or "")
        if result["status"] != "ok":
            return {"message": f"Fichier mis a jour mais echec pipeline: {result['detail']}", "document": {"id": doc.idDocument, "titre": doc.titre, "nom_fichier": doc.nom_fichier}}
        return {"message": "Document mis a jour et re-indexe avec succes", "document": {"id": doc.idDocument, "titre": doc.titre, "nom_fichier": doc.nom_fichier}, "pipeline": {"chunks": result["chunks"], "inserted": result["inserted"]}}
    except Exception as e:
        return {"message": f"Fichier mis a jour mais erreur indexation: {e}", "document": {"id": doc.idDocument, "titre": doc.titre, "nom_fichier": doc.nom_fichier}}
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# ── BACKGROUND RAG PROCESSING (for sync) ──────────────────────────────

def _process_doc_rag(doc_id: int):
    """Run RAG pipeline on a document in background via RAG API."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.idDocument == doc_id).first()
        if not doc:
            print(f"[background] Document {doc_id} not found")
            return

        if doc.contenu:
            fname = doc.nom_fichier or f"doc_{doc_id}"
            temp_path = UPLOAD_DIR / f"__rag_{doc_id}_{fname}"
            try:
                temp_path.write_bytes(bytes(doc.contenu))
                result = _process_file_with_rag(doc_id, temp_path, doc.site or "IIT", doc.titre or "", doc.categorie or "")
                print(f"[background] Processed doc {doc_id}: {result.get('inserted', 0)} chunks inserted")
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
        elif doc.fichier and Path(doc.fichier).exists():
            with httpx.Client(timeout=600.0) as client:
                proc_resp = client.post(
                    f"{RAG_API_URL}/process",
                    json={
                        "file_path": doc.fichier,
                        "site": doc.site or "IIT",
                        "title": doc.titre or "",
                        "category": doc.categorie or "",
                    },
                )
                proc_resp.raise_for_status()
                proc_data = proc_resp.json()
                print(f"[background] Processed doc {doc_id}: {proc_data.get('inserted', 0)} chunks inserted")

                if proc_data.get("inserted", 0) > 0:
                    index_resp = client.post(
                        f"{RAG_API_URL}/index-unindexed",
                        json={"use_graph": True, "gen_synthetic": True, "synth_limit": 20},
                    )
                    index_resp.raise_for_status()
                    print(f"[background] Indexing complete for doc {doc_id}")
    except Exception as e:
        print(f"[background] Error processing doc {doc_id}: {e}")
    finally:
        db.close()


# ── SYNC (scraped files from disk) ────────────────────────────────────

@router.post("/sync", status_code=201)
def sync_documents(background_tasks: BackgroundTasks, db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    known_paths = {
        Path(r.fichier).resolve() for r in db.query(Document.fichier).all() if r.fichier
    }
    found = []
    skipped = 0

    for base_dir in DOC_SCAN_DIRS:
        if not base_dir.exists():
            continue
        for ext in SUPPORTED_EXTS:
            for fp in base_dir.rglob(f"*{ext}"):
                resolved = fp.resolve()
                if resolved in known_paths:
                    skipped += 1
                    continue
                site = _path_to_site(fp)
                cat = _path_to_category(fp)
                title = fp.stem.replace("_", " ").replace("-", " ")[:255]

                from app.models.administrateur import Administrateur
                admin = db.query(Administrateur).first()
                admin_id = admin.id if admin else 1

                doc = Document(
                    titre=title,
                    site=site,
                    source="scraped",
                    categorie=cat,
                    fichier=str(fp),
                    taille=round(fp.stat().st_size / 1024, 2),
                    dateAjout=date.today(),
                    administrateur_id=admin_id,
                )
                db.add(doc)
                db.flush()
                found.append((doc.idDocument, doc.titre))

    db.commit()

    for doc_id, doc_title in found:
        background_tasks.add_task(_process_doc_rag, doc_id)

    return {
        "message": f"Sync complete: {len(found)} nouveaux, {skipped} deja presents",
        "nouveaux": len(found),
        "deja_presents": skipped,
    }


# ── LIST ──────────────────────────────────────────────────────────────

@router.get("", response_model=List[DocumentOut])
def get_all(
    site: Optional[str] = Query(None, description="Filtrer par site (IIT/ISB)"),
    source: Optional[str] = Query(None, description="Filtrer par source (scraped/uploaded/synced)"),
    categorie: Optional[str] = Query(None, description="Filtrer par categorie"),
    search: Optional[str] = Query(None, description="Recherche dans le titre"),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(Document)
    if site:
        q = q.filter(Document.site.ilike(site))
    if source:
        q = q.filter(Document.source == source)
    if categorie:
        q = q.filter(Document.categorie.ilike(f"%{categorie}%"))
    if search:
        q = q.filter(Document.titre.ilike(f"%{search}%"))
    total = q.count()
    docs = q.order_by(Document.dateAjout.desc()).offset(offset).limit(limit).all()
    return docs


# ── STATS ─────────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func
    total = db.query(func.count(Document.idDocument)).scalar()
    by_site = db.query(Document.site, func.count(Document.idDocument)).group_by(Document.site).all()
    by_source = db.query(Document.source, func.count(Document.idDocument)).group_by(Document.source).all()
    return {
        "total": total,
        "par_site": {s or "?" : c for s, c in by_site},
        "par_source": {s or "?" : c for s, c in by_source},
    }


# ── GET ONE ───────────────────────────────────────────────────────────

@router.get("/{doc_id}", response_model=DocumentOut)
def get_one(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.idDocument == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouve")
    return doc


# ── DOWNLOAD ──────────────────────────────────────────────────────────

@router.get("/{doc_id}/download")
def download_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.idDocument == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouve")

    if doc.contenu:
        # Serve from bytea
        fname = doc.nom_fichier or f"document_{doc_id}"
        media_type = "application/octet-stream"
        return StreamingResponse(
            io.BytesIO(bytes(doc.contenu)),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    if doc.fichier and Path(doc.fichier).exists():
        filename = Path(doc.fichier).name
        return FileResponse(doc.fichier, filename=filename, media_type="application/octet-stream")

    raise HTTPException(status_code=404, detail="Fichier introuvable")


# ── DELETE ────────────────────────────────────────────────────────────

@router.delete("/{doc_id}")
def delete(doc_id: int, db: Session = Depends(get_db), _=Depends(require_role("admin"))):
    doc = db.query(Document).filter(Document.idDocument == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouve")

    site = doc.site or ""
    title = doc.titre or ""
    fichier_path = doc.fichier

    db.delete(doc)
    db.commit()

    n_chunks = _call_rag_delete_chunks(site, title) if site and title else 0

    # Only delete from disk if it was a file-path based doc (legacy scraped/uploaded)
    if fichier_path and Path(fichier_path).exists():
        Path(fichier_path).unlink(missing_ok=True)

    return {
        "message": "Document supprime",
        "chunks_supprimes": n_chunks,
    }
